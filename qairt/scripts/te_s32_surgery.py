# -*- coding: utf-8 -*-
"""Tier 1 步1：把 text_encoder 四段的序列长度常量从 20 改成 32。

方案见 scripts/EXP_PLAN_PROMPT_LEN.md。

**为什么不是只改图尾 4 个常量**（2026-08-27 实测教训）：
只改图尾后转换直接失败——
  `ReshapeOp::calculateShape ... node_view_108: input 131072 != output 130560`
根因是注意力层里还有一批 baked 的形状常量把 20 写死了。
`HANDOVER §15.21.5` 早就记过「ONNX 内部 Reshape 常量 {1,20,-1,128}」，
我读到过却没连起来 —— 与本项目多次记录的同类失误一致。

**实测的常量清单（每段恰好 22 个含 20 的小整数常量）**：

    (1, 20, -1, 128)    [B, seq, heads, dim]              ×1
    (1, 32, 20, 128)    [B, heads=32, seq, dim]           ×1
    (1, 8, 4, 20, 128)  GQA [B, kv=8, rep=4, seq, dim]    ×1
    (-1, 20, 128)       [B*heads, seq, dim]               ×8~9
    (1, 32, 128, 20)    转置后 [B, heads, dim, seq]        ×8~9
    (1, 20, -1)         [B, seq, hidden]                  ×1
    (20,)               part4 图尾 Slice 的 end            ×2

🔴 **陷阱（约束 8：操作化先验证）**：上表里的 `32` 是**注意力头数**，与目标 seq=32 撞值。
替换规则因此必须是「**只把值 20 改成 32**」，逐元素做，**绝不能按位置或按整条形状替换**。
已实测确认：除 part4 的 `val_5778=[19]`（其所属节点在本脚本里被删除）外，
不存在其他 seq 派生常量（无 19/21/40/400）。

part4 额外做图尾重构：原本 `caption = Concat(hidden[:,0:20,:], Expand(hidden[:,19:20,:],[1,12,1]))`
（20 真 + 12 份复制），改为 `caption = hidden[:,0:32,:]`（32 真）。**输出名必须保持 caption。**
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SRC_DIR = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
OLD_LEN = 20
# 目标槽数可由命令行给出（默认 32）。2026-08-27 起 Tier 2 用 80。
NEW_LEN = int(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else 32
EXPECT_CONSTS = 22          # 每段含 20 的小整数常量个数（实测）
PARTS = ['text_encoder_part1', 'text_encoder_part2',
         'text_encoder_part3', 'text_encoder_part4']


def small_int_inits(g):
    for t in g.initializer:
        if t.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
            continue
        a = numpy_helper.to_array(t)
        if 0 < a.size <= 16:
            yield t, a


def retarget_seq_consts(g):
    """把所有小整数常量里的值 20 逐元素改成 32；返回改动数与改动明细。"""
    changed = []
    for t, a in list(small_int_inits(g)):
        flat = a.reshape(-1)
        if OLD_LEN not in flat.tolist():
            continue
        before = flat.tolist()
        new = np.where(a == OLD_LEN, NEW_LEN, a).astype(a.dtype)
        # 断言：只有等于 20 的位置变了，其余逐元素不变（防止误伤头数 32）
        mask = (a != OLD_LEN)
        assert np.array_equal(new[mask], a[mask]), t.name
        assert (new[~mask] == NEW_LEN).all(), t.name
        t.CopyFrom(numpy_helper.from_array(new, name=t.name))
        changed.append((t.name, before, new.reshape(-1).tolist()))
    return changed


def rebuild_part4_tail(g):
    """caption = Concat(slice[0:20], Expand(slice[19:20],[1,12,1]))  ->  slice[0:32]"""
    names = {n.name for n in g.node}
    for need in ('node_slice_221', 'node_slice_220', 'node_expand_74', 'node_cat_145'):
        if need not in names:
            raise RuntimeError('part4 缺少节点 %s，源图与方案记录不符' % need)

    def consumers(t):
        return [n.name for n in g.node if t in n.input]
    for t, exp in (('slice_221', ['node_cat_145']),
                   ('slice_220', ['node_expand_74']),
                   ('expand_74', ['node_cat_145'])):
        got = consumers(t)
        if got != exp:
            raise RuntimeError('%s 的消费者是 %s，预期 %s' % (t, got, exp))

    keep = [n for n in g.node
            if n.name not in ('node_slice_220', 'node_expand_74', 'node_cat_145')]
    del g.node[:]
    g.node.extend(keep)
    for n in g.node:
        if n.name == 'node_slice_221':
            n.output[0] = 'caption'
            break
    for o in g.output:
        if o.name == 'caption':
            d = o.type.tensor_type.shape.dim
            d[1].Clear()
            d[1].dim_param = 'seq_len'


def main():
    rc = 0
    for part in PARTS:
        src = os.path.join(SRC_DIR, part + '.onnx')
        dst = os.path.join(SRC_DIR, part + ('_s32.onnx' if NEW_LEN == 32 else '_L%d.onnx' % NEW_LEN))
        m = onnx.load(src, load_external_data=False)
        g = m.graph

        n_before = sum(1 for _, a in small_int_inits(g)
                       if OLD_LEN in a.reshape(-1).tolist())
        if n_before != EXPECT_CONSTS:
            print('FAIL %s: 含 %d 的小整数常量有 %d 个，方案记录是 %d ⇒ 源图已变，停止'
                  % (part, OLD_LEN, n_before, EXPECT_CONSTS))
            return 1

        changed = retarget_seq_consts(g)
        if part.endswith('part4'):
            rebuild_part4_tail(g)

        n_after = sum(1 for _, a in small_int_inits(g)
                      if OLD_LEN in a.reshape(-1).tolist())
        if n_after != 0:
            print('FAIL %s: 改完仍有 %d 个常量含 20' % (part, n_after))
            return 1

        cwd = os.getcwd()
        os.chdir(SRC_DIR)
        try:
            onnx.checker.check_model(m, full_check=False)
        finally:
            os.chdir(cwd)

        produced = {o for n in g.node for o in n.output}
        init_names = {t.name for t in g.initializer}
        in_names = {i.name for i in g.input}
        dangling = sorted({t for n in g.node for t in n.input
                           if t and t not in produced and t not in init_names
                           and t not in in_names})
        if dangling:
            print('FAIL %s: 悬空输入 %s' % (part, dangling[:5]))
            return 1
        for o in g.output:
            if o.name not in produced:
                print('FAIL %s: 图输出 %s 无人产出' % (part, o.name))
                return 1

        onnx.save(m, dst)
        print('%-20s 改了 %2d 个常量, 节点 %d, checker/完整性通过 -> %s'
              % (part, len(changed), len(g.node), os.path.basename(dst)))
    return rc


if __name__ == '__main__':
    sys.exit(main())
