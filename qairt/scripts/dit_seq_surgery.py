# -*- coding: utf-8 -*-
"""Tier 2：把 transformer 四段的 caption 槽数从 32 改成 L（默认 80），不重导 PyTorch。

## 依据（2026-08-27 实测，EXP_PLAN_PROMPT_LEN §九）

| 段 | 依赖 seq 的**数据张量** | 形状常量 |
|---|---|---|
| part1a | 8 个（全是 0 或等差 `1..32`） | 19 |
| part1b / 2a / 2b | **0 个** | 各 5（全含 4128） |

part1a 的 8 个数据张量实测取值：
`val_572`/`new_full_3`/`val_566`/`val_534` **全 0**；`val_520` = **1,2,…,32**（等差）；
`unsqueeze_3` [4096,1] 全 False（4096 是图像 token 数，不变）。
⇒ **没有按位置学到的参数**，可程序化扩展。

## 🔴 替换规则（逐个追消费者后定稿，不是假设）

    32   -> L            （caption 槽数；注意力头数是 **30**，故 32 无歧义）
    4128 -> 4096 + L     （unified = 图像 4096 + caption）
    4096 -> **不动**      （图像 token 数）

⚠️ `32*128 = 4096` 与图像 token 数撞值。已逐个追消费者确认：所有含 4096 的常量都出现在
**序列位**（`(4096,64)`、`(-1,4096,128)`、`(1,4096,30,128)`、`(1,30,128,4096)`、
`(4096,3)`、`(1,4096,3840)`），**全属图像流** ⇒ 一律不动。

## 🟢 不重导的最大好处

**张量名不变** ⇒ §29.8 那套值 +6.74 dB 的 Clip+FP16 白名单继续适用。

用法：python dit_seq_surgery.py [L] [--variant=base|deployed]
      源文件名由 `canonical_sources.py` 给出，**本脚本不得自己拼**（台账 #138）
"""
import os
import sys

import numpy as np
import onnx
from onnx import external_data_helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources          # noqa: E402  # 唯一的源文件名来源

ONNX = canonical_sources.ONNX_DIR
OLD_CAP, IMG = 32, 4096
SEGS = ['part1a', 'part1b', 'part2a', 'part2b']

# 🔴 2026-08-28（台账 #138）：这里原来是**手写清单**
#     PARTS = ['transformer_part1a', 'transformer_part1b',
#              'transformer_part2a', 'transformer_part2b']
# part2a/2b 因此指向 `transformer_part2a.onnx`，而权威 base 是
# `transformer_part2a_fixed.onnx`（实测 1159917 vs 1136771 字节，确实不同）
# ⇒ 同一个取错源的错误的第三次（#61 缺 `_fixed`、#136 缺 `_clip`）。
# 教训：**建了权威模块不等于权威模块被用上** —— 本脚本压根没导入它。
# 现在源名一律由 canonical_sources 给出，脚本不许自己拼。


def resolve(variant):
    """返回 [(seg, 源 ONNX 主干, 绝对路径)]，主干只能来自 canonical_sources。"""
    out = []
    for seg in SEGS:
        p = canonical_sources.onnx_for(seg, variant)          # 不存在会抛错
        stem = os.path.splitext(os.path.basename(p))[0]
        assert stem == canonical_sources.ALL[seg][0 if variant == 'base' else 1],             '源主干 %s 不是 canonical_sources 给出的' % stem
        out.append((seg, stem, p))
    return out
# 🔴 不用手写名字清单 —— 2026-08-27 就是因为手写漏了 `new_full_2`（外部存储、
# dims=[1,32,3840] 全 0），转换失败在 node_view_31（输入 32*3840 != 输出 80*3840）。
# 改为**程序化穷举**：凡 dims 里含旧槽数的 initializer 一律处理，内容按规则外推，
# 遇到无法安全外推的立即停下报错，不猜。


def remap(vals, L):
    """逐元素：32->L, 4128->4096+L, 其余(含 4096)不动。"""
    out = []
    for v in vals:
        if v == OLD_CAP:
            out.append(L)
        elif v == IMG + OLD_CAP:
            out.append(IMG + L)
        else:
            out.append(v)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    variant = 'base'
    for a in sys.argv[1:]:
        if a.startswith('--variant='):
            variant = a.split('=', 1)[1]
    if variant not in ('base', 'deployed'):
        print('FAIL: --variant 只能是 base 或 deployed')
        return 1
    L = int(args[0]) if args else 80
    if L <= OLD_CAP:
        print('FAIL: L 必须 > %d' % OLD_CAP)
        return 1
    print('目标 caption 槽数 L = %d（可用正文 %d token），unified = %d' % (L, L - 8, IMG + L))
    srcs = resolve(variant)
    print('源（来自 canonical_sources，variant=%s）：' % variant)
    for seg, stem, _ in srcs:
        print('   %-8s <- %s.onnx' % (seg, stem))

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        for seg, part, _abs in srcs:
            m = onnx.load(part + '.onnx', load_external_data=False)
            g = m.graph
            n_shape = 0
            for t in g.initializer:
                if t.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                    continue
                if t.data_location == 1:
                    continue
                a = numpy_helper.to_array(t)
                if a.size == 0 or a.size > 16:
                    continue
                v = a.reshape(-1).tolist()
                if not any(x in (OLD_CAP, IMG + OLD_CAP) for x in v):
                    continue
                nv = np.array(remap(v, L), dtype=a.dtype).reshape(a.shape)
                t.CopyFrom(numpy_helper.from_array(nv, name=t.name))
                n_shape += 1

            n_data = 0
            for t in list(g.initializer):
                # 🔴 两种维度都要管：caption 槽数 32，以及 unified 长度 4128
                # （只查 32 会漏掉 `unified_mask` [1,4128]，2026-08-27 被断言抓到）
                if OLD_CAP not in list(t.dims) and (IMG + OLD_CAP) not in list(t.dims):
                    continue
                tt = onnx.TensorProto(); tt.CopyFrom(t)
                if tt.data_location == 1:
                    external_data_helper.load_external_data_for_tensor(tt, '.')
                a = numpy_helper.to_array(tt)
                dims = [L if d == OLD_CAP else (IMG + L) if d == IMG + OLD_CAP else d
                        for d in t.dims]
                u = np.unique(a)
                if u.size == 1:                       # 常量填充 -> 同值填满新形状
                    nv = np.full(dims, u[0], dtype=a.dtype)
                elif (a.dtype.kind == 'i' and a.size == OLD_CAP
                      and np.array_equal(a.reshape(-1), np.arange(1, OLD_CAP + 1))):
                    nv = np.arange(1, L + 1, dtype=a.dtype).reshape(dims)
                elif (a.dtype.kind == 'i' and a.size == OLD_CAP
                      and np.array_equal(a.reshape(-1), np.arange(0, OLD_CAP))):
                    nv = np.arange(0, L, dtype=a.dtype).reshape(dims)
                else:
                    print('FAIL %s: 张量 %s dims=%s 既非常量填充也非等差，'
                          '**无法安全外推**，停止（不猜）' % (part, t.name, list(t.dims)))
                    return 1
                t.CopyFrom(numpy_helper.from_array(nv, name=t.name))
                n_data += 1

            # 🔴 图输入/输出 **以及 `value_info`** 里带 32 / 4128 的维度都要改。
            # 2026-08-28（台账 #139）：原来只改了 g.input / g.output，**漏了 g.value_info**。
            # part1a/1b 恰好有 **0 条** value_info 所以没暴露；
            # part2a_fixed / part2b_fixed 各有约 1000 条，其中 793 / 740 条仍写 4128
            # ⇒ ORT 加载时 `node_mul` 报 `[ShapeInferenceError] Incompatible dimensions`。
            # 实测：part2a/2b 的 value_info 里 **一次都没出现过 32**，只有 4128 ⇒ 重映射无歧义。
            n_io = 0
            for vi in list(g.input) + list(g.output) + list(g.value_info):
                for d in vi.type.tensor_type.shape.dim:
                    if d.HasField('dim_value'):
                        if d.dim_value == OLD_CAP:
                            d.dim_value = L; n_io += 1
                        elif d.dim_value == IMG + OLD_CAP:
                            d.dim_value = IMG + L; n_io += 1

            # 断言：改完不得再有裸 32 / 4128 的小整数常量
            left = 0
            for t in g.initializer:
                if t.data_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32) \
                        and t.data_location != 1:
                    a = numpy_helper.to_array(t)
                    if 0 < a.size <= 16 and any(x in (OLD_CAP, IMG + OLD_CAP)
                                                for x in a.reshape(-1).tolist()):
                        left += 1
            for t in g.initializer:
                if OLD_CAP in list(t.dims) or (IMG + OLD_CAP) in list(t.dims):
                    print('FAIL %s: 张量 %s 的 dims 仍是 %s' % (part, t.name, list(t.dims)))
                    return 1
            for vi in list(g.input) + list(g.output) + list(g.value_info):
                bad = [d.dim_value for d in vi.type.tensor_type.shape.dim
                       if d.HasField('dim_value')
                       and d.dim_value in (OLD_CAP, IMG + OLD_CAP)]
                if bad:
                    print('FAIL %s: value_info/IO %s 仍含 %s' % (part, vi.name, bad))
                    return 1
            if left:
                print('FAIL %s: 仍有 %d 个常量含 32/4128' % (part, left))
                return 1

            onnx.checker.check_model(m, full_check=False)
            dst = '%s_L%d.onnx' % (part, L)
            onnx.save(m, dst)
            print('%-24s 形状常量 %2d  数据张量 %d  IO+value_info 维度 %4d  '
                  'value_info %4d  -> %s'
                  % (part, n_shape, n_data, n_io, len(g.value_info), dst))
    finally:
        os.chdir(cwd)
    return 0


if __name__ == '__main__':
    sys.exit(main())
