# -*- coding: utf-8 -*-
"""在 FP32 TE 上按各版本的 encoding 插 QDQ，直接测「量化后的 caption 误差」。

## 为什么需要它（2026-08-27）

20 槽 18.763 dB / v1 17.486 / v2 17.025 / v3 16.024 —— 三次换校准配方都更差，
但我一直**没有量过真正的因变量**：量化后的 caption 相对 FP32 差多少。
成图 PSNR 是被 8 步扩散 + VAE 放大后的结果，无法定位，也只能一个 prompt 一个 prompt 地试。

原想用 `snpe-net-run` 在宿主跑量化 DLC，实测**封死**：
`error_code=1002 ... No backend could validate Op=node_embedding Type=Gather error code=3110`。

⇒ 改用 QDQ 模拟（同 #124 思路）：在 FP32 ONNX 上对每个有 encoding 的张量插
quantize-dequantize，用 ORT 跑，得到「算术完全正确的 A16」下的 caption。
**零设备成本，可一次跑几十个 prompt** ⇒ 同时回答「根因」与「普遍性」。

## 口径与边界（必须随结论一起说）

1. **只模拟激活，不模拟权重**。权重量化在各版本间是**共同因子**（同一份 fp32 DLC、
   同样的 per-channel min-max）⇒ 对**版本间比较**无影响。给出的是**相对排序**，
   不是绝对误差值。
2. **只覆盖 ONNX 里有同名张量的 encoding**（实测 part1：451/753 = 59.9%，其余是
   转换器生成的 `_fc`/`_pre_reshape`/`_converted_*`）。三版用**同一套覆盖集合**，
   比较仍成立。
3. QDQ 公式（与 QAIRT 一致，已用 `caption` 的 encoding 反算验证：
   `scale*(0+offset)=min`、`scale*(65535+offset)=max`）：
   `real = scale * (q + offset)`，`q ∈ [0, 65535]`
   ⇒ `x' = scale * clip(round(x/scale), offset, 65535+offset)`
   opset 17 无 uint16 QuantizeLinear ⇒ Div/Round/Clip/Mul 四件套。
4. 接线方式：把**生产者的输出改名**为 `<t>__pre`，QDQ 后仍产出原名 `<t>`
   ⇒ 消费者与图输出都不用改，段间衔接不受影响。

用法：python te_qdq_sim.py
"""
import os
import re
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from tokenizers import Tokenizer

ONNX = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
ENC = r'D:\LocalDreamZImage\logs\tier1_20260827'
TOK = r'D:\LocalDreamZImage\logs\dup126_20260826\tokenizer.json'
PAD = 151643
ENC_PAT = re.compile(r'([A-Za-z0-9_.]+) encoding : bitwidth (\d+), min ([-0-9.eE]+), '
                     r'max ([-0-9.eE]+), scale ([-0-9.eE+]+), offset ([-0-9.]+)')
CHAIN_IN = {1: ['input_ids', 'attention_mask'], 2: ['add_2452', 'attention_mask'],
            3: ['add_4828', 'attention_mask'], 4: ['add_7204', 'attention_mask']}
VARIANTS = [('20 槽（部署版）', 'enc20', 20, ''),
            ('32 槽 v1 复制末行', 'encv1', 32, '_s32'),
            ('32 槽 v3 满槽校准', 'encv3', 32, '_s32')]
PROMPTS = ['一条蓝色的鱼', '一只戴帽子的企鹅', '刘亦菲', '赵本山',
           '一位年轻的东方女性，站姿，海滩', '男人戴着一枚小的银色胸针',
           '一个红色的立方体在一个蓝色的球体里面', '一只橘色的猫坐在木桌上',
           'A cinematic shot of a cute cat sitting on a wooden desk',
           'Portrait of a young woman with flowing hair in golden hour light',
           'Aerial view of a misty mountain forest at dawn',
           'Cozy living room interior with warm lighting and bookshelves']


def load_enc(tag, part):
    p = os.path.join(ENC, '%s_p%d.csv' % (tag, part))
    out = {}
    for m in ENC_PAT.finditer(open(p, encoding='utf-8', errors='replace').read()):
        out[m.group(1)] = (float(m.group(5)), float(m.group(6)))
    return out


def insert_qdq(model, enc):
    g = model.graph
    produced = {o: nd for nd in g.node for o in nd.output}
    targets = [t for t in produced if t in enc and enc[t][0] > 0]
    new_nodes, inits = [], []
    for k, t in enumerate(targets):
        scale, offset = enc[t]
        pre = t + '__pre'
        nd = produced[t]
        for i, o in enumerate(nd.output):
            if o == t:
                nd.output[i] = pre
        s = 'qdq%d_' % k
        cs, clo, chi = s + 's', s + 'lo', s + 'hi'
        inits += [numpy_helper.from_array(np.array(scale, np.float32), cs),
                  numpy_helper.from_array(np.array(float(offset), np.float32), clo),
                  numpy_helper.from_array(np.array(float(65535 + offset), np.float32), chi)]
        new_nodes += [
            helper.make_node('Div', [pre, cs], [s + 'd'], name=s + 'div'),
            helper.make_node('Round', [s + 'd'], [s + 'r'], name=s + 'rnd'),
            helper.make_node('Clip', [s + 'r', clo, chi], [s + 'c'], name=s + 'clip'),
            helper.make_node('Mul', [s + 'c', cs], [t], name=s + 'mul')]
    g.node.extend(new_nodes)
    g.initializer.extend(inits)
    # 拓扑排序：新节点追加在末尾，ORT 要求拓扑序
    ready = {i.name for i in g.input} | {i.name for i in g.initializer}
    order, pend = [], list(g.node)
    while pend:
        rest, prog = [], False
        for nd in pend:
            if all((x in ready) or x == '' for x in nd.input):
                order.append(nd); ready.update(nd.output); prog = True
            else:
                rest.append(nd)
        pend = rest
        if not prog:
            raise RuntimeError('拓扑排序卡住，剩 %d 节点' % len(pend))
    del g.node[:]
    g.node.extend(order)
    return len(targets)


def sessions(tag, suffix, so):
    made = []
    total = 0
    for p in (1, 2, 3, 4):
        src = 'text_encoder_part%d%s.onnx' % (p, suffix)
        if tag is None:
            made.append(ort.InferenceSession(src, so, providers=['CPUExecutionProvider']))
            continue
        m = onnx.load(src, load_external_data=False)
        total += insert_qdq(m, load_enc(tag, p))
        tmp = '_qdq_%s_p%d.onnx' % (tag, p)
        onnx.save(m, tmp)
        made.append(ort.InferenceSession(tmp, so, providers=['CPUExecutionProvider']))
    return made, total


def run(sess, ids, seq):
    n = len(ids)
    a = np.full((1, seq), PAD, dtype=np.int64); a[0, :n] = ids
    m = np.zeros((1, seq), dtype=np.int64); m[0, :n] = 1
    t = {'input_ids': a, 'attention_mask': m}
    for p, se in zip((1, 2, 3, 4), sess):
        f = {}
        for i in se.get_inputs():
            v = t[i.name]
            f[i.name] = (v.astype(np.int32) if 'int32' in i.type
                         else v.astype(np.float32) if 'float' in i.type else v)
        for o, v in zip(se.get_outputs(), se.run(None, f)):
            t[o.name] = v
    return t['caption'], n


def main():
    tk = Tokenizer.from_file(TOK)
    wrap = lambda p: "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    toks = []
    for p in PROMPTS:
        ids = tk.encode(wrap(p), add_special_tokens=False).ids
        if len(ids) > 20:
            print('  跳过（%d token 超过 20 槽，无法双方可比）: %r' % (len(ids), p))
            continue
        toks.append((p, ids))
    print('可比 prompt %d 个（都 ≤20 token，两种槽数下都不截断）\n' % len(toks))

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        base, _ = sessions(None, '_s32', so)
        ref = {p: run(base, ids, 32) for p, ids in toks}
        del base
        rows = []
        for name, tag, seq, suffix in VARIANTS:
            se, n_ins = sessions(tag, suffix, so)
            errs = []
            for p, ids in toks:
                cap, n = run(se, ids, seq)
                r = ref[p][0][0, :n]
                e = np.linalg.norm(cap[0, :n] - r) / np.linalg.norm(r) * 100
                errs.append(e)
            del se
            rows.append((name, n_ins, errs))
            print('%-20s 插入 QDQ %d 处   caption 相对误差 中位数 %.4f%%  均值 %.4f%%'
                  % (name, n_ins, float(np.median(errs)), float(np.mean(errs))))
        print('\n逐 prompt（caption 真实槽相对 FP32 的相对 L2 %%）：')
        print('  %-34s %10s %10s %10s' % ('prompt', '20槽', 'v1', 'v3'))
        for i, (p, _) in enumerate(toks):
            print('  %-34s %9.4f %9.4f %9.4f'
                  % (p[:32], rows[0][2][i], rows[1][2][i], rows[2][2][i]))
        best = [min(range(3), key=lambda k: rows[k][2][i]) for i in range(len(toks))]
        print('\n每个 prompt 谁最小: 20槽 %d 次 / v1 %d 次 / v3 %d 次'
              % (best.count(0), best.count(1), best.count(2)))
    finally:
        os.chdir(cwd)
    return 0


if __name__ == '__main__':
    sys.exit(main())
