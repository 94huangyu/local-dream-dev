# -*- coding: utf-8 -*-
"""Tier 2 门 步 2：用 **FP32 ONNX text_encoder** 产出 L=80 的 caption / cap_pad_mask。

方案：`scripts/EXP_PLAN_T2_CLIPSET.md` §四 步 2。

## 为什么不用 `official_caption.py emit`
`emit` 走 **PyTorch bf16** 模型且对 pad 位**补零**；而部署链的 caption 来自
**我们导出的 FP32 ONNX TE**，pad 位是 TE 的真实输出（实测：盘上 `caption.raw`
第 20~31 行是第 19 行的逐字节副本，范数 482，**不是 0**）。
`linear_23` 是 `[1,32,3840]` 的 **caption-only** 张量且在钳位集合里
⇒ pad 行的取值会直接进它的 `|a|max` ⇒ **必须忠实复制部署链的产生方式**。

## 已知样本自检（约束 8）
`known` 模式跑**原 20 槽链**（`text_encoder_partN.onnx`）+ 盘上 `prompt_ids.npy`，
与部署的 `htp_inloop/const/caption.raw` 前 20 行比对。不过就不许往下走。

## cap_pad_mask 极性（实测，不是猜）
盘上 `cap_pad_mask.raw` = `[0]*20 + [1]*12` ⇒ **0 = 真实 token，1 = padding**，
与 TE 的 `attention_mask`（1=真实）**相反**。

用法:
    python t2_caption_L80.py known
    python t2_caption_L80.py emit
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import onnxruntime as ort                                    # noqa: E402

ONNX = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
P0 = r'D:\ZImage_Work\p0_experiments'
HIDDEN, PAD_ID = 2560, 151643
CHAIN_IN = [('input_ids', 'attention_mask'), ('add_2452', 'attention_mask'),
            ('add_4828', 'attention_mask'), ('add_7204', 'attention_mask')]


def run_chain(suffix, ids, slots):
    """逐段建 session、用完即释放（峰值内存更低）。返回 caption [1,slots,2560]。"""
    n = min(len(ids), slots)          # 超出槽数的部分被图内常量截断（#134 的本体）
    a = np.full((1, slots), PAD_ID, dtype=np.int64); a[0, :n] = ids[:n]
    m = np.zeros((1, slots), dtype=np.int64); m[0, :n] = 1
    t = {'input_ids': a, 'attention_mask': m}
    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        for p in (1, 2, 3, 4):
            f = 'text_encoder_part%d%s.onnx' % (p, suffix)
            se = ort.InferenceSession(f, so, providers=['CPUExecutionProvider'])
            feed = {}
            for inp in se.get_inputs():
                v = t[inp.name]
                feed[inp.name] = (v.astype(np.int32) if 'int32' in inp.type
                                  else v.astype(np.float32) if 'float' in inp.type else v)
            for o, v in zip(se.get_outputs(), se.run(None, feed)):
                t[o.name] = v
            del se
            print('   part%d ok' % p, flush=True)
    finally:
        os.chdir(cwd)
    return t['caption'], n


def known():
    ids = np.load(os.path.join(P0, 'prompt_ids.npy')).tolist()
    print('已知样本自检：原 20 槽链，prompt_ids.npy 共 %d token（超出 20 的会被图内截断）'
          % len(ids))
    cap, n = run_chain('', ids, 20)
    cap = np.asarray(cap, np.float32).reshape(-1, HIDDEN)
    ref = np.fromfile(os.path.join(P0, 'htp_inloop', 'const', 'caption.raw'),
                      np.float32).reshape(32, HIDDEN)
    k = min(20, cap.shape[0])
    a, b = cap[:k], ref[:k]
    same = np.array_equal(a, b)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print('\n复现 caption 形状 %s，与部署 caption.raw 前 %d 行比对：' % (cap.shape, k))
    print('  逐字节相同 = %s   相对 L2 = %.6f%%   最大绝对差 = %.3g'
          % (same, rel * 100, float(np.abs(a - b).max())))
    ok = same or rel < 1e-4
    print('  判定: %s' % ('✅ PASS —— 链路复现正确' if ok
                          else '🔴 FAIL —— 我的链路与部署链不一致，停止'))
    return 0 if ok else 1


def emit():
    ids = np.load(os.path.join(P0, 'prompt_ids_L80.npy')).tolist()
    slots = 80
    print('L=80：prompt 共 %d token（正文 %d + 模板 8），占 %d/%d 槽'
          % (len(ids), len(ids) - 8, len(ids), slots))
    cap, n = run_chain('_L80', ids, slots)
    cap = np.ascontiguousarray(np.asarray(cap, np.float32).reshape(-1, HIDDEN))
    assert cap.shape == (slots, HIDDEN), 'caption 形状 %s 不是 (%d,%d)' % (cap.shape, slots, HIDDEN)
    mask = np.ones((1, slots), np.float32); mask[0, :n] = 0.0      # 0=真实 1=pad（实测极性）
    cp = os.path.join(P0, 'caption_L80.raw'); cap.tofile(cp)
    mp = os.path.join(P0, 'cap_pad_mask_L80.raw'); mask.tofile(mp)
    assert os.path.getsize(cp) == slots * HIDDEN * 4
    assert os.path.getsize(mp) == slots * 4
    nz = [i for i in range(slots) if np.linalg.norm(cap[i]) > 0]
    print('\n写出 %s (%d B)  %s (%d B)' % (cp, os.path.getsize(cp), mp, os.path.getsize(mp)))
    print('  非零行 %d / %d   行范数 首=%.0f 末真实=%.0f 首pad=%.0f'
          % (len(nz), slots, np.linalg.norm(cap[0]), np.linalg.norm(cap[n - 1]),
             np.linalg.norm(cap[n]) if n < slots else float('nan')))
    print('  |cap|max = %.4f' % float(np.abs(cap).max()))
    return 0


if __name__ == '__main__':
    sys.exit({'known': known, 'emit': emit}[sys.argv[1]]())
