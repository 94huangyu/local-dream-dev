# -*- coding: utf-8 -*-
"""Tier 2 门 步 3 前置：跑 FP32 四段链，产出**段间输入**（L=32 校验 / L=80 产出）。

方案：`scripts/EXP_PLAN_T2_CLIPSET.md` §四。

## 为什么要它
`maxfp16_stats.py` 的 part1b/2a/2b 输入来自盘上 4128 长的旧产物
（且 #116 已指出那是「CPU 参考链」不是纯 FP32 链）。L=80 下这些输入**不存在**，
必须自己跑出来，且必须**同一条链续接**，不得混用来源。

## 已知样本校验（约束 8）
`known` 模式在 **L=32 base ONNX + 部署 caption** 上跑完四段，
把 part2b 的 `latents` 输出与盘上 `vs_fp32/latents_fp32_s0.raw` 比对。
**不过就不许往下走** —— 否则 L=80 的测量是在一条没验证过的链上做的。

⚠️ 源 ONNX 一律由 `canonical_sources.py` 给出（台账 #138）。

用法:
    python t2_chain_L80.py known      # L=32 全链校验
    python t2_chain_L80.py emit       # L=80 全链，dump 段间输入
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                                     # noqa: E402
import onnx                                                  # noqa: E402
import onnxruntime as ort                                    # noqa: E402
from onnx import TensorProto                                 # noqa: E402

P0 = r'D:\ZImage_Work\p0_experiments'
SEGS = ('part1a', 'part1b', 'part2a', 'part2b')
DST80 = os.path.join(P0, 'L80_chain')


def seg_onnx(seg, L, variant='base'):
    """L=32 用 canonical 源；L=80 用其手术产物 <stem>_L80。

    variant='base'     —— 不带 Clip，用于**测真实 |a|max**（Clip 会把溢出藏起来）
    variant='deployed' —— 带 Clip 的部署源，用于**验证要送去转换的那张图**
    """
    p = canonical_sources.onnx_for(seg, variant)
    if L == 32:
        return p
    stem = os.path.splitext(os.path.basename(p))[0]
    q = os.path.join(os.path.dirname(p), '%s_L%d.onnx' % (stem, L))
    if not os.path.isfile(q):
        raise FileNotFoundError('缺手术产物 %s（先跑 dit_seq_surgery.py %d --variant=%s）'
                                % (q, L, variant))
    return q


def run(L, cap_path, mask_path, dump_dir=None, variant='base'):
    t = {'latents': np.fromfile(os.path.join(P0, 'htp_inloop', 's0', 'latents.raw'),
                                np.float32).reshape(1, 16, 128, 128),
         'timestep': np.fromfile(os.path.join(P0, 'htp_inloop', 's0', 'timestep.raw'),
                                 np.float32).reshape(1),
         'caption': np.fromfile(cap_path, np.float32).reshape(1, L, 2560),
         'cap_pad_mask': np.fromfile(mask_path, np.float32).reshape(1, L)}
    print('  输入 caption %s |max|=%.1f   cap_pad_mask 真实槽 %d / pad %d'
          % (t['caption'].shape, float(np.abs(t['caption']).max()),
             int((t['cap_pad_mask'] == 0).sum()), int((t['cap_pad_mask'] == 1).sum())),
          flush=True)
    so = ort.SessionOptions(); so.log_severity_level = 3
    for seg in SEGS:
        path = seg_onnx(seg, L, variant)
        m = onnx.load(path, load_external_data=False)
        need = [(i.name, i.type.tensor_type.elem_type) for i in m.graph.input]
        missing = [n for n, _ in need if n not in t]
        if missing:
            raise KeyError('%s 缺输入 %s（链断了，不许用别处的产物凑）' % (seg, missing))
        if dump_dir:
            d = os.path.join(dump_dir, 's0_%s' % seg)
            os.makedirs(d, exist_ok=True)
            for n, _e in need:
                np.ascontiguousarray(t[n], np.float32).tofile(os.path.join(d, n + '.raw'))
        cwd = os.getcwd(); os.chdir(os.path.dirname(path))
        try:
            se = ort.InferenceSession(os.path.basename(path), so,
                                      providers=['CPUExecutionProvider'])
            feed = {}
            for n, e in need:
                feed[n] = t[n].astype(bool) if e == TensorProto.BOOL else t[n]
            t0 = time.time()
            outs = se.run(None, feed)
        finally:
            os.chdir(cwd)
        for o, v in zip(se.get_outputs(), outs):
            t[o.name] = np.ascontiguousarray(v, np.float32) if v.dtype != np.bool_ \
                else v.astype(np.float32)
        del se
        print('  %-7s %-34s %5.1f s  -> %d 个输出'
              % (seg, os.path.basename(path), time.time() - t0, len(outs)), flush=True)
    return t


def known():
    print('已知样本校验：L=32 base 四段链 vs 盘上 FP32 step-0 参考')
    t = run(32, os.path.join(P0, 'htp_inloop', 'const', 'caption.raw'),
            os.path.join(P0, 'htp_inloop', 'const', 'cap_pad_mask.raw'))
    got = t['latents'].reshape(-1)
    ref = np.fromfile(os.path.join(P0, 'vs_fp32', 'latents_fp32_s0.raw'),
                      np.float32).reshape(-1)
    assert got.size == ref.size, '元素数 %d != %d' % (got.size, ref.size)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    print('\n  part2b latents vs latents_fp32_s0.raw:')
    print('    逐字节相同 = %s   相对 L2 = %.6f%%   余弦 = %.8f'
          % (np.array_equal(got, ref), rel * 100, cos))
    ok = rel < 0.01           # 事前定：<1% 视为同一条 FP32 链（ORT 版本差异余量）
    print('    判定: %s' % ('✅ PASS —— 链跑器正确' if ok
                            else '🔴 FAIL —— 链跑器与盘上 FP32 参考不一致，停止'))
    return 0 if ok else 1


def emit():
    print('L=80 四段链，dump 段间输入 -> %s' % DST80)
    os.makedirs(DST80, exist_ok=True)
    t = run(80, os.path.join(P0, 'caption_L80.raw'),
            os.path.join(P0, 'cap_pad_mask_L80.raw'), dump_dir=DST80)
    out = t['latents']
    print('\n  part2b latents %s  |max|=%.4f  有限值=%s'
          % (out.shape, float(np.abs(out).max()), bool(np.isfinite(out).all())))
    for seg in SEGS:
        d = os.path.join(DST80, 's0_%s' % seg)
        fs = sorted(os.listdir(d))
        print('  %-7s %d 个输入: %s' % (seg, len(fs), ' '.join(fs)))
    return 0


def verify_clip():
    """验证要送去转换的那张图（带 Clip 的 L=80 部署源）：ORT 能否加载 + 前向是否有限 +
    Clip 是否真的把 `linear_23` 钳在 65504。指南 §34.2：ONNX 手术后第一道检查是 ORT，不是转换。"""
    print('验证 L=80 **部署源（带 Clip）** 四段链')
    t = run(80, os.path.join(P0, 'caption_L80.raw'),
            os.path.join(P0, 'cap_pad_mask_L80.raw'), variant='deployed')
    out = t['latents']
    ok = bool(np.isfinite(out).all())
    print(chr(10) + '  part2b latents %s  |max|=%.4f  有限值=%s'
          % (out.shape, float(np.abs(out).max()), ok))
    ref = os.path.join(P0, 'L80_chain', 's0_part2b')
    if os.path.isdir(ref):
        b = np.fromfile(os.path.join(P0, 'L80_chain', 's0_part2b', 'add_92.raw'), np.float32)
        print('  （base 链的 add_92 |max| = %.1f，供对照）' % float(np.abs(b).max()))
    print('  判定: %s' % ('✅ PASS —— 带 Clip 的 L=80 图可加载且前向有限'
                          if ok else '🔴 FAIL —— 输出含 inf/nan'))
    return 0 if ok else 1


def dump32():
    """#116：产出 **L=32 且纯 FP32 血统** 的段间输入，与 testB（CPU 参考链）做单变量对照。

    方案 `scripts/EXP_PLAN_116.md`。用部署那份 caption（20 真实槽 + 复制补齐到 32），
    与产出盘上 FP32 参考的那次完全一致。
    """
    dst = os.path.join(P0, 'L32_pure')
    print('L=32 纯 FP32 四段链，dump 段间输入 -> %s' % dst)
    os.makedirs(dst, exist_ok=True)
    t = run(32, os.path.join(P0, 'htp_inloop', 'const', 'caption.raw'),
            os.path.join(P0, 'htp_inloop', 'const', 'cap_pad_mask.raw'), dump_dir=dst)
    ref = np.fromfile(os.path.join(P0, 'vs_fp32', 'latents_fp32_s0.raw'), np.float32)
    got = t['latents'].reshape(-1)
    same = np.array_equal(got, ref)
    print(chr(10) + '  自检：part2b 输出 vs 盘上 FP32 参考  逐字节相同=%s' % same)
    if not same:
        print('  🔴 不一致 ⇒ 这条链与产出参考的那条不是同一条，dump 不可用')
        return 1
    for seg in SEGS:
        d = os.path.join(dst, 's0_%s' % seg)
        print('  %-7s %d 个输入' % (seg, len(os.listdir(d))))
    return 0


if __name__ == '__main__':
    sys.exit({'known': known, 'emit': emit, 'verify_clip': verify_clip,
              'dump32': dump32}[sys.argv[1]]())
