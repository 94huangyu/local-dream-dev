# -*- coding: utf-8 -*-
"""Tier 2：为 L=80 的 text_encoder 生成校准数据（混合长度）。

## 配方选择的依据（2026-08-27 实测，EXP_PLAN_S32_VS_FP32 §十）

三种 32 槽配方的 **caption 相对 FP32 误差**（12 个 prompt，QDQ 模拟，工具已自检）：

    20 槽部署版 3.5988%  |  v1 复制末行 3.4528%  |  v3 满槽 3.3956%

三者相差不到 0.2 pp ⇒ **在唯一可信的直接测量上基本等价**。
⇒ 不再纠结配方，改用**最有依据**的一种：官方文档要求
*"a representative set of input data"*，业界共识是校准长度应
*"representative of typical inference scenarios ... use a mix of lengths"*。
L=80 时真实 prompt 的正文会在 8~72 token 之间 ⇒ **用混合长度**。

🚫 已被实测否证、勿再当理由：
  · 「pad 行污染 per-tensor 量程」—— 段间 `add_2452/4828/7204` 实测 ×1.0000
  · 「padding 越多量化越差」—— 无任何文档/文献来源支持
"""
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ONNX = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
TOK = r'D:\LocalDreamZImage\logs\dup126_20260826\tokenizer.json'
DST = r'D:\ZImage_Work\TIER2_L80'
SEQ, HIDDEN, TEMPLATE, PAD = 80, 2560, 8, 151643
PART_IN = {'text_encoder_part1': ['input_ids', 'attention_mask'],
           'text_encoder_part2': ['add_2452', 'attention_mask'],
           'text_encoder_part3': ['add_4828', 'attention_mask'],
           'text_encoder_part4': ['add_7204', 'attention_mask']}
# 目标正文长度（含模板后分别是 20 / 32 / 48 / 64 / 80 槽）
TARGETS = [12, 24, 40, 56, 72]
SEED_TEXT = [
    'A cinematic shot of a cute cat sitting on a wooden desk beside a sunlit window with soft '
    'shadows across scattered paper notes and a chipped ceramic mug holding cold morning coffee '
    'while dust drifts slowly through the quiet room',
    'Portrait of a young woman with flowing hair in golden hour light standing near a quiet harbour '
    'while gulls drift above calm water and distant masts sway against a sky turning amber over '
    'the slow tide coming in',
    'Aerial view of a misty mountain forest at dawn where pale light spills over ridges and a narrow '
    'river winds through dense pines below scattered clouds that catch the first warmth of the '
    'rising sun across the valley',
    'Abstract fluid art with vibrant neon colors on dark background swirling into glossy ribbons that '
    'catch sharp reflections along curling edges while deeper currents fold slowly beneath the '
    'surface in shifting bands of light',
    'Cozy living room interior with warm lighting and bookshelves where a worn armchair faces tall '
    'windows framing a slow grey afternoon rain and a folded blanket rests across the arm beside '
    'a low table stacked with novels',
]


def wrap(p):
    return "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"


def fit(tk, text, want_body):
    """取「正文 token 数不超过 want_body」的最长前缀，返回 (文本, 实际正文 token 数)。

    不强求精确命中：校准要的是**长度混合**，不是某个精确值
    （2026-08-27 第一版强求精确，第 4 条凑不出 56 就整个失败）。
    """
    words = text.split()
    best, best_n = None, 0
    for k in range(1, len(words) + 1):
        cand = ' '.join(words[:k])
        n = len(tk.encode(wrap(cand), add_special_tokens=False).ids) - TEMPLATE
        if n > want_body:
            break
        best, best_n = cand, n
    return best, best_n


def main():
    tk = Tokenizer.from_file(TOK)
    samples = []
    for i, (want, seed) in enumerate(zip(TARGETS, SEED_TEXT)):
        txt, got = fit(tk, seed, want)
        if txt is None or got == 0:
            print('FAIL: 第 %d 条无法构造' % i)
            return 1
        ids = tk.encode(wrap(txt), add_special_tokens=False).ids
        samples.append(ids)
        print('  sample_%04d 目标正文 %2d -> 实际 %2d + 模板 %d = %2d 槽占用（/%d）'
              % (i, want, got, TEMPLATE, len(ids), SEQ))

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess = [ort.InferenceSession('text_encoder_part%d_L80.onnx' % p, so,
                                     providers=['CPUExecutionProvider']) for p in (1, 2, 3, 4)]
    finally:
        os.chdir(cwd)

    for part in PART_IN:
        d = os.path.join(DST, part, 'calibration_raw')
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    lines = {p: [] for p in PART_IN}
    for i, ids in enumerate(samples):
        s = 'sample_%04d' % i
        n = len(ids)
        a = np.full((1, SEQ), PAD, dtype=np.int64); a[0, :n] = ids
        m = np.zeros((1, SEQ), dtype=np.int64); m[0, :n] = 1
        t = {'input_ids': a, 'attention_mask': m}
        for se in sess:
            f = {}
            for inp in se.get_inputs():
                v = t[inp.name]
                f[inp.name] = (v.astype(np.int32) if 'int32' in inp.type
                               else v.astype(np.float32) if 'float' in inp.type else v)
            for o, v in zip(se.get_outputs(), se.run(None, f)):
                t[o.name] = v
        print('  %s caption %s' % (s, t['caption'].shape), flush=True)
        for part, ins in PART_IN.items():
            sd = os.path.join(DST, part, 'calibration_raw', s)
            os.makedirs(sd, exist_ok=True)
            entry = []
            for name in ins:
                arr = np.ascontiguousarray(t[name].astype(np.float32)).reshape(-1)
                p = os.path.join(sd, name + '.raw')
                arr.tofile(p)
                want = SEQ * 4 if name in ('input_ids', 'attention_mask') else SEQ * HIDDEN * 4
                if os.path.getsize(p) != want:
                    print('FAIL %s/%s/%s: %d != %d' % (part, s, name, os.path.getsize(p), want))
                    return 1
                entry.append('%s:=%s' % (name, p))
            lines[part].append(' '.join(entry))

    for part, ls in lines.items():
        il = os.path.join(DST, part, 'input_list_raw.txt')
        with open(il, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(ls) + '\n')
        print('%-20s %d 个样本' % (part, len(ls)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
