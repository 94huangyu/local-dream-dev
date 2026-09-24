# -*- coding: utf-8 -*-
"""Tier 1 校准 v3：用**填满 32 槽的真实 prompt** 做校准，使校准里不含 padding 行。

## 为什么（2026-08-27 实测，EXP_PLAN_S32_VS_FP32 §八/§九）

v1（复制最后一行）17.486 dB、v2（真实 pad token）17.025 dB，都低于 20 槽的 18.763 dB。
实测机制：**pad 行撑宽了 part1 内部张量的 per-tensor 量程**——

    linear_32     [-10.63, 5.73]  ->  [-27.63, 5.95]   ×2.05
    view_14 等    [-0.628, 0.568] ->  [-0.924, 1.200]  ×1.78

per-tensor 量化下真实行与 pad 行共用一个 scale ⇒ **步长被 pad 行拖粗，真实行受损**。
且方向一致：v2 量程变化 >1% 的有 164 个（v1 只有 66 个），v2 也确实更差。

⇒ 让校准里**根本没有 pad 行**：把原 5 条 prompt 扩写到正文 24 token（+8 模板 = 32 槽满）。

## 单变量纪律

**只改「有没有 pad 行」**，不改语种/题材 —— 沿用原来那 5 条英文摄影风 prompt 的主题，
只在同风格下续写，避免把「换语料」混进来（本项目已因非单变量栽过四次，见 §3.6）。

🚫 已否证的假设，勿重试：
  · 「pad 行完全不撑宽量程」——只对段间 `add_2452/4828/7204` 成立（×1.0000），内部张量不成立
  · 「−10000 掩码偏置只在 seq32 出现」——实测 `val_217` 两版都是 [−10000, 0]（×1.00）
"""
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ONNX = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
TOK = r'D:\LocalDreamZImage\logs\dup126_20260826\tokenizer.json'
DST_ROOT = r'D:\ZImage_Work\TIER1_S32'
SEQ, HIDDEN, TEMPLATE = 32, 2560, 8
PAD = 151643

# 原 5 条（正文 10~12 token）在同风格下续写，目标：含模板正好 32 token
BASE = [
    'A cinematic shot of a cute cat sitting on a wooden desk beside a sunlit window '
    'with soft shadows falling across scattered paper notes',
    'Portrait of a young woman with flowing hair in golden hour light standing near '
    'a quiet harbour while gulls drift above calm water',
    'Aerial view of a misty mountain forest at dawn where pale light spills over '
    'ridges and a narrow river winds through dense pines',
    'Abstract fluid art with vibrant neon colors on dark background swirling into '
    'glossy ribbons that catch sharp reflections along their curling edges',
    'Cozy living room interior with warm lighting and bookshelves where a worn '
    'armchair faces tall windows framing a slow grey afternoon rain',
]
CHAIN = [('text_encoder_part1_s32.onnx', 'add_2452'),
         ('text_encoder_part2_s32.onnx', 'add_4828'),
         ('text_encoder_part3_s32.onnx', 'add_7204'),
         ('text_encoder_part4_s32.onnx', 'caption')]
PART_INPUTS = {
    'text_encoder_part1': ['input_ids', 'attention_mask'],
    'text_encoder_part2': ['add_2452', 'attention_mask'],
    'text_encoder_part3': ['add_4828', 'attention_mask'],
    'text_encoder_part4': ['add_7204', 'attention_mask'],
}


def wrap(p):
    return "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"


def fit(tk, text):
    """把 prompt 截到「含模板正好 SEQ 个 token」；不足则报错（要求填满）。"""
    ids = tk.encode(wrap(text), add_special_tokens=False).ids
    if len(ids) == SEQ:
        return ids
    if len(ids) < SEQ:
        return None
    # 超了就从正文末尾逐词砍，直到正好 SEQ
    words = text.split()
    while words:
        words.pop()
        ids = tk.encode(wrap(' '.join(words)), add_special_tokens=False).ids
        if len(ids) == SEQ:
            return ids
        if len(ids) < SEQ:
            return None
    return None


def main():
    tk = Tokenizer.from_file(TOK)
    prompts = []
    for i, b in enumerate(BASE):
        ids = fit(tk, b)
        if ids is None:
            print('FAIL: 第 %d 条无法调到正好 %d token，请改写' % (i, SEQ))
            return 1
        prompts.append(ids)
        print('  sample_%04d 正文 %d token（+%d 模板 = %d，**无 padding**）'
              % (i, SEQ - TEMPLATE, TEMPLATE, SEQ))

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = {m: ort.InferenceSession(m, so, providers=['CPUExecutionProvider'])
                for m, _ in CHAIN}
    finally:
        os.chdir(cwd)

    for part in PART_INPUTS:
        d = os.path.join(DST_ROOT, part, 'calibration_raw')
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    lines = {p: [] for p in PART_INPUTS}
    for i, ids in enumerate(prompts):
        s = 'sample_%04d' % i
        a = np.array([ids], dtype=np.int64)
        m = np.ones((1, SEQ), dtype=np.int64)          # 全真，无 padding
        t = {'input_ids': a, 'attention_mask': m}
        for mdl, _ in CHAIN:
            se = sess[mdl]
            feeds = {}
            for inp in se.get_inputs():
                v = t[inp.name]
                feeds[inp.name] = (v.astype(np.int32) if 'int32' in inp.type
                                   else v.astype(np.float32) if 'float' in inp.type else v)
            for o, v in zip(se.get_outputs(), se.run(None, feeds)):
                t[o.name] = v
        print('  %s caption %s' % (s, t['caption'].shape), flush=True)

        for part, ins in PART_INPUTS.items():
            d = os.path.join(DST_ROOT, part, 'calibration_raw', s)
            os.makedirs(d, exist_ok=True)
            entry = []
            for name in ins:
                arr = np.ascontiguousarray(t[name].astype(np.float32)).reshape(-1)
                p = os.path.join(d, name + '.raw')
                arr.tofile(p)
                want = SEQ * 4 if name in ('input_ids', 'attention_mask') else SEQ * HIDDEN * 4
                if os.path.getsize(p) != want:
                    print('FAIL %s/%s/%s 字节数不符' % (part, s, name))
                    return 1
                entry.append('%s:=%s' % (name, p))
            lines[part].append(' '.join(entry))

    for part, ls in lines.items():
        il = os.path.join(DST_ROOT, part, 'input_list_raw.txt')
        with open(il, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(ls) + '\n')
        print('%-20s %d 个样本（全部 32 槽满）' % (part, len(ls)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
