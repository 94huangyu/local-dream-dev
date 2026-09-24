# -*- coding: utf-8 -*-
"""T3 失败归因：在 **FP32 ONNX** 上比对旧图(seq20) 与新图(seq32) 的前 N 个位置。

目的：把「图的数学」与「量化/设备」分开。

  · 若 FP32 下 hidden[0:token_count] **逐位相同** ⇒ 图层面没问题，
    T3 的差异来自量化或设备执行。
  · 若 FP32 下就不同 ⇒ 我「因果性 ⇒ 前面位置不受影响」的前提是错的，
    padding 会向前泄漏。

用法：python scripts/te_s32_fp32_probe.py <part编号>
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ONNX = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
TOK = r'D:\LocalDreamZImage\logs\dup126_20260826\tokenizer.json'
PAD = 151643          # kQwenPadId，与 app 的 TextEncoder.hpp:613 一致
PROMPT = '一条蓝色的鱼'


def build(seq):
    tk = Tokenizer.from_file(TOK)
    text = "<|im_start|>user\n" + PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tk.encode(text, add_special_tokens=False).ids
    n = len(ids)
    if n > seq:
        raise SystemExit('prompt %d token 超过 seq=%d' % (n, seq))
    a = np.full((1, seq), PAD, dtype=np.int64)
    a[0, :n] = ids
    m = np.zeros((1, seq), dtype=np.int64)
    m[0, :n] = 1
    return a, m, n


def run(model, seq):
    a, m, n = build(seq)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(os.path.join(ONNX, model), so,
                                providers=['CPUExecutionProvider'])
    feeds = {}
    for i in sess.get_inputs():
        arr = a if i.name == 'input_ids' else m
        # 按图声明的 dtype 喂
        want = i.type
        if 'int32' in want:
            arr = arr.astype(np.int32)
        elif 'float' in want:
            arr = arr.astype(np.float32)
        feeds[i.name] = arr
    out = sess.run(None, feeds)[0]
    return out, n


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else '1'
    old = 'text_encoder_part%s.onnx' % part
    new = 'text_encoder_part%s_s32.onnx' % part
    cwd = os.getcwd()
    os.chdir(ONNX)          # external data 按当前工作目录解析
    try:
        o, n = run(old, 20)
        print('旧图 seq=20 输出 %s  真实 token 数=%d' % (o.shape, n))
        p, _ = run(new, 32)
        print('新图 seq=32 输出 %s' % (p.shape,))
    finally:
        os.chdir(cwd)

    A, B = o[0, :n], p[0, :n]
    same = np.array_equal(A, B)
    print('\n前 %d 个位置（真实 token）逐位相同: %s' % (n, same))
    if not same:
        d = np.abs(A - B)
        rel = np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-12)
        print('  最大绝对差 %.6g   平均 %.6g   相对 L2 %.4f%%' % (d.max(), d.mean(), rel * 100))
        per = [(i, float(np.abs(A[i] - B[i]).max())) for i in range(n)]
        print('  逐位置最大差:', [('%d:%.3g' % t) for t in per])
    # 顺带看 pad 区（预期不同，且不影响下游）
    print('pad 区 [%d:20) 最大差: %.6g' % (n, np.abs(o[0, n:20] - p[0, n:20]).max()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
