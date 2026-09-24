# -*- coding: utf-8 -*-
"""为 Tier 2 门产出 token id：先用**已知样本**验证模板复现正确，再算新 prompt。

约束 8：模板拼法是本脚本唯一的自由度，定稿前必须先复现盘上的 22-token 样本。
模板（抄自 official_caption.py 文档串，与 prompt_ids.npy 实测一致，前 3 后 5 共 8 token）：
    <|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer                      # noqa: E402

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
KNOWN_PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"
KNOWN_NPY = os.path.join(P0, "prompt_ids.npy")

TPL = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"


def encode(tok, p):
    return tok(TPL % p, add_special_tokens=False)["input_ids"]


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL, "tokenizer"))

    # --- G0-known：必须逐元素复现盘上的 22 token ---
    want = np.load(KNOWN_NPY).tolist()
    got = encode(tok, KNOWN_PROMPT)
    ok = (got == want)
    print("G0-known 模板复现：盘上 %d token / 复现 %d token  %s"
          % (len(want), len(got), "✅ 逐元素相同" if ok else "🔴 不同"))
    if not ok:
        print("  盘上: %s" % want)
        print("  复现: %s" % got)
        return 1

    if len(sys.argv) < 3:
        print("用法: t2_make_ids.py <输出npy> <prompt>")
        return 1
    out, prompt = sys.argv[1], sys.argv[2]
    ids = encode(tok, prompt)
    print("\n新 prompt 共 %d token（正文 %d + 模板 8）" % (len(ids), len(ids) - 8))
    if len(ids) > 80:
        print("🔴 超过 L=80 槽，会被截断。请缩短 prompt。")
        return 1
    np.save(out, np.array(ids, dtype=np.int64))
    print("写出 -> %s（占 %d/80 槽，剩 %d 槽 padding）" % (out, len(ids), 80 - len(ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
