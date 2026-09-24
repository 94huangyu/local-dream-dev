"""用【完整 22 token】chat 模板算 caption，与截断到 20 的版本做对照。

## 背景（2026-08-18 实测）
部署的 text_encoder 被固定成 `input_ids [1,20]`，而本项目 prompt 的完整 chat 模板
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` 是 **22 个 token**
⇒ 末尾 `<|im_start|>assistant\n` **被整个截掉**。

⚠️ ONNX 源图是**动态形状**（`input_ids: ['?','?']`）⇒ 限制来自**导出时的固定化**，
不是模型本身，因此可以用 ONNX 直接跑 22 token 做对照。

本脚本只产出 caption，供 fp32_step_runner 使用（单变量：只改 token 数）。
"""
import os, sys, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
TOKENIZER = os.path.join("D:", os.sep, "ZIMAGE", "tokenizer", "tokenizer.json")
QWEN_PAD_ID = 151643
PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"


def load(n):
    t0 = time.time()
    s = ort.InferenceSession(os.path.join(E, n + ".onnx"), providers=["CPUExecutionProvider"])
    print(f"  load {n} {time.time()-t0:.0f}s", flush=True)
    return s


def run(s, **kw):
    names = [o.name for o in s.get_outputs()]
    return dict(zip(names, s.run(names, kw)))


def main():
    max_len = int(sys.argv[1]); out = sys.argv[2]
    tok = Tokenizer.from_file(TOKENIZER)
    ids = tok.encode(f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n").ids
    n = min(len(ids), max_len)
    print(f"完整模板 {len(ids)} token；本次用 max_len={max_len} ⇒ 实际 {n}"
          f"{'（截断，丢 %d 个）' % (len(ids)-max_len) if len(ids)>max_len else ''}", flush=True)
    input_ids = np.full((1, max_len), QWEN_PAD_ID, dtype=np.int32)
    attn = np.zeros((1, max_len), dtype=np.int32)
    input_ids[0, :n] = ids[:n]; attn[0, :n] = 1

    t = {"input_ids": input_ids, "attention_mask": attn}
    import gc
    for name, need in [("text_encoder_part1", ["input_ids", "attention_mask"]),
                       ("text_encoder_part2", ["add_2452", "attention_mask"]),
                       ("text_encoder_part3", ["add_4828", "attention_mask"]),
                       ("text_encoder_part4", ["add_7204", "attention_mask"])]:
        s = load(name); t.update(run(s, **{k: t[k] for k in need})); del s; gc.collect()
    cap = t["caption"].astype(np.float32)
    print(f"caption shape={cap.shape}  std={cap.std():.4f}", flush=True)
    np.ascontiguousarray(cap).tofile(out)
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
