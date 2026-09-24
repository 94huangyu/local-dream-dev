"""
Test A（方案见 scripts/EXP_PLAN_V2.md）：检验 text encoder 量化是否破坏了 caption 条件。

路径：同一组 input_ids/attention_mask
  FP32 侧  : onnxruntime 跑 text_encoder_part1~4          -> caption_fp32
  量化侧   : snpe-net-run 跑 4 个 *_quantized.dlc          -> caption_quant
指标：逐 token 余弦相似度 + 相对误差；随后把 caption_quant 喂 FP32 transformer+VAE 出图。

链路（取自 zimage_fp32_pipeline.py，已核对）：
  part1: input_ids, attention_mask -> add_2452
  part2: add_2452,  attention_mask -> add_4828
  part3: add_4828,  attention_mask -> add_7204
  part4: add_7204,  attention_mask -> caption
"""
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
DLC_DIR = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline"
TOKENIZER_PATH = r"D:\ZIMAGE\tokenizer\tokenizer.json"
WORK = r"D:\ZImage_Work\p0_experiments\testA"
SNPE = r"D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\snpe-net-run.exe"
QNN_LIB = r"D:\qairt\2.48.0.260626\lib\x86_64-windows-msvc"
QNN_BIN = r"D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc"

TEXT_MAX_LEN = 20
QWEN_PAD_ID = 151643
PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"

CHAIN = [
    ("text_encoder_part1", ["input_ids", "attention_mask"], "add_2452"),
    ("text_encoder_part2", ["add_2452", "attention_mask"], "add_4828"),
    ("text_encoder_part3", ["add_4828", "attention_mask"], "add_7204"),
    ("text_encoder_part4", ["add_7204", "attention_mask"], "caption"),
]


def tokenize():
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    text = f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text).ids
    input_ids = np.full((1, TEXT_MAX_LEN), QWEN_PAD_ID, dtype=np.int32)
    attention_mask = np.zeros((1, TEXT_MAX_LEN), dtype=np.int32)
    n = min(len(ids), TEXT_MAX_LEN)
    input_ids[0, :n] = ids[:n]
    attention_mask[0, :n] = 1
    return input_ids, attention_mask, n


def run_fp32(input_ids, attention_mask):
    tensors = {"input_ids": input_ids, "attention_mask": attention_mask}
    for name, ins, out_name in CHAIN:
        t0 = time.time()
        sess = ort.InferenceSession(f"{EVIDENCE}\\{name}.onnx", providers=["CPUExecutionProvider"])
        outs = [o.name for o in sess.get_outputs()]
        res = dict(zip(outs, sess.run(outs, {k: tensors[k] for k in ins})))
        tensors.update(res)
        del sess
        print(f"  [FP32] {name}: {out_name} shape={tensors[out_name].shape} "
              f"dtype={tensors[out_name].dtype} ({time.time()-t0:.1f}s)")
    return tensors


def write_raw(arr, path):
    arr.tofile(path)
    expect = arr.size * arr.dtype.itemsize
    got = os.path.getsize(path)
    # V1 有效性检查：字节数必须等于 shape 连乘 × dtype 字节数
    assert got == expect, f"V1 FAIL {path}: {got} != {expect}"
    return path


def run_quant(input_ids, attention_mask, fp32_tensors):
    os.makedirs(WORK, exist_ok=True)
    env = dict(os.environ, PATH=f"{QNN_LIB};{QNN_BIN};" + os.environ["PATH"])
    tensors = {"input_ids": input_ids, "attention_mask": attention_mask}

    for name, ins, out_name in CHAIN:
        raws = {}
        for k in ins:
            a = tensors[k]
            # 量化侧输入必须和 FP32 侧同 dtype；float 张量统一 float32
            if a.dtype not in (np.int32, np.float32):
                a = a.astype(np.float32)
            raws[k] = write_raw(np.ascontiguousarray(a), f"{WORK}\\{name}_{k}.raw")

        lst = f"{WORK}\\{name}_list.txt"
        with open(lst, "w") as f:
            f.write(f"%{out_name}\n")
            f.write(" ".join(f"{k}:={v}" for k, v in raws.items()) + "\n")

        outdir = f"{WORK}\\{name}_out"
        cmd = [SNPE, "--container", f"{DLC_DIR}\\{name}\\{name}_quantized.dlc",
               "--input_list", lst, "--output_dir", outdir]
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise RuntimeError(f"snpe-net-run failed for {name}")

        ref = fp32_tensors[out_name]
        got = np.fromfile(f"{outdir}\\Result_0\\{out_name}.raw", dtype=np.float32)
        assert got.size == ref.size, f"V1 FAIL {out_name}: {got.size} != {ref.size}"
        tensors[out_name] = got.reshape(ref.shape)
        e = np.abs(tensors[out_name] - ref)
        print(f"  [量化] {name}: {out_name} 误差max={e.max():.6f} "
              f"rel={e.max()/max(np.abs(ref).max(),1e-9)*100:.4f}% ({time.time()-t0:.1f}s)")
    return tensors


def main():
    input_ids, attention_mask, n = tokenize()
    print(f"prompt token_count = {n}")
    print("\n=== FP32 侧 ===")
    fp32 = run_fp32(input_ids, attention_mask)
    print("\n=== 量化侧 ===")
    quant = run_quant(input_ids, attention_mask, fp32)

    cf, cq = fp32["caption"], quant["caption"]
    np.savez(f"{WORK}\\caption_compare.npz", caption_fp32=cf, caption_quant=cq,
             input_ids=input_ids, attention_mask=attention_mask)

    print("\n" + "=" * 84)
    print("caption 对比（这是喂给 transformer 的条件张量）")
    print("=" * 84)
    a, b = cf[0], cq[0]                     # (32, 2560)
    err = np.abs(b - a)
    print(f"shape={cf.shape}  FP32范围[{a.min():.4f}, {a.max():.4f}]")
    print(f"误差 max={err.max():.6f}  rms={np.sqrt((err**2).mean()):.6f}  "
          f"相对(max/|ref|max)={err.max()/np.abs(a).max()*100:.4f}%")
    print()
    print(f"{'token':>6} {'余弦相似度':>14} {'该token误差max':>16} {'是否真实token':>14}")
    for t in range(a.shape[0]):
        cos = float(a[t] @ b[t] / (np.linalg.norm(a[t]) * np.linalg.norm(b[t]) + 1e-12))
        print(f"{t:>6} {cos:>14.6f} {err[t].max():>16.6f} {'是' if t < n else '(padding)':>14}")

    real = slice(0, n)
    cos_all = float((a[real] * b[real]).sum() /
                    (np.linalg.norm(a[real]) * np.linalg.norm(b[real]) + 1e-12))
    print()
    print(f">>> 真实 token 段整体余弦相似度 = {cos_all:.6f}")
    print()
    print("按 EXP_PLAN_V2 的验收标准判读：")
    print("  余弦相似度 > 0.99  -> caption 条件基本完好，需继续用它出图确认")
    print("  余弦相似度 < 0.9   -> caption 条件已被严重破坏，C2 很可能就是根因")


if __name__ == "__main__":
    main()
