"""
主线检查：transformer_part1a 的 caption 输入编码是
    bitwidth 16, min=-4556.255371, max=13753.419922, scale=0.279387742281, offset=-16308
（来自 snpe-dlc-info 读取已部署的 transformer_part1a_quantized.dlc）

若 caption 的真实数值量级远小于 scale，条件信息会被量化抹平。
本脚本用真实 prompt 算出 FP32 caption，再按上述编码量化+反量化，量化 caption 还剩多少信息。

判据：
  - 若量化后余弦相似度仍 > 0.99 -> caption 通道没问题，排除
  - 若显著下降           -> prompt 条件在进入 transformer 之前就被破坏了，这是根因级发现
"""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
TOKENIZER_PATH = r"D:\ZIMAGE\tokenizer\tokenizer.json"
TEXT_MAX_LEN, QWEN_PAD_ID = 20, 151643
PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"

# 来自 snpe-dlc-info 的实际编码
SCALE, OFFSET, BW = 0.279387742281, -16308.0, 16
ENC_MIN, ENC_MAX = -4556.255371093750, 13753.419921875000

tok = Tokenizer.from_file(TOKENIZER_PATH)
ids = tok.encode(f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n").ids
input_ids = np.full((1, TEXT_MAX_LEN), QWEN_PAD_ID, dtype=np.int32)
attention_mask = np.zeros((1, TEXT_MAX_LEN), dtype=np.int32)
n = min(len(ids), TEXT_MAX_LEN)
input_ids[0, :n] = ids[:n]
attention_mask[0, :n] = 1

CHAIN = [("text_encoder_part1", ["input_ids", "attention_mask"]),
         ("text_encoder_part2", ["add_2452", "attention_mask"]),
         ("text_encoder_part3", ["add_4828", "attention_mask"]),
         ("text_encoder_part4", ["add_7204", "attention_mask"])]

t = {"input_ids": input_ids, "attention_mask": attention_mask}
for name, ins in CHAIN:
    s = ort.InferenceSession(f"{EVIDENCE}\\{name}.onnx", providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()]
    t.update(dict(zip(outs, s.run(outs, {k: t[k] for k in ins}))))
    del s

cap = t["caption"].astype(np.float32)
print("=" * 88)
print("FP32 caption（transformer 的 prompt 条件输入）真实统计")
print("=" * 88)
print(f"shape={cap.shape}")
print(f"min={cap.min():.6f}  max={cap.max():.6f}  mean={cap.mean():.6f}  std={cap.std():.6f}")
print(f"|值| 的分位数: 50%={np.percentile(np.abs(cap),50):.6f}  "
      f"90%={np.percentile(np.abs(cap),90):.6f}  99%={np.percentile(np.abs(cap),99):.6f}  "
      f"max={np.abs(cap).max():.6f}")
print()
print("=" * 88)
print("已部署 DLC 对 caption 使用的量化编码")
print("=" * 88)
print(f"bitwidth={BW}  scale={SCALE}  offset={OFFSET}")
print(f"校准范围 = [{ENC_MIN:.3f}, {ENC_MAX:.3f}]   跨度 = {ENC_MAX-ENC_MIN:.1f}")
print(f"本 prompt 实际范围 = [{cap.min():.3f}, {cap.max():.3f}]   跨度 = {cap.max()-cap.min():.3f}")
print(f"实际跨度只占校准跨度的 {(cap.max()-cap.min())/(ENC_MAX-ENC_MIN)*100:.4f}%")
print()
eff = (cap.max() - cap.min()) / SCALE
print(f">>> 本 prompt 的 caption 实际只用到约 {eff:.1f} 个量化档位（总共 {2**BW} 个）")
print()

q = np.clip(np.rint(cap / SCALE - OFFSET), 0, 2**BW - 1)
deq = ((q + OFFSET) * SCALE).astype(np.float32)
err = np.abs(deq - cap)

print("=" * 88)
print("按该编码量化+反量化之后，caption 还剩多少信息")
print("=" * 88)
print(f"误差 max={err.max():.6f}  rms={np.sqrt((err**2).mean()):.6f}")
print(f"相对误差(rms误差/信号rms) = {np.sqrt((err**2).mean())/np.sqrt((cap**2).mean())*100:.2f}%")
print()
a, b = cap[0], deq[0]
print(f"{'token':>6} {'余弦相似度':>14} {'FP32 |max|':>13} {'量化后唯一值个数':>18}")
for i in range(a.shape[0]):
    na, nb = np.linalg.norm(a[i]), np.linalg.norm(b[i])
    cos = float(a[i] @ b[i] / (na * nb)) if na > 0 and nb > 0 else float("nan")
    print(f"{i:>6} {cos:>14.6f} {np.abs(a[i]).max():>13.6f} {len(np.unique(b[i])):>18}")

cos_all = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print()
print(f">>> 整体余弦相似度 = {cos_all:.6f}")
print(f">>> 量化后整个 caption 张量的唯一取值个数 = {len(np.unique(deq))}（FP32 侧 {len(np.unique(cap))}）")
print()
print("判读：余弦 > 0.99 则 caption 通道无问题；显著低于 0.99 则 prompt 条件在进入 transformer 前已被破坏。")

np.savez(r"D:\ZImage_Work\p0_experiments\caption_quant_check.npz",
         caption_fp32=cap, caption_dequant=deq)
