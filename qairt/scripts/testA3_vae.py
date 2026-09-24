"""
实验 A3（方案见 scripts/EXP_PLAN_A3.md）：VAE 量化是否是设备失败的原因。

用同一份真实 latents，对比 FP32 VAE 与量化 VAE 的解码结果。
量化侧统一走 snpe_runner.py（强制契约检查，禁止自己拼 input_list）。
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

import snpe_runner

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
LAT = EVIDENCE + r"\dlc_pipeline\vae_decoder\calibration_raw\sample_0000\latent_sample.raw"
OUT = r"D:\LocalDreamZImage\scratch_runs"
WORK = r"D:\ZImage_Work\p0_experiments\testA3"

SHAPE = (1, 16, 128, 128)


def to_png(px, path):
    """px: (1,3,1024,1024) in [-1,1]"""
    img = np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(img).save(path)
    return img


n = os.path.getsize(LAT)
exp = int(np.prod(SHAPE)) * 4
print(f"latents 文件 {n} 字节，期望 {exp} 字节 -> {'OK' if n == exp else '不符!'}")
assert n == exp, "latents 尺寸不符，实验无效"

lat = np.fromfile(LAT, dtype=np.float32).reshape(SHAPE)
print(f"latents 统计: min={lat.min():.4f} max={lat.max():.4f} "
      f"mean={lat.mean():.4f} std={lat.std():.4f}")

# ---------- FP32 侧 ----------
print("\n=== FP32 VAE (onnxruntime) ===")
s = ort.InferenceSession(f"{EVIDENCE}\\onnx\\vae_decoder.onnx", providers=["CPUExecutionProvider"])
print("  ONNX 输入名:", [i.name for i in s.get_inputs()])
px_f = s.run(["pixels"], {"vae_latents": lat})[0]
del s
img_f = to_png(px_f, f"{OUT}\\testA3_vae_fp32.png")
print(f"  pixels: min={px_f.min():.4f} max={px_f.max():.4f} std={px_f.std():.4f}")
print(f"  已保存 {OUT}\\testA3_vae_fp32.png")

# ---------- 量化侧（走强制入口）----------
print("\n=== 量化 VAE (snpe_runner，含契约强制检查) ===")
res = snpe_runner.run("vae_decoder", {"vae_latents": lat}, ["pixels"], WORK)
px_q = res["pixels"]
img_q = to_png(px_q, f"{OUT}\\testA3_vae_quant.png")
print(f"  pixels: min={px_q.min():.4f} max={px_q.max():.4f} std={px_q.std():.4f}")
print(f"  已保存 {OUT}\\testA3_vae_quant.png")

# ---------- 对比 ----------
a, b = img_f.astype(np.float32), img_q.astype(np.float32)
d = np.abs(a - b)
mse = ((a - b) ** 2).mean()
psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
print("\n" + "=" * 72)
print("A3 结果")
print("=" * 72)
print(f"  平均|像素差| = {d.mean():.2f}    最大 = {d.max():.1f}    PSNR = {psnr:.2f} dB")
print()
print("按 EXP_PLAN_A3 第6节的验收标准：")
print("  PSNR > 20 dB 且内容一致 -> A3 被否定，转查 B（C++ 实现）")
print("  量化解码是色块/乱码      -> A3 就是根因")
print("  PSNR 10~20 dB           -> A3 有贡献但可能不是全部")
print()
print("注意：本实验【不能】验证设备 C++ 侧的 VAE 前后处理系数是否正确（属候选 B），")
print("      也【不能】代表 HTP 上跑 .bin 的行为（本实验是 SNPE CPU 参考实现跑 .dlc）。")
