"""🔴 2026-08-21 作废声明（HANDOVER 15.22 / 台账 #80）
本脚本的 ONNX 臂喂的是 lat/0.3611+0.1159，而 vae_decoder.onnx 图内首节点已有
Div(vae_latents,0.3611) ⇒ 除了两遍。**本脚本产出的一切数字与图像一律作废，不得引用。**
正确喂法与三臂对照见 scripts/vae_input_convention.py。
按约束 2 保留原文，不删除。
"""
"""VAE 双臂对照：同一份 final latents，分别用官方 PyTorch VAE 与我们的 ONNX VAE 解码。

## 为什么做
官方 vs 我们的 ONNX 全流程：逐步 noise_std 差 ±0.24%（几乎一致），
但成图平均像素差 13.63、**高频能量高出 30.7%**（毛发蜡质/过锐）。
⇒ 放大发生在哪一环？本实验把 VAE 单独摘出来。

## 单变量
两臂吃**完全相同**的 latents（`fp32_steps/lat_8.raw`，来自 22-token ONNX 那次，std=1.0996）。
⚠️ 官方那次的 final latents 未保存（脚本疏漏），但两条轨迹末步只差 0.17%
（1.1001 vs 1.0996），对"隔离 VAE"这个目的无影响 —— 关键是**两臂同输入**。

判读：
· 两臂高频能量接近 ⇒ VAE 不是放大点，缺陷在 transformer 导出
· 我们的 ONNX VAE 高频明显更高 ⇒ **VAE 导出就是放大点**

用法: python vae_ab.py <latents.raw> <out_prefix>
"""
import os, sys, time
import numpy as np

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
SCALING, SHIFT = 0.3611, 0.1159


def grad_energy(a):
    g = np.gradient(a.astype(np.float64).mean(axis=2))
    return float(np.sqrt(g[0] ** 2 + g[1] ** 2).mean())


def to_img(px):
    return np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def main():
    lat_path, prefix = sys.argv[1], sys.argv[2]
    lat = np.fromfile(lat_path, np.float32).reshape(1, 16, 128, 128)
    vae_in = (lat.astype(np.float64) / SCALING + SHIFT).astype(np.float32)
    print(f"latents std={lat.std():.4f}  vae_in std={vae_in.std():.4f}", flush=True)
    from PIL import Image

    # --- 臂 A：官方 PyTorch VAE ---
    import torch
    from diffusers import AutoencoderKL
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    vae.eval()
    with torch.no_grad():
        px = vae.decode(torch.from_numpy(vae_in), return_dict=False)[0].numpy()
    a_img = to_img(px)
    Image.fromarray(a_img).save(prefix + "_pytorch.png")
    print(f"官方 PyTorch VAE 解码 {time.time()-t0:.0f}s -> {prefix}_pytorch.png", flush=True)
    del vae

    # --- 臂 B：我们的 ONNX VAE ---
    import onnxruntime as ort
    t0 = time.time()
    s = ort.InferenceSession(os.path.join(ONNX, "vae_decoder.onnx"),
                             providers=["CPUExecutionProvider"])
    out = s.run(["pixels"], {"vae_latents": vae_in})[0]
    b_img = to_img(out)
    Image.fromarray(b_img).save(prefix + "_onnx.png")
    print(f"我们的 ONNX VAE 解码 {time.time()-t0:.0f}s -> {prefix}_onnx.png", flush=True)

    # --- 判读 ---
    A, B = a_img.astype(np.float64), b_img.astype(np.float64)
    d = np.abs(A - B)
    ga, gb = grad_energy(A), grad_energy(B)
    print("\n=== 同一份 latents，唯一变量 = VAE 实现 ===")
    print(f"  平均|像素差| {d.mean():.3f}   最大 {d.max():.0f}   "
          f"PSNR {10*np.log10(255**2/max(((A-B)**2).mean(),1e-9)):.2f} dB")
    print(f"  高频能量: 官方 {ga:.3f}   我们的 ONNX {gb:.3f}   "
          f"相对 {(gb/ga-1)*100:+.1f}%")
    print(f"\n  判读: {'🔴 VAE 导出是放大点' if abs(gb/ga-1) > 0.10 else '✅ VAE 不是放大点 ⇒ 缺陷在 transformer 导出'}")
    print("  （全流程对照里我们的成图高频高出 +30.7%，若此处远小于该值则 VAE 只是次要因素）")


if __name__ == "__main__":
    main()
