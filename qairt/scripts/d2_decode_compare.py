# -*- coding: utf-8 -*-
"""D2 收尾：把设备跑出的终点 latents 在宿主解码，并与 FP32 参考比。

## 两臂必须用【完全相同的解码路径】
否则 PSNR 差异里会混进 VAE 实现的差异。本脚本对两臂一律用
**同一个 ONNX VAE**（按各自比例的手术产物），喂法用 #80 的正确形式
`lat + VAE_SHIFT*VAE_SCALING`（图内首节点已做 Div，外面只能加 shift*scale）。

## 判据（执行前锁定）
本实验回答的是「**换比例有没有额外代价**」，不是「量化好不好」。
所以判据是**两臂之差**，不是绝对值：
    D = PSNR(1:1 量化 vs 1:1 FP32) - PSNR(新比例 量化 vs 新比例 FP32)
    D <= 2.0 dB   🟢 换比例无额外代价
    2 < D <= 5    🟡 有代价，需看图判断可接受性
    D >  5.0 dB   🔴 换比例有明显代价，须查
🔴 **绝对值不能单独用作判据**：#143 实测同一套量化在三个 prompt 上是
   22.58 / 14.25 / 13.47 dB，跨度 9 dB —— 单个绝对值说明不了任何事。
⚠️ 按约束 1，**两张图都必须自己打开看**，不得只看 PSNR 下结论。

用法: python d2_decode_compare.py <tag> <宽> <高>
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
REFN = os.path.join(P0, "vae_calib_1184x896")     # 4:3 的按 seed 隔离的 FP32 参考
VAE_SCALING, VAE_SHIFT = 0.3611, 0.1159          # 抄自 t2_fp32_ref.py，勿改


def decode(lat_path, shape, vae_onnx, png_path):
    lat = np.fromfile(lat_path, np.float32)
    exp = int(np.prod(shape))
    assert lat.size == exp, "%s 元素数 %d != %d" % (lat_path, lat.size, exp)
    lat = lat.reshape(*shape)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    cwd = os.getcwd()
    os.chdir(E)
    try:
        s = ort.InferenceSession(vae_onnx, so, providers=["CPUExecutionProvider"])
        px = s.run(None, {"vae_latents":
                          (lat.astype(np.float64) + VAE_SHIFT * VAE_SCALING).astype(np.float32)})[0]
    finally:
        os.chdir(cwd)
    arr = ((np.clip(px[0], -1, 1) + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(arr).save(png_path)
    return arr


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")


def arm(tag, W, H, name, dev_lat, fp32_lat, vae_onnx):
    lh, lw = H // 8, W // 8
    if not os.path.isfile(dev_lat):
        print("  [%s] 缺设备产物 %s —— 该臂跳过" % (name, dev_lat))
        return None
    q = decode(dev_lat, (1, 16, lh, lw), vae_onnx,
               os.path.join(OUT, "d2_%s_quant.png" % name))
    r = decode(fp32_lat, (1, 16, lh, lw), vae_onnx,
               os.path.join(OUT, "d2_%s_fp32.png" % name))
    v = psnr(q, r)
    print("  [%-4s] %dx%d  量化 vs FP32 = %6.2f dB   图: d2_%s_quant.png / d2_%s_fp32.png"
          % (name, W, H, v, name, name), flush=True)
    return v


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print("=== D2 解码与比较（两臂同一解码路径）===", flush=True)
    # 🔴🔴 2026-09-03：原本引用【固定名】的 FP32 参考，而那两个文件会被
    # 每次 t2_fp32_ref.py drive 覆盖 => 实际比的是「本 seed 的量化输出 vs 另一 seed 的
    # FP32 参考」，落在 12 dB（无关图基线），害我一度误判「换比例回退 9.61 dB」。
    # 现在一律用【按 seed 隔离】的参考文件；缺文件就跳过该臂，绝不退回固定名。
    seed = "42"
    if "--seed" in sys.argv:
        seed = sys.argv[sys.argv.index("--seed") + 1]
    new = arm(tag, W, H, "new",
              os.path.join(OUT, "d2_final_latents_%s_s%s.raw" % (tag, seed)),
              os.path.join(REFN, "latent_s%s.raw" % seed),
              "vae_decoder_%s.onnx" % tag)
    ctrl = arm(tag, 1024, 1024, "ctrl",
               os.path.join(OUT, "d2_final_latents_1x1_s%s.raw" % seed),
               os.path.join(P0, "fp32_1x1_s%s.raw" % seed),
               "vae_decoder.onnx")

    print("")
    if new is None or ctrl is None:
        print("两臂不齐，**不下结论**（缺对照就无法归因，R1）")
        return 1
    d = ctrl - new
    print("对照臂(1:1) %.2f dB  -  新比例 %.2f dB  =  差 %.2f dB" % (ctrl, new, d))
    print("判定: %s" % ("🟢 换比例无额外代价（<=2 dB）" if d <= 2.0 else
                        "🟡 有代价（2~5 dB），需看图判断" if d <= 5.0 else
                        "🔴 明显代价（>5 dB），须查"))
    print("⚠️ 约束 1：两张图都必须自己打开看，不得只看 PSNR 下结论。")
    return 0


sys.exit(main())
