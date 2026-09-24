# -*- coding: utf-8 -*-
"""Tier 2 的三层指标（MAINLINE §2.5），参考系是**L=80 自己的 FP32**。

## 为什么不能用 panel3.py
`panel3.py` 的 `REF_FIN`/`REF_IMG` 写死指向 **L=32** 的 FP32 参考。
Tier 2 必须与**它自己的** FP32 参考比（§2.1.1：基准是 FP32，不是上一个量化版本）——
而且实测两个 FP32 参考之间就差 25.15 dB（L=32 那条链在第 20 个 token 处截断，
少了结尾的 assistant 标记；L=80 完整带上 22 个）。

## 三层
  L1  step-0 噪声预测相对 L2（E_all）
  L2  终点 latents 相对 L2 + 主体余弦
  L3  成图 PSNR / 像素相对 L2      <- **报告结论必须给这一层**

## 前提核对（#74：同 seed != 同噪声）
两臂的初始 latents 与 caption 必须逐字节相同，本脚本用 md5 核对，不一致直接拒绝。

用法: python t2_l3.py
"""
import hashlib
import io
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image                                        # noqa: E402

P0 = r"D:\ZImage_Work\p0_experiments"
S = r"D:\LocalDreamZImage\scratch_runs"
TAG = os.environ.get("T2_TAG", "L80_fp16")
REF = os.environ.get("T2_REF", "cat")        # 参考图/终点 latents 的名字后缀
STEPS = 8


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:20]


def sigmas(n, shift=3.0):
    f = lambda s: shift * s / (1.0 + (shift - 1.0) * s)      # noqa: E731
    out = [f(1.0 if n == 1 else 1.0 - (1.0 - 1.0 / n) * i / (n - 1)) for i in range(n)]
    return np.array(out + [0.0], dtype=np.float64)


def rel(a, b):
    return 100.0 * float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def bulk(a, b):
    """主体口径（|b| <= p99）—— 约束 7：全量相对 L2 会被离群值主导。"""
    thr = np.percentile(np.abs(b), 99)
    m = np.abs(b) <= thr
    return rel(a[m], b[m]), cos(a[m], b[m])


def conc1(b):
    """前 1% 元素占 ||b||^2 的比例（约束 7 强制同时报）。"""
    e = b ** 2
    k = max(1, int(0.01 * e.size))
    return 100.0 * float(np.sort(e)[-k:].sum() / e.sum())


def main():
    work = os.path.join(P0, "htp_inloop_%s" % TAG)
    dev_s0 = os.path.join(work, "s0", "latents_dev.raw")
    dev_png = os.path.join(S, "htp_inloop_transformer_%s.png" % TAG)
    _sfx = "" if REF == "cat" else "_%s" % REF
    ref_fin = os.path.join(S, "fp32_final_latents%s_L80.raw"
                           % ("" if REF == "cat" else "_" + REF))
    ref_png = os.path.join(S, "fp32ref_%s_L80.png" % REF)
    fp32_steps = os.path.join(P0, "fp32_steps%s_L80"
                              % ("" if REF == "cat" else "_" + REF))

    miss = [p for p in (dev_s0, dev_png, ref_fin, ref_png) if not os.path.isfile(p)]
    if miss:
        print("🔴 缺文件：")
        for p in miss:
            print("   %s" % p)
        return 2

    # --- 前提：两臂同噪声同 caption ---
    print("=== 前提核对（#74：同 seed != 同噪声）===")
    a = os.path.join(P0, "htp_inloop", "s0", "latents.raw")          # FP32 臂的 lat_init
    b = os.path.join(fp32_steps, "lat_0.raw")
    print("  初始 latents  FP32臂 %s / 设备臂输入 %s" % (md5(b), md5(a)))
    if md5(a) != md5(b):
        print("  🔴 初始噪声不同，比较无意义")
        return 1
    c1 = os.path.join(P0, "caption_%s_L80.raw" % REF)
    c2 = os.path.join(work, "const", "caption.raw")
    if os.path.isfile(c2):
        print("  caption      FP32臂 %s / 设备臂 %s" % (md5(c1), md5(c2)))
        if md5(c1) != md5(c2):
            print("  🔴 caption 不同，比较无意义")
            return 1
    print("  ✅ 同噪声同 caption")

    # --- L1：step-0 噪声预测 ---
    print()
    print("=== L1  step-0 噪声预测（vs L=80 的 FP32）===")
    l0 = np.fromfile(os.path.join(fp32_steps, "lat_0.raw"), np.float32).astype(np.float64)
    l1 = np.fromfile(os.path.join(fp32_steps, "lat_1.raw"), np.float32).astype(np.float64)
    ts = sigmas(STEPS)[:-1] * 1000.0
    dt = ts[1] / 1000.0 - ts[0] / 1000.0
    fp32_noise = (l0 - l1) / dt                      # 由 Euler 更新反解，精确
    dev_noise = np.fromfile(dev_s0, np.float32).astype(np.float64)
    if dev_noise.size != fp32_noise.size:
        print("  🔴 元素数不符 %d vs %d" % (dev_noise.size, fp32_noise.size))
        return 1
    print("  E_all = %.4f%%   集中度(前1%%占能量) = %.2f%%"
          % (rel(dev_noise, fp32_noise), conc1(fp32_noise)))

    # --- L2：终点 latents ---
    print()
    print("=== L2  终点 latents ===")
    # in-loop 脚本不落盘终点 latents（它在内存里算完就送去 VAE）。
    # 这里按 Euler 更新**精确重建**：final = lat_in(s7) - dt7 * noise(s7)，
    # dt7 = 0 - ts[7]/1000（最后一步的 sigma 归零）。不是近似。
    fin_cache = os.path.join(work, "final_latents.raw")
    if not os.path.isfile(fin_cache):
        s7 = os.path.join(work, "s%d" % (STEPS - 1))
        li = os.path.join(s7, "latents.raw")
        nd = os.path.join(s7, "latents_dev.raw")
        if os.path.isfile(li) and os.path.isfile(nd):
            ts_all = sigmas(STEPS)[:-1] * 1000.0
            dt7 = 0.0 - ts_all[STEPS - 1] / 1000.0
            fin = (np.fromfile(li, np.float32).astype(np.float64)
                   - dt7 * np.fromfile(nd, np.float32).astype(np.float64))
            np.ascontiguousarray(fin, np.float32).tofile(fin_cache)
            print("  （由 s%d 按 Euler 重建终点 latents，dt=%.4f）" % (STEPS - 1, dt7))
    dev_fin = fin_cache if os.path.isfile(fin_cache) else None
    if dev_fin:
        d = np.fromfile(dev_fin, np.float32).astype(np.float64)
        r = np.fromfile(ref_fin, np.float32).astype(np.float64)
        br, bc = bulk(d, r)
        print("  相对 L2 = %.4f%%   余弦 = %.6f" % (rel(d, r), cos(d, r)))
        print("  主体口径 相对 L2 = %.4f%%   余弦 = %.6f   集中度 = %.2f%%"
              % (br, bc, conc1(r)))
    else:
        print("  ⚠️ 找不到设备终点 latents，跳过 L2")

    # --- L3：成图 ---
    print()
    print("=== L3  成图（报告结论必须给这一层）===")
    x = np.asarray(Image.open(dev_png).convert("RGB"), np.float64)
    y = np.asarray(Image.open(ref_png).convert("RGB"), np.float64)
    mse = float(np.mean((x - y) ** 2))
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print("  PSNR = %.2f dB   像素相对 L2 = %.2f%%" % (psnr, rel(x.ravel(), y.ravel())))
    print()
    print("  对照：L=32 的 Clip+FP16 配置对**它自己的** FP32 参考是 18.67 dB")
    print("  ⚠️ 两个数各自对自己的参考，读的是「量化流水线跟得多紧」，")
    print("     **不是**两张图谁更好；且各自 n=1（#131：弱/强 prompt 可差 5.6 dB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
