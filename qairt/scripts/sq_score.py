# -*- coding: utf-8 -*-
"""给 sq_e2e 的三臂算三层指标（MAINLINE §2.5 的面板，参考原点一字不改）。

L1  step-0 噪声预测相对 L2      原点 p0_experiments/vs_fp32/latents_fp32_s0.raw
L2  终点 latents 相对 L2 + 主体余弦  原点 scratch_runs/fp32_final_latents_inloopref_vaefix.raw
L3  成图 PSNR / 像素相对 L2      原点 scratch_runs/recheck_new_fp32ref.png

🔴 措辞：这是**在 QDQ 模拟上**，不是设备实测。
⚠️ 约束 7：相对 L2 必须同时报「前 1% 元素占 ||a||^2 的比例」。
"""
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
SR = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
REF1 = os.path.join(P0, "vs_fp32", "latents_fp32_s0.raw")
REF2 = os.path.join(SR, "fp32_final_latents_inloopref_vaefix.raw")
REF3 = os.path.join(SR, "recheck_new_fp32ref.png")
ARMS = [("H", "H 部署 encoding（基线）"),
        ("S0", "S0 重标定 encoding"),
        ("S5", "S5 重标定 + SmoothQuant a=0.5")]
EPS = 1e-12


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    d = b - a
    full = float(np.linalg.norm(d) / (np.linalg.norm(a) + EPS))
    p99 = np.percentile(np.abs(a), 99)
    m = np.abs(a) <= p99
    bulk = float(np.linalg.norm(d[m]) / (np.linalg.norm(a[m]) + EPS))
    ca = a[m] - a[m].mean()
    cb = b[m] - b[m].mean()
    cos = float((ca * cb).sum() / (np.linalg.norm(ca) * np.linalg.norm(cb) + EPS))
    e2 = a ** 2
    k = max(1, int(round(0.01 * e2.size)))
    conc = float(np.sort(e2)[-k:].sum() / (e2.sum() + EPS))
    return full, bulk, cos, conc


def img(p):
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def main():
    r1 = np.fromfile(REF1, dtype=np.float32)
    r2 = np.fromfile(REF2, dtype=np.float32)
    r3 = img(REF3)
    print("参考原点：L1 %s / L2 %s / L3 %s\n"
          % (os.path.basename(REF1), os.path.basename(REF2), os.path.basename(REF3)))
    print("%-32s %10s %10s %10s | %10s %8s | %8s %9s"
          % ("臂", "L1全量", "L1主体", "L1集中", "L2全量", "L2余弦", "L3 PSNR", "L3像素L2"))
    print("-" * 108)
    out = {}
    for key, lbl in ARMS:
        w = os.path.join(P0, "sim_sq_%s" % key)
        n0 = os.path.join(w, "noise_0.raw")
        l8 = os.path.join(w, "lat_8.raw")
        png = os.path.join(SR, "sim_sq_%s.png" % key)
        if not (os.path.isfile(n0) and os.path.isfile(l8)):
            print("  %-30s 产物缺失" % lbl)
            continue
        a1 = np.fromfile(n0, dtype=np.float32)
        f1, b1, _c1, k1 = stats(r1, a1)
        a2 = np.fromfile(l8, dtype=np.float32)
        f2, _b2, c2, _k2 = stats(r2, a2)
        if os.path.isfile(png):
            y = img(png)
            if y.shape != r3.shape:
                ps, pl2 = float("nan"), float("nan")
            else:
                mse = float(((y - r3) ** 2).mean())
                ps = 10 * np.log10(255.0 ** 2 / max(mse, EPS))
                pl2 = float(np.linalg.norm(y - r3) / (np.linalg.norm(r3) + EPS))
        else:
            ps, pl2 = float("nan"), float("nan")
        out[key] = (f1, b1, f2, c2, ps, pl2)
        print("%-32s %9.2f%% %9.2f%% %10.4f | %9.2f%% %8.4f | %7.2f dB %8.2f%%"
              % (lbl, 100 * f1, 100 * b1, k1, 100 * f2, c2, ps, 100 * pl2))

    if "H" in out and "S0" in out:
        print("\n=== ① 重标定 encoding 的效应（H → S0）===")
        h, z = out["H"], out["S0"]
        print("  L1 全量 %+.1f%%   L2 全量 %+.1f%%   **L3 %+.2f dB**"
              % (100 * (h[0] - z[0]) / h[0], 100 * (h[2] - z[2]) / h[2], z[4] - h[4]))
    if "S0" in out and "S5" in out:
        print("\n=== ② SmoothQuant 的净效应（S0 → S5，共用同一份 encoding）===")
        z, f = out["S0"], out["S5"]
        print("  L1 全量 %+.1f%%   L2 全量 %+.1f%%   **L3 %+.2f dB**"
              % (100 * (z[0] - f[0]) / z[0], 100 * (z[2] - f[2]) / z[2], f[4] - z[4]))
    print("\n🔴 这是**在 QDQ 模拟上**，不是设备实测（#123 出现过模拟/设备比值 1.826 的离群）。")
    print("🔴 报告结论必须给 L3（§2.4）。")


if __name__ == "__main__":
    main()
