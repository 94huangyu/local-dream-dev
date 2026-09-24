"""评估 MSE/SQNR 校准：在"钳位损失"与"步长损失"之间求最优，而非硬砍分位。

percentile 已被证否（p99.99 钳掉 48~99.97% 的能量，钳位本身引入 30~99% 误差）。
MSE 校准的思路不同：搜索一个量程 r，使
    量化误差 = ||Q(clip(x,±r)) - x||
最小。若最优 r 接近 max|x|，说明这些张量本来就"没有多余量程可省"，
即分布病态是本质，换校准方式救不了 —— 这同样是有价值的结论。

零成本：只用已有的 CPU 参考张量，不跑设备、不重量化。
"""
import os

import numpy as np

DIRS = [
    r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu\out\Result_0",
    r"D:\ZImage_Work\p0_experiments\layer_probe\cpu\out\Result_0",
]
EXTRA = {
    "add_138": r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1a\out\Result_0\add_138.raw",
    "latents": r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part2\out\Result_0\latents.raw",
}
BW = 16


def quant_err(x, r):
    """对称量程 ±r、BW 位均匀量化后的相对误差（含钳位）。"""
    if r <= 0:
        return np.inf
    s = 2 * r / (2 ** BW - 1)
    q = np.clip(np.round(np.clip(x, -r, r) / s), -(2 ** (BW - 1)), 2 ** (BW - 1) - 1)
    return np.linalg.norm(q * s - x) / np.linalg.norm(x)


def main():
    files = {}
    for d in DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".raw"):
                    files.setdefault(f[:-4], os.path.join(d, f))
    files.update(EXTRA)

    print(f"{'张量':<32}{'minmax误差%':>12}{'MSE最优误差%':>13}{'改善倍数':>10}"
          f"{'最优r/max':>11}{'最优有效bit':>12}{'minmax有效bit':>14}")
    print("=" * 108)
    gains = []
    for name, path in sorted(files.items()):
        x = np.fromfile(path, np.float32).astype(np.float64)
        if x.size < 1000:
            continue
        # 抽样加速（保序、保极值）
        if x.size > 2_000_000:
            idx = np.random.default_rng(0).choice(x.size, 2_000_000, replace=False)
            xs = np.concatenate([x[idx], [x.min(), x.max()]])
        else:
            xs = x
        mx = np.abs(xs).max()
        e_mm = quant_err(xs, mx)
        # 在 [p90, max] 上搜索最优量程
        lo = np.percentile(np.abs(xs), 90)
        cands = np.geomspace(max(lo, mx * 1e-6), mx, 40)
        errs = [quant_err(xs, r) for r in cands]
        j = int(np.argmin(errs))
        r_opt, e_opt = cands[j], errs[j]
        ratio = e_mm / e_opt if e_opt > 0 else np.inf
        s_mm, s_opt = 2 * mx / 65535, 2 * r_opt / 65535
        gains.append((name, ratio, r_opt / mx, e_mm, e_opt))
        print(f"{name:<32}{100*e_mm:>12.4f}{100*e_opt:>13.4f}{ratio:>10.2f}"
              f"{r_opt/mx:>11.4f}{np.log2(max(xs.std()/s_opt,1)):>12.2f}"
              f"{np.log2(max(xs.std()/s_mm,1)):>14.2f}")

    print()
    print("=" * 108)
    print("判定")
    print("=" * 108)
    g = np.array([x[1] for x in gains])
    rr = np.array([x[2] for x in gains])
    print(f"  张量数 {len(gains)}")
    print(f"  MSE 相对 min-max 的量化误差改善倍数：中位 {np.median(g):.2f}x  "
          f"最大 {g.max():.2f}x  最小 {g.min():.2f}x")
    print(f"  最优量程 / max|x|：中位 {np.median(rr):.4f}")
    n_big = int((g >= 2).sum())
    print(f"  改善 ≥2 倍的张量：{n_big}/{len(gains)}")
    print()
    if np.median(g) >= 2:
        print("  => MSE 校准有实质收益，值得投入重量化验证。")
    elif np.median(g) >= 1.3:
        print("  => 收益有限（中位 <2 倍），需权衡：重量化成本约 1 小时/段。")
    else:
        print("  => 收益微弱：说明这些张量的最优量程已接近 max，")
        print("     分布病态是本质，换校准方式救不了。应转向其它方向。")


if __name__ == "__main__":
    main()
