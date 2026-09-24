"""【动手前的离线验证】percentile 校准到底能带来多少余量、代价是什么？

按约束 8：判据/方案的操作化必须先用已知样本验证，再投入数小时。

percentile 校准把 encoding 量程从 min-max 改成 |x| 的 p99.99（或其它分位），
超出部分被**钳位**。所以它同时有两个效果：
  收益：主体的量化步长变小 -> 有效位宽提升
  代价：被钳掉的元素值被彻底改变

关键风险：若被钳掉的那部分承载了大量能量，钳位本身就是破坏。
本脚本对每个已有 CPU 参考张量同时算收益与代价，用数据判断可行性。
"""
import os

import numpy as np

PAIRS = [
    (r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu\out\Result_0", None),
    (r"D:\ZImage_Work\p0_experiments\layer_probe\cpu\out\Result_0", None),
]
EXTRA = {
    "add_138": r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1a\out\Result_0\add_138.raw",
    "latents": r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part2\out\Result_0\latents.raw",
}
PCTS = [99.9, 99.99, 99.999]


def analyse(name, a):
    out = {"name": name, "n": a.size, "std": a.std()}
    lo, hi = a.min(), a.max()
    half_mm = max(abs(lo), abs(hi))
    scale_mm = (hi - lo) / 65535.0
    out["mm"] = (scale_mm, np.log2(max(a.std() / scale_mm, 1)))
    rows = []
    for p in PCTS:
        r = np.percentile(np.abs(a), p)
        scale_p = 2 * r / 65535.0
        eff = np.log2(max(a.std() / scale_p, 1))
        clipped = np.abs(a) > r
        n_clip = int(clipped.sum())
        e_clip = float((a[clipped] ** 2).sum() / (a ** 2).sum()) if n_clip else 0.0
        # 钳位后的误差：被钳元素变成 ±r
        b = np.clip(a, -r, r)
        err = 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)
        rows.append((p, scale_p, eff, n_clip, 100 * n_clip / a.size, 100 * e_clip, err))
    out["p"] = rows
    return out


def main():
    files = {}
    for d, _ in PAIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".raw"):
                files.setdefault(f[:-4], os.path.join(d, f))
    files.update(EXTRA)

    print(f"{'张量':<32}{'minmax有效bit':>14}{'分位':>8}{'新有效bit':>11}"
          f"{'增益bit':>9}{'钳位元素%':>11}{'钳掉能量%':>11}{'钳位误差%':>11}")
    print("=" * 110)
    risky, good = [], []
    for name, path in sorted(files.items()):
        a = np.fromfile(path, np.float32).astype(np.float64)
        if a.size < 1000:
            continue
        r = analyse(name, a)
        base = r["mm"][1]
        for (p, sc, eff, nc, pc, ec, err) in r["p"]:
            if p != 99.99:
                continue
            gain = eff - base
            print(f"{name:<32}{base:>14.2f}{p:>8}{eff:>11.2f}{gain:>+9.2f}"
                  f"{pc:>11.4f}{ec:>11.2f}{err:>11.2f}")
            (risky if ec > 5 else good).append((name, gain, ec, err))

    print()
    print("=" * 110)
    print("可行性判定（p99.99）")
    print("=" * 110)
    print(f"  安全（钳掉能量 ≤5%）：{len(good)} 个张量")
    if good:
        g = np.array([x[1] for x in good])
        print(f"    有效位宽增益 中位 {np.median(g):+.2f} bit  最大 {g.max():+.2f} bit")
    print(f"  ⚠ 危险（钳掉能量 >5%）：{len(risky)} 个张量")
    for n, gain, ec, err in sorted(risky, key=lambda x: -x[2])[:10]:
        print(f"    {n:<32} 钳掉能量 {ec:6.2f}%  钳位本身引入误差 {err:6.2f}%  "
              f"（增益 {gain:+.2f} bit）")


if __name__ == "__main__":
    main()
