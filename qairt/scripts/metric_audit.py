"""【度量审计】用正确口径重算本轮所有关键数字，判定哪些结论受影响。

发现的问题：相对 L2 = ||b-a||/||a|| 被 ||a|| 里的离群值主导。
对"量程被撑大数千倍"的张量，全量口径可低估达 48 倍（linear_153_fc: 5.68% -> 274.99%）。

三种口径并列，任何结论都必须说明用的是哪一种：
  full  : ||b-a||/||a||                 全量（旧口径，离群主导）
  bulk  : 只取 |a| <= p99 的元素          主体（剔除离群主导）
  medrel: median(|b-a|/|a|)，|a|>0.1σ    逐元素相对误差中位数（最不受分布影响）
  cos   : 余弦相似度                      与幅值无关，只看方向
"""
import os

import numpy as np

TB = r"D:\ZImage_Work\p0_experiments\testB"
HV = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu"
VF = r"D:\ZImage_Work\p0_experiments\vs_fp32"
CC = r"D:\ZImage_Work\p0_experiments\ch85_causal"


def metrics(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    full = 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)
    p99 = np.percentile(np.abs(a), 99)
    m = np.abs(a) <= p99
    bulk = 100.0 * np.linalg.norm((b - a)[m]) / np.linalg.norm(a[m])
    sig = np.abs(a) > 0.1 * a.std()
    med = 100.0 * np.median(np.abs(b - a)[sig] / np.abs(a)[sig]) if sig.sum() else np.nan
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # 分布形态：量程被离群撑大的倍数
    stretch = np.abs(a).max() / max(np.percentile(np.abs(a), 99.9), 1e-12)
    return full, bulk, med, cos, stretch


def rd(p):
    return np.fromfile(p, np.float32) if os.path.exists(p) else None


CASES = [
    ("逐段隔离 part1a add_138",
     f"{TB}\\s0_transformer_part1a\\out\\Result_0\\add_138.raw", f"{HV}\\htp\\add_138.raw"),
    ("逐段隔离 part1b unified",
     f"{TB}\\s0_transformer_part1b\\out\\Result_0\\unified.raw", f"{HV}\\A_part1b\\unified.raw"),
    ("逐段隔离 part2 latents",
     f"{TB}\\s0_transformer_part2\\out\\Result_0\\latents.raw", f"{HV}\\A_part2\\latents.raw"),
    ("级联 part1b unified",
     f"{TB}\\s0_transformer_part1b\\out\\Result_0\\unified.raw", f"{HV}\\B_part1b\\unified.raw"),
    ("★级联最终 latents（噪声预测）",
     f"{TB}\\s0_transformer_part2\\out\\Result_0\\latents.raw", f"{HV}\\B_part2\\latents.raw"),
    ("CPU参考 vs FP32 (part1b unified)",
     f"{VF}\\unified_fp32.raw", f"{TB}\\s0_transformer_part1b\\out\\Result_0\\unified.raw"),
    ("HTP vs FP32 (part1b unified)",
     f"{VF}\\unified_fp32.raw", f"{HV}\\A_part1b\\unified.raw"),
    ("ch85: 全HTP unified 喂 CPU part2",
     f"{TB}\\s0_transformer_part2\\out\\Result_0\\latents.raw",
     f"{CC}\\全HTP_U_htp\\out\\Result_0\\latents.raw"),
]

print(f"{'用例':<36}{'full%':>9}{'bulk%':>9}{'medrel%':>10}{'余弦':>10}{'量程撑大':>10}{'倍差':>7}")
print("=" * 92)
for tag, pa, pb in CASES:
    a, b = rd(pa), rd(pb)
    if a is None or b is None or a.size != b.size:
        print(f"{tag:<36}(缺文件或尺寸不符)")
        continue
    f, bk, me, c, st = metrics(a, b)
    r = bk / f if f > 1e-9 else np.nan
    flag = "  <<<" if r > 2 else ""
    print(f"{tag:<36}{f:>9.2f}{bk:>9.2f}{me:>10.2f}{c:>10.5f}{st:>10.1f}{r:>7.1f}{flag}")

print()
print("=" * 92)
print("关键判定：标定点 15.988% 与当前数字是否可比？")
print("=" * 92)
a = rd(f"{TB}\\s0_transformer_part2\\out\\Result_0\\latents.raw")
if a is not None:
    aa = a.astype(np.float64)
    st = np.abs(aa).max() / np.percentile(np.abs(aa), 99.9)
    print(f"  latents（噪声预测）的分布：std={aa.std():.4f}  "
          f"max|x|={np.abs(aa).max():.4f}  p99.9={np.percentile(np.abs(aa),99.9):.4f}")
    print(f"  量程被离群撑大 {st:.2f} 倍")
    print(f"  -> {'分布温和，full 与 bulk 口径接近，标定点可比' if st < 3 else '离群显著，两口径会分叉，标定点需重算'}")
