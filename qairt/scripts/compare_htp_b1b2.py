"""EXP_PLAN_HTP_VS_CPU_B1B2 的比对脚本。判据见方案第 3 节，此处只算数、不改判据。

实验A（逐段隔离）：两侧都喂 CPU 参考的上游输出 -> 该段自身的 HTP vs CPU 差异
实验B（级联）    ：HTP 全程用自己的上游输出   -> 设备上真实累积的总差异
"""
import numpy as np
import os

TB = r"D:\ZImage_Work\p0_experiments\testB"
HV = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu"
CALIB = 15.988


def stats(cpu_path, htp_path):
    a = np.fromfile(cpu_path, np.float32).astype(np.float64)
    b = np.fromfile(htp_path, np.float32).astype(np.float64)
    assert a.size == b.size, f"元素数不一致 {a.size} vs {b.size}"
    d = b - a
    rel = 100.0 * np.linalg.norm(d) / np.linalg.norm(a)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return a, b, d, rel, cos


CASES = [
    ("A 隔离 part1a  add_138", f"{TB}\\s0_transformer_part1a\\out\\Result_0\\add_138.raw",
     f"{HV}\\htp\\add_138.raw"),
    ("A 隔离 part1b  unified", f"{TB}\\s0_transformer_part1b\\out\\Result_0\\unified.raw",
     f"{HV}\\A_part1b\\unified.raw"),
    ("A 隔离 part2   latents", f"{TB}\\s0_transformer_part2\\out\\Result_0\\latents.raw",
     f"{HV}\\A_part2\\latents.raw"),
    ("B 级联 part1b  unified", f"{TB}\\s0_transformer_part1b\\out\\Result_0\\unified.raw",
     f"{HV}\\B_part1b\\unified.raw"),
    ("B 级联 part2   latents", f"{TB}\\s0_transformer_part2\\out\\Result_0\\latents.raw",
     f"{HV}\\B_part2\\latents.raw"),
]

print(f"{'用例':<26}{'元素数':>10}{'相对L2%':>11}{'余弦':>11}"
      f"{'最大|差|':>12}{'CPU std':>11}{'最大/σ':>9}")
print("=" * 92)
res = {}
for tag, p_cpu, p_htp in CASES:
    if not os.path.exists(p_htp):
        print(f"{tag:<26}  缺文件 {p_htp}")
        continue
    a, b, d, rel, cos = stats(p_cpu, p_htp)
    res[tag] = (rel, cos, a, d)
    print(f"{tag:<26}{a.size:>10}{rel:>11.4f}{cos:>11.6f}"
          f"{np.abs(d).max():>12.6g}{a.std():>11.6g}{np.abs(d).max()/a.std():>9.1f}")

print()
print("=" * 92)
print("判据 A（归因）：三段各自隔离条件下的相对 L2")
print("=" * 92)
iso = [(t, res[t][0]) for t in res if t.startswith("A ")]
for t, r in iso:
    print(f"  {t:<26} {r:.4f}%")
mx, mn = max(r for _, r in iso), min(r for _, r in iso)
print(f"  最大/最小 = {mx/mn:.2f} 倍  ->  ", end="")
print("存在主要差异源（≥3 倍）" if mx / mn >= 3 else "三段量级相当，差异是全局性的，非某个坏算子")

print()
print("=" * 92)
print("判据 B（量级判定，本轮核心）：级联后最终 latents")
print("=" * 92)
key = "B 级联 part2   latents"
if key in res:
    rel, cos, a, d = res[key]
    print(f"  相对 L2 = {rel:.4f}%     余弦 = {cos:.6f}")
    print(f"  标定点  = {CALIB}%（Test B 实测：噪声预测达此值时成图仍完好）")
    print()
    if rel < CALIB:
        print(f"  => {rel:.4f}% < {CALIB}%：【假设 C 不足以解释橙色色块】")
        print("     按方案第 3/5 节：必须回顶层重提假设，不得继续在 C 上深挖。")
    else:
        print(f"  => {rel:.4f}% >= {CALIB}%：C 超出已标定无害上界，成为根因【候选】")
        print("     但仍须单独论证'该量级能否造成色块'，且需补做 vs FP32 的对照。")
    print()
    print("  判据 C（离群形态）——最终 latents 的 |差| 分位数:")
    ad = np.abs(d)
    for q in [50, 90, 99, 99.9, 99.99, 100]:
        print(f"    p{q:<7} = {np.percentile(ad, q):.6g}   ({np.percentile(ad, q)/a.std():.2f}σ)")
    n20 = int((ad > 20 * a.std()).sum())
    print(f"    > 20σ 的元素数 = {n20}  ({100.0*n20/ad.size:.6f}%)")
    print("    -> " + ("存在显著离群点，需定位" if n20 > 0 else "无 >20σ 离群点，误差为弥散型"))

print()
print("=" * 92)
print("上游差异的传递（B级联 vs A隔离，同一张量）")
print("=" * 92)
for seg, ka, kb in [("part1b unified", "A 隔离 part1b  unified", "B 级联 part1b  unified"),
                    ("part2  latents", "A 隔离 part2   latents", "B 级联 part2   latents")]:
    if ka in res and kb in res:
        print(f"  {seg}: 隔离 {res[ka][0]:.4f}%  ->  级联 {res[kb][0]:.4f}%  "
              f"(放大 {res[kb][0]/res[ka][0]:.2f} 倍)")
