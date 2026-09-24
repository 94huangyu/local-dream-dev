"""HTP 的误差是不是"幅值越大压得越狠"？——零成本，用已有的三方数据直接分箱。

数据：part1b 的 unified，1585 万元素，同一输入下的
  FP32(真值) / SNPE CPU 参考 / 设备 HTP
按 |FP32| 幅值分箱，看每个箱里 HTP 与 CPU 各自的行为。

这是 15.11 节②推论「HTP 系统性把幅值算小，且幅值越大压得越狠」的直接检验。
"""
import numpy as np

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b\out\Result_0\unified.raw"
HV = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu\A_part1b\unified.raw"
FP = r"D:\ZImage_Work\p0_experiments\vs_fp32\unified_fp32.raw"

f = np.fromfile(FP, np.float32).astype(np.float64)
c = np.fromfile(TB, np.float32).astype(np.float64)
h = np.fromfile(HV, np.float32).astype(np.float64)
assert f.size == c.size == h.size

af = np.abs(f)
sd = f.std()
print(f"元素数 {f.size}   FP32 std = {sd:.4f}")
print()

# 按 |FP32| / std 分箱
edges = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 1e9]
print(f"{'|FP32|/σ 区间':<18}{'元素数':>11}{'占比%':>8}"
      f"{'CPU斜率':>10}{'HTP斜率':>10}{'CPU相对err%':>13}{'HTP相对err%':>13}")
print("=" * 84)

rows = []
for i in range(len(edges) - 1):
    lo, hi = edges[i] * sd, edges[i + 1] * sd
    m = (af >= lo) & (af < hi)
    n = int(m.sum())
    if n < 100:
        continue
    fv, cv, hv = f[m], c[m], h[m]
    # 斜率 = <x,y>/<x,x>：该箱里输出相对真值的整体缩放系数
    k_cpu = float(cv @ fv / (fv @ fv))
    k_htp = float(hv @ fv / (fv @ fv))
    e_cpu = 100.0 * np.linalg.norm(cv - fv) / np.linalg.norm(fv)
    e_htp = 100.0 * np.linalg.norm(hv - fv) / np.linalg.norm(fv)
    lbl = f"[{edges[i]:g}, {edges[i+1]:g})" if edges[i + 1] < 1e8 else f"[{edges[i]:g}, inf)"
    rows.append((lbl, n, k_cpu, k_htp, e_cpu, e_htp))
    print(f"{lbl:<18}{n:>11}{100.0*n/f.size:>8.3f}"
          f"{k_cpu:>10.4f}{k_htp:>10.4f}{e_cpu:>13.3f}{e_htp:>13.3f}")

print()
print("=" * 84)
print("解读")
print("=" * 84)
k_small = [r[3] for r in rows[:3]]
k_large = [r[3] for r in rows[-3:]]
print(f"  小幅值箱的 HTP 斜率 ≈ {np.mean(k_small):.4f}")
print(f"  大幅值箱的 HTP 斜率 ≈ {np.mean(k_large):.4f}")
if np.mean(k_large) < np.mean(k_small) - 0.05:
    print("  => HTP 斜率随幅值【下降】：确认『幅值越大压得越狠』")
elif abs(np.mean(k_large) - np.mean(k_small)) <= 0.05:
    print("  => HTP 斜率基本不随幅值变化：『越大压得越狠』不成立，")
    print("     误差更像是与幅值无关的加性噪声")
else:
    print("  => HTP 斜率随幅值【上升】，与推论相反")

print()
print("全局斜率（整张量）:")
print(f"  CPU: {float(c @ f / (f @ f)):.6f}")
print(f"  HTP: {float(h @ f / (f @ f)):.6f}")

print()
print("加性噪声视角：各箱内 (HTP-FP32) 的标准差是否恒定？")
print(f"{'|FP32|/σ 区间':<18}{'std(HTP-FP32)':>16}{'std(CPU-FP32)':>16}")
print("-" * 52)
for i in range(len(edges) - 1):
    lo, hi = edges[i] * sd, edges[i + 1] * sd
    m = (af >= lo) & (af < hi)
    if int(m.sum()) < 100:
        continue
    lbl = f"[{edges[i]:g}, {edges[i+1]:g})" if edges[i + 1] < 1e8 else f"[{edges[i]:g}, inf)"
    print(f"{lbl:<18}{(h[m]-f[m]).std():>16.4f}{(c[m]-f[m]).std():>16.4f}")
