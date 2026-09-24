"""EXP_PLAN_ONDEVICE_PREPARE 判据：区分"宿主离线 prepare"与"HTP 执行语义"。

三方比对 part1b 的 `unified`（同一份输入：testB s0 的 CPU 参考上游输出）：
  CPU 参考(.dlc on SNPE CPU)  <- 基准
  宿主 .bin on HTP            <- 已实测 19.47%
  设备在线建图 .dlc on HTP    <- 本轮
"""
import numpy as np
import os

CPU = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b\out\Result_0\unified.raw"
BIN = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu\A_part1b\unified.raw"
DLC = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu\D_part1b\unified.raw"

R_BIN_KNOWN = 19.4727  # 宿主 .bin vs CPU 参考，来自 EXP_PLAN_HTP_VS_CPU_B1B2


def rel_cos(a, b):
    d = b - a
    return (100.0 * np.linalg.norm(d) / np.linalg.norm(a),
            float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            np.abs(d).max())


for p in (CPU, BIN, DLC):
    if not os.path.exists(p):
        print(f"缺文件: {p}")
        raise SystemExit(1)

cpu = np.fromfile(CPU, np.float32).astype(np.float64)
bin_ = np.fromfile(BIN, np.float32).astype(np.float64)
dlc = np.fromfile(DLC, np.float32).astype(np.float64)

print(f"元素数: CPU={cpu.size}  宿主.bin={bin_.size}  设备建图={dlc.size}")
exp = 15851520
ok = dlc.size == exp
print(f"判据 0: 设备建图输出 {dlc.size} == {exp} ? {'✅ 通过' if ok else '❌ 失败，本次无效'}")
if not ok:
    raise SystemExit(1)
print()

r_bin, c_bin, m_bin = rel_cos(cpu, bin_)
r_dlc, c_dlc, m_dlc = rel_cos(cpu, dlc)
r_bd, c_bd, m_bd = rel_cos(bin_, dlc)

print(f"{'对照':<34}{'相对L2%':>11}{'余弦':>12}{'最大|差|':>12}")
print("=" * 70)
print(f"{'宿主 .bin  vs CPU 参考  (R_bin)':<34}{r_bin:>11.4f}{c_bin:>12.6f}{m_bin:>12.4g}")
print(f"{'设备建图   vs CPU 参考  (R_dlc)':<34}{r_dlc:>11.4f}{c_dlc:>12.6f}{m_dlc:>12.4g}")
print(f"{'设备建图   vs 宿主 .bin (交叉)':<34}{r_bd:>11.4f}{c_bd:>12.6f}{m_bd:>12.4g}")

print()
print("=" * 70)
print("判据 1（EXP_PLAN_ONDEVICE_PREPARE 第 3 节，事前定稿）")
print("=" * 70)
lo, hi = R_BIN_KNOWN * 0.8, R_BIN_KNOWN * 1.2
print(f"  R_bin = {r_bin:.4f}%   R_dlc = {r_dlc:.4f}%   同量级区间 = [{lo:.2f}%, {hi:.2f}%]")
print()
if r_dlc < 2.0:
    print("  => R_dlc < 2%：设备在线建图与 CPU 参考基本一致")
    print("     【宿主离线 prepare 是元凶】—— 可修的工程问题。")
    print("     下一步：转向 prepare 参数 / SDK 版本，并在 part2 上复核。")
elif lo <= r_dlc <= hi:
    print("  => R_dlc 与 R_bin 同量级")
    print("     【prepare 无罪，HTP 执行语义本身与 SNPE CPU 参考不等价】")
    print("     下一步：转向精度配置（act_bitwidth / use_dynamic_16_bit_weights / per-channel）。")
else:
    print("  => 落在中间：两者都有贡献，不得只归因一边。")
    print(f"     设备建图消掉的份额 ≈ {(1 - r_dlc/r_bin)*100:.1f}%（按相对 L2 粗算）")

print()
print("判据 2（交叉检查）：设备建图 vs 宿主 .bin")
if r_bd < 1e-9:
    print("  ❌ 逐位相同 —— 疑似 --dlc_path 实际仍加载了缓存 .bin，结论无效，必须查清。")
else:
    print(f"  ✅ 两条路径产出不同结果（相对 L2 {r_bd:.4f}%），说明确实各自建了图。")
