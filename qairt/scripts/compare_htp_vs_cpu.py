"""EXP_PLAN_HTP_VS_CPU 的比对脚本：设备 HTP 跑 .bin  vs  SNPE CPU 参考跑 .dlc。

两侧输入逐字节相同（同一批 testB step0 的 float32 .raw），
两侧输出都是 float32（qnn-net-run 非 native 模式；SNPE 侧见 EXECUTION_MODEL 规则 2）。

判据见 scripts/EXP_PLAN_HTP_VS_CPU.md 第 4 节，此处只算数、不改判据。
"""
import numpy as np
import os

CPU = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1a\out\Result_0"
HTP = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu\htp"

CALIB = 15.988  # Test B 实测标定点：噪声预测相对 L2 达此值时成图仍完好

names = sorted(f for f in os.listdir(CPU) if f.endswith(".raw"))
print(f"{'张量':<20}{'元素数':>10}{'相对L2%':>12}{'余弦':>12}{'最大绝对差':>14}"
      f"{'CPU std':>12}{'HTP std':>12}")
print("=" * 92)

rows = []
for n in names:
    p_cpu, p_htp = os.path.join(CPU, n), os.path.join(HTP, n)
    if not os.path.exists(p_htp):
        print(f"{n:<20}  <- HTP 侧缺失")
        continue
    a = np.fromfile(p_cpu, dtype=np.float32)
    b = np.fromfile(p_htp, dtype=np.float32)
    if a.size != b.size:
        print(f"{n:<20}  <- 元素数不一致 CPU={a.size} HTP={b.size}（判据0失败）")
        continue
    diff = b.astype(np.float64) - a.astype(np.float64)
    na = np.linalg.norm(a.astype(np.float64))
    rel = 100.0 * np.linalg.norm(diff) / na if na > 0 else float("nan")
    denom = na * np.linalg.norm(b.astype(np.float64))
    cos = float(a.astype(np.float64) @ b.astype(np.float64) / denom) if denom > 0 else float("nan")
    rows.append((n.replace(".raw", ""), a.size, rel, cos, np.abs(diff).max(),
                 a.std(), b.std()))
    print(f"{rows[-1][0]:<20}{a.size:>10}{rel:>12.4f}{cos:>12.6f}"
          f"{np.abs(diff).max():>14.6g}{a.std():>12.6g}{b.std():>12.6g}")

print()
print("=" * 92)
print("判据 1（EXP_PLAN_HTP_VS_CPU 第 4 节，事前定稿）")
print("=" * 92)
worst = max(rows, key=lambda r: r[2])
best_cos = min(rows, key=lambda r: r[3])
print(f"  最大相对 L2 : {worst[2]:.4f}%   （张量 {worst[0]}）")
print(f"  最小余弦     : {best_cos[3]:.6f}   （张量 {best_cos[0]}）")
print()
if all(r[2] < 1.0 for r in rows) and all(r[3] > 0.9999 for r in rows):
    print("  => 全部 < 1% 且余弦 > 0.9999：【H-C 被否定】HTP 与 CPU 参考等价。")
    print("     注意：按第 5 节，必须先确认本次确实跑在 HTP 上，否则结论无效。")
elif any(r[2] > CALIB for r in rows):
    bad = [r for r in rows if r[2] > CALIB]
    print(f"  => 存在 {len(bad)} 个张量相对 L2 > {CALIB}%（标定点）：【H-C 成立】")
    print("     且差异达到'可能致命'量级。第一个显著偏离的张量见下：")
    for r in sorted(bad, key=lambda x: -x[2]):
        print(f"       {r[0]:<20} 相对L2={r[2]:.4f}%  余弦={r[3]:.6f}")
else:
    print(f"  => 全部落在 1% ~ {CALIB}% 之间：差异存在，但在【已知无害区间】内。")
    print("     按判据：不得据此宣称找到根因。")
