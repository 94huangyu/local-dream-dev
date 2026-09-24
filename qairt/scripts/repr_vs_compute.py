"""决定性判别：HTP 的误差来自【量化表示不足】还是【定点计算不足】？

这决定了修复路径：
  表示不足 -> 通道迁移（#20）可在 v79 上解决，零体积零速度代价
  计算不足 -> 需要 v81 的 PRECISION_COMPENSATION 或浮点，v79 无解

方法：把张量按"是否属于 massive activation 通道"分成两部分，
分别计算【纯量化表示误差】（用该张量自己的 encoding 量化再反量化）。

先前算的 0.32% 是范数加权的，而范数 99.58% 来自通道 85 ——
那个数字只说明通道 85 表示得好，没说明主体。
"""
import numpy as np

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b\out\Result_0\unified.raw"
HV = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu\A_part1b\unified.raw"
FP = r"D:\ZImage_Work\p0_experiments\vs_fp32\unified_fp32.raw"

# unified 的 encoding（来自 maskfix_part2_dlcinfo，作为 part2 输入）
LO, HI, BW = -145.277114868164, 6265.992675781250, 16
C = 3840


def quantize(x, lo, hi, bw):
    s = (hi - lo) / (2 ** bw - 1)
    q = np.clip(np.round((x - lo) / s), 0, 2 ** bw - 1)
    return q * s + lo, s


def rel(a, b):
    return 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)


def main():
    fp = np.fromfile(FP, np.float32).astype(np.float64)
    cpu = np.fromfile(TB, np.float32).astype(np.float64)
    htp = np.fromfile(HV, np.float32).astype(np.float64)

    F = fp.reshape(-1, C)
    e = (F ** 2).sum(axis=0)
    hot = int(np.argmax(e))
    share = e[hot] / e.sum()
    print(f"unified: 最高能通道 = ch{hot}，占总能量 {100*share:.2f}%")
    print(f"encoding: [{LO:.3f}, {HI:.3f}]  bw={BW}")
    print()

    qf, s = quantize(fp, LO, HI, BW)
    print(f"量化步长 s = {s:.6f}")
    print()
    print("=" * 82)
    print("【纯量化表示误差】把 FP32 真值用该 encoding 量化再反量化（不含任何计算）")
    print("=" * 82)

    mask_hot = np.zeros(C, bool)
    mask_hot[hot] = True
    idx_hot = np.tile(mask_hot, fp.size // C)
    idx_bulk = ~idx_hot

    for tag, m in [("全张量（范数加权）", slice(None)),
                   (f"仅 ch{hot}（massive activation）", idx_hot),
                   (f"其余 {C-1} 个通道（主体）", idx_bulk)]:
        a, b = fp[m], qf[m]
        print(f"  {tag:<34} 表示误差 = {rel(a, b):8.3f}%   "
              f"std={a.std():9.4f}  该部分占总能量 "
              f"{100*(a**2).sum()/(fp**2).sum():6.2f}%")

    print()
    print("=" * 82)
    print("对照：CPU 参考与 HTP 各自的【实测】误差（同样分部位看）")
    print("=" * 82)
    for tag, m in [("全张量", slice(None)),
                   (f"仅 ch{hot}", idx_hot),
                   (f"其余通道（主体）", idx_bulk)]:
        print(f"  {tag:<26} CPU {rel(fp[m], cpu[m]):8.3f}%   HTP {rel(fp[m], htp[m]):8.3f}%")

    print()
    print("=" * 82)
    print("判别")
    print("=" * 82)
    r_repr = rel(fp[idx_bulk], qf[idx_bulk])
    r_cpu = rel(fp[idx_bulk], cpu[idx_bulk])
    r_htp = rel(fp[idx_bulk], htp[idx_bulk])
    print(f"  主体通道：表示极限 {r_repr:.2f}%  |  CPU 实测 {r_cpu:.2f}%  |  HTP 实测 {r_htp:.2f}%")
    print()
    if r_repr > 30:
        print("  => 【表示不足是主因】主体通道的量化表示误差本身就很大，")
        print("     说明量程被 massive activation 绑架。")
        print("     ⇒ 通道迁移（#20）可在 v79 上解决，不必换芯片。")
    elif r_cpu < 5 and r_htp > 30:
        print("  => 【计算不足是主因】表示与 CPU 都没问题，只有 HTP 崩。")
        print("     ⇒ 指向 HTP 定点计算精度，v79 上可能无解（PRECISION_COMPENSATION 要 v81+）。")
    else:
        print("  => 两者都有贡献，按比例报告。")


if __name__ == "__main__":
    main()
