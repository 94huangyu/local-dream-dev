# -*- coding: utf-8 -*-
"""SmoothQuant 的**代价门 G0-cost**（方案 `EXP_PLAN_SMOOTHQUANT.md` §二，判据事前锁定）。

问题：逐通道迁移之后，**激活的纯表示误差降了多少？权重的纯表示误差涨了多少？**

做法（全部在宿主，零设备）：对每个静态权重 MatMul
  s[j] = ax[j]^alpha / aw[j]^(1-alpha)      （SmoothQuant 原式）
  X' = X / s   （逐输入通道）
  W' = W * s   （逐输入通道，折进权重，运行时零代价）
然后分别做 quantize-dequantize：
  激活：per-tensor 非对称 uFxp_16（与部署一致）
  权重：per-channel 对称 int8（per-row，#38 实测部署版就是 sFxp_8 per-channel）

🔴 判据（事前锁定，不得改）：
  所有 alpha 上「激活改善」都小于「权重恶化」        => ❌ 直接否决
  存在 alpha 使**净表示误差改善 >= 20%（相对）**      => 🟢 进 §三 全模型模拟
  介于两者之间                                        => 🟡 记录，按成本再决定

⚠️ 这道门**只能否决不能肯定**：#52 已证表示误差不是 HTP 误差的预测量（37.6 倍）。

⚠️ 本脚本只用**逐通道 max**（ax/aw）重建量程，不是重跑真实张量 —— 这是 SmoothQuant
   原论文的做法，但意味着**激活误差是按「均匀分布在量程内」估的上界**，
   不是真实分布下的误差。这一点必须在结论里标注（②推算，非①实测）。

用法: python sq_g0cost.py [part1b]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
SQ = os.path.join(P0, "smoothquant")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-8


def q_err_pertensor_u16(mx):
    """per-tensor 非对称 uFxp_16 的**相对表示误差上界**（均匀量化，误差 ~ step/sqrt(12)）。

    量程 [0, mx]（取 |max|），step = mx / (2^16 - 1)。
    相对误差以「量程内均方根值」为分母；用同一分母比较迁移前后，分母可约掉，
    所以这里直接返回 step（正比于误差）。
    """
    return mx / 65535.0


def q_err_perchannel_i8(w_rowmax):
    """per-channel 对称 int8 的 step（每行一个 scale）。"""
    return w_rowmax / 127.0


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else "part1b"
    f = os.path.join(SQ, "%s_L80_chstats.json" % seg)
    d = json.load(open(f, encoding="utf-8"))
    print("读 %s  （源 %s）" % (os.path.basename(f), os.path.basename(d["onnx"])))
    print("  静态权重 MatMul %d 个｜激活张量 %d 个\n" % (d["n_matmul_static"], d["n_act"]))

    act = {k: np.asarray(v, dtype=np.float64) for k, v in d["act_chmax"].items()}
    wmx = {k: np.asarray(v, dtype=np.float64) for k, v in d["w_rowmax"].items()}

    print("%-6s %10s %10s %10s %10s" % ("alpha", "激活step", "权重step", "净(几何)", "vs alpha=0"))
    print("-" * 52)
    base = None
    rows = []
    for al in ALPHAS:
        ea, ew, n = 0.0, 0.0, 0
        for pr in d["pairs"]:
            a, w = pr["act"], pr["w"]
            if a not in act or w not in wmx:
                continue
            ax, aw = act[a], wmx[w]
            if ax.shape != aw.shape:
                continue                      # K 对不上的（转置位/Gemm）跳过
            s = np.power(np.maximum(ax, EPS), al) / np.power(np.maximum(aw, EPS), 1.0 - al)
            s = np.maximum(s, EPS)
            # 迁移后：激活量程按逐通道除 s 之后的**全局 max**（per-tensor 量化看全局）
            ax2 = (ax / s).max()
            aw2 = (aw * s)
            ea += q_err_pertensor_u16(ax2)
            ew += q_err_perchannel_i8(aw2).mean()
            n += 1
        # 几何平均作为“净”指标：两项量纲不同，不能直接相加
        net = float(np.sqrt((ea / n) * (ew / n)))
        if base is None:
            base = net
        rows.append((al, ea / n, ew / n, net, 100.0 * (base - net) / base))
        print("%-6.2f %10.3e %10.3e %10.3e %9.1f%%" % rows[-1])

    print("\n  （对照的 %d 个 MatMul；alpha=0 即不做迁移，等于现状）" % n)
    best = max(rows, key=lambda r: r[4])
    print("\n=== G0-cost 判据（事前锁定）===")
    print("  最优 alpha = %.2f，净表示误差改善 **%.1f%%**" % (best[0], best[4]))
    if best[4] >= 20.0:
        print("  ⇒ 🟢 PASS（>= 20%）：进入 §三 全模型 QDQ 模拟")
    elif best[4] <= 0.0:
        print("  ⇒ ❌ 否决：所有 alpha 上激活改善都不敌权重恶化，E 关闭")
    else:
        print("  ⇒ 🟡 介于两者之间（0 < %.1f%% < 20%%）：记录，按 §三 的成本再决定" % best[4])
    out = os.path.join(SQ, "%s_g0cost.json" % seg)
    json.dump({"seg": seg, "n_pairs": n,
               "rows": [{"alpha": r[0], "act_step": r[1], "w_step": r[2],
                         "net": r[3], "improve_pct": r[4]} for r in rows]},
              open(out, "w", encoding="utf-8"), indent=1)
    print("  写出 %s" % out)


if __name__ == "__main__":
    main()
