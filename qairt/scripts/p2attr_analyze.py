"""EXP_PLAN_P2_ATTR 判据落档。判据逐字取自方案 §四（事前锁定）。

真值 = FP32 ONNX（方案 §三 已明写改道：SNPE CPU 跑不了 per-row DLC，见 #105）。
⚠️ 参照系：cos(X) 是「HTP vs FP32 真值」，**包含量化误差本身**，不只是 HTP 特有误差。

度量（约束 7）：主体相对 L2（|a|<=p99）+ 主体余弦 + 前 1% 能量占比。
🔴 V2 可比性门：参与比较的张量主体余弦必须 > 0.9（#97 教训：废墟里定位无意义）。
"""
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
FP32, HTP = os.path.join(R, "fp32"), os.path.join(R, "htp")
# ONNX 名 -> 设备侧 DLC 名（FullyConnected 融合后带 _fc 后缀）
DLCNAME = {"linear_1": "linear_1_fc", "linear_4": "linear_4_fc",
           "linear_5": "linear_5_fc", "linear_9": "linear_9_fc"}


def met(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    bulk = 100 * np.linalg.norm(y - x) / np.linalg.norm(x)
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    e = np.sort(a * a)[::-1]
    return bulk, cos, 100 * e[:max(1, e.size // 100)].sum() / e.sum()


def main():
    chain = json.load(open(os.path.join(R, "chain.json"), encoding="utf-8"))
    rows, miss = {}, []
    for t, info in chain.items():
        pf = os.path.join(FP32, t + ".raw")
        ph = os.path.join(HTP, DLCNAME.get(t, t) + ".raw")
        if not (os.path.exists(pf) and os.path.exists(ph)):
            miss.append(t); continue
        a = np.fromfile(pf, np.float32); b = np.fromfile(ph, np.float32)
        if a.size != b.size:
            print("  %-30s ❌ 元素数 %d vs %d" % (t, a.size, b.size)); continue
        bulk, cos, top1 = met(a, b)
        rows[t] = dict(op=info["op"], bulk=bulk, cos=cos, top1=top1, up=info["up"],
                       inf=int((~np.isfinite(b)).sum()))
    if miss:
        print("缺产物（设备侧未拉回？）:", miss)
    if not rows:
        print("\n没有可分析的配对 —— 先把设备侧 12 个张量拉到", HTP); return

    print("%-30s %-14s %10s %10s %8s %6s" % ("tensor", "op", "主体相对L2", "主体余弦", "top1%", "inf"))
    print("-" * 84)
    for t, r in rows.items():
        print("%-30s %-14s %9.4f%% %10.6f %7.2f%% %6d"
              % (t, r["op"], r["bulk"], r["cos"], r["top1"], r["inf"]))

    print("\n=== V2 可比性门（主体余弦 > 0.9）===")
    ok = {t: r for t, r in rows.items() if r["cos"] > 0.9}
    print("  通过 %d / %d ；退出定量比较: %s"
          % (len(ok), len(rows), [t for t in rows if t not in ok]))

    print("\n=== 主判据：按算子类型聚合 Δcos ===")
    agg = {}
    for t, r in ok.items():
        ups = [u for u in r["up"] if u in ok]
        c_in = max((ok[u]["cos"] for u in ups), default=1.0)   # 图输入误差为 0 => cos=1
        d = c_in - r["cos"]
        agg.setdefault(r["op"], []).append((t, d))
        print("  %-30s %-14s cos_in=%.6f -> cos_out=%.6f   Δcos=%+.6f"
              % (t, r["op"], c_in, r["cos"], d))
    print()
    tot = {k: sum(d for _, d in v) for k, v in agg.items()}
    for k in sorted(tot, key=lambda z: -tot[z]):
        print("  %-16s Δcos 总和 %+.6f  （%d 个算子，中位 %+.6f）"
              % (k, tot[k], len(agg[k]), np.median([d for _, d in agg[k]])))
    rms = tot.get("Mul", 0.0)                # RmsNorm 输出节点在 ONNX 里是 Mul
    other = sum(v for k, v in tot.items() if k != "Mul")
    print("\n  RmsNorm(Mul) %+.6f   其余合计 %+.6f" % (rms, other))
    if rms >= 2 * other:
        v = "✅ 与 part1b 同机制（RmsNorm 主导）⇒ #98 成立，无新杠杆，收口"
    elif other >= 2 * rms:
        v = "🟢 非 RmsNorm 主导 ⇒ **新目标**，按该算子类型展开"
    else:
        v = "🔶 两者相当 ⇒ 报告比例，不得宣称任一方"
    print("  判定：%s" % v)
    print("\n🔴 参照系：cos 是「HTP vs FP32 真值」，含量化误差本身（方案 §三 改道声明）")
    json.dump({t: {k: r[k] for k in ("op", "bulk", "cos", "top1")} for t, r in rows.items()},
              open(os.path.join(R, "verdict.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
