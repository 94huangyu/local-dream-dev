"""EXP_PLAN_ACTFP16 判据落档。判据逐字取自方案 §四（事前锁定）。

真值 = vs_fp32/unified_fp32.raw（FP32 参考）。
度量按约束 7：全量 + 主体（|a|<=p99）+ 主体余弦 + 前 1% 能量占比。
🔴 unified 集中度 99.83% ⇒ **以主体口径为准**（全量口径对它不合法）。
🔴 参照臂必须是**同装置重跑**的 per-row，不引用历史数字（#95）。
"""
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
REF = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "vs_fp32", "unified_fp32.raw")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "actfp16")
COS_BASE = 0.7986      # #47 记录的 per-row 主体余弦（仅作参考，判据用同装置重跑值）


def met(a, b):
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    full = 100 * np.linalg.norm(b - a) / np.linalg.norm(a)
    x, y = a[m], b[m]
    bulk = 100 * np.linalg.norm(y - x) / np.linalg.norm(x)
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    e = np.sort(a * a)[::-1]
    return full, bulk, cos, 100 * e[:max(1, e.size // 100)].sum() / e.sum()


def main():
    a = np.fromfile(REF, np.float32).astype(np.float64)
    rows = {}
    for tag in ("perrow", "actfp16"):
        p = os.path.join(W, "unified_%s.raw" % tag)
        if not os.path.exists(p):
            print("  %-8s 缺产物 %s" % (tag, p)); continue
        b = np.fromfile(p, np.float32).astype(np.float64)
        if b.size != a.size:
            print("  %-8s ❌ 元素数不符 %d vs %d" % (tag, b.size, a.size)); continue
        nbad = int((~np.isfinite(b)).sum())
        f, bl, c, t1 = met(a, b)
        rows[tag] = dict(full=f, bulk=bl, cos=c, inf_nan=nbad)
        print("  %-8s 全量 %8.4f%%  主体 %8.4f%%  主体余弦 %.4f   inf/nan %d"
              % (tag, f, bl, c, nbad))
    if not rows:
        return
    print("\n[约束 7] 真值前 1%% 能量占比 %.2f%% ⇒ 以主体口径为准" % met(a, a * 1.0)[3])
    print("\n=== G4 数值门 ===")
    for t, r in rows.items():
        print("  %-8s inf/nan = %d  %s" % (t, r["inf_nan"], "PASS" if r["inf_nan"] == 0 else "FAIL"))
    if "actfp16" not in rows:
        return
    cf = rows["actfp16"]["cos"]
    base = rows.get("perrow", {}).get("cos")
    print("\n=== 主判据（方案 §四）===")
    print("  actfp16 主体余弦 = %.4f" % cf)
    if base is not None:
        print("  同装置重跑 per-row 主体余弦 = %.4f （#47 历史记录 %.4f，仅供对照）" % (base, COS_BASE))
    if rows["actfp16"]["inf_nan"]:
        v = "❌ G4 未过（出现 inf/nan）⇒ 关闭"
    elif cf > 0.90:
        v = "🟢 显著改善 ⇒ 立刻推进 part2，不得在 part1b 继续打磨"
    elif cf > 0.80:
        v = "🟡 有改善 ⇒ 评估是否值得四段全做"
    else:
        v = "❌ 无改善或倒退 ⇒ 关闭 #102"
    print("  判定：%s" % v)
    print("\n🔴 划界：本试点只做 part1b；结论不得外推 part2 / 端到端 / 成图质量")
    print("   （#69/#70/#84：三把标量尺与成图质量反相关，最终仍须人工看图）")
    print("🔴 变量不纯：激活 + gamma + bias 三者一起 FP16（方案 §三·补）")
    json.dump(rows, open(os.path.join(W, "verdict.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
