"""EXP_PLAN_FP16_SURGICAL 判据落档（阈值事前锁定，§四）。"""
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")


def met(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return (100 * np.linalg.norm(y - x) / np.linalg.norm(x),
            float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
            int((~np.isfinite(b)).sum()))


def main():
    ref = np.fromfile(os.path.join(W, "fp32", "add_92.raw"), np.float32)
    rows = {}
    for tag in ("ctrl", "fp16"):
        p = os.path.join(W, "htp", "add_92_%s.raw" % tag)
        if not os.path.exists(p):
            print("缺 %s" % p); continue
        b = np.fromfile(p, np.float32)
        e, c, nb = met(ref, b)
        rows[tag] = dict(bulk=e, cos=c, inf=nb)
        print("  %-5s 主体相对L2 %8.4f%%   主体余弦 %.6f   inf/nan %d" % (tag, e, c, nb))
    if len(rows) < 2:
        print("\n两臂未齐"); return
    print("\n=== G4 数值门 ===")
    for t, r in rows.items():
        print("  %-5s inf/nan=%d  %s" % (t, r["inf"], "PASS" if r["inf"] == 0 else "FAIL"))
    d = rows["fp16"]["cos"] - rows["ctrl"]["cos"]
    print("\n=== 主判据（方案 §四，事前锁定）===")
    print("  控制臂主体余弦 %.6f  ->  FP16 臂 %.6f   提升 %+.6f" % (rows["ctrl"]["cos"], rows["fp16"]["cos"], d))
    if rows["fp16"]["inf"]:
        v = "❌ G4 未过（inf/nan）⇒ 关闭"
    elif d >= 0.03:
        v = "🟢 有效 ⇒ 推广到四段"
    elif d >= 0.005:
        v = "🟡 有限 ⇒ 评估性价比"
    else:
        v = "❌ 无改善或倒退 ⇒ 关闭 #106，「把激活换成浮点」整族穷尽"
    print("  判定：%s" % v)
    print("\n🔴 划界：唯一变量 = 8 个张量（7 SwiGLU + linear_55）不给 encoding；")
    print("   但浮点实际扩散到 153 个激活（8 + 其输入锥）⇒ 措辞应为「这一组改动」。")
    print("   本试点只做 part2a，不得外推四段/端到端/成图质量（#69/#70/#84）。")
    json.dump(rows, open(os.path.join(W, "verdict_106.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
