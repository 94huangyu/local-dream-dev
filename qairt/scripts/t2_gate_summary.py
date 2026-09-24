# -*- coding: utf-8 -*-
"""Tier 2 门 最终汇总：判决 + 余量 + L=32/L=80 同口径配对统计。

方案 `scripts/EXP_PLAN_T2_CLIPSET.md`。本脚本只做汇总，不改判据。
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_clipset_gate as G                                  # noqa: E402

W, FP16MAX = G.W, G.FP16MAX
SEGS = ("part1a", "part1b", "part2a", "part2b")


def base(t):
    return t[:-3] if t.endswith("_fc") else t


def main():
    print("=== 一、四段余量（L=80 实测）===")
    print("%-8s %5s %5s %-24s %12s %9s" %
          ("段", "S32", "S80", "未入选里最高的张量", "L80 |a|max", "余量"))
    for seg in SEGS:
        s32 = set(json.load(io.open(os.path.join(W, "%s_clipped.json" % seg),
                                    encoding="utf-8")))
        picked, _w, allv = G.select(seg, os.path.join(W, "%s_L80_amax.json" % seg))
        un = sorted(((h, o) for o, (h, src) in allv.items()
                     if o not in picked and src == "实测"), reverse=True)
        h, o = un[0]
        print("%-8s %5d %5d %-24s %12.0f %8.2fx" %
              (seg, len(s32), len(picked), o, h, FP16MAX / h))

    print()
    print("=== 二、同口径 L=32 vs L=80 配对（唯一能谈「L 的影响」的证据）===")
    pairs = []
    # part1b：全量 552 个实测
    a32 = {k: v["amax"] for k, v in json.load(
        io.open(os.path.join(W, "part1b_amax.json"), encoding="utf-8"))["tensors"].items()}
    a80 = {k: v["amax"] for k, v in json.load(
        io.open(os.path.join(W, "part1b_L80_amax.json"), encoding="utf-8"))["tensors"].items()}
    for k in a32:
        if k in a80 and a32[k] > 0:
            pairs.append(("part1b", k, a32[k], a80[k]))
    # part2a：risky_stats.json（已核实产出脚本 p2_risky_stats.py 的 BASE=transformer_part2a_fixed）
    r32 = json.load(io.open(os.path.join(W, "risky_stats.json"), encoding="utf-8"))
    r80 = {}
    for k, v in json.load(io.open(os.path.join(W, "part2a_L80_amax.json"),
                                  encoding="utf-8"))["tensors"].items():
        r80.setdefault(base(k), v["amax"])
    for k, v in r32.items():
        if k in r80 and v["amax"] > 0:
            pairs.append(("part2a", k, v["amax"], r80[k]))
    # part1a：#110 记录在 MAINLINE 的 mul_131 实测
    m80 = json.load(io.open(os.path.join(W, "part1a_L80_amax.json"),
                            encoding="utf-8"))["tensors"]["mul_131"]["amax"]
    pairs.append(("part1a", "mul_131(#110)", 27819.0, m80))

    r = np.array([c / b for _s, _k, b, c in pairs])
    big = [(c / b, s, k, b, c) for s, k, b, c in pairs if b > 5000]
    rb = np.array([x[0] for x in big])
    print("全部配对 n=%d：比值 min %.4f 中位 %.4f max %.4f" % (len(r), r.min(), np.median(r), r.max()))
    print("决策相关区间（L32 实测 >5000）n=%d：比值 min %.4f 中位 %.4f max %.4f，>1.05 的 %d 个"
          % (len(rb), rb.min(), np.median(rb), rb.max(), int((rb > 1.05).sum())))
    print()
    print("  最大的 6 个张量：")
    print("  %-8s %-18s %11s %11s %8s" % ("段", "tensor", "L32 实测", "L80 实测", "比值"))
    for _rr, s, k, b, c in sorted(big, key=lambda x: -x[3])[:6]:
        print("  %-8s %-18s %11.0f %11.0f %8.4f" % (s, k, b, c, c / b))
    print()
    print("⚠️ 这些配对同时差着三样：长度、caption 内容（猫 vs 渔夫）、"
          "输入链来源（L32 侧是 testB 的 CPU 参考链，#116）")
    print("   ⇒ 三者**捆绑不可分离**；但对安全性问题，该散布**框住了三者合计的影响**。")

    print()
    print("=== 三、标定 vs L=80 实测（这才是真正的缺陷所在）===")
    print("%-8s %-14s %12s %12s %8s" % ("段", "tensor", "标定", "L80 实测", "标定/实测"))
    kind, pt = G.load_enc("part1a")
    j = json.load(io.open(os.path.join(W, "part1a_L80_amax.json"), encoding="utf-8"))["tensors"]
    for t in ("linear_18", "linear_49", "mul_131", "linear_23_fc", "linear_7", "linear_10"):
        if t not in pt or t not in j:
            continue
        bw, sc, off = pt[t]
        cal = max(abs((2 ** bw - 1 + off) * sc), abs(off * sc))
        me = j[t]["amax"]
        print("%-8s %-14s %12.0f %12.0f %8.2fx" % ("part1a", t, cal, me, cal / me))
    return 0


if __name__ == "__main__":
    sys.exit(main())
