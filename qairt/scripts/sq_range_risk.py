# -*- coding: utf-8 -*-
"""量化「重标定 encoding」的跨 prompt 风险：我的量程比出厂的紧多少？

背景（#150）：重标定让 part1b 单段误差降 ~22%，且跨**去噪步**留出验证通过。
但留出的是同一条 prompt 的不同步，**不是不同 prompt**。
我的 min-max 只在一条 prompt 上取 —— 换 prompt 若超出量程就会被钳。
这正是 #53/#67 的死法（percentile 砍量程换分辨率 => 净损害）。

本脚本不需要第二条 prompt，先回答一个更便宜的问题：
**我的量程相对出厂量程紧了多少？** 紧得越多，跨 prompt 越危险。

判据（事前锁定）：
  中位收紧比 <= 1.3x 且 无张量 > 3x  => 🟢 风险低，可继续
  中位 1.3~2x                        => 🟡 需要真做跨 prompt 检验
  中位 > 2x 或 有张量 > 5x            => 🔴 高危，等同 percentile，不得上机
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import sim_qdq

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else "part1b"
    kind, ch, pt = sim_qdq.parse(seg)
    st = json.load(io.open(os.path.join(P0, "smoothquant", "%s_sim32_chstats.json" % seg),
                           encoding="utf-8"))
    amin = {k: np.asarray(v, np.float64) for k, v in st["act_chmin_signed"].items()}
    amax = {k: np.asarray(v, np.float64) for k, v in st["act_chmax_signed"].items()}

    rows = []
    for a in sorted(set(amin) & set(pt)):
        bw, sc, off = pt[a]
        if bw != 16 or kind.get(a, ("", ""))[1] == "STATIC":
            continue
        dep_lo, dep_hi = off * sc, (2 ** bw - 1 + off) * sc
        dep_span = dep_hi - dep_lo
        my_lo, my_hi = float(amin[a].min()), float(amax[a].max())
        my_span = my_hi - my_lo
        if my_span <= 0 or dep_span <= 0:
            continue
        rows.append((a, dep_span, my_span, dep_span / my_span, dep_lo, dep_hi, my_lo, my_hi))

    if not rows:
        sys.exit("没有可比的张量")
    r = np.array([x[3] for x in rows])
    print("段 %s   可比激活 %d 个" % (seg, len(rows)))
    print("\n收紧比 = 出厂量程 / 我的量程（>1 表示我更紧）")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print("   p%-3d  %6.2fx" % (q, np.percentile(r, q)))
    print("\n  我更紧的 %d 个（%.0f%%）｜我更宽的 %d 个"
          % ((r > 1).sum(), 100 * (r > 1).mean(), (r <= 1).sum()))

    print("\n收紧最多的 8 个（跨 prompt 最危险）：")
    for x in sorted(rows, key=lambda z: -z[3])[:8]:
        print("   %-22s 出厂[%.4g, %.4g]  我的[%.4g, %.4g]  紧 %.2fx"
              % (x[0][:22], x[4], x[5], x[6], x[7], x[3]))

    med = float(np.median(r))
    mx = float(r.max())
    print("\n=== 判据（事前锁定）===")
    print("  中位收紧比 %.2fx｜最大 %.2fx" % (med, mx))
    if med <= 1.3 and mx <= 3.0:
        print("  ⇒ 🟢 风险低：出厂量程本就与实测接近，收益不是靠砍量程换来的")
    elif med <= 2.0:
        print("  ⇒ 🟡 中等：**必须真做跨 prompt 检验**再上机")
    else:
        print("  ⇒ 🔴 高危：等同 percentile 收紧（#53 已实测净损害），不得上机")


if __name__ == "__main__":
    main()
