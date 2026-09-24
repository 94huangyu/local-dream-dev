# -*- coding: utf-8 -*-
"""SmoothQuant 勘察：枚举某段里所有 MatMul 的 (激活输入, 权重, K, M)。

方案 `scripts/EXP_PLAN_SMOOTHQUANT.md` §四的前置：先知道要测哪些张量、规模多大，
再决定逐通道激活统计怎么跑。

用法: python sq_survey.py [part1a|part1b|part2a|part2b]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import onnx

import canonical_sources as cs


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else "part1b"
    p = cs.onnx_for(seg)
    print("段 %s   源 %s" % (seg, p))
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    init = {i.name: list(i.dims) for i in g.initializer}
    node_out = {o for n in g.node for o in n.output}
    gin = {i.name for i in g.input}

    rows = []
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        a, w = n.input[0], n.input[1]
        # 权重必须是 initializer（静态权重）；否则是 attention 里的 A@B，不适用 SmoothQuant
        if w in init:
            rows.append((n.output[0], a, w, init[w], "静态权重"))
        elif a in init:
            rows.append((n.output[0], w, a, init[a], "静态权重(转置位)"))
        else:
            rows.append((n.output[0], a, w, None, "动态x动态"))

    static = [r for r in rows if r[3] is not None]
    dyn = [r for r in rows if r[3] is None]
    print("  MatMul/Gemm 共 %d 个：静态权重 %d、动态×动态 %d（后者是注意力 A@B，"
          "**SmoothQuant 不适用**）" % (len(rows), len(static), len(dyn)))

    print("\n  === 静态权重的（SmoothQuant 的作用对象）===")
    shapes = Counter()
    acts = {}
    for out, a, w, wd, _ in static:
        shapes[tuple(wd)] += 1
        acts.setdefault(a, []).append(out)
    for sh, c in shapes.most_common():
        print("    权重形状 %-16s x%d" % (str(list(sh)), c))
    print("\n  唯一的激活输入张量 **%d** 个（这就是要做逐通道统计的集合）" % len(acts))
    multi = {a: v for a, v in acts.items() if len(v) > 1}
    if multi:
        print("  其中 %d 个被多个 MatMul 共用（QKV 那类）：" % len(multi))
        for a, v in list(multi.items())[:5]:
            print("     %-24s -> %s" % (a, ",".join(x[:16] for x in v[:4])))
    print("\n  前 10 个激活输入：")
    for a in list(acts)[:10]:
        src = "图输入" if a in gin else ("节点输出" if a in node_out else "?")
        print("     %-26s (%s)" % (a, src))


if __name__ == "__main__":
    main()
