# -*- coding: utf-8 -*-
"""mode S 实现前的前置检查：喂 MatMul 的激活还有没有别的消费者？

若「只喂静态权重 MatMul」，则可以把 A 整个换成 A/s、把 s 折进各个 W —— 图上等价、
运行时零代价（真实部署里 Div 还能折进上游的 RmsNorm 乘法，连这个 Div 都不存在）。
若还有别的消费者，就得为它们补回一个 Mul，实现复杂且不再零代价。

约束 8：这是 mode S 的**操作化前提**，动手写之前先用图本身验证。
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
    p = cs.onnx_for(seg, L=cs.DEPLOYED_L)
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    init = {i.name for i in g.initializer}
    gout = {o.name for o in g.output}

    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    acts = {}
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        a, w = n.input[0], n.input[1]
        if w in init:
            acts.setdefault(a, []).append(n)

    print("段 %s   喂静态权重 MatMul 的激活 %d 个" % (seg, len(acts)))
    clean, dirty = [], []
    kinds = Counter()
    for a, mms in acts.items():
        others = [n for n in cons.get(a, []) if n not in mms]
        if a in gout:
            others = others + ["<图输出>"]
        if others:
            dirty.append((a, len(mms), others))
            for o in others:
                kinds[o if isinstance(o, str) else o.op_type] += 1
        else:
            clean.append((a, len(mms)))

    print("  \U0001F7E2 只喂 MatMul（可整体替换 A -> A/s）: **%d** 个" % len(clean))
    print("  \U0001F534 还有别的消费者（需补回 Mul）      : **%d** 个" % len(dirty))
    if kinds:
        print("     其他消费者的算子分布: %s" % dict(kinds))
    for a, nm, others in dirty[:10]:
        oo = [o if isinstance(o, str) else "%s(%s)" % (o.op_type, o.output[0][:14])
              for o in others]
        print("     %-24s 喂 %d 个 MatMul + 其他 %s" % (a[:24], nm, ",".join(oo[:3])))


if __name__ == "__main__":
    main()
