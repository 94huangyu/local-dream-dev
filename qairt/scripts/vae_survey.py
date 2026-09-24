# -*- coding: utf-8 -*-
"""#86 前置勘察：VAE decoder 里与空间维绑定的东西有哪些？

part1a 的经验（已实测）：空间结构几乎不在形状常量里，只在 `stack_1` 这张坐标表上。
VAE 是卷积网络，Conv 权重与空间维无关；风险在 Reshape / 注意力块 / value_info。

用法: python vae_survey.py
"""
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
from onnx import numpy_helper

P = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx",
                 "vae_decoder.onnx")
# latent 128x128 -> 像素 1024x1024。与空间有关的数：
SPATIAL = {128: "latent边", 1024: "像素边", 16384: "128^2", 256: "?", 512: "中间边",
           64: "?", 32: "?", 1048576: "1024^2"}


def main():
    m = onnx.load(P, load_external_data=False)
    g = m.graph
    print("VAE %s" % P)
    print("  节点 %d｜initializer %d｜value_info %d"
          % (len(g.node), len(g.initializer), len(g.value_info)))
    print("  IO: %s -> %s"
          % ([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
              for i in g.input],
             [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
              for o in g.output]))

    print("\n=== 小整数 initializer 里含空间维的（改比例要动的形状常量）===")
    hits = []
    for ini in g.initializer:
        if ini.data_type not in (6, 7):        # int32 / int64
            continue
        n = 1
        for d in ini.dims:
            n *= d
        if n > 16:
            continue
        a = numpy_helper.to_array(ini)
        v = a.reshape(-1).tolist()
        if any(x in SPATIAL for x in v):
            hits.append((ini.name, v))
    for nm, v in hits:
        print("   %-24s %s" % (nm[:24], v))
    print("   合计 **%d** 个" % len(hits))

    print("\n=== 大整型 initializer（可能是烘焙的索引表）===")
    big = []
    for ini in g.initializer:
        if ini.data_type not in (6, 7):
            continue
        n = 1
        for d in ini.dims:
            n *= d
        if n >= 64:
            big.append((n, ini.name, list(ini.dims)))
    for n, nm, d in sorted(big, reverse=True)[:8]:
        print("   %-24s dims=%-16s 元素=%d" % (nm[:24], str(d), n))
    if not big:
        print("   （无）⇒ 没有烘焙的索引表要重算")

    print("\n=== value_info 里带空间维的（Tier2 §34.1：这些也必须改）===")
    c = Counter()
    for v in list(g.value_info) + list(g.input) + list(g.output):
        dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
        for x in dims:
            if x in SPATIAL:
                c[x] += 1
    for k, n in sorted(c.items(), key=lambda z: -z[1]):
        print("   维值 %-8d 出现在 %3d 个 value_info 里   (%s)" % (k, n, SPATIAL[k]))

    print("\n=== 注意力块（MatMul/Transpose）的形状 ===")
    init = {i.name: list(i.dims) for i in g.initializer}
    for n in g.node:
        if n.op_type in ("MatMul", "Softmax", "Transpose"):
            w = [i for i in n.input if i in init]
            print("   %-10s out=%-22s 权重=%s"
                  % (n.op_type, n.output[0][:22], [init[x] for x in w] or "-"))


if __name__ == "__main__":
    main()
