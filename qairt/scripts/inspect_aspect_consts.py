# -*- coding: utf-8 -*-
"""读 part1a 里与「改比例」相关的几个烘焙常量的**内容**（不只是维度）。

为 #86 补齐最后两个未验证项：unsqueeze_3 与 unified_mask。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx

import canonical_sources as cs

NP = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16}
NAME = {1: "float32", 6: "int32", 7: "int64", 9: "bool", 10: "float16"}
WANT = ["unsqueeze_3", "unified_mask", "stack_1", "val_572", "val_566"]


def load(p, t):
    if not t.external_data:
        from onnx import numpy_helper
        return numpy_helper.to_array(t)
    d = {e.key: e.value for e in t.external_data}
    loc = os.path.join(os.path.dirname(p), d["location"])
    with open(loc, "rb") as f:
        f.seek(int(d.get("offset", 0)))
        raw = f.read(int(d["length"]))
    return np.frombuffer(raw, dtype=NP[t.data_type]).reshape([x for x in t.dims])


def main():
    p = cs.onnx_for("part1a")
    print("源: %s" % p)
    m = onnx.load(p, load_external_data=False)
    ini = {i.name: i for i in m.graph.initializer}
    for nm in WANT:
        if nm not in ini:
            print("  %-14s 不存在" % nm)
            continue
        t = ini[nm]
        a = load(p, t)
        u = np.unique(a)
        head = a.reshape(-1)[:8].tolist()
        print("  %-14s %-8s dims=%-16s 唯一值=%-6d 前8=%s" %
              (nm, NAME.get(t.data_type, "?"), str(list(t.dims)), u.size, head))
        if u.size <= 4:
            print("                 全部取值 = %s" % u.tolist())
        elif np.issubdtype(a.dtype, np.number):
            flat = a.reshape(-1)
            arith = np.array_equal(flat, np.arange(flat.size, dtype=flat.dtype))
            print("                 范围=(%s, %s)  是否 0..N-1 等差 = %s"
                  % (a.min(), a.max(), arith))


if __name__ == "__main__":
    main()
