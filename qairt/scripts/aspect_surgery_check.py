# -*- coding: utf-8 -*-
"""EXP_PLAN_MULTIGRAPH §6.4 的门 S2 / S4 / S5（结构类，零推理）。

S2 接缝    段间张量形状逐对相等，且都等于新的 unified 长度
S4 无误伤  caption(80) / 模型维度(3840,128,30) 相关常量逐个未变；
           RoPE 频率表 [1536,16] / [512,24] 逐字节未变
S5 名不变  手术前后张量名集合完全相同（否则 §29.8 的 Clip+FP16 白名单失效）

用法: python aspect_surgery_check.py <tag> <宽> <高>
"""
import os
import sys

import numpy as np
import onnx
from onnx import external_data_helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                      # noqa: E402

ONNX = canonical_sources.ONNX_DIR
SEGS = ["part1a", "part1b", "part2a", "part2b"]
CAP = 80
FREQ_DIMS = ([1536, 16], [512, 24])
FAIL = []


def resolve(t):
    tt = onnx.TensorProto()
    tt.CopyFrom(t)
    if tt.data_location == 1:
        external_data_helper.load_external_data_for_tensor(tt, ".")
    return numpy_helper.to_array(tt)


def shapes(g):
    out = {}
    for vi in list(g.input) + list(g.output):
        out[vi.name] = [d.dim_value if d.HasField("dim_value") else d.dim_param
                        for d in vi.type.tensor_type.shape.dim]
    return out


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    LAT_W, LAT_H = W // 8, H // 8
    GW, GH = LAT_W // 2, LAT_H // 2
    UNI = CAP + GW * GH
    print("检查 %s: latent %dx%d 网格 %dx%d unified %d" % (tag, LAT_H, LAT_W, GH, GW, UNI))

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        old_m, new_m = {}, {}
        for s in SEGS:
            stem = os.path.splitext(os.path.basename(
                canonical_sources.onnx_for(s, L=canonical_sources.DEPLOYED_L)))[0]
            old_m[s] = onnx.load(stem + ".onnx", load_external_data=False)
            new_m[s] = onnx.load("%s_%s.onnx" % (stem, tag), load_external_data=False)

        # ---------- S5 张量名集合不变 ----------
        print("\n=== S5 张量名不变 ===")
        for s in SEGS:
            a = set(t.name for t in old_m[s].graph.initializer)
            b = set(t.name for t in new_m[s].graph.initializer)
            na = set(n.name for n in old_m[s].graph.node)
            nb = set(n.name for n in new_m[s].graph.node)
            ok = (a == b) and (na == nb)
            print("  %-8s initializer %d==%d  node %d==%d  %s"
                  % (s, len(a), len(b), len(na), len(nb), "OK" if ok else "FAIL"))
            if not ok:
                FAIL.append("S5 %s: 名集合变了 (+%s -%s)" % (s, list(b - a)[:3], list(a - b)[:3]))

        # ---------- S2 接缝 ----------
        print("\n=== S2 段间接缝 ===")
        chain = [("part1a", "part1b"), ("part1b", "part2a"), ("part2a", "part2b")]
        for up, dn in chain:
            so = shapes(new_m[up].graph)
            si = shapes(new_m[dn].graph)
            common = [n.name for n in new_m[dn].graph.input if n.name in so]
            bad = [(n, so[n], si[n]) for n in common if so[n] != si[n]]
            print("  %s -> %s : 共有张量 %d，形状不一致 %d" % (up, dn, len(common), len(bad)))
            for n, x, y in bad[:4]:
                print("     FAIL %s  上游 %s != 下游 %s" % (n, x, y))
            if bad:
                FAIL.append("S2 %s->%s 有 %d 个接缝不一致" % (up, dn, len(bad)))
            if not common:
                FAIL.append("S2 %s->%s 没有共有张量，装置有误" % (up, dn))

        # unified 长度确实是新值
        u = shapes(new_m["part1b"].graph).get("unified")
        print("  part1b 输出 unified = %s （应含 %d）" % (u, UNI))
        if not u or UNI not in u:
            FAIL.append("S2 unified 长度不是 %d" % UNI)

        # part1a 输入 latents
        la = shapes(new_m["part1a"].graph).get("latents")
        print("  part1a 输入 latents = %s （应为 [1,16,%d,%d]）" % (la, LAT_H, LAT_W))
        if la != [1, 16, LAT_H, LAT_W]:
            FAIL.append("S2 part1a latents 形状是 %s" % la)

        # ---------- S4 无误伤 ----------
        print("\n=== S4 无误伤 ===")
        for s in SEGS:
            od = {t.name: t for t in old_m[s].graph.initializer}
            changed, freq_changed, cap_changed = [], [], []
            for t in new_m[s].graph.initializer:
                o = od.get(t.name)
                if o is None:
                    continue
                # RoPE 频率表：必须逐字节未变
                if list(o.dims) in FREQ_DIMS:
                    if (list(t.dims) != list(o.dims)
                            or t.data_location != o.data_location
                            or (t.data_location != 1 and t.raw_data != o.raw_data)):
                        freq_changed.append(t.name)
                    continue
                # 只检查被本脚本改成内联的小整数常量
                if t.data_location == 1 or o.data_location == 1:
                    continue
                if t.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                    continue
                av, bv = numpy_helper.to_array(o), numpy_helper.to_array(t)
                if av.shape != bv.shape or not np.array_equal(av, bv):
                    changed.append(t.name)
                    if CAP in [int(x) for x in av.reshape(-1)] and av.size <= 16:
                        # 含 80 的常量：只有当它同时含 4096/4176 时才允许被改
                        if not any(int(x) in (4096, 4176) for x in av.reshape(-1)):
                            cap_changed.append(t.name)
            print("  %-8s 改动常量 %2d | RoPE 频率表被改 %d | 误伤 caption 常量 %d"
                  % (s, len(changed), len(freq_changed), len(cap_changed)))
            if freq_changed:
                FAIL.append("S4 %s: RoPE 频率表被改 %s" % (s, freq_changed[:3]))
            if cap_changed:
                FAIL.append("S4 %s: 误伤纯 caption 常量 %s" % (s, cap_changed[:3]))
    finally:
        os.chdir(cwd)

    print("")
    if FAIL:
        print("结论：FAIL")
        for f in FAIL:
            print("   - " + f)
        return 1
    print("结论：S2 / S4 / S5 全过")
    return 0


sys.exit(main())
