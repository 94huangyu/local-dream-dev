# -*- coding: utf-8 -*-
"""把 transformer 四段从 1024x1024 改到目标长宽比，**不从 PyTorch 重导**。

方案与判据：`scripts/EXP_PLAN_MULTIGRAPH.md` §六（执行前定稿）。

与 `dit_seq_surgery.py` 的关系：复用它的**机制**（外部张量拷贝→解析→改→转内联；
value_info 也要改，#139 的教训），但**替换规则相反** —— 它是「32→80，4096 不动」，
本脚本是「80 不动，4096→新 token 数」。

🔴 **64 / 128 有歧义**（64 = axes_dims/2，128 = head_dim = 3840/30），
   因此涉及它们的张量**一律按名字处理，禁止按值替换**（约束 8）。
🔴 源文件名由 `canonical_sources.py` 给出，不得自己拼（#138）。

用法: python dit_aspect_surgery.py <tag> <宽> <高> [--variant=base|deployed]
      例: python dit_aspect_surgery.py r4x3 1152 864 --variant=base
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
CAP = 80                                       # caption 槽数，本手术不动
OLD_LAT, OLD_GRID, OLD_IMG = 128, 64, 4096
OLD_UNI = CAP + OLD_IMG                        # 4176


def resolve(t):
    """把（可能是外部存储的）initializer 读成 numpy。"""
    tt = onnx.TensorProto()
    tt.CopyFrom(t)
    if tt.data_location == 1:
        external_data_helper.load_external_data_for_tensor(tt, ".")
    return numpy_helper.to_array(tt)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    # 🔴 两个变体都要做：base 供 FP32 参考（t2_fp32_ref.py 刻意用它，Clip 是量化辅助
    #    手段，FP32 真值不该带），clip/deployed 供量化与建图。
    variant = "deployed"
    for x in sys.argv[4:]:
        if x.startswith("--variant="):
            variant = x.split("=", 1)[1]
    assert variant in ("base", "deployed"), "variant 只能是 base / deployed"
    assert W % 16 == 0 and H % 16 == 0, "像素必须能被 16 整除（VAE 8 x patch 2）"
    LAT_W, LAT_H = W // 8, H // 8
    assert LAT_W % 2 == 0 and LAT_H % 2 == 0, "latent 两维必须是偶数"
    GW, GH = LAT_W // 2, LAT_H // 2
    IMG, UNI = GW * GH, CAP + GW * GH
    print("变体 %s" % variant, flush=True)
    print("目标 %s: 像素 %dx%d | latent H=%d W=%d | 网格 GH=%d GW=%d | tokens %d | unified %d"
          % (tag, W, H, LAT_H, LAT_W, GH, GW, IMG, UNI), flush=True)

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        srcs = []
        for s in SEGS:
            p = canonical_sources.onnx_for(s, variant, L=canonical_sources.DEPLOYED_L)
            srcs.append((s, os.path.splitext(os.path.basename(p))[0]))

        for seg, stem in srcs:
            m = onnx.load(stem + ".onnx", load_external_data=False)
            g = m.graph
            n_named = n_val = n_dims = n_io = 0

            # ---- 1) 按名字处理（涉及 64/128 歧义的，禁止按值替换）----
            named = {}
            if seg == "part1a":
                named = {
                    "stack_1": ("grid_pos", None),
                    "val_70": ("exact", [16, 1, 1, GH, 2, GW, 2]),
                    "latents_shape": ("exact", [1, LAT_H, LAT_W]),
                }
            elif seg == "part2b":
                named = {
                    "shape_120_fixed": ("exact", [1, GH, GW, 1, 2, 2, 16]),
                    "shape_unsafe_fixed": ("exact", [1, 16, LAT_H, LAT_W]),
                }

            for t in list(g.initializer):
                if t.name not in named:
                    continue
                kind, val = named[t.name]
                a = resolve(t)
                if kind == "grid_pos":
                    old = a[0]
                    assert len(np.unique(old[..., 0])) == 1, "stack_1 轴0 不是常量，装置假设不成立"
                    ax0 = int(np.unique(old[..., 0])[0])
                    assert (old[..., 1] == np.arange(OLD_GRID)[:, None]).all(), "轴1 不是行号"
                    assert (old[..., 2] == np.arange(OLD_GRID)[None, :]).all(), "轴2 不是列号"
                    nv = np.zeros((1, GH, GW, 3), dtype=a.dtype)
                    nv[0, ..., 0] = ax0        # 🔴 原样保留（见台账 #154）
                    nv[0, ..., 1] = np.arange(GH)[:, None]
                    nv[0, ..., 2] = np.arange(GW)[None, :]
                else:
                    assert a.size == len(val), "%s 元素数 %d != %d" % (t.name, a.size, len(val))
                    nv = np.array(val, dtype=a.dtype).reshape(a.shape)
                t.CopyFrom(numpy_helper.from_array(nv, name=t.name))
                n_named += 1

            # ---- 2) 按值替换：只动 4096 / 4176（已实测零撞值）----
            for t in g.initializer:
                if t.name in named:
                    continue
                if t.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                    continue
                if t.data_location == 1:
                    continue
                a = numpy_helper.to_array(t)
                if a.size == 0 or a.size > 16:
                    continue
                v = [int(x) for x in a.reshape(-1)]
                if not any(x in (OLD_IMG, OLD_UNI) for x in v):
                    continue
                nv = np.array([IMG if x == OLD_IMG else UNI if x == OLD_UNI else x for x in v],
                              dtype=a.dtype).reshape(a.shape)
                t.CopyFrom(numpy_helper.from_array(nv, name=t.name))
                n_val += 1

            # ---- 3) dims 含 4096 / 4176 的数据张量（unsqueeze_3、unified_mask 等）----
            for t in list(g.initializer):
                if t.name in named:
                    continue
                if OLD_IMG not in list(t.dims) and OLD_UNI not in list(t.dims):
                    continue
                a = resolve(t)
                dims = [IMG if d == OLD_IMG else UNI if d == OLD_UNI else d for d in t.dims]
                u = np.unique(a)
                assert u.size == 1, ("张量 %s dims=%s 不是常量填充，无法安全外推（不猜）"
                                     % (t.name, list(t.dims)))
                nv = np.full(dims, u[0], dtype=a.dtype)
                t.CopyFrom(numpy_helper.from_array(nv, name=t.name))
                n_dims += 1

            # ---- 4) IO / value_info（#139：part2a/2b 各约 1000 条）----
            for vi in list(g.input) + list(g.output) + list(g.value_info):
                if seg == "part1a" and vi.name == "latents":
                    d = vi.type.tensor_type.shape.dim
                    got = [x.dim_value for x in d]
                    assert got == [1, 16, OLD_LAT, OLD_LAT], "part1a latents 形状是 %s" % got
                    d[2].dim_value, d[3].dim_value = LAT_H, LAT_W
                    n_io += 2
                    continue
                for d in vi.type.tensor_type.shape.dim:
                    if not d.HasField("dim_value"):
                        continue
                    if d.dim_value == OLD_IMG:
                        d.dim_value = IMG
                        n_io += 1
                    elif d.dim_value == OLD_UNI:
                        d.dim_value = UNI
                        n_io += 1

            # ---- 门 S3：不得残留 4096 / 4176 ----
            for t in g.initializer:
                if (t.data_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32)
                        and t.data_location != 1):
                    a = numpy_helper.to_array(t)
                    if 0 < a.size <= 16 and any(int(x) in (OLD_IMG, OLD_UNI)
                                                for x in a.reshape(-1)):
                        print("FAIL %s: 常量 %s 仍含 4096/4176" % (seg, t.name))
                        return 1
                if OLD_IMG in list(t.dims) or OLD_UNI in list(t.dims):
                    print("FAIL %s: 张量 %s dims 仍是 %s" % (seg, t.name, list(t.dims)))
                    return 1
            for vi in list(g.input) + list(g.output) + list(g.value_info):
                bad = [d.dim_value for d in vi.type.tensor_type.shape.dim
                       if d.HasField("dim_value") and d.dim_value in (OLD_IMG, OLD_UNI)]
                if bad:
                    print("FAIL %s: %s 仍含 %s" % (seg, vi.name, bad))
                    return 1

            # ---- 门 S1 ----
            onnx.checker.check_model(m, full_check=False)
            dst = "%s_%s.onnx" % (stem, tag)
            onnx.save(m, dst)
            print("  %-36s 按名 %d | 按值 %d | dims %d | IO/vi %4d -> %s"
                  % (seg, n_named, n_val, n_dims, n_io, dst), flush=True)

        print("", flush=True)
        print("四段手术完成（门 S1/S3 已过）。S2/S4/S5/S6 由 aspect_surgery_check.py 验。",
              flush=True)
    finally:
        os.chdir(cwd)
    return 0


sys.exit(main())
