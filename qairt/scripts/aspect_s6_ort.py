# -*- coding: utf-8 -*-
"""EXP_PLAN_MULTIGRAPH §6.4 门 S6：ORT 跑 FP32 前向。

判据（事前锁定）：四段串起来能跑通；part2b 输出 latents 形状 = [1,16,LAT_H,LAT_W]；
全程无 NaN / Inf。

⚠️ 本门只保证「结构正确、能算」，**不保证画质** —— 画质要等量化 + 建图 + 设备实测。

用法: python aspect_s6_ort.py <tag> <宽> <高> [--seg part1a]
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                      # noqa: E402

ONNX = canonical_sources.ONNX_DIR
CAP = 80
SEGS = ["part1a", "part1b", "part2a", "part2b"]


def stem_of(seg):
    return os.path.splitext(os.path.basename(
        canonical_sources.onnx_for(seg, L=canonical_sources.DEPLOYED_L)))[0]


def check(name, a):
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    print("     %-16s %-22s min=%9.3f max=%9.3f  NaN=%d Inf=%d"
          % (name, str(a.shape), np.nanmin(a), np.nanmax(a), nan, inf), flush=True)
    return nan == 0 and inf == 0


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    only = None
    if "--seg" in sys.argv:
        only = sys.argv[sys.argv.index("--seg") + 1]
    LAT_W, LAT_H = W // 8, H // 8
    UNI = CAP + (LAT_W // 2) * (LAT_H // 2)
    print("S6 %s: latent %dx%d unified %d" % (tag, LAT_H, LAT_W, UNI), flush=True)

    rng = np.random.default_rng(42)
    vals = {
        "latents": rng.standard_normal((1, 16, LAT_H, LAT_W)).astype(np.float32),
        "timestep": np.array([1.0], dtype=np.float32),
        "caption": rng.standard_normal((1, CAP, 2560)).astype(np.float32) * 0.1,
        "cap_pad_mask": np.zeros((1, CAP), dtype=bool),
    }
    vals["cap_pad_mask"][0, :14] = True

    so = ort.SessionOptions()
    so.log_severity_level = 3
    cwd = os.getcwd()
    os.chdir(ONNX)
    ok = True
    try:
        for seg in SEGS:
            if only and seg != only:
                continue
            f = "%s_%s.onnx" % (stem_of(seg), tag)
            print("\n--- %s (%s) ---" % (seg, f), flush=True)
            t0 = time.time()
            sess = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
            names = [i.name for i in sess.get_inputs()]
            missing = [n for n in names if n not in vals]
            if missing:
                print("     缺输入 %s ⇒ 跳过该段（需上游先跑）" % missing, flush=True)
                continue
            feed = {n: vals[n] for n in names}
            outs = sess.run(None, feed)
            for o, v in zip(sess.get_outputs(), outs):
                vals[o.name] = v
                if v.dtype.kind == "f":
                    ok &= check(o.name, v)
                else:
                    print("     %-16s %-22s dtype=%s" % (o.name, str(v.shape), v.dtype), flush=True)
            print("     用时 %.1f 分钟" % ((time.time() - t0) / 60.0), flush=True)
            del sess
    finally:
        os.chdir(cwd)

    print("")
    if "latents" in vals and not only:
        got = list(vals["latents"].shape)
        want = [1, 16, LAT_H, LAT_W]
        print("最终 latents 形状 %s （应为 %s）" % (got, want))
        if got != want:
            print("结论：FAIL 形状不对")
            return 1
    print("结论：S6 %s" % ("PASS（无 NaN/Inf）" if ok else "FAIL（出现 NaN/Inf）"))
    return 0 if ok else 1


sys.exit(main())
