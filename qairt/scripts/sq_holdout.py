# -*- coding: utf-8 -*-
"""留出集检验：把 s0 上标定的 encoding 拿到 s1/s4 上评测。

🔴 为什么必须做：`sq_actstats --sim-src` 的统计取自 **testB/s0**，
   而 `sq_validate` 的评测**也用 s0** ⇒ 「重标定 encoding 好 23%」是 train-on-test，
   不能当收益。s1~s7 未参与标定，是干净的留出集。

⚠️ 对照关系要分清：
  · 「H vs S(alpha=0)」= **重标定 encoding** 的效应 —— 受泄漏影响，必须看留出集
  · 「S(alpha=0) vs S(alpha=0.5)」= **SmoothQuant 本身**的净效应 ——
    两臂共用同一份（同样泄漏的）encoding，泄漏在共同基线里抵消，故该对照在 s0 上也成立

用法: python sq_holdout.py [part1b] [--steps 0,1,4]
"""
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
STEM = {"part1b": "transformer_part1b"}
INS = [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
       ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
       ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))]
EPS = 1e-12
ARMS = [("H", "H", None), ("S0", "S", 0.0), ("S5", "S", 0.5)]
LBL = {"H": "H（部署 encoding）", "S0": "S a=0（重标定，未平滑）", "S5": "S a=0.5（重标定+平滑）"}


def feeds_for(seg, step):
    d = os.path.join(P0, "testB", "s%d_transformer_%s" % (step, seg))
    f = {}
    for n, sh in INS:
        p = os.path.join(d, n + ".raw")
        want = 4
        for x in sh:
            want *= x
        got = os.path.getsize(p)
        if got != want:
            sys.exit("FAIL s%d/%s 字节 %d != %d" % (step, n, got, want))
        f[n] = np.fromfile(p, dtype=np.float32).reshape(sh)
    return f


def run(path, feeds):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()]
    tgt = "unified" if "unified" in outs else outs[0]
    r = s.run([tgt], feeds)[0]
    del s
    return np.asarray(r, dtype=np.float32)


def rel(a, b):
    d = (b - a).astype(np.float64).ravel()
    af = a.astype(np.float64).ravel()
    full = float(np.linalg.norm(d) / (np.linalg.norm(af) + EPS))
    p99 = np.percentile(np.abs(af), 99)
    msk = np.abs(af) <= p99
    bulk = float(np.linalg.norm(d[msk]) / (np.linalg.norm(af[msk]) + EPS))
    return full, bulk


def build(seg, key, mode, alpha):
    env = dict(os.environ)
    if alpha is not None:
        env["SQ_ALPHA"] = str(alpha)
    r = subprocess.run([sys.executable, os.path.join(HERE, "sim_qdq.py"), seg, mode],
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit("FAIL 建图 %s:\n%s" % (key, ((r.stdout or "") + (r.stderr or ""))[-800:]))
    src = os.path.join(D, "%s_sim%s.onnx" % (STEM[seg], mode))
    dst = os.path.join(D, "%s_ho%s.onnx" % (STEM[seg], key))
    shutil.copyfile(src, dst)
    return dst


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "part1b"
    steps = [0, 1, 4]
    if "--steps" in sys.argv:
        steps = [int(x) for x in sys.argv[sys.argv.index("--steps") + 1].split(",")]

    paths = {}
    for key, mode, al in ARMS:
        paths[key] = build(seg, key, mode, al)
    print("三张图已建好（各存为独立文件，避免同名互相覆盖）\n", flush=True)

    ref_p = os.path.join(D, STEM[seg] + ".onnx")
    res = {}
    for st in steps:
        tag = "标定集（s0）" if st == 0 else "**留出集**"
        print("=== step %d  %s ===" % (st, tag), flush=True)
        feeds = feeds_for(seg, st)
        t0 = time.time()
        ref = run(ref_p, feeds)
        for key, _m, _a in ARMS:
            y = run(paths[key], feeds)
            res[(st, key)] = rel(ref, y)
            print("  %-24s E全量 %7.4f%%   E主体 %7.4f%%"
                  % (LBL[key], 100 * res[(st, key)][0], 100 * res[(st, key)][1]), flush=True)
            del y
        del ref, feeds
        print("  （本步用时 %.0f s）\n" % (time.time() - t0), flush=True)

    print("=== ① 重标定 encoding 的效应（H -> S a=0）—— 受泄漏影响，看留出集 ===")
    for st in steps:
        h, z = res[(st, "H")], res[(st, "S0")]
        print("  step %d  全量 %+6.1f%%   主体 %+6.1f%%   %s"
              % (st, 100 * (h[0] - z[0]) / h[0], 100 * (h[1] - z[1]) / h[1],
                 "← 标定集，可能虚高" if st == 0 else "← 留出集，这个才算数"))

    print("\n=== ② SmoothQuant 本身的净效应（S a=0 -> S a=0.5，共用同一份 encoding）===")
    for st in steps:
        z, f = res[(st, "S0")], res[(st, "S5")]
        print("  step %d  全量 %+6.1f%%   主体 %+6.1f%%"
              % (st, 100 * (z[0] - f[0]) / z[0], 100 * (z[1] - f[1]) / z[1]))

    print("\n⚠️ 集中度 0.9983 ⇒ 约束 7：**全量口径不得用于描述该张量整体**，"
          "决策看主体口径（EXP_PLAN_CH85_CAUSAL：主体占 36% 能量却造成 99% 下游损害）。")
    print("⚠️ 这是单段 QDQ 模拟，不是端到端更不是设备；"
          "#52 已证宿主会低估权重量化在 HTP 上的代价（37.6 倍）⇒ 对 SmoothQuant 偏乐观。")


if __name__ == "__main__":
    main()
