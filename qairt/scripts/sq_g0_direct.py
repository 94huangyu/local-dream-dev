# -*- coding: utf-8 -*-
"""G0-cost 的**正确操作化**：直接量 MatMul 的**输出误差**，不用合成指标。

🔴 为什么重写（约束 8）：`sq_g0cost.py` 用 `净 = sqrt(激活step × 权重step)` 当综合指标，
   两项量纲不同、合成方式无依据，且 alpha=1 时激活 step 必然崩塌（把每通道除以自己的
   max，所有通道 max 变成 1）——那是把定义代入自己，不是收益。它报的 97.8% 作废。

**判据线不变**（事前锁定，不得改）：净改善 >= 20% => 🟢；<= 0 => ❌；之间 => 🟡。
改的只是「怎么量」，因为原操作化从未被验证过。

正确的量：对真实的 X 与 W，
    Y_ref = X @ W                                    (fp32)
    Y_now = QDQ_a(X)       @ QDQ_w(W)                (现状)
    Y_sq  = QDQ_a(X / s)   @ QDQ_w(W * s)            (SmoothQuant)
比 ||Y - Y_ref|| / ||Y_ref||。s 折进权重是离线的 => 运行时零代价。

量化口径与部署一致：
  · 激活 per-tensor 非对称 uFxp_16（min-max，#53 已证其他校准法不收紧量程）
  · 权重 per-channel(per-row) 对称 sFxp_8（#38 实测部署版就是这个）

⚠️ 同时报**全量口径**与**主体口径**（|Y|<=p99），约束 7 强制。

用法: python sq_g0_direct.py [part1b] [--n 8]
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
from onnx import TensorProto, helper

import canonical_sources as cs

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
CH80 = os.path.join(P0, "L80_chain")
SQ = os.path.join(P0, "smoothquant")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-8
ROWS = 0          # 0 = 全部行；命令行 --rows N 设定
INS = [("add_138", (1, 4176, 3840)), ("add_131", (1, 1, 3840)),
       ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4176, 1, 64)),
       ("select_46", (1, 4176, 1, 64)), ("adaln_input", (1, 256))]


def qdq_act(x):
    """per-tensor 非对称 uFxp_16（min-max），与部署一致。"""
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return x.copy()
    sc = (mx - mn) / 65535.0
    q = np.clip(np.round((x - mn) / sc), 0, 65535)
    return (q * sc + mn).astype(np.float32)


def qdq_w(w):
    """per-channel(per-row) 对称 sFxp_8。w: [K, M]，每行一个 scale。"""
    a = np.abs(w).max(axis=1, keepdims=True)
    sc = np.maximum(a / 127.0, EPS)
    q = np.clip(np.round(w / sc), -127, 127)
    return (q * sc).astype(np.float32)


def rel(a, b):
    """b 相对 a 的相对 L2（全量），以及主体口径（|a|<=p99）与集中度。"""
    d = (b - a).ravel()
    af = a.ravel()
    full = float(np.linalg.norm(d) / (np.linalg.norm(af) + EPS))
    p99 = np.percentile(np.abs(af), 99)
    m = np.abs(af) <= p99
    bulk = float(np.linalg.norm(d[m]) / (np.linalg.norm(af[m]) + EPS))
    e2 = af.astype(np.float64) ** 2
    k = max(1, int(round(0.01 * e2.size)))
    conc = float(np.sort(e2)[-k:].sum() / (e2.sum() + EPS))
    return full, bulk, conc


def main():
    import onnxruntime as ort
    seg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "part1b"
    nsample = 8
    if "--n" in sys.argv:
        nsample = int(sys.argv[sys.argv.index("--n") + 1])
    global ROWS
    if "--rows" in sys.argv:
        ROWS = int(sys.argv[sys.argv.index("--rows") + 1])
    print("  行采样 ROWS=%s" % (ROWS or "全部"))

    st = json.load(open(os.path.join(SQ, "%s_L80_chstats.json" % seg), encoding="utf-8"))
    p = cs.onnx_for(seg, L=cs.DEPLOYED_L)
    print("段 %s   源 %s" % (seg, os.path.basename(p)))

    # 每种权重形状各取若干个，覆盖四类
    byshape = {}
    for pr in st["pairs"]:
        byshape.setdefault((pr["K"], pr["M"]), []).append(pr)
    if nsample <= 0:                      # --n 0 = 全测
        sample = list(st["pairs"])
        print("  **全测**：%d 个静态权重 MatMul" % len(sample))
    else:
        per = max(1, nsample // len(byshape))
        sample = []
        for sh, lst in sorted(byshape.items()):
            sample.extend(lst[:per])
        print("  权重形状 %d 类，各取 %d 个 ⇒ 样本 %d 个 MatMul"
              % (len(byshape), per, len(sample)))
        for pr in sample:
            print("     %-22s W%s" % (pr["act"], [pr["K"], pr["M"]]))

    # --- 取真实激活 ---
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    gin = {i.name for i in g.input}
    feeds = {}
    for n, sh in INS:
        f = os.path.join(CH80, "s0_%s" % seg, n + ".raw")
        feeds[n] = np.fromfile(f, dtype=np.float32).reshape(sh)
    need = sorted({pr["act"] for pr in sample if pr["act"] not in gin})
    base_out = list(g.output)
    for t in need:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    dst = os.path.join(os.path.dirname(p), "%s_sqdirect.onnx"
                       % os.path.splitext(os.path.basename(p))[0])
    onnx.save(m, dst)
    t0 = time.time()
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    outs = sess.run(need, feeds)
    X = dict(zip(need, outs))
    for t in {pr["act"] for pr in sample} & gin:
        X[t] = feeds[t]
    del outs, sess
    gc.collect()
    print("  取到 %d 个真实激活，用时 %.0f s" % (len(X), time.time() - t0), flush=True)

    ini = {i.name: i for i in g.initializer}

    def load_w(name):
        t = ini[name]
        d = {e.key: e.value for e in t.external_data}
        loc = os.path.join(os.path.dirname(p), d["location"])
        with open(loc, "rb") as f:
            f.seek(int(d.get("offset", 0)))
            raw = f.read(int(d["length"]))
        return np.frombuffer(raw, dtype=np.float32).reshape([x for x in t.dims])

    print("\n%-22s %-6s %10s %10s %10s" % ("MatMul激活", "alpha", "全量E", "主体E", "vs a=0"))
    print("-" * 64)
    agg = {a: [] for a in ALPHAS}
    for pr in sample:
        a, w = pr["act"], pr["w"]
        x = np.asarray(X[a], dtype=np.float32)
        x2 = x.reshape(-1, x.shape[-1])
        # 行采样：误差是统计量，取前 ROWS 行足够（#39 的 numpy 定点模拟器就用前 512 行）。
        # 🔴 ROWS 的合法性由 --verify-rows 用「已知样本」核对（约束 8），不得直接假定。
        if ROWS and x2.shape[0] > ROWS:
            x2 = np.ascontiguousarray(x2[:ROWS])
        W = load_w(w)
        if W.shape[0] != x2.shape[1]:
            # 权重以 [M, K] 存放（Gemm/转置位）=> 转成 [K, M] 再算，别整类跳过。
            # 原先直接 continue 把 adaLN 那 7 个 MatMul 整类漏掉了。
            if W.shape[1] == x2.shape[1]:
                W = np.ascontiguousarray(W.T)
            else:
                print("  跳过 %s（K 两侧都不匹配 %s vs %d）" % (a, W.shape, x2.shape[1]))
                continue
        yref = x2.astype(np.float32) @ W.astype(np.float32)
        ax = np.abs(x2).max(axis=0)
        aw = np.abs(W).max(axis=1)
        e0 = None
        for al in ALPHAS:
            s = np.power(np.maximum(ax, EPS), al) / np.power(np.maximum(aw, EPS), 1.0 - al)
            s = np.maximum(s, EPS).astype(np.float32)
            xq = qdq_act(x2 / s)
            wq = qdq_w(W * s[:, None])
            y = xq.astype(np.float32) @ wq.astype(np.float32)
            full, bulk, conc = rel(yref, y)
            if e0 is None:
                e0 = full
            imp = 100.0 * (e0 - full) / (e0 + EPS)
            agg[al].append(full)
            print("%-22s %-6.2f %9.4f%% %9.4f%% %9.1f%%"
                  % (a[:22] if al == ALPHAS[0] else "", al, 100 * full, 100 * bulk, imp))
            del xq, wq, y
        del W, yref, x2
        gc.collect()

    # 🔴 约束 7：均值会被离群 MatMul 主导，必须同时报中位数与逐个改善的分布
    print("\n=== 汇总（%d 个 MatMul）===" % len(agg[0.0]))
    base_v = np.asarray(agg[0.0])
    base = float(base_v.mean())
    best = (None, -1e9)
    for al in ALPHAS:
        v = np.asarray(agg[al])
        mu, md = float(v.mean()), float(np.median(v))
        imp_mean = 100.0 * (base - mu) / (base + EPS)
        per_mm = 100.0 * (base_v - v) / (base_v + EPS)      # 逐 MatMul 的改善
        imp_med = float(np.median(per_mm))
        print("  alpha=%.2f  E均值=%.4f%% 中位=%.4f%% | 改善: 均值口径 %+.1f%%  "
              "**逐MatMul中位 %+.1f%%**  (最差 %+.1f%%, 最好 %+.1f%%)"
              % (al, 100 * mu, 100 * md, imp_mean, imp_med,
                 per_mm.min(), per_mm.max()))
        if imp_mean > best[1]:
            best = (al, imp_mean, imp_med)
    print("\n  ⚠️ 均值口径会被少数「massive activation」MatMul 主导（约束 7）；"
          "决策时**两个口径都要看**。")
    print("\n=== G0-cost 判据（判据线未改：>=20% 过、<=0 否决）===")
    print("  最优 alpha=%.2f，改善：均值口径 **%+.1f%%**，逐MatMul中位 **%+.1f%%**" % best)
    if best[1] >= 20.0:
        print("  ⇒ 🟢 PASS：进入 §三 全模型 QDQ 模拟")
    elif best[1] <= 0.0:
        print("  ⇒ ❌ 否决：SmoothQuant 在本模型上不降低 MatMul 输出误差，E 关闭")
    else:
        print("  ⇒ 🟡 0 < %.1f%% < 20%%：记录，按 §三 成本再决定" % best[1])
    json.dump({"seg": seg, "n": len(agg[0.0]),
               "mean_full_E": {str(a): float(np.mean(agg[a])) for a in ALPHAS}},
              open(os.path.join(SQ, "%s_g0_direct.json" % seg), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
