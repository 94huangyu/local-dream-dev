# -*- coding: utf-8 -*-
"""mode S 的**已知样本检验**（约束 8），以及 part1b 单段的第一手信号。

🔴 为什么必须先做这个：`sim_qdq` 的 mode Z 自检验的是**原有**插入机制；
   mode S 新增了 Div 节点、重算的 encoding、权重 Mul，**一条都没对过已知答案**。

已知样本：**S(alpha=0) 应当 ≈ mode H**。
  alpha=0 => s 恒为 1 => Div 是恒等运算，
  唯一差别是「重算的 encoding」vs「录得的 encoding」。
  两者若差很多，说明我的 encoding 重算写错了 ⇒ alpha=0.5 的结果全部作废。

判据（事前锁定）：
  |E(S,a=0) - E(H)| / E(H)  <= 5%   => 🟢 机制可信，可以看 alpha=0.5
                             > 5%   => 🔴 机制有问题，先修，不得用 alpha=0.5 的数

单段前向约 30 秒，比 sim_runner 的四段全链（10 分钟/配置）便宜得多。

用法: python sq_validate.py [part1b]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
ONNX = {"part1b": "transformer_part1b"}
# 与 sim_qdq 同源的 L=32 输入（testB）
INS = [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
       ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
       ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))]
OUT_T = "unified"
EPS = 1e-12


def feeds_for(seg):
    d = os.path.join(P0, "testB", "s0_transformer_%s" % seg)
    f = {}
    for n, sh in INS:
        p = os.path.join(d, n + ".raw")
        want = 4
        for x in sh:
            want *= x
        got = os.path.getsize(p)
        if got != want:
            sys.exit("FAIL 输入 %s 字节 %d != %d" % (n, got, want))
        f[n] = np.fromfile(p, dtype=np.float32).reshape(sh)
    return f


def run(path, feeds):
    import onnxruntime as ort
    t0 = time.time()
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()]
    tgt = OUT_T if OUT_T in outs else outs[0]
    r = s.run([tgt], feeds)[0]
    del s
    return np.asarray(r, dtype=np.float32), time.time() - t0, tgt


def rel(a, b):
    d = (b - a).astype(np.float64).ravel()
    af = a.astype(np.float64).ravel()
    full = float(np.linalg.norm(d) / (np.linalg.norm(af) + EPS))
    p99 = np.percentile(np.abs(af), 99)
    msk = np.abs(af) <= p99
    bulk = float(np.linalg.norm(d[msk]) / (np.linalg.norm(af[msk]) + EPS))
    e2 = af ** 2
    k = max(1, int(round(0.01 * e2.size)))
    conc = float(np.sort(e2)[-k:].sum() / (e2.sum() + EPS))
    return full, bulk, conc


def build(seg, mode, alpha=None):
    env = dict(os.environ)
    if alpha is not None:
        env["SQ_ALPHA"] = str(alpha)
    r = subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_qdq.py"),
                        seg, mode],
                       env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        sys.exit("FAIL 建图 %s/%s:\n%s" % (mode, alpha, (r.stdout or "") + (r.stderr or ""))[-800:])
    return os.path.join(D, "%s_sim%s.onnx" % (ONNX[seg], mode))


def main():
    seg = sys.argv[1] if len(sys.argv) > 1 else "part1b"
    feeds = feeds_for(seg)
    print("段 %s   输入 6 个（testB，L=32）字节全对" % seg)

    ref_p = os.path.join(D, ONNX[seg] + ".onnx")
    ref, t, tgt = run(ref_p, feeds)
    print("  FP32 参考：%s %s  用时 %.0fs\n" % (tgt, ref.shape, t))

    res = {}
    for tag, mode, al in (("H（部署配置）", "H", None),
                          ("S alpha=0（对照臂）", "S", 0.0),
                          ("S alpha=0.5", "S", 0.5)):
        p = build(seg, mode, al)
        y, t, _ = run(p, feeds)
        full, bulk, conc = rel(ref, y)
        res[tag] = full
        print("  %-20s E全量 %7.4f%%   E主体 %7.4f%%   集中度 %.4f   用时 %.0fs"
              % (tag, 100 * full, 100 * bulk, conc, t))
        del y

    print("\n=== 已知样本检验（判据事前锁定：偏差 ≤5% 才可信）===")
    eh = res["H（部署配置）"]
    e0 = res["S alpha=0（对照臂）"]
    dev = abs(e0 - eh) / (eh + EPS)
    print("  E(S,a=0)=%.4f%%  vs  E(H)=%.4f%%   偏差 **%.1f%%**"
          % (100 * e0, 100 * eh, 100 * dev))
    ok = dev <= 0.05
    print("  ⇒ %s" % ("🟢 机制可信" if ok else
                      "🔴 FAIL：encoding 重算或 Div 插入有问题，alpha=0.5 的数**不得使用**"))
    if ok:
        e5 = res["S alpha=0.5"]
        imp = 100.0 * (e0 - e5) / (e0 + EPS)
        print("\n=== 第一手信号（part1b 单段，对照臂 = S alpha=0）===")
        print("  alpha=0.5 相对对照臂改善 **%+.1f%%**（%.4f%% -> %.4f%%）"
              % (imp, 100 * e0, 100 * e5))
        print("  ⚠️ 这是**单段 QDQ 模拟**，不是端到端、更不是设备。"
              "#121 已证 L1 无法换算成 L3；#52 已证宿主会**低估权重量化在 HTP 上的代价**"
              "（37.6 倍）⇒ 本数偏乐观。")


if __name__ == "__main__":
    main()
