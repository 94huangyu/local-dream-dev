# -*- coding: utf-8 -*-
"""逐段查：端到端 S5 塌成噪声，是哪一段坏的？（约束 4：先分执行错 vs 方向错）

part1b 单段测过 mode S，只是轻微变化（全量 −17%），不是灾难；
而端到端四段一起用就塌到余弦 0.2346。=> 另外三段里大概率有一段是坏的。

做法：每段单独跑 FP32 / H / S5 三张图，同一份 L32_pure 输入，比输出误差。
某段 S5 的误差远大于 H => 那段的 mode S 实现有 bug（执行错），
四段都正常 => 才轮到「误差跨段累积」这类方向性解释。
"""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
from onnx import TensorProto

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
CH32 = os.path.join(P0, "L32_pure")
STEM = {"part1a": "transformer_part1a", "part1b": "transformer_part1b",
        "part2a": "transformer_part2a_fixed", "part2b": "transformer_part2b_fixed"}
INS = {
    "part1a": [("latents", (1, 16, 128, 128)), ("timestep", (1,)),
               ("caption", (1, 32, 2560)), ("cap_pad_mask", (1, 32))],
    "part1b": [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
               ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
               ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))],
    "part2a": [("unified", (1, 4128, 3840)), ("unified_mask", (1, 4128)),
               ("unified_freqs", (1, 4128, 64, 2)), ("adaln_input", (1, 256))],
    "part2b": [("add_92", (1, 4128, 3840)), ("select", (1, 4128, 1, 64)),
               ("select_1", (1, 4128, 1, 64)), ("split_7_split_2", (1, 1, 3840)),
               ("split_7_split_3", (1, 1, 3840)), ("val_105", (1, 1, 1, 4128)),
               ("adaln_input", (1, 256))],
}
EPS = 1e-12


def feeds_for(seg, gin_bool):
    d = os.path.join(CH32, "s0_%s" % seg)
    f = {}
    for n, sh in INS[seg]:
        p = os.path.join(d, n + ".raw")
        f[n] = np.fromfile(p, dtype=np.float32).reshape(sh)
    for n in gin_bool:
        if n in f:
            f[n] = f[n].astype(bool)
    return f


def run(path, feeds):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    outs = s.run(names, feeds)
    del s
    return dict(zip(names, outs))


def relerr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + EPS))


def main():
    segs = sys.argv[1:] or ["part1a", "part1b", "part2a", "part2b"]
    print("%-9s %-22s %10s %10s   %s" % ("段", "输出张量", "H", "S5", "判定"))
    print("-" * 74)
    for seg in segs:
        base = os.path.join(D, STEM[seg] + ".onnx")
        m = onnx.load(base, load_external_data=False)
        gb = [i.name for i in m.graph.input
              if i.type.tensor_type.elem_type == TensorProto.BOOL]
        feeds = feeds_for(seg, gb)
        ref = run(base, feeds)
        got = {}
        for arm, path in (("H", os.path.join(D, STEM[seg] + "_simH.onnx")),
                          ("S5", os.path.join(D, STEM[seg] + "_simS.onnx"))):
            if not os.path.isfile(path):
                print("  %s 缺 %s" % (seg, os.path.basename(path)))
                got[arm] = None
                continue
            got[arm] = run(path, feeds)
        for k in ref:
            if ref[k].dtype != np.float32 or ref[k].size < 16:
                continue
            eh = relerr(ref[k], got["H"][k]) if got.get("H") else float("nan")
            es = relerr(ref[k], got["S5"][k]) if got.get("S5") else float("nan")
            flag = ""
            if es == es and eh == eh:
                if es > max(0.5, 5 * eh):
                    flag = "🔴 **该段 S5 已毁 ⇒ 执行错**"
                elif es > 1.5 * eh:
                    flag = "⚠️ 明显变差"
            print("%-9s %-22s %9.2f%% %9.2f%%   %s" % (seg, k[:22], 100 * eh, 100 * es, flag))
        del ref, got, feeds
        gc.collect()
    print("\n⚠️ mode S 的产物是 `_simS.onnx`，两个 alpha 会互相覆盖 —— "
          "本脚本读的是**最后一次建的那份**（sq_e2e 最后建的是 alpha=0.5）。")


if __name__ == "__main__":
    main()
