"""模拟器反向自检（EXP_PLAN_SIM_CEILING §3.3）：不过就不许用它下结论。

三项：
  1. **零编码对照**：mode Z（什么都不插）必须与纯 FP32 前向**逐位相同**
     —— 这是「图手术本身没有改变语义」的唯一证明（约束 8：判据先用已知样本验证）。
  2. mode C（只量权重）输出必须**有限**，且与 FP32 的偏差应当很小但非零。
  3. mode A（权重+激活）输出必须**有限**，偏差应当大于 C。

用法: python sim_selfcheck.py part1b
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
CFG = {
    "part1b": ("transformer_part1b", "unified",
               [(n, os.path.join(P0, "testB", "s0_transformer_part1b", n + ".raw"), s)
                for n, s in [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
                             ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
                             ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))]]),
}


def main():
    import onnx
    import onnxruntime as ort
    seg = sys.argv[1]
    base, probe, ins = CFG[seg]
    feeds = {}
    for n, p, s in ins:
        feeds[n] = np.fromfile(p, np.float32).reshape(s)
    m = onnx.load(os.path.join(D, base + ".onnx"), load_external_data=False)
    for i in m.graph.input:
        if i.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
            feeds[i.name] = feeds[i.name].astype(bool)

    def go(name):
        t0 = time.time()
        s = ort.InferenceSession(os.path.join(D, name + ".onnx"),
                                 providers=["CPUExecutionProvider"])
        ts = time.time() - t0
        t0 = time.time()
        r = s.run([probe], feeds)[0].astype(np.float64).ravel()
        print("    %-34s session %4.0fs 前向 %4.0fs  finite=%s |a|max=%.4g"
              % (name, ts, time.time() - t0, bool(np.isfinite(r).all()), np.abs(r).max()),
              flush=True)
        del s
        return r

    print("[%s] 探针张量 = %s" % (seg, probe), flush=True)
    ref = go(base)
    z = go(base + "_simZ")
    c = go(base + "_simC")
    a = go(base + "_simA")

    def rel(x):
        return 100.0 * np.linalg.norm(x - ref) / np.linalg.norm(ref)

    bitZ = np.array_equal(ref, z)
    print("")
    print("  [自检 1] mode Z 与纯 FP32 **逐位相同**: %s" % ("PASS" if bitZ else "🔴 FAIL"))
    print("  [自检 2] mode C（只量权重）相对 FP32: %.4f%%  有限=%s"
          % (rel(c), bool(np.isfinite(c).all())))
    print("  [自检 3] mode A（权重+激活）相对 FP32: %.4f%%  有限=%s"
          % (rel(a), bool(np.isfinite(a).all())))
    ok = bitZ and np.isfinite(c).all() and np.isfinite(a).all() and rel(a) > rel(c) > 0
    print("")
    print("  ⇒ %s" % ("✅ 三项自检通过，模拟器可用" if ok else
                      "🔴 未通过，**不得**用它下结论"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
