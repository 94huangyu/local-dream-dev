"""#114 步 1 的前置：造出 part2b 的 step-0 输入。

为什么需要：`p0_experiments/p2split/` 下只有**校准样本**，没有 step-0 的那一份；
而 part2b 的 6 个输入全是 part2a 的图输出。

🔴 **provenance 必须与其余三段同一条链**：part1b / part2a 用的是
`p0_experiments/testB/`（= SNPE **CPU 参考**跑量化 DLC 的逐段输出，见 `compare_images.py`
的标注「CPU 跑量化 transformer (Test B)」；按 #35，CPU 是**浮点执行**、权重 per-tensor 8-bit）。
=> 本脚本用**同一份 testB 输入**跑 part2a 的 FP32 ONNX，取其 6 个图输出，
   这样 part2b 的测量是同一条链的续接，而不是换了一个来源。
   ⚠️ 这不是「纯 FP32 链」——③ 该差异必须在结论里标注。

用法: python maxfp16_p2b_inputs.py
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
SRC = os.path.join(P0, "testB", "s0_transformer_part2")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr", "fp32_s0_part2b")
INS = {"unified": (1, 4128, 3840), "unified_mask": (1, 4128),
       "unified_freqs": (1, 4128, 64, 2), "adaln_input": (1, 256)}
# part2b 的 7 个输入里，6 个是 part2a 的图输出；adaln_input 与 part2a 共用同一份
NEED = ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3", "val_105"]


def main():
    import onnx
    import onnxruntime as ort

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(D, "transformer_part2a_fixed.onnx")
    m = onnx.load(p, load_external_data=False)
    gout = [o.name for o in m.graph.output]
    miss = [t for t in NEED if t not in gout]
    if miss:
        sys.exit("FAIL 这些不是 part2a 的图输出: %s（实际输出: %s）" % (miss, gout))
    print("part2a 图输出: %s" % gout, flush=True)

    feeds = {}
    for n, shape in INS.items():
        f = os.path.join(SRC, n + ".raw")
        if not os.path.exists(f):
            sys.exit("FAIL 缺输入 %s" % f)
        feeds[n] = np.fromfile(f, np.float32).reshape(shape)
    for i in m.graph.input:
        if i.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
            feeds[i.name] = feeds[i.name].astype(bool)

    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    r = s.run(NEED, feeds)
    for n, a in zip(NEED, r):
        a = np.ascontiguousarray(a, dtype=np.float32)
        a.tofile(os.path.join(OUT, n + ".raw"))
        print("  %-18s %-22s %10.1f MB  |a|max=%.4g"
              % (n, str(a.shape), a.nbytes / 1e6, float(np.abs(a).max())), flush=True)
    # adaln_input 与 part2a 共用
    src = os.path.join(SRC, "adaln_input.raw")
    np.fromfile(src, np.float32).tofile(os.path.join(OUT, "adaln_input.raw"))
    print("  adaln_input        （从 %s 复制）" % src, flush=True)
    print("⇒ 写出 %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
