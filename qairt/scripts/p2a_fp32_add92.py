"""判定要用的 FP32 真值：part2a 的图输出 add_92（参照臂 HTP 值今天已在 p2attr/htp/）。"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "testB", "s0_transformer_part2")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr", "fp32")
IN_SHAPES = {"unified": (1, 4128, 3840), "unified_mask": (1, 4128),
             "unified_freqs": (1, 4128, 64, 2), "adaln_input": (1, 256)}
dst = os.path.join(OUT, "add_92.raw")
if os.path.exists(dst):
    print("已存在，复用:", dst); sys.exit(0)
import onnx, onnxruntime as ort
from onnx import TensorProto
p = os.path.join(D, "transformer_part2a_fixed.onnx")
m = onnx.load(p, load_external_data=False)
feeds = {}
for n, s in IN_SHAPES.items():
    feeds[n] = np.fromfile(os.path.join(SRC, n + ".raw"), np.float32).reshape(s)
for i in m.graph.input:
    if i.type.tensor_type.elem_type == TensorProto.BOOL:
        feeds[i.name] = feeds[i.name].astype(bool)
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
r = s.run(["add_92"], feeds)[0].astype(np.float32)
np.ascontiguousarray(r).tofile(dst)
print("写出 %s  shape=%s std=%.6f" % (dst, r.shape, r.std()))
