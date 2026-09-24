"""
decompose_linear33.py 的修正版。

上一版的缺陷：模拟时用的是 FP32 的 mul_107 作输入，但设备实际用的是它自己那份带误差的
mul_107（实测误差 max 8.39）。本版用【设备实测的 mul_107】重新算，回答一个关键问题：

  linear_33 上 350.13 的误差，是
    (A) 从 mul_107 继承来、再被通道85那根超大权重列放大的？   -> 修复要往上游走
    (B) 还是在 MatMul 内部产生的（累加器饱和/溢出等硬件行为）？ -> 属于硬件/编译层问题

判据：用设备的 mul_107 @ FP32权重，如果能复现出 ~350 的误差，就是 (A)；否则是 (B)。
"""
import json
import numpy as np
import onnx
from onnx import numpy_helper

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
ONNX_DIR = EVIDENCE + r"\onnx"
QDIR = r"D:\ZImage_Work\p0_experiments\snpe_caption_ffn\Result_0"
ENC = json.load(open(r"D:\ZImage_Work\p0_experiments\dump_test_out_encoding.json"))
CH = 85

m = onnx.load(ONNX_DIR + r"\transformer_part1a.onnx", load_external_data=False)
node = next(n for n in m.graph.node if n.output[0] == "linear_33")
init_map = {i.name: i for i in m.graph.initializer}
wname = next(i for i in node.input if i in init_map)
W = numpy_helper.to_array(init_map[wname], base_dir=ONNX_DIR).astype(np.float32)

ref = np.load(ONNX_DIR + r"\caption_ffn_reference.npz")
x_fp32 = ref["mul_107"].astype(np.float32)[0]
y_fp32 = ref["linear_33"].astype(np.float32)[0]
x_dev = np.fromfile(QDIR + r"\mul_107.raw", dtype=np.float32).reshape(x_fp32.shape)
y_dev = np.fromfile(QDIR + r"\linear_33.raw", dtype=np.float32).reshape(y_fp32.shape)
if W.shape[0] != x_fp32.shape[1]:
    W = W.T


def quant_w_per_tensor(W, bits=8):
    lo, hi = W.min(), W.max()
    s = (hi - lo) / (2 ** bits - 1)
    off = np.rint(lo / s)
    return ((np.clip(np.rint(W / s - off), 0, 2 ** bits - 1) + off) * s).astype(np.float32)


def quant_w_per_channel(W, bits=8):
    lo, hi = W.min(axis=0, keepdims=True), W.max(axis=0, keepdims=True)
    s = (hi - lo) / (2 ** bits - 1)
    s[s == 0] = 1e-12
    off = np.rint(lo / s)
    return ((np.clip(np.rint(W / s - off), 0, 2 ** bits - 1) + off) * s).astype(np.float32)


cases = {
    "设备mul_107 @ FP32权重":            x_dev @ W,
    "设备mul_107 @ 8bit权重(per-tensor)": x_dev @ quant_w_per_tensor(W),
    "设备mul_107 @ 8bit权重(per-channel)": x_dev @ quant_w_per_channel(W),
    "FP32 mul_107 @ FP32权重 (基准)":     x_fp32 @ W,
}

print("=" * 100)
print("用【设备实测的 mul_107】重新分解 linear_33 的误差")
print("=" * 100)
print(f"输入 mul_107 的设备误差: max={np.abs(x_dev-x_fp32).max():.4f}, "
      f"rms={np.sqrt(((x_dev-x_fp32)**2).mean()):.6f}")
print(f"通道85 权重列: |max|={np.abs(W[:, CH]).max():.4f}, rms={np.sqrt((W[:, CH]**2).mean()):.4f} "
      f"(全矩阵 rms={W.std():.4f})")
print()
print(f"{'对照组':<38} {'全张量误差max':>14} {'通道85误差':>13} {'距设备实测':>13}")
print("-" * 100)
for label, y in cases.items():
    e = np.abs(y - y_fp32)
    print(f"{label:<38} {e.max():>14.4f} {e[:, CH].max():>13.4f} "
          f"{np.abs(y - y_dev).max():>13.4f}")
dev_e = np.abs(y_dev - y_fp32)
print("-" * 100)
print(f"{'设备实测':<38} {dev_e.max():>14.4f} {dev_e[:, CH].max():>13.4f} {0.0:>13.4f}")

print()
print("=" * 100)
print("通道 85 上，设备输出 vs FP32 输出的逐 token 对照（32 个 caption token）")
print("=" * 100)
print(f"{'token':>6} {'FP32值':>14} {'设备值':>14} {'误差':>12} {'比值':>10}")
for t in range(32):
    a, b = y_fp32[t, CH], y_dev[t, CH]
    print(f"{t:>6} {a:>14.4f} {b:>14.4f} {b-a:>12.4f} {(b/a if abs(a)>1e-6 else float('nan')):>10.5f}")

print()
print("=" * 100)
print("判读")
print("=" * 100)
inherit = np.abs(cases["设备mul_107 @ FP32权重"] - y_fp32)[:, CH].max()
print(f"仅靠继承 mul_107 的误差能解释的通道85误差 = {inherit:.4f}")
print(f"设备实测通道85误差                       = {dev_e[:, CH].max():.4f}")
if inherit > dev_e[:, CH].max() * 0.5:
    print(">>> (A) 误差主要是从 mul_107 继承、被通道85的超大权重列放大的。修复应针对 mul_107 的量化。")
else:
    print(">>> (B) 继承解释不了。误差是在 MatMul 内部产生的 —— 指向累加器饱和/溢出等硬件层行为，")
    print(">>>     或者设备上这一层用了和我们假设不同的计算精度。下一步应查该层在 DLC 里的实际")
    print(">>>     数据类型与 encoding（qairt-quantizer --dump_encoding_json 里 val_802 的 param 编码）。")
