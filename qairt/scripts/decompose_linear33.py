"""
机制分解：linear_33 = MatMul(mul_107, W) 这一步把误差放大 41.7 倍，误差 100% 落在通道 85。
本脚本在 numpy 里分离各个误差来源，确定到底是"输入量化"还是"权重量化"造成的。

对照组：
  (0) FP32 基准             : x_fp32 @ W_fp32
  (1) 只量化输入            : Q(x) @ W_fp32
  (2) 只量化权重(per-tensor): x_fp32 @ Q_pt(W)
  (3) 只量化权重(per-channel): x_fp32 @ Q_pc(W)
  (4) 输入+权重都量化       : Q(x) @ Q(W)
  (5) 设备实测              : snpe-net-run 导出的 linear_33.raw

比对 (1)~(4) 哪个最接近 (5)，就知道设备上真正的误差来源是什么。
"""
import json
import numpy as np
import onnx
from onnx import numpy_helper, external_data_helper

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
ONNX_DIR = EVIDENCE + r"\onnx"
QDIR = r"D:\ZImage_Work\p0_experiments\snpe_caption_ffn\Result_0"
ENC = json.load(open(r"D:\ZImage_Work\p0_experiments\dump_test_out_encoding.json"))
CH = 85

# ---------- 取出 linear_33 这个 MatMul 的权重 ----------
m = onnx.load(ONNX_DIR + r"\transformer_part1a.onnx", load_external_data=False)
node = next(n for n in m.graph.node if n.output[0] == "linear_33")
init_map = {i.name: i for i in m.graph.initializer}
wname = next(i for i in node.input if i in init_map)
print(f"linear_33 = MatMul({[i for i in node.input if i not in init_map]}, {wname})")

wt = init_map[wname]
# 注意：必须给 to_array 传 base_dir，否则它会用 cwd 去找 .onnx.data 而报
# "should be stored in xxx.onnx.data, but it is not regular file"
W = numpy_helper.to_array(wt, base_dir=ONNX_DIR).astype(np.float32)
print(f"权重 shape = {W.shape}, 范围 [{W.min():.5f}, {W.max():.5f}], std={W.std():.5f}")
print()

# ---------- 权重的逐通道统计：通道 85 是否异常 ----------
# W 形状 (10240, 3840)，输出通道是第 2 维
col_absmax = np.abs(W).max(axis=0)
col_rms = np.sqrt((W ** 2).mean(axis=0))
order = np.argsort(col_absmax)[::-1]
print("=" * 92)
print("权重矩阵按输出通道统计（看通道 85 是否是异常大的那一列）")
print("=" * 92)
print(f"通道 85       : |max|={col_absmax[CH]:.5f}  rms={col_rms[CH]:.5f}")
print(f"全部通道中位数: |max|={np.median(col_absmax):.5f}  rms={np.median(col_rms):.5f}")
print(f"通道 85 的 |max| 排名: 第 {list(order).index(CH)+1} / {len(col_absmax)}")
print(f"|max| 最大的 5 个通道: {order[:5].tolist()}  值={col_absmax[order[:5]].round(5).tolist()}")
print()

# ---------- 量化工具 ----------
def quant_act(x, name):
    e = ENC["activation_encodings"][name][0]
    s, off = e["scale"], e["offset"]
    q = np.clip(np.rint(x / s - off), 0, 2 ** e["bitwidth"] - 1)
    return ((q + off) * s).astype(np.float32)

def quant_w_per_tensor(W, bits=8):
    lo, hi = W.min(), W.max()
    s = (hi - lo) / (2 ** bits - 1)
    off = np.rint(lo / s)
    q = np.clip(np.rint(W / s - off), 0, 2 ** bits - 1)
    return ((q + off) * s).astype(np.float32)

def quant_w_per_channel(W, bits=8):
    lo = W.min(axis=0, keepdims=True)
    hi = W.max(axis=0, keepdims=True)
    s = (hi - lo) / (2 ** bits - 1)
    s[s == 0] = 1e-12
    off = np.rint(lo / s)
    q = np.clip(np.rint(W / s - off), 0, 2 ** bits - 1)
    return ((q + off) * s).astype(np.float32)

# ---------- 载入输入与设备实测 ----------
ref = np.load(ONNX_DIR + r"\caption_ffn_reference.npz")
x = ref["mul_107"].astype(np.float32)[0]          # (32, 10240)
y_fp32 = ref["linear_33"].astype(np.float32)[0]   # (32, 3840)
y_dev = np.fromfile(QDIR + r"\linear_33.raw", dtype=np.float32).reshape(32, 3840)

# 有些 MatMul 权重存的是 (in, out)，核对一下方向
if W.shape[0] != x.shape[1]:
    W = W.T
print(f"输入 x {x.shape} @ 权重 {W.shape} -> 输出 {y_fp32.shape}")
print()

xq = quant_act(x, "mul_107")
Wpt = quant_w_per_tensor(W)
Wpc = quant_w_per_channel(W)

cases = {
    "(1) 只量化输入":            xq @ W,
    "(2) 只量化权重 per-tensor":  x @ Wpt,
    "(3) 只量化权重 per-channel": x @ Wpc,
    "(4) 输入+权重 per-tensor":   xq @ Wpt,
    "(5) 输入+权重 per-channel":  xq @ Wpc,
}

print("=" * 92)
print("各误差来源的贡献（对照设备实测）")
print("=" * 92)
print(f"{'对照组':<28} {'全张量误差max':>14} {'通道85误差':>13} {'和设备实测的差距':>18}")
print("-" * 92)
dev_err = np.abs(y_dev - y_fp32)
for label, y in cases.items():
    e = np.abs(y - y_fp32)
    gap = np.abs(y - y_dev).max()
    print(f"{label:<28} {e.max():>14.4f} {e[:, CH].max():>13.4f} {gap:>18.4f}")
print("-" * 92)
print(f"{'设备实测 (snpe-net-run)':<28} {dev_err.max():>14.4f} {dev_err[:, CH].max():>13.4f} {0.0:>18.4f}")

print()
print("=" * 92)
print("判读")
print("=" * 92)
best = min(cases.items(), key=lambda kv: np.abs(kv[1] - y_dev).max())
print(f">>> 最接近设备实测的是：{best[0]}")
e_in = np.abs(cases["(1) 只量化输入"] - y_fp32)[:, CH].max()
e_wpt = np.abs(cases["(2) 只量化权重 per-tensor"] - y_fp32)[:, CH].max()
e_wpc = np.abs(cases["(3) 只量化权重 per-channel"] - y_fp32)[:, CH].max()
print(f">>> 通道85 上：输入量化贡献 {e_in:.2f}，权重per-tensor贡献 {e_wpt:.2f}，"
      f"权重per-channel贡献 {e_wpc:.2f}")
if e_wpt > e_in * 3:
    print(">>> 主因是【权重量化】。若 per-channel 明显更好，说明设备上这层没吃到 per-channel。")
elif e_in > e_wpt * 3:
    print(">>> 主因是【输入 mul_107 的激活量化】——它 std 只有 2.65 却要覆盖 ±862 的离群值，")
    print(">>> per-tensor scale 太粗，10240 项累加后放大。这属于激活离群值问题。")
else:
    print(">>> 两者贡献相当，需要一起处理。")
