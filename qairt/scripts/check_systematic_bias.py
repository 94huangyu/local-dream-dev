"""
新发现引出的问题：在 (caption token 4, channel 85) 位置上，mul_107 的值只有 0.0223，
误差却有 -0.0358（比数值本身还大，符号翻转）。而 linear_33 = mul_107 @ W 累加 10240 项后
误差达到 350。

若量化误差是随机的，10240 项累加只会得到约 0.4（sqrt(10240) * 0.007 * 0.553）。
实测 350 是它的 900 倍，所以误差必然是【系统性】的。

本脚本检验"系统性偏置"假设：
  若 mul_107 的量化误差存在常数偏置 b（例如 zero-point 不可精确表示导致所有小值同向偏移），
  则累加误差 ≈ b * Σw，可以直接验算。
"""
import json
import numpy as np
import onnx
from onnx import numpy_helper

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
ONNX_DIR = EVIDENCE + r"\onnx"
QDIR = r"D:\ZImage_Work\p0_experiments\snpe_caption_ffn\Result_0"
ENC = json.load(open(r"D:\ZImage_Work\p0_experiments\dump_test_out_encoding.json"))
CH, TOK = 85, 4

m = onnx.load(ONNX_DIR + r"\transformer_part1a.onnx", load_external_data=False)
node = next(n for n in m.graph.node if n.output[0] == "linear_33")
init_map = {i.name: i for i in m.graph.initializer}
wname = next(i for i in node.input if i in init_map)
W = numpy_helper.to_array(init_map[wname], base_dir=ONNX_DIR).astype(np.float32)

ref = np.load(ONNX_DIR + r"\caption_ffn_reference.npz")
x_ref = ref["mul_107"].astype(np.float32)[0]
x_dev = np.fromfile(QDIR + r"\mul_107.raw", dtype=np.float32).reshape(x_ref.shape)
if W.shape[0] != x_ref.shape[1]:
    W = W.T

dx = x_dev - x_ref               # (32, 10240) 量化误差
w = W[:, CH]                     # (10240,) 通道85的权重列

e = ENC["activation_encodings"]["mul_107"][0]
print("=" * 92)
print("mul_107 的编码与误差统计")
print("=" * 92)
print(f"编码: bitwidth={e['bitwidth']}, scale={e['scale']:.8f}, offset={e['offset']}, "
      f"min={e['min']:.3f}, max={e['max']:.3f}")
print(f"round-to-nearest 理论误差上限 = scale/2 = {e['scale']/2:.8f}")
print(f"实测 |dx| max = {np.abs(dx).max():.6f}   （超过理论上限 {np.abs(dx).max()/(e['scale']/2):.1f} 倍）")
print()

row = dx[TOK]
print("=" * 92)
print(f"caption token {TOK} 这一行（10240 个元素）的误差结构")
print("=" * 92)
print(f"  误差均值 (偏置 b) = {row.mean():.8f}")
print(f"  误差标准差        = {row.std():.8f}")
print(f"  |误差| 最大       = {np.abs(row).max():.6f}")
print(f"  误差>0 的比例     = {(row > 0).mean()*100:.2f}%")
print()
print("=" * 92)
print("通道 85 权重列统计")
print("=" * 92)
print(f"  Σw   = {w.sum():.6f}")
print(f"  |w|max = {np.abs(w).max():.6f}   rms = {np.sqrt((w**2).mean()):.6f}")
print()

actual = float(row @ w)
bias_pred = float(row.mean() * w.sum())
rand_pred = float(np.sqrt(10240) * row.std() * np.sqrt((w**2).mean()))

print("=" * 92)
print("三种模型对 linear_33 误差的预测 vs 实测")
print("=" * 92)
print(f"  实测误差 (dx · w)          = {actual:>14.4f}   <- 真值")
print(f"  纯偏置模型 (mean(dx)·Σw)   = {bias_pred:>14.4f}")
print(f"  纯随机模型 (√n·σ_dx·rms_w) = {rand_pred:>14.4f}")
print()
print(f"  偏置模型解释了 {bias_pred/actual*100:.2f}% 的实测误差")
print(f"  随机模型解释了 {rand_pred/actual*100:.2f}% 的实测误差")
print()

# 相关性：误差是否与权重同向
corr = float(np.corrcoef(row, w)[0, 1])
print(f"  误差与权重的相关系数 = {corr:.6f}")
print()

# 按 |x| 分组看误差，判断是不是"小值被 scale 吞掉"
print("=" * 92)
print("按输入幅值分组，看误差集中在哪一档（判断是否'小值被粗 scale 吞掉'）")
print("=" * 92)
xr = x_ref[TOK]
bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1), (1, 10), (10, 1000)]
print(f"{'|x| 区间':<16} {'元素数':>8} {'|误差|均值':>13} {'该组对输出的贡献':>18}")
print("-" * 92)
for lo, hi in bins:
    msk = (np.abs(xr) >= lo) & (np.abs(xr) < hi)
    if msk.sum() == 0:
        continue
    contrib = float(row[msk] @ w[msk])
    print(f"[{lo:>6.2f},{hi:>7.2f}) {msk.sum():>8} {np.abs(row[msk]).mean():>13.6f} {contrib:>18.4f}")

print()
print("=" * 92)
print("判读")
print("=" * 92)
if abs(bias_pred / actual) > 0.5:
    print(">>> 系统性偏置是主因：mul_107 的量化误差存在常数偏移，被 Σw 放大。")
    print(">>> 这类问题通常来自 zero-point 无法精确表示，或校准 min/max 不对称。")
elif abs(corr) > 0.05:
    print(">>> 误差与权重显著相关：说明量化误差不是独立随机的，与权重结构耦合。")
else:
    print(">>> 既不是纯偏置也不是纯随机，需要看上面的分组贡献表定位是哪一档幅值的元素在贡献误差。")
