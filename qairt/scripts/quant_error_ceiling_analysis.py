"""
收益上限评估（纯计算，不跑 SDK，不改任何文件）

核心问题：把残差张量标成 float / 提高它自己的编码精度，最多能挽回多少误差？

逻辑：
  一个张量用 scale=s 的定点编码存储，如果喂给它的是**完全正确**的浮点值，
  那么"存储这一步"引入的误差上限就是 s/2（round-to-nearest）。
  所以：
    - 若 观测误差 >> s/2  =>  误差不是这个张量自己的编码造成的，改它的编码无效。
    - 若 观测误差 ≈ s/2   =>  误差确实来自这个张量自己的量化，改它的编码有效。
"""
import json
import numpy as np

# 第 8.4 节实测的逐层最大误差（HANDOVER 文档 8.4 节表格，来自 trace_residual_corruption.py）
OBSERVED_MAX_ERR = {
    "add_54": 31.8,
    "add_66": 34.3,
    "add_78": 48.2,
    "add_90": 47.3,
    "add_102": 60.5,
    "add_114": 87.3,
    "add_126": 149.3,
    "add_138": 321.8,
}

ENC_PATH = r"D:/ZImage_Work/p0_experiments/dump_test_out_encoding.json"
NPZ_PATH = r"D:/ZImage_Work/ZImage_QNN_Evidence/onnx/residual_chain_reference.npz"

enc = json.load(open(ENC_PATH))
act = enc["activation_encodings"]

print("=" * 100)
print("表 1：每个残差张量『自己的编码』最多能贡献多少误差 vs 实际观测到的误差")
print("=" * 100)
print(f"{'张量':<10} {'own scale':>14} {'自身量化误差上限':>18} {'实测最大误差':>14} {'实测/自身上限':>14} {'可归因于自身编码':>16}")
print("-" * 100)

rows = []
for name, observed in OBSERVED_MAX_ERR.items():
    e = act.get(name)
    if e is None:
        print(f"{name:<10} {'<缺失>':>14}")
        continue
    scale = e[0]["scale"]
    own_limit = scale / 2.0          # round-to-nearest 的最大误差
    ratio = observed / own_limit
    attributable = own_limit / observed * 100.0
    rows.append((name, scale, own_limit, observed, ratio, attributable))
    print(f"{name:<10} {scale:>14.6g} {own_limit:>18.6g} {observed:>14.1f} {ratio:>14,.0f}x {attributable:>15.4f}%")

print()
print("=" * 100)
print("表 2：用 FP32 参考值交叉验证编码 scale 是否自洽（排除『scale 本身记错了』这个可能）")
print("=" * 100)

npz = np.load(NPZ_PATH)
print("npz 里的数组:", list(npz.keys())[:20])
print()
print(f"{'张量':<10} {'ref min':>14} {'ref max':>14} {'(max-min)/65535':>18} {'dump 里的 scale':>18} {'相符?':>8}")
print("-" * 100)

for name in OBSERVED_MAX_ERR:
    if name not in npz:
        print(f"{name:<10} <npz 中没有这个数组>")
        continue
    arr = npz[name]
    lo, hi = float(arr.min()), float(arr.max())
    implied = (hi - lo) / 65535.0
    e = act.get(name)
    if e is None:
        continue
    scale = e[0]["scale"]
    agree = "是" if abs(implied - scale) / max(scale, 1e-12) < 0.35 else "否"
    print(f"{name:<10} {lo:>14.4f} {hi:>14.4f} {implied:>18.6g} {scale:>18.6g} {agree:>8}")

print()
print("=" * 100)
print("表 3：add_138 的误差到底是不是『自己存储时舍入』造成的——逐元素分解")
print("=" * 100)

if "add_138" in npz:
    ref = npz["add_138"].astype(np.float32)
    e = act["add_138"][0]
    scale, offset = e["scale"], e["offset"]
    # 模拟：如果喂进来的是完全正确的 FP32 值，按这个编码量化再反量化，会得到什么
    q = np.rint(ref / scale - offset)
    q = np.clip(q, 0, 65535)
    deq = (q + offset) * scale
    err_if_input_were_perfect = np.abs(deq - ref)
    print(f"add_138 shape={ref.shape}  ref范围=[{ref.min():.2f}, {ref.max():.2f}]")
    print(f"编码: scale={scale:.6g}, offset={offset}")
    print()
    print(f"  【假设输入完全正确】只因这一层存储产生的误差:")
    print(f"      最大 = {err_if_input_were_perfect.max():.6f}")
    print(f"      均值 = {err_if_input_were_perfect.mean():.6f}")
    print(f"      有多少元素误差 > 1.0 : {(err_if_input_were_perfect > 1.0).sum()}")
    print()
    print(f"  【实际观测到的】最大误差 = {OBSERVED_MAX_ERR['add_138']}")
    print()
    print(f"  => 差距 {OBSERVED_MAX_ERR['add_138'] / err_if_input_were_perfect.max():,.0f} 倍")
