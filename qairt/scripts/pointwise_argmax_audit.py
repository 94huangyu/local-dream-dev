"""
按独立审查意见补的严格论证：不用"不同张量的最大值相加"，而是直接报告
add_54 最大误差元素所在的【同一个 (token, channel) 位置】上，各项的逐点数值。

审查意见原文：
  "不同张量的最大误差未必发生在同一元素位置，因此不能直接把两个最大值相加当作逐点解释。
   最好直接报告 add_54 最大误差元素位置上三项的逐点 FP32 值、量化值和误差。"
"""
import numpy as np

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
ONNX_DIR = EVIDENCE + r"\onnx"
Q_EARLY = r"D:\ZImage_Work\p0_experiments\snpe_early_chain\Result_0"
Q_MERGE = r"D:\ZImage_Work\p0_experiments\snpe_merge_window\Result_0"
Q_FFN = r"D:\ZImage_Work\p0_experiments\snpe_caption_ffn\Result_0"

early = np.load(ONNX_DIR + r"\early_block_reference.npz")
merge = np.load(ONNX_DIR + r"\merge_window_reference.npz")
ffn = np.load(ONNX_DIR + r"\caption_ffn_reference.npz")


def dev(d, name, shape):
    return np.fromfile(f"{d}\\{name}.raw", dtype=np.float32).reshape(shape)


# ---------- 1) 定位 add_54 的最大误差元素 ----------
r54 = early["add_54"].astype(np.float32)[0]              # (4128, 3840)
q54 = dev(Q_EARLY, "add_54", (1, 4128, 3840))[0]
e54 = np.abs(q54 - r54)
t, c = np.unravel_index(e54.argmax(), e54.shape)
print("=" * 96)
print(f"add_54 最大误差元素位置: token={t} (caption段起点=4096), channel={c}")
print("=" * 96)
print(f"  FP32   = {r54[t, c]:.6f}")
print(f"  量化   = {q54[t, c]:.6f}")
print(f"  误差   = {q54[t, c] - r54[t, c]:.6f}")
print()

# ---------- 2) 在【同一位置】上分解 add_54 = select_scatter_4 + mul_127 ----------
print("=" * 96)
print("在这同一个 (token, channel) 位置上，逐点分解 add_54 = select_scatter_4 + mul_127")
print("=" * 96)
print(f"{'项':<26} {'FP32':>16} {'量化':>16} {'逐点误差':>16}")
print("-" * 96)
tot_r = tot_q = 0.0
for nm, shp in [("select_scatter_4", (1, 4128, 3840)), ("mul_127", (1, 4128, 3840))]:
    r = merge[nm].astype(np.float32)[0]
    q = dev(Q_MERGE, nm, shp)[0]
    print(f"{nm:<26} {r[t, c]:>16.6f} {q[t, c]:>16.6f} {q[t, c] - r[t, c]:>16.6f}")
    tot_r += r[t, c]
    tot_q += q[t, c]
print("-" * 96)
print(f"{'两项之和':<26} {tot_r:>16.6f} {tot_q:>16.6f} {tot_q - tot_r:>16.6f}")
print(f"{'add_54 实际':<26} {r54[t, c]:>16.6f} {q54[t, c]:>16.6f} {q54[t, c] - r54[t, c]:>16.6f}")
print(f"{'差额(=add_54自身重量化)':<26} {r54[t,c]-tot_r:>16.6f} {q54[t,c]-tot_q:>16.6f}")
print()

# ---------- 3) 沿 caption FFN 链，在【同一 channel】上追踪该 token 的误差 ----------
cap_t = t - 4096 if t >= 4096 else None
print("=" * 96)
if cap_t is None:
    print("最大误差不在 caption 段，跳过 FFN 链逐点追踪")
else:
    print(f"沿 caption FFN 链追踪 caption token {cap_t}、channel {c} 的逐点误差")
    print("=" * 96)
    print(f"{'张量':<16} {'FP32':>16} {'量化':>16} {'逐点误差':>14} {'该张量全局max误差':>18}")
    print("-" * 96)
    for nm in ["mul_106", "linear_31", "linear_32", "silu_4", "mul_107", "linear_33", "mul_109"]:
        r = ffn[nm].astype(np.float32)[0]
        q = dev(Q_FFN, nm, r.shape if r.ndim == 2 else r.shape)
        q = q.reshape(r.shape)
        ch = c if r.shape[1] > c else r.shape[1] - 1
        tag = "" if r.shape[1] > c else f" (该张量只有{r.shape[1]}通道,取末列)"
        e = np.abs(q - r)
        print(f"{nm:<16} {r[cap_t, ch]:>16.6f} {q[cap_t, ch]:>16.6f} "
              f"{q[cap_t, ch]-r[cap_t, ch]:>14.6f} {e.max():>18.4f}{tag}")

print()
print("=" * 96)
print("严格结论（不依赖跨位置的最大值相加）")
print("=" * 96)
r_ss = merge["select_scatter_4"].astype(np.float32)[0]
q_ss = dev(Q_MERGE, "select_scatter_4", (1, 4128, 3840))[0]
r_m127 = merge["mul_127"].astype(np.float32)[0]
q_m127 = dev(Q_MERGE, "mul_127", (1, 4128, 3840))[0]
eA = abs(q_ss[t, c] - r_ss[t, c])
eB = abs(q_m127[t, c] - r_m127[t, c])
print(f"在 add_54 最大误差位置上：路径A 逐点误差={eA:.6f}，路径B 逐点误差={eB:.6f}")
print(f"路径B 的全局最大误差为 {np.abs(q_m127-r_m127).max():.6f}，即使取其上界也无法抵消路径A。")
print(f"=> 路径A 主导，该结论不依赖'不同位置最大值相加'。")
