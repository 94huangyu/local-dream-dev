"""
判定 add_54 上 31.84 的误差是从哪条路径进来的。

add_54 = Add(select_scatter_4, mul_127)
  路径 A（残差载体）: add_24[图像] + add_45[caption] -> Concat -> Pad(constant_pad_nd_4)
                      -> ScatterElements -> select_scatter_4
  路径 B（注意力）  : select_scatter_4 -> RMSNorm(mul_112) -> QKV -> attention
                      -> scaled_dot_product_attention_4 -> linear_38 -> mul_126 -> mul_127

判读：哪个张量在 caption token 段（unified 序列的 [4096:4128]）/ 通道 85 上首次出现 ~30 量级误差。
"""
import numpy as np

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
REF = np.load(EVIDENCE + r"\onnx\merge_window_reference.npz")
QDIR = r"D:\ZImage_Work\p0_experiments\snpe_merge_window\Result_0"

TENSORS = ["constant_pad_nd_4", "select_scatter_4", "mul_112",
           "scaled_dot_product_attention_4", "linear_38", "mul_127"]
IMG_N = 4096
CH = 85   # 13.12 定位到的问题通道

print("=" * 112)
print("路径判定：各张量的误差，以及误差在 caption 段 / 通道 85 上的分布")
print("=" * 112)
print(f"{'张量':<32} {'shape':<22} {'ref|max|':>10} {'误差max':>10} "
      f"{'图像段误差':>11} {'caption段误差':>13} {'通道85误差':>11}")
print("-" * 112)

for name in TENSORS:
    r = REF[name].astype(np.float32)
    q = np.fromfile(f"{QDIR}\\{name}.raw", dtype=np.float32)
    if q.size != r.size:
        print(f"{name:<32} 尺寸不符 ref={r.size} q={q.size}")
        continue
    q = q.reshape(r.shape)
    err = np.abs(q - r)

    # 把张量整理成 (token, channel) 视图
    if r.ndim == 3 and r.shape[1] == 4128 and r.shape[2] == 3840:
        e2 = err[0]
    elif r.ndim == 2 and r.shape[0] == 4128:
        e2 = err
    elif name == "scaled_dot_product_attention_4":     # (1,30,4128,128)
        e2 = err[0].transpose(1, 0, 2).reshape(4128, -1)
    else:
        e2 = None

    if e2 is not None:
        img_e = e2[:IMG_N].max()
        cap_e = e2[IMG_N:].max()
        ch_e = e2[:, CH].max() if e2.shape[1] > CH else float("nan")
        print(f"{name:<32} {str(r.shape):<22} {np.abs(r).max():>10.2f} {err.max():>10.4f} "
              f"{img_e:>11.4f} {cap_e:>13.4f} {ch_e:>11.4f}")
    else:
        print(f"{name:<32} {str(r.shape):<22} {np.abs(r).max():>10.2f} {err.max():>10.4f} "
              f"{'(无法按token拆分)':>37}")

print()
print("=" * 112)
print("判读")
print("=" * 112)


def cap_err(name):
    r = REF[name].astype(np.float32)
    q = np.fromfile(f"{QDIR}\\{name}.raw", dtype=np.float32).reshape(r.shape)
    e = np.abs(q - r)
    e2 = e[0] if e.ndim == 3 else e
    return e2[IMG_N:].max()


a = cap_err("select_scatter_4")
b = cap_err("mul_127")
print(f"路径A 残差载体 select_scatter_4  caption段误差 = {a:.4f}")
print(f"路径B 注意力输出 mul_127         caption段误差 = {b:.4f}")
print(f"两者相加 ≈ {a + b:.4f}   （add_54 实测 caption 段误差 = 31.8380）")
print()
if a > 10:
    print(">>> 误差从【路径A 残差载体】进来：说明它在合并之前就已经存在，")
    print(">>> 即产生在 caption 流最后一个 block（add_42 -> add_45，算子 495~515）。")
    print(">>> 注意 add_45 在 DLC 里被融合掉了，需要用 mul_106/linear_31/32/33/mul_109 这些")
    print(">>> 可导出的张量继续二分。")
elif b > 10:
    print(">>> 误差从【路径B 注意力】进来：产生在第一个 unified attention block 内部（算子 532~617）。")
else:
    print(">>> 两条路径都没有 ~30 量级误差 —— 说明误差是在 Add 这一步本身产生的，")
    print(">>> 最可能的机制：add_54 的 per-tensor 量化编码要同时覆盖两条量级差 15 倍的流。")
    print(">>> 这种情况下应该去看 add_54 自身的 encoding 和 QNN 的 Add 算子重量化行为。")
