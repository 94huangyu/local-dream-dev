"""EXP_PLAN_ATTENTION 判据 1：沿注意力依赖链看主体误差与主体余弦在哪一步崩。

度量按约束 7：
  主体相对 L2（|a| <= p99）、主体余弦、前 1% 元素占 ||a||^2 的比例。
  余弦 < 0.3 的张量退出定量比较，只标"已毁"。
  前 1% 能量 > 50% 的张量，全量相对 L2 不得用于结论。
"""
import os

import numpy as np

CPU = r"D:\ZImage_Work\p0_experiments\attn_probe\cpu\out\Result_0"
HTP = r"D:\ZImage_Work\p0_experiments\attn_probe\htp"
# 第二轮已有的两个，纳入同一条链
CPU2 = r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu\out\Result_0"
HTP2 = r"D:\ZImage_Work\p0_experiments\layer_probe2\htp"

# 依赖顺序（trace_dag.py 验证过），不是 Id 顺序
CHAIN = [
    ("transpose_72", "Q（Transpose）"),
    ("val_2587", "Q x scale（Eltwise）"),
    ("val_2590", "K^T x scale（Eltwise）"),
    ("val_2590_converted_unsigned_symmetric", "Convert(K) 对称化"),
    ("transpose_74", "V（Transpose）"),
    ("transpose_74_converted_unsigned_symmetric", "Convert(V) 对称化"),
    ("scaled_dot_product_attention_18", "注意力输出（MatMul）"),
]


def load(n):
    for c, h in ((CPU, HTP), (CPU2, HTP2)):
        pa, pb = os.path.join(c, n + ".raw"), os.path.join(h, n + ".raw")
        if os.path.exists(pa) and os.path.exists(pb):
            a = np.fromfile(pa, np.float32).astype(np.float64)
            b = np.fromfile(pb, np.float32).astype(np.float64)
            if a.size == b.size:
                return a, b
    return None, None


print(f"{'张量':<44}{'说明':<22}{'主体L2%':>10}{'主体余弦':>10}{'前1%能量':>10}{'状态':>10}")
print("=" * 108)
rows = []
for n, desc in CHAIN:
    a, b = load(n)
    if a is None:
        print(f"{n:<44}{desc:<22}(缺文件)")
        continue
    p99 = np.percentile(np.abs(a), 99)
    m = np.abs(a) <= p99
    bulk = 100.0 * np.linalg.norm((b - a)[m]) / np.linalg.norm(a[m])
    ca, cb = a[m], b[m]
    cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))
    e = a ** 2
    top1 = np.sort(e)[::-1][:max(1, e.size // 100)].sum() / e.sum()
    st = "已毁" if cos < 0.3 else ("劣化" if cos < 0.9 else "健康")
    rows.append((n, desc, bulk, cos, top1, st))
    print(f"{n:<44}{desc:<22}{bulk:>10.2f}{cos:>10.4f}{100*top1:>9.1f}%{st:>10}")

print()
print("=" * 108)
print("判据 1（EXP_PLAN_ATTENTION 第 3 节，事前定稿）")
print("=" * 108)
if not rows:
    raise SystemExit("无数据")

# 找第一个余弦跌破 0.9 的环节
first = next((r for r in rows if r[3] < 0.9), None)
inputs = [r for r in rows[:-1]]
out = rows[-1] if rows[-1][0] == "scaled_dot_product_attention_18" else None

print("  入口张量（Q/K/V 及其 Convert）的主体余弦：")
for r in inputs:
    print(f"    {r[0]:<46}{r[3]:.4f}  {r[5]}")
if out:
    print(f"  注意力输出：{out[3]:.4f}  {out[5]}")
print()

worst_in = min(inputs, key=lambda r: r[3]) if inputs else None
if worst_in and worst_in[3] < 0.7:
    print(f"  => 【入口就已劣化】最差入口 {worst_in[0]} 余弦 {worst_in[3]:.4f}")
    print("     引入点在注意力段【上游】，注意力只是传递误差。H-attn 被否定。")
elif worst_in and out and worst_in[3] > 0.9 and out[3] < 0.7:
    print(f"  => 【入口健康、输出崩塌】入口最差 {worst_in[3]:.4f}，输出 {out[3]:.4f}")
    print("     引入点在注意力段【内部】(MatMul/Softmax/加权MatMul)。")
    print("     需补测 val_2591(打分) 与 val_2592(Softmax 输出) 才能进一步区分。")
elif worst_in and out and out[3] > 0.9:
    print("  => 全链余弦均 >0.9，注意力段不是引入点。H-attn 被否定，转台账 #19。")
else:
    print("  => 介于中间，按各环节余弦逐段报告，不下单一结论。")

print()
print("Convert 算子的单独影响（对称化 MatMul 的 B 输入，台账 #25）：")
for base, conv in [("val_2590", "val_2590_converted_unsigned_symmetric"),
                   ("transpose_74", "transpose_74_converted_unsigned_symmetric")]:
    rb = next((r for r in rows if r[0] == base), None)
    rc = next((r for r in rows if r[0] == conv), None)
    if rb and rc:
        print(f"  {base:<16} 余弦 {rb[3]:.4f}  ->  Convert 后 {rc[3]:.4f}   "
              f"变化 {rc[3]-rb[3]:+.4f}")
