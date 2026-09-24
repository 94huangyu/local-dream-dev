"""
第三轮二分：caption 流最后一个 FFN block（算子 495~515）。
add_42 误差 1.07 -> add_45 误差 31.92，找出这条链上误差跳变的具体位置。

链条: add_42 -> [RMSNorm融合] -> mul_106 -> linear_31/linear_32 -> silu_4
      -> mul_107 -> linear_33 -> [RMSNorm融合] -> mul_109 -> add_45
"""
import numpy as np

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
REF = np.load(EVIDENCE + r"\onnx\caption_ffn_reference.npz")
QDIR = r"D:\ZImage_Work\p0_experiments\snpe_caption_ffn\Result_0"

CHAIN = ["mul_106", "linear_31", "linear_32", "silu_4", "mul_107", "linear_33", "mul_109"]
CH = 85

print("=" * 104)
print("caption FFN block 链条上的误差演变（起点 add_42 误差=1.07，终点 add_45 误差=31.92）")
print("=" * 104)
print(f"{'张量':<14} {'shape':<18} {'ref min':>11} {'ref max':>11} "
      f"{'误差max':>11} {'通道85误差':>11} {'相对误差':>10}")
print("-" * 104)

prev = None
jump_at = None
for name in CHAIN:
    r = REF[name].astype(np.float32)
    q = np.fromfile(f"{QDIR}\\{name}.raw", dtype=np.float32)
    if q.size != r.size:
        print(f"{name:<14} 尺寸不符 ref={r.size} q={q.size}")
        continue
    q = q.reshape(r.shape)
    err = np.abs(q - r)
    e2 = err[0] if err.ndim == 3 else err
    ch_e = e2[:, CH].max() if e2.ndim == 2 and e2.shape[1] > CH else float("nan")
    rel = err.max() / max(np.abs(r).max(), 1e-9) * 100

    print(f"{name:<14} {str(r.shape):<18} {r.min():>11.3f} {r.max():>11.3f} "
          f"{err.max():>11.4f} {ch_e:>11.4f} {rel:>9.3f}%")

    if prev is not None and prev[1] > 0 and err.max() / prev[1] > 5:
        jump_at = (prev[0], name, prev[1], err.max())
    prev = (name, err.max())

print()
print("=" * 104)
print("判读")
print("=" * 104)
if jump_at:
    a, b, ea, eb = jump_at
    print(f">>> 误差在 {a} (误差 {ea:.4f}) -> {b} (误差 {eb:.4f}) 之间跳了 {eb/ea:.1f} 倍")
    print(f">>> 根因算子就是产生 {b} 的那个算子。")
else:
    print(">>> 链条上没有出现 5 倍以上的单步跳变，误差是渐进累积的，")
    print(">>> 或者跳变发生在被 DLC 融合掉、无法导出的 RMSNorm 段里。")

# 额外看通道 85 在 FP32 参考里的绝对量级，判断它是不是天然的 massive activation 通道
print()
print("通道 85 在各张量里的 FP32 数值量级（看它是否天生就是超大值通道）：")
for name in CHAIN:
    r = REF[name].astype(np.float32)
    r2 = r[0] if r.ndim == 3 else r
    if r2.ndim == 2 and r2.shape[1] > CH:
        col = r2[:, CH]
        allmax = np.abs(r2).max()
        print(f"  {name:<14} 通道85 |max| = {np.abs(col).max():>11.3f}   "
              f"整张量 |max| = {allmax:>11.3f}   占比 {np.abs(col).max()/allmax*100:>6.1f}%")
