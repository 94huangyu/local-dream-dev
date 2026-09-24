"""
对比前 4 个 block 的 FP32 参考值 vs 量化实测值，找出误差的真实起点。

判据（和 HANDOVER 13.8 的收益上限分析同一套逻辑）：
  某张量的误差如果 >> 它自身编码的 round-to-nearest 上限 (scale/2)，
  说明误差是"带进来的"，不是这一层存储造成的。
  沿着残差链往前走，第一个出现这种情况的位置，就是误差的起点区间。
"""
import json
import numpy as np

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
REF_NPZ = EVIDENCE + r"\onnx\early_block_reference.npz"
QUANT_DIR = r"D:\ZImage_Work\p0_experiments\snpe_early_chain\Result_0"
ENC = r"D:\ZImage_Work\p0_experiments\dump_test_out_encoding.json"

CHECKPOINTS = ["add_9", "add_12", "add_21", "add_32", "add_35", "add_42", "add_54"]

# 8.4 节已测的后段（用于把前后段接成一条完整曲线）
KNOWN_LATER = {
    "add_54": 31.8, "add_66": 34.3, "add_78": 48.2, "add_90": 47.3,
    "add_102": 60.5, "add_114": 87.3, "add_126": 149.3, "add_138": 321.8,
}

enc = json.load(open(ENC))["activation_encodings"]
ref = np.load(REF_NPZ)

print(f"{'张量':<10} {'shape':<20} {'ref |max|':>11} {'自身上限s/2':>12} "
      f"{'实测最大误差':>13} {'倍数':>9} {'离群元素最大误差':>16}")
print("-" * 100)

rows = []
for name in CHECKPOINTS:
    r = ref[name].astype(np.float32)
    q = np.fromfile(f"{QUANT_DIR}\\{name}.raw", dtype=np.float32)
    if q.size != r.size:
        print(f"{name:<10} 尺寸不匹配: ref={r.size} quant={q.size}，跳过")
        continue
    q = q.reshape(r.shape)

    err = np.abs(q - r)
    scale = enc[name][0]["scale"]
    own_limit = scale / 2.0
    max_err = float(err.max())

    # 和 8.4 节同样的"离群元素"划分：|真值| >= 5
    outlier = np.abs(r) >= 5.0
    out_max = float(err[outlier].max()) if outlier.any() else 0.0

    ratio = max_err / own_limit
    rows.append((name, max_err, own_limit, ratio, out_max))
    print(f"{name:<10} {str(r.shape):<20} {np.abs(r).max():>11.2f} {own_limit:>12.5f} "
          f"{max_err:>13.2f} {ratio:>8,.0f}x {out_max:>16.2f}")

print()
print("=" * 100)
print("完整残差链误差曲线（前段=本次实测，后段=8.4 节已测）")
print("=" * 100)
for name, max_err, own_limit, ratio, out_max in rows:
    bar = "#" * min(60, int(max_err / 5) + 1)
    print(f"{name:<10} {max_err:>8.2f}  {bar}")
for name, v in KNOWN_LATER.items():
    if name in [r[0] for r in rows]:
        continue
    bar = "#" * min(60, int(v / 5) + 1)
    print(f"{name:<10} {v:>8.2f}  {bar}   (8.4 节)")

print()
if rows:
    first = rows[0]
    print(f"最早测量点 {first[0]}：误差 {first[1]:.2f}，是自身编码上限的 {first[3]:,.0f} 倍")
    if first[3] > 100:
        print(">>> 误差在第 1 个 block 就已经远超自身编码能解释的范围，")
        print(">>> 起点还在更前面（embedding / patchify / 输入量化），需要继续往前测。")
    else:
        print(">>> 第 1 个 block 的误差还在自身编码量级，起点在本次测量的区间内，可以二分定位。")
