"""验证机制假设：RmsNorm 的 gamma 权重被 8-bit min-max 量化后，
多数元素只落在极少数量化级上 —— 相对精度崩塌。

定位结果（EXP_PLAN_LAYER_LOCALIZE 第二轮实测）：
  误差放大点全是 RmsNorm，且放大倍数与"幅值压缩倍数"相关：
    mul_476  压缩  13x -> 误差 0.65x（降低）
    mul_479  压缩 293x -> 误差 5.54x
    mul_483  压缩 265x -> 误差 16.9x，增益 k=0.331

本脚本从 FP32 ONNX 取出对应 gamma 的真实分布，核对其 8-bit 量化的有效精度。
"""
import numpy as np
import onnx
from onnx import numpy_helper

ONNX = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1b.onnx"

# (gamma 名, DLC 里的 8-bit encoding range, 该 RmsNorm 的误差表现)
TARGETS = [
    ("layers.14.attention_norm2.weight", (-0.5551, 4.687), "mul_476  压缩13x  误差0.65x(降低)"),
    ("layers.14.ffn_norm1.weight",       (-0.3523, 1.409), "mul_479  压缩293x 误差5.54x"),
    ("layers.14.ffn_norm2.weight",       (-0.4365, 8.126), "mul_483  压缩265x 误差16.9x k=0.331"),
]


def main():
    m = onnx.load(ONNX, load_external_data=True)
    inits = {t.name: t for t in m.graph.initializer}
    print(f"ONNX initializer 共 {len(inits)} 个\n")

    for name, (lo, hi), note in TARGETS:
        t = inits.get(name)
        if t is None:
            cands = [k for k in inits if name.split(".")[-2] in k and k.endswith("weight")]
            print(f"[缺] {name}   相近候选: {cands[:3]}")
            continue
        w = numpy_helper.to_array(t).astype(np.float64).ravel()
        scale = (hi - lo) / 255.0          # 8-bit min-max
        print("=" * 78)
        print(f"{name}")
        print(f"  {note}")
        print(f"  DLC encoding: range=[{lo}, {hi}]  scale={scale:.6g}  bw=8")
        print(f"  实际权重: n={w.size}  min={w.min():.6g}  max={w.max():.6g}")
        print(f"            mean={w.mean():.6g}  std={w.std():.6g}  median={np.median(w):.6g}")
        # 多数元素占用多少个量化级
        for p in (50, 90, 99):
            v = np.percentile(np.abs(w), p)
            print(f"    |w| 的 p{p:<3} = {v:.6g}  -> {v/scale:8.1f} 个量化级 "
                  f"(有效 {np.log2(max(v/scale,1)):.2f} bit)")
        # 相对量化误差：每个元素的 scale/2 相对自身幅值
        rel = (scale / 2) / np.maximum(np.abs(w), 1e-12)
        print(f"    逐元素相对量化误差 (s/2)/|w|:")
        print(f"      中位 {np.median(rel)*100:.2f}%   p90 {np.percentile(rel,90)*100:.2f}%"
              f"   均值 {rel.mean()*100:.2f}%")
        n_bad = int((rel > 0.10).sum())
        print(f"      相对误差 >10% 的元素: {n_bad}/{w.size} ({100.0*n_bad/w.size:.1f}%)")
        # 被大值撑开的程度
        print(f"    量程利用: max|w|={np.abs(w).max():.4g}，"
              f"而 p99={np.percentile(np.abs(w),99):.4g} "
              f"-> 量程被前 1% 的值撑大 {np.abs(w).max()/max(np.percentile(np.abs(w),99),1e-12):.1f} 倍")
        print()


if __name__ == "__main__":
    main()
