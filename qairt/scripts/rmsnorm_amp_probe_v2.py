"""#97 重跑（V2）—— 修正 V3 门失败的原因：运行时 gamma 是**量化后**的，不是 ONNX 里的 fp32 gamma。

方案：scripts/EXP_PLAN_RMSNORM_AMP.md（判据事前锁定，本次**未修改任何判据**）
本文件属于该方案 §五「V3 不过 ⇒ 我对 RmsNorm 的复现不对（如漏了 gamma、eps、归一化轴），修正后重跑」。

v1 的 V3 门 3/4 未过（1.31%/1.55%/1.45%，要求 <1%）。诊断过程：
  1. 查 ONNX：Pow 指数 = 2.0、eps(val_240) = 1e-05、ReduceMean axes=[-1] keepdims=1
     ⇒ 公式三个假设**全部正确**（约束 8：操作化先验证）
  2. 逐通道最小二乘拟合 y = (x/rms(x)) * g_eff ⇒ 残差 **0.0001%**
     ⇒ 结构精确，唯一偏差是 gamma 本身
  3. CPU 参考跑的是 transformer_part1b_quantized.dlc 且未加 --enable_cpu_fxp（#35）
     ⇒ 浮点执行**量化 DLC** ⇒ 运行时 gamma = 反量化后的量化 gamma
  4. DLC encoding 实查：这些 gamma 是 **uFxp_8**（8-bit per-tensor）

⚠️ 不用拟合值回填（那会让 V3 变成循环论证）。用**从 DLC 独立取出**的 encoding 做 Q/DQ。
"""
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_probe", "htp")
C = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_probe", "cpu", "out", "Result_0")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx", "transformer_part1b.onnx")
EPS = 1e-5

# 从 part1b_baseline_encodings.csv 实查（snpe-dlc-info -d -s csv），全部 uFxp_8 per-tensor
ENC = {
    "layers.7.ffn_norm2.weight":       dict(scale=0.019397212192, offset=-12.0, bw=8),
    "layers.8.ffn_norm2.weight":       dict(scale=0.023391544819, offset=-24.0, bw=8),
    "layers.8.attention_norm2.weight": dict(scale=0.011427695863, offset=-18.0, bw=8),
    "layers.8.attention.norm_q.weight":dict(scale=0.007904412225, offset=0.0,   bw=8),
}

CASES = [
    (14,  "linear_97_fc",  "mul_308",  "layers.7.ffn_norm2.weight",        (4128, 3840)),
    (101, "linear_105_fc", "mul_333",  "layers.8.ffn_norm2.weight",        (4128, 3840)),
    (84,  "linear_102_fc", "mul_326",  "layers.8.attention_norm2.weight",  (4128, 3840)),
    (35,  "linear_99_fc",  "mul_314",  "layers.8.attention.norm_q.weight", (4128, 30, 128)),
]


def qdq(x, scale, offset, bw):
    """SNPE uFxp 约定：dq = (q + offset) * scale，q in [0, 2^bw-1]
    自检：(0+offset)*scale == min 且 (2^bw-1+offset)*scale == max（见脚本头部实查值）"""
    n = (1 << bw) - 1
    q = np.clip(np.round(x / scale) - offset, 0, n)
    return (q + offset) * scale


def bulk(a, b):
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return (100.0 * np.linalg.norm(y - x) / np.linalg.norm(x),
            float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))


def rmsnorm(x, g, eps=EPS):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * g


def main():
    import onnx
    from onnx import numpy_helper
    m = onnx.load(ONNX, load_external_data=True)
    inits = {t.name: t for t in m.graph.initializer}

    rows = []
    for oid, fin, fout, gname, shape in CASES:
        pin_c, pin_h = os.path.join(C, fin + ".raw"), os.path.join(H, fin + ".raw")
        pou_c, pou_h = os.path.join(C, fout + ".raw"), os.path.join(H, fout + ".raw")
        g_fp = numpy_helper.to_array(inits[gname]).astype(np.float64).ravel()
        e = ENC[gname]
        g_q = qdq(g_fp, e["scale"], e["offset"], e["bw"])

        xc = np.fromfile(pin_c, np.float32).astype(np.float64).reshape(shape)
        xh = np.fromfile(pin_h, np.float32).astype(np.float64).reshape(shape)
        yc = np.fromfile(pou_c, np.float32).astype(np.float64).reshape(shape)
        yh = np.fromfile(pou_h, np.float32).astype(np.float64).reshape(shape)

        e_in,  c_in  = bulk(xc, xh)
        e_out, c_out = bulk(yc, yh)
        A_htp = e_out / max(e_in, 1e-12)

        # 互验：从 DLC 取的 g_q 应与「只用 CPU 臂做的逐通道最小二乘拟合」一致
        n_c = (xc / np.sqrt(np.mean(xc * xc, axis=-1, keepdims=True) + EPS)).reshape(-1, shape[-1])
        Y = yc.reshape(-1, shape[-1])
        g_fit = (n_c * Y).sum(0) / (n_c * n_c).sum(0)
        agree = 100.0 * np.linalg.norm(g_q - g_fit) / np.linalg.norm(g_fit)

        yc_sim, yh_sim = rmsnorm(xc, g_q), rmsnorm(xh, g_q)
        v3, _ = bulk(yc, yc_sim)                       # V3 门
        e_sim, c_sim = bulk(yc_sim, yh_sim)
        A_float = e_sim / max(e_in, 1e-12)

        v3_ok = v3 < 1.0
        v2_ok = min(c_in, c_out, c_sim) > 0.9
        if A_htp <= 1.3 * A_float:   verdict = "固有"
        elif A_htp >= 2.0 * A_float: verdict = "HTP"
        else:                        verdict = "部分"
        rows.append((oid, gname, e_in, e_out, A_htp, A_float, A_htp / A_float,
                     v3, v3_ok, v2_ok, agree, verdict))

        print("OpID %-4d %s" % (oid, gname))
        print("   [互验] DLC 量化 gamma vs CPU 臂最小二乘拟合: %.4f%%（应 <<1%%，否则两条独立路径不一致）" % agree)
        print("   [V3] 复现 vs 真实 CPU 输出 : fp32 gamma 版 %.4f%%  ->  量化 gamma 版 %.4f%%  %s"
              % (bulk(yc, rmsnorm(xc, g_fp))[0], v3, "PASS" if v3_ok else "FAIL"))
        print("   [V2] 余弦 输入 %.4f 输出 %.4f 模拟 %.4f  %s" % (c_in, c_out, c_sim, "PASS" if v2_ok else "FAIL"))
        print("   [V1] 输入误差 %.4f%%   输出误差 %.4f%%" % (e_in, e_out))
        print("   A_htp %.2f   A_float %.2f   A_htp/A_float = %.3f   => %s"
              % (A_htp, A_float, A_htp / A_float, verdict if (v3_ok and v2_ok) else "门未过"))
        print()

    print("=" * 92)
    print("%-6s %-34s %8s %8s %9s %7s %8s" % ("OpID", "gamma", "A_htp", "A_float", "比值", "V3", "判定"))
    print("-" * 92)
    for oid, gn, ei, eo, ah, af, r, v3, v3ok, v2ok, ag, vd in rows:
        print("%-6d %-34s %8.2f %8.2f %9.3f %6.3f%% %8s"
              % (oid, gn[:34], ah, af, r, v3, vd if (v3ok and v2ok) else "门未过"))


if __name__ == "__main__":
    main()
