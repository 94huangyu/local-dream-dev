"""EXP_PLAN_P2_ATTR §四 并行条款：mul_23 的放大是**固有**还是 **HTP 缺陷**？

装置照抄 §15.30（#97 的 v2）：float64 精确复现 RmsNorm + **DLC 里的量化 gamma**。
🔴 必须用量化 gamma —— #97 第一版用 ONNX 的 fp32 gamma，V3 门 3/4 未过。

判据（方案 §四，事前锁定）：A_htp / A_float <= 1.3 => 固有；>= 2.0 => HTP 缺陷。
V 门：V3 复现门（float64 复现 vs 真值，相对差 < 1%）；V2 余弦 > 0.9。
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
FP32, HTP = os.path.join(R, "fp32"), os.path.join(R, "htp")
EPS = 1e-5                                    # ONNX val_11 实查
GAMMA = "layers.15.ffn_norm2.weight"
G_ENC = dict(bw=8, scale=0.041475184262, offset=-9.0)      # part2a DLC 实取
X_ENC = dict(bw=16, scale=1.92004442215, offset=-25696.0)  # linear_7 的激活 encoding
SHAPE = (4128, 3840)


def qdq(x, e):
    n = (1 << e["bw"]) - 1
    return (np.clip(np.round(x / e["scale"]) - e["offset"], 0, n) + e["offset"]) * e["scale"]


def bulk(a, b):
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return (100.0 * np.linalg.norm(y - x) / np.linalg.norm(x),
            float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))


def rmsnorm(x, g):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)) * g


def main():
    import onnx
    from onnx import numpy_helper
    m = onnx.load(os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx",
                               "transformer_part2a_fixed.onnx"), load_external_data=True)
    g_fp = numpy_helper.to_array({t.name: t for t in m.graph.initializer}[GAMMA]).astype(np.float64)
    g_q = qdq(g_fp, G_ENC)
    print("gamma: fp32 min/max %.5f/%.5f   量化后与 fp32 的相对差 %.4f%%"
          % (g_fp.min(), g_fp.max(), 100 * np.linalg.norm(g_q - g_fp) / np.linalg.norm(g_fp)))

    xf = np.fromfile(os.path.join(FP32, "linear_7.raw"), np.float32).astype(np.float64).reshape(SHAPE)
    yf = np.fromfile(os.path.join(FP32, "mul_23.raw"), np.float32).astype(np.float64).reshape(SHAPE)

    # ---- 不需要设备的部分：linear_7 自身量化代价，以及它经 RmsNorm 后的放大 ----
    xq = qdq(xf, X_ENC)
    e_in_q, c_in_q = bulk(xf, xq)
    y_from_xq = rmsnorm(xq, g_q)
    e_out_q, c_out_q = bulk(yf, y_from_xq)
    print("\n[纯量化代价，无需设备]")
    print("  linear_7 自身 uFxp_16 表示误差（主体）: %.4f%%   余弦 %.6f" % (e_in_q, c_in_q))
    print("  经 RmsNorm(量化 gamma) 后 vs FP32 真值 : %.4f%%   余弦 %.6f" % (e_out_q, c_out_q))
    print("  ⇒ 仅「linear_7 量化 + gamma 量化」就放大 %.2f 倍" % (e_out_q / max(e_in_q, 1e-12)))

    # ---- V3 复现门 ----
    # 🔴 参考系决定用哪个 gamma：本次真值来自 **FP32 ONNX**（fp32 gamma）；
    #    #97 那次参考是**量化 DLC 的 CPU 执行**（量化 gamma）。用错就复现不出来。
    v3, _ = bulk(yf, rmsnorm(xf, g_fp))
    v3q, _ = bulk(yf, rmsnorm(xf, g_q))
    print("\n[V3 复现门] float64 复现(fp32 gamma) vs FP32 真值 mul_23: %.4f%%  %s"

          % (v3, "PASS" if v3 < 1.0 else "FAIL（复现不对，本次作废）"))
    print("            对照：用量化 gamma 复现 = %.4f%% —— 差值即 gamma 量化本身的代价" % v3q)

    # ---- 需要设备的部分 ----
    ph = os.path.join(HTP, "linear_7_fc.raw")
    if not os.path.exists(ph):
        ph = os.path.join(HTP, "linear_7.raw")
    if not os.path.exists(ph):
        print("\n[等设备] 缺 linear_7 的 HTP 值，A_htp/A_float 无法计算"); return
    xh = np.fromfile(ph, np.float32).astype(np.float64).reshape(SHAPE)
    yh = np.fromfile(os.path.join(HTP, "mul_23.raw"), np.float32).astype(np.float64).reshape(SHAPE)
    e_in, c_in = bulk(xf, xh)
    e_out, c_out = bulk(yf, yh)
    A_htp = e_out / max(e_in, 1e-12)
    # 🔴🔴 2026-08-22 数据到达**之前**发现并修正的口径错误（明写，不得事后粉饰）：
    #   原写法 A_float = relerr(rmsnorm(xf,g_q), rmsnorm(xh,g_q)) —— 两臂都用量化 gamma
    #   ⇒ gamma 量化代价在 A_float 里被约掉，却仍留在 A_htp 的分子里（真值用 fp32 gamma）
    #   ⇒ 比值被系统性抬高，**偏向"HTP 缺陷"的结论**。
    #   修正：A_float 的分子与 A_htp 的分子**用同一个真值 yf**，模拟臂用 HTP 实际使用的量化 gamma。
    y_sim_h = rmsnorm(xh, g_q)          # 精确 float64 + HTP 自己的输入 + HTP 实际用的量化 gamma
    e_sim, c_sim = bulk(yf, y_sim_h)
    A_float = e_sim / max(e_in, 1e-12)
    # 直接判据（更强，不依赖比值）：HTP 输出 vs「同输入同 gamma 的精确计算」
    e_excess, c_excess = bulk(y_sim_h, yh)
    print("\n[主判据]")
    print("  输入 linear_7 : 误差 %.4f%%  余弦 %.6f" % (e_in, c_in))
    print("  输出 mul_23   : 误差 %.4f%%  余弦 %.6f" % (e_out, c_out))
    print("  A_htp   = %.3f" % A_htp)
    print("  A_float = %.3f   （同一输入误差在精确 float64 下传播出的输出误差 %.4f%%）" % (A_float, e_sim))
    r = A_htp / max(A_float, 1e-12)
    print("  A_htp / A_float = %.3f" % r)
    print()
    print("  🔴 [直接判据·更强] HTP 输出 vs「同输入、同量化 gamma 的精确 float64 计算」")
    print("      超出误差 = %.4f%%   余弦 %.6f" % (e_excess, c_excess))
    print("      ⇒ 该值小 ⇒ HTP 的算术本身是对的，误差全部来自上游 + 表示；"
          "该值大 ⇒ HTP 这一步算错了")
    v2 = min(c_in, c_out, c_sim) > 0.9
    print("  [V2 余弦门] 输入 %.4f 输出 %.4f 模拟 %.4f  %s" % (c_in, c_out, c_sim, "PASS" if v2 else "FAIL"))
    if not (v2 and v3 < 1.0):
        print("\n  ❌ V 门未过，不落判据"); return
    v = ("✅ 固有（非 HTP 缺陷）⇒ #98 成立，收口" if r <= 1.3 else
         "🔴 HTP 缺陷 ⇒ 新目标" if r >= 2.0 else "🔶 部分，记录比例")
    print("\n  判定：%s" % v)


if __name__ == "__main__":
    main()
