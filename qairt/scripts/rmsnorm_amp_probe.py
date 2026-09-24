"""#97：RmsNorm 的误差放大是 HTP 的问题，还是算子固有的误差传播？

方案：scripts/EXP_PLAN_RMSNORM_AMP.md（判据事前锁定）
全程宿主，零设备占用——用的全是已在盘上的 CPU/HTP 配对产物。

🔴 回查记录（约束 6）：#15「RmsNorm 是误差放大点」曾被**撤回**，理由是
   *"那些张量主体余弦仅 0.11~0.48，在废墟里比大小无意义"*。
   本次余弦 0.9285~0.9996，**远超 0.3 阈值** ⇒ 撤回理由不适用。
   #12（gamma 量化质量）与 #14（输出 encoding 余量）已被排除 ⇒ 旧机制解释不了。
"""
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
H = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_probe", "htp")
C = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_probe", "cpu", "out", "Result_0")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx", "transformer_part1b.onnx")
EPS = 1e-5          # 图内实测：Pow 2.0 -> ReduceMean -> Add 1e-05 -> Sqrt

# (OpID, 输入张量文件, 输出张量文件, gamma 名, 归一化后的形状)
CASES = [
    (14,  "linear_97_fc",  "mul_308",  "layers.7.ffn_norm2.weight",        (4128, 3840)),
    (101, "linear_105_fc", "mul_333",  "layers.8.ffn_norm2.weight",        (4128, 3840)),
    (84,  "linear_102_fc", "mul_326",  "layers.8.attention_norm2.weight",  (4128, 3840)),
    (35,  "linear_99_fc",  "mul_314",  "layers.8.attention.norm_q.weight", (4128, 30, 128)),
]


def bulk(a, b):
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return (100.0 * np.linalg.norm(y - x) / np.linalg.norm(x),
            float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))


def rmsnorm(x, g, eps=EPS, plus_one=False):
    """精确 float64 RmsNorm，沿最后一维。
    plus_one: DiT 常用 x/rms * (1+gamma)（图头部见过 Add 常量 1.0 的模式）"""
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / r) * ((1.0 + g) if plus_one else g)


def main():
    import onnx
    from onnx import numpy_helper
    m = onnx.load(ONNX, load_external_data=True)
    inits = {t.name: t for t in m.graph.initializer}

    print("%-5s %-34s %10s %10s %10s %10s %8s" %
          ("OpID", "gamma", "输入误差", "输出误差", "A_htp", "A_float", "判定"))
    print("-" * 96)
    for oid, fin, fout, gname, shape in CASES:
        pin_c, pin_h = os.path.join(C, fin + ".raw"), os.path.join(H, fin + ".raw")
        pou_c, pou_h = os.path.join(C, fout + ".raw"), os.path.join(H, fout + ".raw")
        if not all(os.path.exists(p) for p in (pin_c, pin_h, pou_c, pou_h)):
            print("  OpID %-4d 缺产物，跳过" % oid); continue
        if gname not in inits:
            print("  OpID %-4d ONNX 里找不到 %s，跳过" % (oid, gname)); continue
        g = numpy_helper.to_array(inits[gname]).astype(np.float64).ravel()

        xc = np.fromfile(pin_c, np.float32).astype(np.float64).reshape(shape)
        xh = np.fromfile(pin_h, np.float32).astype(np.float64).reshape(shape)
        yc = np.fromfile(pou_c, np.float32).astype(np.float64).reshape(shape)
        yh = np.fromfile(pou_h, np.float32).astype(np.float64).reshape(shape)
        assert g.size == shape[-1], "gamma 尺寸 %d != 最后一维 %d" % (g.size, shape[-1])

        e_in, c_in = bulk(xc, xh)
        e_out, c_out = bulk(yc, yh)
        A_htp = e_out / max(e_in, 1e-12)

        # V3 门：我复现的 RmsNorm 必须与真实 CPU 输出吻合。
        # 两种 gamma 约定各试一次，取更贴近的那种（并报出来，避免我凭猜选）
        cands = {}
        for po in (False, True):
            cands[po] = bulk(yc, rmsnorm(xc, g, plus_one=po))[0]
        po = min(cands, key=cands.get)
        print("        [gamma 约定] x*gamma=%.4f%%  x*(1+gamma)=%.4f%%  => 采用 %s"
              % (cands[False], cands[True], "x*(1+gamma)" if po else "x*gamma"))
        yc_sim = rmsnorm(xc, g, plus_one=po)
        v3, v3c = bulk(yc, yc_sim)

        yh_sim = rmsnorm(xh, g, plus_one=po)
        e_sim, c_sim = bulk(yc_sim, yh_sim)
        A_float = e_sim / max(e_in, 1e-12)

        if A_htp <= 1.3 * A_float:
            v = "✅固有"
        elif A_htp >= 2.0 * A_float:
            v = "🔴HTP"
        else:
            v = "🔶部分"
        ok23 = (v3 < 1.0) and min(c_in, c_out, c_sim) > 0.9
        print("%-5d %-34s %9.4f%% %9.4f%% %10.2f %10.2f %8s" %
              (oid, gname[:33], e_in, e_out, A_htp, A_float, v if ok23 else "门未过"))
        print("        [V3] 复现 RmsNorm vs 真实 CPU 输出: %.4f%%（要求<1%%）  %s" %
              (v3, "✅" if v3 < 1.0 else "❌ 复现不对，本行作废"))
        print("        [V2] 余弦 输入%.4f 输出%.4f 模拟%.4f（要求>0.9）  %s" %
              (c_in, c_out, c_sim, "✅" if min(c_in, c_out, c_sim) > 0.9 else "❌"))
        print("        固有传播下输出误差 = %.4f%%   实测 HTP = %.4f%%   超出 %.2f 倍" %
              (e_sim, e_out, e_out / max(e_sim, 1e-12)))
        print()


if __name__ == "__main__":
    main()
