"""EXP_PLAN_FXP_SIM：自建 A16W8 定点模拟器，判别 HTP 做的是「正确的定点」还是「错的定点」。

前提已验证（fxp_sim_premise.py，P1~P4 全过）：
  · 权重按名对应 ONNX（val_1791 等在 ONNX 里同名）
  · 语义是 y = x @ W（W 为 (K,N)），方阵方向已实测确认
  · 权重 per-tensor 量化，zero_point = -offset
  · 用【反量化权重】复算 vs CPU 参考：0.0001% ~ 0.65%

精度说明：整数累加用 float64 承载。最大乘积 |q_x-z|·|q_w-z| <= 65535*255 ≈ 1.7e7，
K<=10240 ⇒ 累加上界 ≈ 1.7e11 << 2^53 ≈ 9e15，**float64 可精确表示所有整数**，无舍入。
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, r"D:\LocalDreamZImage\scripts")
import map_opid as M

ONNX = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1b.onnx"
DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"
XIN = r"D:\ZImage_Work\p0_experiments\fc_inputs\cpu\out\Result_0"
YCPU = r"D:\ZImage_Work\p0_experiments\fc_probe\cpu\out\Result_0"
YHTP = r"D:\ZImage_Work\p0_experiments\fc_probe\htp"

CASES = [
    (12, "linear_97_fc", "node_linear_97_pre_reshape", "val_1791", (4128, 10240), (4128, 3840)),
    (29, "linear_99_fc", "node_linear_99_pre_reshape", "val_1800", (4128, 3840), (4128, 3840)),
    (31, "linear_100_fc", "node_linear_100_pre_reshape", "val_1801", (4128, 3840), (4128, 3840)),
    (82, "linear_102_fc", "node_linear_102_pre_reshape", "val_1896", (4128, 3840), (4128, 3840)),
    (99, "linear_105_fc", "node_linear_105_pre_reshape", "val_1908", (4128, 10240), (4128, 3840)),
]
ROWS = 512          # 只用前 512 行做模拟，够统计且省内存


def bulk(a, b):
    """主体口径（|a|<=p99）相对 L2 与余弦（约束 7）"""
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    rel = 100.0 * np.linalg.norm(y - x) / np.linalg.norm(x)
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    return rel, cos


def main():
    ops = M.parse(DUMP)
    m = onnx.load(ONNX, load_external_data=True)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    print(f"{'FC':<16}{'E_sim':>9}{'E_htp':>9}{'D(sim,htp)':>12}"
          f"{'cos(sim,fp)':>12}{'cos(htp,fp)':>12}{'cos(sim,htp)':>13}")
    print("=" * 84)
    summary = []
    for oid, yname, xname, wdlc, xsh, ysh in CASES:
        op = ops[oid]
        ew, ey = op["enc"][wdlc], op["enc"][yname]
        K, N = xsh[1], ysh[1]

        # 输入激活的 encoding：取上游（Reshape 透明）
        ex = None
        for o2 in ops.values():
            for t in o2["out"]:
                if t["name"] == xname and xname in o2["enc"]:
                    ex = o2["enc"][xname]
        if ex is None:   # Reshape 无 encoding -> 用其上游
            prod = {t["name"]: o2 for o2 in ops.values() for t in o2["out"]}
            cur = xname
            for _ in range(6):
                o2 = prod.get(cur)
                if o2 is None: break
                if cur in o2["enc"]: ex = o2["enc"][cur]; break
                ins = [i["name"] for i in o2["in"] if i["ttype"] != "STATIC"]
                if not ins: break
                cur = ins[0]
        if ex is None:
            print(f"{yname:<16} 找不到输入 encoding，跳过"); continue

        w = inits[wdlc].astype(np.float64)
        if w.shape != (K, N):
            w = w.T
        x = np.fromfile(f"{XIN}\\{xname}.raw", np.float32).reshape(xsh)[:ROWS].astype(np.float64)
        ycpu = np.fromfile(f"{YCPU}\\{yname}.raw", np.float32).reshape(ysh)[:ROWS].astype(np.float64)
        yhtp = np.fromfile(f"{YHTP}\\{yname}.raw", np.float32).reshape(ysh)[:ROWS].astype(np.float64)

        sx, zx = ex["scale"], -ex["offset"]
        sw, zw = ew["scale"], -ew["offset"]
        sy, zy = ey["scale"], -ey["offset"]
        qmax_x = 2 ** ex["bw"] - 1
        qmax_w = 2 ** ew["bw"] - 1

        qx = np.clip(np.round(x / sx) + zx, 0, qmax_x)
        qw = np.clip(np.round(w / sw) + zw, 0, qmax_w)
        cx, cw = qx - zx, qw - zw            # 去零点后的整数

        acc = cx @ cw                        # 精确整数累加（float64 承载）
        ysim = acc * (sx * sw)
        # 输出 requant（HTP 会把结果量化到 out encoding）
        qy = np.clip(np.round(ysim / sy) + zy, 0, 2 ** ey["bw"] - 1)
        ysim_rq = (qy - zy) * sy

        e_sim, c_sim = bulk(ycpu, ysim_rq)
        e_htp, c_htp = bulk(ycpu, yhtp)
        d, c_sh = bulk(yhtp, ysim_rq)
        summary.append((yname, e_sim, e_htp, d, c_sim, c_htp, c_sh, acc, cx, cw, zw, K))
        print(f"{yname:<16}{e_sim:>9.3f}{e_htp:>9.3f}{d:>12.3f}"
              f"{c_sim:>12.5f}{c_htp:>12.5f}{c_sh:>13.5f}")

    print()
    print("=" * 84)
    print("判据 2：offset 项量级（审查者 §3.8 机制假说）")
    print("  acc = Σq_x·q_w − z_w·Σq_x − z_x·Σq_w + K·z_x·z_w")
    print("=" * 84)
    print(f"{'FC':<16}{'|主项|中位':>14}{'|z_w·Σq_x|中位':>16}{'比值':>10}{'z_w':>7}")
    print("-" * 68)
    for (yname, *_ , acc, cx, cw, zw, K) in summary:
        qx = cx + 0            # cx = qx - zx
        main_t = np.abs((cx @ (cw + zw)))          # Σ q_x·q_w 的等价（cw+zw = qw）
        off_t = np.abs(zw * cx.sum(axis=1, keepdims=True))
        print(f"{yname:<16}{np.median(main_t):>14.4g}{np.median(off_t):>16.4g}"
              f"{np.median(off_t)/max(np.median(main_t),1e-9):>10.3f}{zw:>7.0f}")

    print()
    print("=" * 84)
    print("判据 3：累加器实际范围 vs int32 边界")
    print("=" * 84)
    print(f"{'FC':<16}{'|acc| max':>14}{'int32 上限':>14}{'占比':>10}{'K':>8}")
    print("-" * 62)
    for (yname, *_ , acc, cx, cw, zw, K) in summary:
        mx = np.abs(acc).max()
        print(f"{yname:<16}{mx:>14.4g}{2**31:>14.4g}{mx/2**31:>10.4f}{K:>8}")

    print()
    print("=" * 84)
    print("判据 1（主判据，事前定稿）")
    print("=" * 84)
    for (yname, e_sim, e_htp, d, *_rest) in summary:
        if e_htp < 1e-9: continue
        if d < 0.3 * e_htp:
            v = "H-A：模拟重现了 HTP ⇒ HTP 做的是正确的定点"
        elif e_sim < 0.3 * e_htp:
            v = "H-B：正确定点误差远小于 HTP ⇒ HTP 特有缺陷"
        else:
            v = "两者皆不显著，按比例报告"
        print(f"  {yname:<16} E_sim={e_sim:7.3f}%  E_htp={e_htp:7.3f}%  D={d:7.3f}%  -> {v}")


if __name__ == "__main__":
    main()
