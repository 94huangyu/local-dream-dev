"""#30 探针判读：HTP 内部有没有与 K 无关的硬性精度地板？

方案 `scripts/EXP_PLAN_MATMUL_PROBE.md` 第 2/3 节，**判据事前定稿，不得改**：

  y_exact —— float64 精确值（真值）
  y_fxp   —— **正确的定点实现**：int 累加、末端一次 requant，用与 HTP 完全相同的 encoding
  y_htp   —— 设备实测

  E_htp / E_fxp 三个 K 上的曲线 = H-floor 与 H-acc 的区分标志
    · 三个 K 都接近 1（<1.5）        => 两假设均否定，单算子上 HTP 是对的 => 转 #33/#34
    · 随 K 显著增长（K=3840 >5、K=64 <2） => H-acc（累加/分块）
    · 所有 K 都大且大致相同（>5）    => H-floor => 量化配置这条路到头，转 #32
    · 等效位宽稳定落在 11~13 bit     => 强支持 H-floor（呼应 15.13.2 的 11.34 bit）

  有效性自检：E_fxp 必须显著小于 E_htp 才有判别力；E_fxp 本身 >5% ⇒ 量程设计不合理，本次无效。

定点语义沿用 `fxp_sim.py` 已验证的前提（P1~P4）：y = x @ W、zero_point = -offset、
整数累加用 float64 承载（|积| <= 65535*255，K<=3840 ⇒ 上界 6.4e10 << 2^53，无舍入）。
"""
import os
import re
import subprocess

import numpy as np

ROOT = r"D:\ZImage_Work\p0_experiments\matmul_probe"
SDK = r"D:\qairt\2.48.0.260626"
TOOL = r"D:\LocalDreamZImage\scripts\qairt_tool.py"
KS = [64, 1024, 3840]
N_TEST = 32

ENC = re.compile(r"(\w+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                 r"scale ([\d.eE+-]+), offset ([-\d.]+)")


def encodings(k):
    """从量化后的 DLC 读 x / W / y 的真实 encoding（不从 ONNX 或印象推导）。"""
    d = os.path.join(ROOT, f"k{k}")
    env = dict(os.environ, PYTHONPATH=os.path.join(SDK, "lib", "python"))
    out = subprocess.run(["python", TOOL, "qairt-dlc-info", "--input_dlc",
                          os.path.join(d, f"probe_k{k}_quant.dlc")],
                         capture_output=True, text=True, env=env).stdout
    e = {}
    for m in ENC.finditer(out):
        name, bw, lo, hi, sc, off = m.groups()
        if name not in e:
            e[name] = dict(bw=int(bw), min=float(lo), max=float(hi),
                           scale=float(sc), offset=float(off))
    for t in ("x", "W", "y"):
        assert t in e, f"K={k} 没解析到 {t} 的 encoding（拿到 {list(e)}）"
    return e


def fxp(x, w, e):
    """正确的定点实现：量化 -> 整数累加 -> 末端一次 requant。"""
    sx, zx, bx = e["x"]["scale"], -e["x"]["offset"], e["x"]["bw"]
    sw, zw, bw_ = e["W"]["scale"], -e["W"]["offset"], e["W"]["bw"]
    sy, zy, by = e["y"]["scale"], -e["y"]["offset"], e["y"]["bw"]
    qx = np.clip(np.round(x / sx) + zx, 0, 2 ** bx - 1)
    qw = np.clip(np.round(w / sw) + zw, 0, 2 ** bw_ - 1)
    acc = (qx - zx) @ (qw - zw)
    y = acc * (sx * sw)
    qy = np.clip(np.round(y / sy) + zy, 0, 2 ** by - 1)
    return (qy - zy) * sy


def rel(a, b):
    return 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)


def bits(err_std, rng):
    """等效位宽，与 15.13.2 同一算法：均匀舍入噪声 std = step/sqrt(12)。"""
    return np.log2(rng / (err_std * np.sqrt(12))) if err_std > 0 else float("inf")


def main():
    rows = []
    for k in KS:
        d = os.path.join(ROOT, f"k{k}")
        e = encodings(k)
        w = np.load(os.path.join(d, "W.npy")).astype(np.float64)
        x = np.stack([np.fromfile(os.path.join(d, "test", f"x_{i:04d}.raw"), np.float32)
                      for i in range(N_TEST)]).astype(np.float64)
        y_exact = x @ w
        y_fxp = fxp(x, w, e)

        hd = os.path.join(d, "htp")
        if not os.path.isdir(hd):
            print(f"K={k}: 缺设备输出（{hd}）——先跑 matmul_probe_run.py")
            continue
        y_htp = np.stack([np.fromfile(os.path.join(hd, f"y_{i:04d}.raw"), np.float32)
                          for i in range(N_TEST)]).astype(np.float64)

        e_fxp, e_htp = rel(y_exact, y_fxp), rel(y_exact, y_htp)
        rng = e["y"]["max"] - e["y"]["min"]
        rows.append(dict(k=k, e_fxp=e_fxp, e_htp=e_htp, ratio=e_htp / max(e_fxp, 1e-12),
                         b_fxp=bits((y_fxp - y_exact).std(), rng),
                         b_htp=bits((y_htp - y_exact).std(), rng),
                         sy=e["y"]["scale"], rng=rng,
                         steps=(y_htp - y_exact).std() / e["y"]["scale"]))

    if not rows:
        return
    print(f"{'K':>6}{'E_fxp':>10}{'E_htp':>10}{'E_htp/E_fxp':>13}"
          f"{'bit_fxp':>10}{'bit_htp':>10}{'HTP误差/量化步':>16}")
    print("-" * 75)
    for r in rows:
        print(f"{r['k']:>6}{r['e_fxp']:>9.4f}%{r['e_htp']:>9.4f}%{r['ratio']:>13.2f}"
              f"{r['b_fxp']:>10.2f}{r['b_htp']:>10.2f}{r['steps']:>16.2f}")

    print("\n" + "=" * 75)
    print("有效性自检（方案第 3 节）")
    bad = [r for r in rows if r["e_fxp"] > 5.0]
    if bad:
        print(f"  ❌ E_fxp > 5% 于 K={[r['k'] for r in bad]} ⇒ 量程设计不合理，"
              f"**本次无效，需重设计**，不得据此判读")
        return
    print(f"  ✅ E_fxp 全部 <= 5%（最大 {max(r['e_fxp'] for r in rows):.4f}%）⇒ 有判别力")

    print("\n" + "=" * 75)
    print("判据（事前定稿）")
    ratios = [r["ratio"] for r in rows]
    r64 = next(r["ratio"] for r in rows if r["k"] == 64)
    r3840 = next(r["ratio"] for r in rows if r["k"] == 3840)
    if all(x < 1.5 for x in ratios):
        v = ("H-floor 与 H-acc **均被否定**：HTP 在单算子上是正确的\n"
             "  ⇒ 问题出在【多算子图】层面（融合 / 中间张量 requant）⇒ 转 #33 / #34")
    elif r3840 > 5 and r64 < 2:
        v = ("**H-acc 成立**：误差随 K 显著增长 ⇒ 累加/分块策略是元凶\n"
             "  ⇒ 查 --vtcm_mb（#34）、算子融合（#33）")
    elif all(x > 5 for x in ratios):
        v = ("**H-floor 成立**：存在与 K 无关的固定精度地板\n"
             "  ⇒ **量化配置这条路到头**，转 #32（SDK 2.28 vs 2.48 A/B）")
    else:
        v = ("落在四条分支之外（比值 "
             + ", ".join(f"K={r['k']}:{r['ratio']:.2f}" for r in rows)
             + "）⇒ 按比例如实报告，**不得**硬套某一分支")
    print("  " + v)
    bh = [r["b_htp"] for r in rows]
    if all(11 <= b <= 13 for b in bh):
        print(f"  等效位宽全部落在 11~13 bit（{[f'{b:.2f}' for b in bh]}）"
              f"⇒ 与 15.13.2 的 11.34 bit 吻合，**强支持 H-floor**")
    else:
        print(f"  等效位宽 {[f'{b:.2f}' for b in bh]}（判据要求 11~13 才算强支持 H-floor）")

    print("\n⚠️ 适用范围（不得外推）：本探针的输入是**良态**的"
          "（N(0,σ)，无 massive activation 离群点）。\n"
          "   若结论是『HTP 正常』，**不能**推出真实模型里也正常"
          "——真实模型 `unified` 的 99.83% 能量集中在前 1%。")


if __name__ == "__main__":
    main()
