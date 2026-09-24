"""【决定性实验】CPU 定点执行 vs CPU 浮点执行 vs HTP —— 三方对照。

发现（2026-08-15，采纳外部审查）：`snpe-net-run` 有 `--enable_cpu_fxp`
（"Enable the fixed point execution on CPU runtime"），而本项目**从未使用**。
⇒ 此前全部"CPU 参考"都是**浮点执行**（反量化后 float 计算），不是定点。
⇒ "HTP 比正确的 16-bit 实现丢 4 bit"这一框架失去前提。

本实验回答：**真定点执行本身就会产生大误差，还是只有 HTP 会？**

  若 CPU-fxp 也大幅偏离 FP32  -> 模型对【真定点】不友好，HTP 不算"坏掉"
                                 ⇒ 方向转向"让模型适配定点"（通道重缩放等）
  若 CPU-fxp 接近 FP32        -> 定点本身没问题，是 HTP 特有 ⇒ 继续查 HTP 链路

判据事前定稿，见下。
"""
import os
import subprocess
import sys

import numpy as np

SDK = r"D:\qairt\2.48.0.260626"
SNPE = f"{SDK}\\bin\\x86_64-windows-msvc\\snpe-net-run.exe"
QNN_LIB = f"{SDK}\\lib\\x86_64-windows-msvc"
QNN_BIN = f"{SDK}\\bin\\x86_64-windows-msvc"
DLC = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\transformer_part1b\transformer_part1b_quantized.dlc"
TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
WORK = r"D:\ZImage_Work\p0_experiments\cpu_fxp"
FP32 = r"D:\ZImage_Work\p0_experiments\vs_fp32\unified_fp32.raw"
HTP = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu\A_part1b\unified.raw"
CPU_FLOAT = f"{TB}\\out\\Result_0\\unified.raw"

INS = ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]
C = 3840


def run_fxp():
    os.makedirs(WORK, exist_ok=True)
    lst = os.path.join(WORK, "input_list.txt")
    with open(lst, "w") as f:
        f.write("%unified\n")
        f.write(" ".join(f"{n}:={TB}\\{n}.raw" for n in INS) + "\n")
    env = dict(os.environ, PATH=f"{QNN_LIB};{QNN_BIN};" + os.environ["PATH"])
    cmd = [SNPE, "--container", DLC, "--input_list", lst,
           "--output_dir", os.path.join(WORK, "out"), "--enable_cpu_fxp"]
    print("执行：snpe-net-run ... --enable_cpu_fxp", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"snpe-net-run 失败 rc={r.returncode}")
    p = os.path.join(WORK, "out", "Result_0", "unified.raw")
    a = np.fromfile(p, np.float32)
    exp = 15851520
    assert a.size == exp, f"判据0 FAIL: 输出 {a.size} != {exp}"
    print(f"判据 0 通过：输出 {a.size} 元素")
    return a


def main():
    out = os.path.join(WORK, "out", "Result_0", "unified.raw")
    fxp = np.fromfile(out, np.float32) if os.path.exists(out) else run_fxp()
    fxp = fxp.astype(np.float64)
    fp = np.fromfile(FP32, np.float32).astype(np.float64)
    cpu = np.fromfile(CPU_FLOAT, np.float32).astype(np.float64)
    htp = np.fromfile(HTP, np.float32).astype(np.float64)

    same = np.array_equal(fxp.astype(np.float32), cpu.astype(np.float32))
    print(f"判据 0b：CPU-fxp 与 CPU-float 是否逐位相同？{'是（开关未生效！）' if same else '否（开关已生效）'}")
    if same:
        print("  => 开关未改变结果，本次无效，需查 --enable_cpu_fxp 是否被接受")
        return

    F = fp.reshape(-1, C)
    e = (F ** 2).sum(axis=0)
    hot = int(np.argmax(e))
    idx_hot = np.tile(np.eye(C, dtype=bool)[hot], fp.size // C)
    idx_bulk = ~idx_hot

    def rel(a, b, m=slice(None)):
        return 100.0 * np.linalg.norm((b - a)[m]) / np.linalg.norm(a[m])

    def cos(a, b, m):
        x, y = a[m], b[m]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    print()
    print("=" * 78)
    print("三方对照（基准 = FP32 真值）")
    print("=" * 78)
    print(f"{'执行方式':<28}{'全张量%':>10}{f'ch{hot}%':>10}{'non-ch85%':>12}{'non-ch85余弦':>13}")
    print("-" * 78)
    for tag, v in [("CPU 浮点（此前的'参考'）", cpu),
                   ("**CPU 真定点 (--enable_cpu_fxp)**", fxp),
                   ("设备 HTP", htp)]:
        print(f"{tag:<28}{rel(fp, v):>10.3f}{rel(fp, v, idx_hot):>10.3f}"
              f"{rel(fp, v, idx_bulk):>12.3f}{cos(fp, v, idx_bulk):>13.4f}")

    print()
    print(f"{'CPU定点 vs HTP':<28}{rel(fxp, htp):>10.3f}{'':>10}"
          f"{rel(fxp, htp, idx_bulk):>12.3f}{cos(fxp, htp, idx_bulk):>13.4f}")

    print()
    print("=" * 78)
    print("判据 1（事前定稿）")
    print("=" * 78)
    r_fxp = rel(fp, fxp, idx_bulk)
    r_htp = rel(fp, htp, idx_bulk)
    r_flt = rel(fp, cpu, idx_bulk)
    print(f"  non-ch85 子集：CPU浮点 {r_flt:.2f}%  |  CPU定点 {r_fxp:.2f}%  |  HTP {r_htp:.2f}%")
    print()
    if r_fxp > 0.5 * r_htp:
        print("  => 【真定点本身就大幅偏离】CPU 定点误差与 HTP 同量级。")
        print("     ⇒ **模型对真定点执行不友好，HTP 不算'坏掉'**。")
        print("     ⇒ 方向应转为『让模型适配定点』（通道重缩放 / 混合精度），")
        print("        而非继续在 HTP 链路里找 bug。")
    elif r_fxp < 2 * r_flt:
        print("  => 【真定点没问题】CPU 定点接近 CPU 浮点。")
        print("     ⇒ 定点算术本身不是原因，**问题是 HTP 特有的** ⇒ 继续查 HTP 链路。")
    else:
        print("  => 介于两者之间：定点有贡献但不足以解释 HTP，按比例报告。")


if __name__ == "__main__":
    main()
