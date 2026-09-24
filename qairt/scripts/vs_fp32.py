"""EXP_PLAN_VS_FP32：HTP 是"算错了"还是"另一种合法舍入"？

用 FP32 ONNX 当真值，看 SNPE CPU 参考与设备 HTP 各自离它多远。
输入与"实验A 隔离 part1b"完全相同（testB s0 的 CPU 参考上游输出）。
"""
import os

import numpy as np
import onnxruntime as ort

EVID = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
HV = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu"
OUT = r"D:\ZImage_Work\p0_experiments\vs_fp32"

SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840),
          "add_138": (1, 4128, 3840), "select_45": (1, 4128, 1, 64),
          "select_46": (1, 4128, 1, 64), "tanh_19": (1, 1, 3840)}


def main():
    os.makedirs(OUT, exist_ok=True)
    fp32_path = f"{OUT}\\unified_fp32.raw"

    if os.path.exists(fp32_path):
        print(f"复用已有 FP32 结果: {fp32_path}")
        u_fp32 = np.fromfile(fp32_path, np.float32)
    else:
        feeds = {}
        for n, s in SHAPES.items():
            feeds[n] = np.fromfile(f"{TB}\\{n}.raw", np.float32).reshape(s)
            print(f"  输入 {n:<14} {feeds[n].shape}")
        print("跑 FP32 part1b (onnxruntime CPU)...", flush=True)
        sess = ort.InferenceSession(f"{EVID}\\transformer_part1b.onnx",
                                    providers=["CPUExecutionProvider"])
        outs = [o.name for o in sess.get_outputs()]
        print(f"  ONNX 输出: {outs}")
        r = sess.run(["unified"], feeds)[0]
        u_fp32 = np.ascontiguousarray(r.reshape(-1).astype(np.float32))
        u_fp32.tofile(fp32_path)
        del sess
        print(f"  已保存 {fp32_path}")

    u_cpu = np.fromfile(f"{TB}\\out\\Result_0\\unified.raw", np.float32)
    u_htp = np.fromfile(f"{HV}\\A_part1b\\unified.raw", np.float32)
    assert u_fp32.size == u_cpu.size == u_htp.size, \
        f"元素数不一致 {u_fp32.size} {u_cpu.size} {u_htp.size}"

    f = u_fp32.astype(np.float64)
    c = u_cpu.astype(np.float64)
    h = u_htp.astype(np.float64)
    nf = np.linalg.norm(f)

    e_cpu_v, e_htp_v = c - f, h - f
    E_cpu = 100.0 * np.linalg.norm(e_cpu_v) / nf
    E_htp = 100.0 * np.linalg.norm(e_htp_v) / nf
    E_ch = 100.0 * np.linalg.norm(h - c) / np.linalg.norm(c)

    print()
    print("=" * 72)
    print("实测（基准 = FP32 onnxruntime）")
    print("=" * 72)
    print(f"  FP32 std={f.std():.4f}  CPU std={c.std():.4f}  HTP std={h.std():.4f}")
    print(f"  E_cpu (CPU 参考 vs FP32) = {E_cpu:.4f}%")
    print(f"  E_htp (设备 HTP vs FP32) = {E_htp:.4f}%")
    print(f"  参考：HTP vs CPU         = {E_ch:.4f}%   (先前实测 19.4727%)")

    print()
    print("判据 3（有效性自检）：", end="")
    if E_cpu >= 100:
        print(f"[FAIL] E_cpu={E_cpu:.2f}% >= 100%，FP32 参考疑似跑错，本次无效")
        return
    if np.array_equal(u_fp32, u_cpu):
        print("[FAIL] FP32 与 CPU 逐位相同 —— 误把量化结果当 FP32，必须查清")
        return
    print(f"[OK] 通过 (E_cpu={E_cpu:.4f}% < 100%，且与 CPU 不同)")

    ratio = E_htp / E_cpu
    print()
    print("=" * 72)
    print("判据 1（主判据，EXP_PLAN_VS_FP32 第 3 节，事前定稿）")
    print("=" * 72)
    print(f"  E_htp / E_cpu = {ratio:.3f}")
    print()
    if ratio > 2.0:
        print("  => 【甲】HTP 显著更远离 FP32 —— 是 HTP 算错。")
        print("     下一步：查 HTP 算子实现 / 精度配置，方向正确。")
    elif ratio < 1.5:
        print("  => 【乙】两者与 FP32 同量级 —— HTP 不算'错'，")
        print("     是【模型对量化过于敏感】。")
        print("     下一步应改量化策略（提高 act_bitwidth / per-channel / 混合精度），")
        print("     不要再去查 HTP 算子实现。")
    else:
        print("  => 证据不足以区分（1.5~2.0），需补做 part2 的同类对照再判。")

    cosang = float(e_cpu_v @ e_htp_v /
                   (np.linalg.norm(e_cpu_v) * np.linalg.norm(e_htp_v)))
    print()
    print("判据 2（误差方向，仅用于解释判据 1）")
    print(f"  cos∠(CPU误差, HTP误差) = {cosang:.4f}")
    if cosang > 0.7:
        print("  -> 两者误差【同向】，HTP 像是 CPU 误差的放大版（同一机制、程度不同）")
    elif abs(cosang) < 0.3:
        print("  -> 两者误差【近正交】，是互相独立的两种误差源（不同实现各自舍入）")
    else:
        print("  -> 部分相关，介于两者之间")


if __name__ == "__main__":
    main()
