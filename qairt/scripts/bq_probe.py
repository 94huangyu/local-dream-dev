"""#91：分块量化（BQ）在 HTP 上可行吗？

方案：scripts/EXP_PLAN_BQ.md（判据事前锁定）
V0 支持性门是本实验的主要价值：qairt-quantizer 没有 BQ 的 CLI 开关，
只能经 overrides 注入 schema 2.0.0 的 BQ 格式，而官方例子全用 int4 ⇒ int8 是否支持未知。

⚠️ 必须主动查"静默降级"：#45 的教训是 overrides 失效时不报错。
"""
import os, re, io, sys, json, subprocess
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
PY = sys.executable
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
M, K, N = 4128, 3840, 3840
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 128
WORK = os.path.join(P0B, "standalone_bq%d" % BLOCK)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    assert K % BLOCK == 0, "BLOCK 必须整除 K=3840"
    os.makedirs(WORK, exist_ok=True)
    W = np.load(os.path.join(P0B, "val_1800_fp32.npy")).astype(np.float64)
    pr = json.load(open(os.path.join(P0B, "linear_99_encodings.json"), encoding="utf-8"))["activations"]
    ex, ey = pr["node_linear_99_pre_reshape"], pr["linear_99_fc"]

    nb = K // BLOCK
    Wb = W.reshape(nb, BLOCK, N)
    s_b = np.maximum(np.abs(Wb).max(axis=1) / 127.0, 1e-12)      # [nb, N]
    print("[配置] BQ block=%d  ⇒ %d 块 × %d 通道 = %d 个 scale" % (BLOCK, nb, N, nb * N))
    np.save(os.path.join(WORK, "bq_scales.npy"), s_b)

    # ---- ONNX ----
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    onnx_p = os.path.join(WORK, "fc99.onnx")
    node = helper.make_node("MatMul", ["x", "val_1800"], ["linear_99_fc"], name="node_linear_99")
    g = helper.make_graph([node], "fc99",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
                          [helper.make_tensor_value_info("linear_99_fc", TensorProto.FLOAT, [M, N])],
                          [numpy_helper.from_array(W.astype(np.float32), "val_1800")])
    mo = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mo.ir_version = 9
    onnx.save(mo, onnx_p)

    # ---- overrides：整份用 schema 2.0.0（BQ 只有 2.0.0 有）----
    # 文档 Appendix C：y_scale 形状 [通道][块]，axis = 分块所沿的轴，y_zero_point = -offset
    ov = {"version": "2.0.0", "encodings": [
        {"name": "x", "output_dtype": "uint16",
         "y_scale": ex["scale"], "y_zero_point": -ex["offset"]},
        {"name": "linear_99_fc", "output_dtype": "uint16",
         "y_scale": ey["scale"], "y_zero_point": -ey["offset"]},
        {"name": "val_1800", "output_dtype": "int8",
         "y_scale": s_b.T.tolist(),                       # [N 通道][nb 块]
         # 🔴 2026-08-22：给二维 y_zero_point 会让转换器在 contain_decimal_num() 里
         #    对 ndarray 调 round() 而 TypeError。y_zero_point 是可选项（默认 0），
         #    本例本来就是对称（全 0）=> 直接省略。
         "axis": 0, "block_size": BLOCK},
    ]}
    ov_p = os.path.join(WORK, "fc99_overrides.json")
    json.dump(ov, open(ov_p, "w", encoding="utf-8"))
    print("[overrides] %.1f MB（schema 2.0.0）" % (os.path.getsize(ov_p) / 1e6))
    lst = os.path.join(WORK, "input_list.txt")
    open(lst, "w").write("x:=" + X_RAW + "\n")

    fdlc, qdlc = os.path.join(WORK, "fc99_fp32.dlc"), os.path.join(WORK, "fc99_quantized.dlc")
    print("")
    print("=== V0 支持性门（本实验的主要价值）===")
    print("[1/3] converter ...", flush=True)
    rc, out = sh([PY, TOOL, "qairt-converter", "--input_network", onnx_p, "--output_path", fdlc,
                  "--quantization_overrides", ov_p, "--float_bitwidth", "32"])
    for l in out.splitlines():
        if "quantization encoding" in l.lower():
            print("    " + l.strip())
    if rc != 0:
        print("  ❌ converter 失败 rc=%d" % rc)
        print("  报错原文（判归因用，见方案 §四）：")
        print("  " + out[-1200:].replace("\n", "\n  "))
        return
    print("  converter OK")

    print("[2/3] quantizer ...", flush=True)
    rc, out = sh([PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                  "--input_list", lst, "--weights_bitwidth", "8", "--act_bitwidth", "16",
                  "--bias_bitwidth", "32", "--act_quantizer_calibration", "min-max",
                  "--param_quantizer_calibration", "min-max"])
    if rc != 0:
        print("  ❌ quantizer 失败 rc=%d" % rc)
        print("  " + out[-1200:].replace("\n", "\n  "))
        return
    print("  quantizer OK")

    # ---- 是否被静默降级 ----
    csv = os.path.join(WORK, "fc99_enc.csv")
    sh([PY, TOOL, "snpe-dlc-info", "-i", qdlc, "-d", "-s", csv])
    txt = io.open(csv, encoding="utf-8", errors="replace").read() if os.path.exists(csv) else ""
    n_ch = len(set(re.findall(r"val_1800 encoding for channel_(\d+):", txt)))
    n_blk = len(re.findall(r"block", txt, re.I))
    has_pt = bool(re.search(r"val_1800 encoding : bitwidth", txt))
    print("")
    print("  读回 val_1800：per-channel 条目 %d   含 'block' 字样 %d 处   per-tensor 条目 %s"
          % (n_ch, n_blk, has_pt))
    if n_blk == 0:
        print("  🔴 **未见分块 encoding** ⇒ 被静默降级（#45 类）或工具链不支持 BQ")
        print("  ⇒ V0 不过：#91 关闭。判据 §二：这是**方向结论**，不是执行错误")
        print("     （converter/quantizer 都返回 0，说明格式被接受但语义没落地 —— 典型的静默失效）")
        return
    print("  ✅ V0 通过：分块 encoding 已落地，进入主判据")

    print("[3/3] context-binary-generator ...", flush=True)
    ctx = os.path.join(WORK, "ctx")
    os.makedirs(ctx, exist_ok=True)
    rc, out = sh([os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-generator.exe"),
                  "--backend", os.path.join(SDK, "lib", "x86_64-windows-msvc", "QnnHtp.dll"),
                  "--dlc_path", qdlc, "--binary_file", "fc99_bq%d_ctx" % BLOCK,
                  "--output_dir", ctx, "--htp_socs", "sm8750"])
    bins = [f for f in os.listdir(ctx) if f.endswith(".bin")]
    if not bins:
        print("  ❌ context 生成失败 rc=%d ⇒ **HTP 后端不支持 FC 的 BQ**（方向结论）" % rc)
        print("  " + out[-1200:].replace("\n", "\n  "))
        return
    for b in bins:
        print("    %s  %.2f MB" % (b, os.path.getsize(os.path.join(ctx, b)) / 1e6))
    print("")
    print("[完成] V0 全过，可上机跑主判据")


if __name__ == "__main__":
    main()
