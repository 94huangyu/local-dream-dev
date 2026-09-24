"""P0-B：建单算子 FC 重放（可选 baseline / perrow 两种配置）。

方案：scripts/EXP_PLAN_P0B_FC_REPLAY.md（判据事前锁定）

用法:  python scripts/p0b_build.py baseline     <- 与 #39 的 2.72~2.87% 可比，**主实验**
       python scripts/p0b_build.py perrow       <- 与设备当前部署配置一致（附带）

🔴 为什么主实验必须用 baseline（2026-08-22 差点搞错）：
   #39 测出的 2.72~2.87% 来自 `fxp_sim.py`，而它读的 dump 是
   `dlc_pipeline/.../transformer_part1b_quantized.dlc` = **基线 per-tensor**；
   `fc_probe/htp/linear_99_fc.raw`（08-15 15:31）也是同一轮。
   若拿 per-row encoding 去重放，判据锚点（#39）就落空 —— **两者不可比。**

🔴 轴向（仅 perrow 用到，已用已知答案验证）：
   DLC 的 axis=0 对应 **ONNX 的 dim1（列）**，因为 FullyConnected 权重从 [in,out] 转置成 [out,in]。
   实测按 dim1 反推 3840 个 scale 100% 吻合；按 dim0 只有 0.99%。**方阵掩盖了这一点。**
"""
import os, re, sys, json, io, subprocess
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
MODE = (sys.argv[1] if len(sys.argv) > 1 else "baseline").lower()
assert MODE in ("baseline", "perrow", "symtensor"), "用法: p0b_build.py [baseline|perrow|symtensor]"

P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
PY = sys.executable
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
M, K, N = 4128, 3840, 3840
WORK = os.path.join(P0B, "standalone_" + MODE)


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    os.makedirs(WORK, exist_ok=True)
    W = np.load(os.path.join(P0B, "val_1800_fp32.npy")).astype(np.float32)
    assert W.shape == (K, N)

    if MODE == "perrow":
        sc = np.load(os.path.join(P0B, "val_1800_perrow_scales.npy"))
        acts = json.load(open(os.path.join(P0B, "linear_99_encodings.json"),
                              encoding="utf-8"))["activations"]
        # 轴向门（已知答案样本，约束 8）
        cand = np.abs(W.astype(np.float64)).max(axis=0) / 127.0
        rel = np.abs(cand - sc) / sc
        frac = float(np.mean(rel < 1e-6))
        print("[轴向门] 按 ONNX dim1 反推：吻合率 %.2f%%  中位相对差 %.2e" % (frac * 100, np.median(rel)))
        if frac < 0.999:
            sys.exit("❌ 轴向门不过，停止")
        wenc = [{"bitwidth": 8, "scale": float(s), "offset": -128,
                 "min": float(-128 * s), "max": float(127 * s), "is_symmetric": "True"}
                for s in sc]
        qextra = ["--use_per_row_quantization"]
        n_expect = N
    elif MODE == "symtensor":
        # P0-B2：对称 per-tensor —— 唯一变量是权重 offset（−126 -> 0），激活保持基线不变
        b = json.load(open(os.path.join(P0B, "linear_99_encodings_baseline.json"), encoding="utf-8"))
        acts = {"node_linear_99_pre_reshape": b["node_linear_99_pre_reshape"],
                "linear_99_fc": b["linear_99_fc"]}
        s_sym = float(np.abs(W.astype(np.float64)).max() / 127.0)
        wenc = [{"bitwidth": 8, "scale": s_sym, "offset": -128,
                 "min": -128 * s_sym, "max": 127 * s_sym, "is_symmetric": "True"}]
        qextra = []
        n_expect = 1
        print("[配置] symtensor 对称 per-tensor：scale=%.12g offset=0（唯一变量 = offset）" % s_sym)
    else:
        b = json.load(open(os.path.join(P0B, "linear_99_encodings_baseline.json"), encoding="utf-8"))
        e = b["val_1800"]
        acts = {"node_linear_99_pre_reshape": b["node_linear_99_pre_reshape"],
                "linear_99_fc": b["linear_99_fc"]}
        wenc = [{"bitwidth": e["bitwidth"], "scale": e["scale"], "offset": e["offset"],
                 "min": e["min"], "max": e["max"], "is_symmetric": "False"}]
        qextra = []
        n_expect = 1
        print("[配置] baseline per-tensor：权重 scale=%.12g offset=%g（与 #39 同一配置）"
              % (e["scale"], e["offset"]))

    # ---- 单算子 ONNX ----
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    onnx_p = os.path.join(WORK, "fc99.onnx")
    node = helper.make_node("MatMul", ["x", "val_1800"], ["linear_99_fc"], name="node_linear_99")
    g = helper.make_graph([node], "fc99",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
                          [helper.make_tensor_value_info("linear_99_fc", TensorProto.FLOAT, [M, N])],
                          [numpy_helper.from_array(W, "val_1800")])
    mo = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mo.ir_version = 9
    onnx.save(mo, onnx_p)

    def aenc(e):
        return [{"bitwidth": e["bitwidth"], "min": e["min"], "max": e["max"],
                 "scale": e["scale"], "offset": e["offset"], "is_symmetric": "False"}]
    ov = {"version": "0.6.1",
          "activation_encodings": {"x": aenc(acts["node_linear_99_pre_reshape"]),
                                   "linear_99_fc": aenc(acts["linear_99_fc"])},
          "param_encodings": {"val_1800": wenc}}
    ov_p = os.path.join(WORK, "fc99_overrides.json")
    json.dump(ov, open(ov_p, "w", encoding="utf-8"), indent=1)
    lst = os.path.join(WORK, "input_list.txt")
    open(lst, "w").write("x:=" + X_RAW + "\n")
    print("[建图] %s  权重 encoding %d 条" % (os.path.basename(onnx_p), len(wenc)))

    fdlc, qdlc = os.path.join(WORK, "fc99_fp32.dlc"), os.path.join(WORK, "fc99_quantized.dlc")
    print("[1/3] converter ...", flush=True)
    rc, out = sh([PY, TOOL, "qairt-converter", "--input_network", onnx_p, "--output_path", fdlc,
                  "--quantization_overrides", ov_p, "--float_bitwidth", "32"])
    for l in out.splitlines():
        if "quantization encoding" in l.lower():
            print("    " + l.strip())
    if rc != 0:
        sys.exit("❌ converter rc=%d\n%s" % (rc, out[-1500:]))

    print("[2/3] quantizer ...", flush=True)
    rc, out = sh([PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                  "--input_list", lst, "--weights_bitwidth", "8", "--act_bitwidth", "16",
                  "--bias_bitwidth", "32", "--act_quantizer_calibration", "min-max",
                  "--param_quantizer_calibration", "min-max"] + qextra)
    if rc != 0:
        sys.exit("❌ quantizer rc=%d\n%s" % (rc, out[-1500:]))

    # ---- V1 门 ----
    csv = os.path.join(WORK, "fc99_enc.csv")
    rc, out = sh([PY, TOOL, "snpe-dlc-info", "-i", qdlc, "-d", "-s", csv])
    CH = re.compile(r"val_1800 encoding for channel_(\d+):.*?scale ([-\d.eE+]+)")
    PT = re.compile(r"val_1800 encoding : bitwidth \d+, min [-\d.]+, max [-\d.]+, "
                    r"scale ([-\d.eE+]+), offset ([-\d.]+)")
    got, pt = {}, None
    with io.open(csv, encoding="utf-8", errors="replace") as f:
        for line in f:
            for mm in CH.finditer(line):
                got[int(mm.group(1))] = float(mm.group(2))
            mm = PT.search(line)
            if mm and pt is None:
                pt = (float(mm.group(1)), float(mm.group(2)))
    print("")
    print("=== V1 门：注入后的 encoding 与真实是否逐个吻合 ===")
    if MODE == "perrow":
        print("  读回通道数 %d（期望 %d）" % (len(got), n_expect))
        if len(got) != n_expect:
            sys.exit("❌ V1 不过：通道数不符")
        gv = np.array([got[i] for i in range(N)])
        rel = np.abs(gv - sc) / sc
        print("  相对差 中位=%.2e 最大=%.2e  %s" % (np.median(rel), rel.max(),
                                                "✅" if rel.max() < 1e-6 else "❌"))
        if rel.max() >= 1e-6:
            sys.exit("❌ V1 不过")
    else:
        want = (wenc[0]["scale"], 0.0 if MODE == "symtensor" else wenc[0]["offset"])
        print("  读回 per-tensor: scale=%s offset=%s   期望 scale=%.12g offset=%g"
              % (pt[0] if pt else None, pt[1] if pt else None, want[0], want[1]))
        if pt is None or abs(pt[0] - want[0]) / want[0] > 1e-6:
            sys.exit("❌ V1 不过：per-tensor scale 未按注入值落地")
        print("  per-channel 条目数 %d（期望 0）  %s" % (len(got), "✅" if len(got) == 0 else "❌"))
        if len(got) != 0:
            sys.exit("❌ V1 不过：意外出现 per-channel")
        print("  ✅ 通过")

    # ---- context ----
    print("")
    print("[3/3] context-binary-generator --htp_socs sm8750 ...", flush=True)
    ctx = os.path.join(WORK, "ctx")
    os.makedirs(ctx, exist_ok=True)
    rc, out = sh([os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-generator.exe"),
                  "--backend", os.path.join(SDK, "lib", "x86_64-windows-msvc", "QnnHtp.dll"),
                  "--dlc_path", qdlc, "--binary_file", "fc99_%s_ctx" % MODE,
                  "--output_dir", ctx, "--htp_socs", "sm8750"])
    bins = [f for f in os.listdir(ctx) if f.endswith(".bin")]
    if not bins:
        sys.exit("❌ 未生成 context rc=%d\n%s" % (rc, out[-1500:]))
    for b in bins:
        print("    %s  %.2f MB  %s" % (b, os.path.getsize(os.path.join(ctx, b)) / 1e6,
                                       "✅ .SM8750 后缀（#64）" if ".SM8750" in b else "❌ 缺后缀"))
    print("")
    print("[完成] 配置=%s  下一步上机" % MODE)


if __name__ == "__main__":
    main()
