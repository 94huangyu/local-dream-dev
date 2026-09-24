"""EXP_PLAN_WFXP_ACTFP 阶段 1（纯宿主）：工具链会不会产出 `FP16 激活 + SFxp8 权重`？

判据见 scripts/EXP_PLAN_WFXP_ACTFP.md（**事前定稿，不得修改**）。

配方依据（文档实查，见指南 §二十七）：
  · `--enable_float_fallback` 在 2.48 的 quantizer 已是 no-op；浮点回退在 **converter**，
    机制 = 「**缺 encoding 的张量落成浮点**」 ⇒ overrides 里**故意不给激活 encoding**
  · `--keep_weights_quantized`：*"keep the weights quantized even when the output of the op
    is in floating point ... Required to enable wFxp_actFP"*

🔴 context 构建照抄 scripts/o_dlbc_probe.py 的**订正后**写法（带 --config_file）。
   **不要照抄 p0b_build.py**——它只传 --htp_socs，正是 #95 那个把整套结论作废的装置错误。
"""
import os, re, sys, json, io, shutil, hashlib, subprocess
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
WORK = os.path.join(P0B, "wfxp")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
M, K, N = 4128, 3840, 3840
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")


def sh(cmd, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_arm(tag, keep_wq, float_bw, calib=True):
    """一臂 = (是否 --keep_weights_quantized, --float_bitwidth, 用 --input_list 还是 --enable_float_fallback)

    calib=False 走 --enable_float_fallback：文档明写它与 --input_list 互斥且二者必居其一
    ⇒ 不给校准集 = 官方表达的「不要量化激活」。"""
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    W = np.load(os.path.join(P0B, "val_1800_fp32.npy")).astype(np.float32)
    assert W.shape == (K, N)

    import onnx
    from onnx import helper, numpy_helper, TensorProto
    onnx_p = os.path.join(d, "fc99.onnx")
    node = helper.make_node("MatMul", ["x", "val_1800"], ["linear_99_fc"], name="node_linear_99")
    g = helper.make_graph([node], "fc99",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
                          [helper.make_tensor_value_info("linear_99_fc", TensorProto.FLOAT, [M, N])],
                          [numpy_helper.from_array(W, "val_1800")])
    mo = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mo.ir_version = 9
    onnx.save(mo, onnx_p)

    # 🔴 只给权重 encoding，**故意不给 activation_encodings**（= 让激活落成浮点）
    # SFxp8 对称 per-row：OpDef 里 FC 的 wFxp_actFP 行要求 SFIXED_POINT_8 + AXIS_SCALE_OFFSET + Symmetric
    sc = np.load(os.path.join(P0B, "val_1800_perrow_scales.npy"))
    wenc = [{"bitwidth": 8, "scale": float(s), "offset": -128,
             "min": float(-128 * s), "max": float(127 * s), "is_symmetric": "True"} for s in sc]
    ov = {"version": "0.6.1", "activation_encodings": {}, "param_encodings": {"val_1800": wenc}}
    ov_p = os.path.join(d, "ov.json")
    json.dump(ov, open(ov_p, "w", encoding="utf-8"), indent=1)
    lst = os.path.join(d, "input_list.txt")
    open(lst, "w").write("x:=" + X_RAW + "\n")

    fdlc, qdlc = os.path.join(d, "fc99_fp32.dlc"), os.path.join(d, "fc99_quantized.dlc")
    rc, out = sh([PY, TOOL, "qairt-converter", "--input_network", onnx_p, "--output_path", fdlc,
                  "--quantization_overrides", ov_p, "--float_bitwidth", str(float_bw)])
    proc = [l.strip() for l in out.splitlines() if "processed" in l.lower() and "encoding" in l.lower()]
    if rc != 0:
        print("  [%s] ❌ converter rc=%d\n%s" % (tag, rc, out[-1200:])); return None
    q = [PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
         "--weights_bitwidth", "8", "--bias_bitwidth", "32",
         "--param_quantizer_calibration", "min-max", "--use_per_row_quantization"]
    q += (["--input_list", lst, "--act_bitwidth", "16",
           "--act_quantizer_calibration", "min-max"] if calib else ["--enable_float_fallback"])
    if keep_wq:
        q.append("--keep_weights_quantized")
    rc, out = sh(q)
    if rc != 0:
        print("  [%s] ❌ quantizer rc=%d\n%s" % (tag, rc, out[-1200:])); return None
    return dict(tag=tag, dir=d, fdlc=fdlc, qdlc=qdlc, proc=proc)


def datatypes(qdlc, d):
    """从 snpe-dlc-info CSV 读回三个张量的真实 data type（#96：读元数据，不看命令行）"""
    csv = os.path.join(d, "enc.csv")
    sh([PY, TOOL, "snpe-dlc-info", "-i", qdlc, "-d", "-s", csv])
    want = {"x": None, "linear_99_fc": None, "val_1800": None}
    pat = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+)")
    with io.open(csv, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in pat.finditer(line):
                if m.group(1) in want and want[m.group(1)] is None:
                    want[m.group(1)] = m.group(2)
    return want


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    arms = {}
    for tag, kwq, fbw, cal in [("kwq_f16", True, 16, True), ("nokwq_f16", False, 16, True),
                               ("kwq_f16_ffb", True, 16, False), ("nokwq_f16_ffb", False, 16, False)]:
        print("[建] %s  keep_wq=%s  float_bw=%d  %s"
              % (tag, kwq, fbw, "--input_list" if cal else "--enable_float_fallback"), flush=True)
        a = build_arm(tag, kwq, fbw, cal)
        if a is None:
            continue
        a["dt"] = datatypes(a["qdlc"], a["dir"])
        a["md5"] = md5(a["qdlc"])
        arms[tag] = a
        print("     converter: %s" % (a["proc"][:1] or ["(无 Processed 行)"]))
        print("     DLC 数据类型: x=%(x)s  val_1800=%(val_1800)s  out=%(linear_99_fc)s" % a["dt"])
        print("     quantized dlc md5 %s" % a["md5"][:16])

    print("\n" + "=" * 78)
    print("=== V2 生效性门：两个不同取值必须给出两个不同产物（§7.3 纪律）===")
    if len(arms) == 2:
        same = arms["kwq_f16"]["md5"] == arms["nokwq_f16"]["md5"]
        print("  带/不带 --keep_weights_quantized 的 quantized DLC md5 %s"
              % ("相同 ❌ 选项被静默忽略" if same else "不同 ✅"))
    else:
        print("  ❌ 两臂未都产出，无法判定")

    print("\n=== V1 产出门（主判据①）===")
    a = arms.get("kwq_f16")
    if a is None:
        print("  ❌ kwq 臂未产出")
        return
    dt = a["dt"]
    ok_act = (dt["x"] or "").lower().startswith("fp16") or "FLOAT_16" == dt["x"] or (dt["x"] or "") in ("Float_16", "fp16")
    print("  期望: x=FLOAT_16 / FP16 ,  val_1800=sFxp_8 ,  out=FLOAT_16 / FP16")
    print("  实测: x=%s , val_1800=%s , out=%s" % (dt["x"], dt["val_1800"], dt["linear_99_fc"]))
    print("  ⇒ 判定见方案 §四；本脚本只报事实，不解释")


if __name__ == "__main__":
    main()
