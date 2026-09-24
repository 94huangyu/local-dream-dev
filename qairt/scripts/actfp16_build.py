"""EXP_PLAN_ACTFP16 第 2 步：转换 + 量化 part1b（选择性 FP16 激活）。

参数逐条照抄现有 per-row DLC 里保存的 Converter/Quantizer command，**只改一个变量**：
  · converter  float_bitwidth  32 -> 16   （#46：必须显式；这是让"缺 encoding 的张量落成 FP16"）
  · quantizer  --input_list -> --enable_float_fallback + --keep_weights_quantized
权重/gamma 的 encoding 全部由 overrides 逐字钉死 ⇒ 唯一变量 = 激活数据类型。

源 ONNX 从 perrow/02_quant_perrow.log 核实（#61：不得由文件名推断）。
"""
import os, sys, subprocess, time
sys.stdout.reconfigure(encoding="utf-8")
BF16 = ("--bf16" in sys.argv)
TAG = "actbf16" if BF16 else "actfp16"
FBW = "bf16" if BF16 else "16"   # 🔴 converter 接受 32/16/bf16；quantizer 只接受 32/16
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "actfp16")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx", "transformer_part1b.onnx")
OVR = os.path.join(W, "part1b_%s_overrides.json" % TAG)
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable
# 🔴 2.48 的 flag 是 --source_model_input_shape / -s；DLC 里 dump 出来的参数名叫 input_dim，
#    直接照抄参数名会 unrecognized arguments（约束 5：查 --help，不猜）
DIMS = [("add_138", "1,4128,3840"), ("add_131", "1,1,3840"), ("tanh_19", "1,1,3840"),
        ("select_45", "1,4128,1,64"), ("select_46", "1,4128,1,64"), ("adaln_input", "1,256")]


def run(name, cmd, timeout):
    print("[%s] 开始 %s" % (name, time.strftime("%H:%M:%S")), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(W, "%s.log" % name), "w", encoding="utf-8").write(out)
    print("[%s] rc=%d  %.1f 分钟" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in out.splitlines():
        low = l.lower()
        if "processed" in low and "encoding" in low:
            print("    🔴 " + l.strip(), flush=True)       # #45：必查这一行
        elif "error" in low or "warning: quant" in low:
            print("    " + l.strip()[:160], flush=True)
    if r.returncode != 0:
        sys.exit("❌ %s 失败，日志见 %s.log" % (name, name))
    return out


def main():
    os.makedirs(W, exist_ok=True)
    f = os.path.join(W, "part1b_%s_fp32.dlc" % TAG)
    q = os.path.join(W, "part1b_%s_quantized.dlc" % TAG)
    cmd = [PY, TOOL, "qairt-converter", "--input_network", ONNX, "--output_path", f,
           "--quantization_overrides", OVR, "--float_bitwidth", FBW]
    for n, d in DIMS:
        cmd += ["--source_model_input_shape", n, d]
    run("01_converter_%s" % TAG, cmd, 7200)
    run("02_quantizer_%s" % TAG,
        [PY, TOOL, "qairt-quantizer", "--input_dlc", f, "--output_dlc", q,
         "--weights_bitwidth", "8", "--bias_bitwidth", "32",
         "--param_quantizer_calibration", "min-max", "--use_per_row_quantization",
         "--keep_weights_quantized", "--enable_float_fallback"], 14400)
    print("\n产物 %s  %.1f MB" % (q, os.path.getsize(q) / 1e6))


if __name__ == "__main__":
    main()
