"""part2a 手术式 FP16：转换 + 量化。参数照抄 part2a 现有 per-row DLC 的记录，只改浮点相关项。"""
import os, sys, subprocess, time
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx",
                    "transformer_part2a_fixed.onnx")
OVR = os.path.join(W, "part2a_fp16_overrides.json")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable
DIMS = [("unified", "1,4128,3840"), ("unified_mask", "1,4128"),
        ("unified_freqs", "1,4128,64,2"), ("adaln_input", "1,256")]


def run(name, cmd, t):
    print("[%s] %s" % (name, time.strftime("%H:%M:%S")), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=t)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(W, name + ".log"), "w", encoding="utf-8").write(out)
    print("[%s] rc=%d %.1f 分钟" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for l in out.splitlines():
        if "processed" in l.lower() and "encoding" in l.lower():
            print("    🔴 " + l.strip(), flush=True)
        elif "ERROR" in l:
            print("    " + l.strip()[:170], flush=True)
    if r.returncode:
        sys.exit("❌ %s 失败" % name)


f = os.path.join(W, "part2a_fp16_fp32.dlc")
q = os.path.join(W, "part2a_fp16_quantized.dlc")
cmd = [PY, TOOL, "qairt-converter", "--input_network", ONNX, "--output_path", f,
       "--quantization_overrides", OVR, "--float_bitwidth", "16"]
for n, d in DIMS:
    cmd += ["--source_model_input_shape", n, d]
run("01_conv_p2a_fp16", cmd, 7200)
run("02_quant_p2a_fp16",
    [PY, TOOL, "qairt-quantizer", "--input_dlc", f, "--output_dlc", q,
     "--weights_bitwidth", "8", "--bias_bitwidth", "32",
     "--param_quantizer_calibration", "min-max", "--use_per_row_quantization",
     "--keep_weights_quantized", "--enable_float_fallback"], 14400)
print("产物 %s  %.1f MB" % (q, os.path.getsize(q) / 1e6))
