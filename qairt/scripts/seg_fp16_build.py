"""Generic: build one segment's surgical-FP16 variant (overrides -> convert -> quantize -> context).

Usage: python seg_fp16_build.py <part1a|part1b|part2b>

Discipline (all learned the hard way today):
  - per-axis encodings must be symmetric, and "symmetric" means offset==0 (constraint 8)
  - must apply the symmetry fix: inject *_converted_unsigned_symmetric encodings onto the
    base tensor names, else MatMul op validation fails (#109)
  - whitelisted tensors get NO encoding => their producer op runs float (#107/#108)
  - context build MUST pass --config_file with devices soc_model 69 / dsp_arch v79 (#95)
"""
import os, re, io, sys, json, shutil, subprocess, time

sys.stdout.reconfigure(encoding="utf-8")
SEG = sys.argv[1]
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable

CFG = {
    "part1a": ("transformer_part1a", [("latents", "1,16,128,128"), ("timestep", "1"),
                                      ("caption", "1,32,2560"), ("cap_pad_mask", "1,32")]),
    "part1b": ("transformer_part1b", [("add_138", "1,4128,3840"), ("add_131", "1,1,3840"),
                                      ("tanh_19", "1,1,3840"), ("select_45", "1,4128,1,64"),
                                      ("select_46", "1,4128,1,64"), ("adaln_input", "1,256")]),
    # 2026-08-25 补：原先缺 part2a（08-23 时它是用专用脚本 p2a_fp16_build.py 建的），
    # 导致四段批量建图时 part2a 直接 KeyError 秒退。输入契约与 seg_dtype_build.py 同源。
    "part2a": ("transformer_part2a_fixed", [("unified", "1,4128,3840"),
                                            ("unified_mask", "1,4128"),
                                            ("unified_freqs", "1,4128,64,2"),
                                            ("adaln_input", "1,256")]),
    "part2b": ("transformer_part2b_fixed", [("add_92", "1,4128,3840"), ("select", "1,4128,1,64"),
                                            ("select_1", "1,4128,1,64"),
                                            ("split_7_split_2", "1,1,3840"),
                                            ("split_7_split_3", "1,1,3840"),
                                            ("val_105", "1,1,1,4128"), ("adaln_input", "1,256")]),
}[SEG]
ONNX, DIMS = CFG

# 🔴 台账 #138：本脚本的 CFG 也是**手写源清单**。它目前是对的
#（造出了部署产物），但"对"不能靠运气 —— 加一条断言钉死它与权威模块一致。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                                      # noqa: E402
assert ONNX == canonical_sources.ALL[SEG][0], (
    "CFG 里 %s 的源主干 %r 与 canonical_sources 的 %r 不符"
    % (SEG, ONNX, canonical_sources.ALL[SEG][0]))

# --- Tier 2 (2026-08-29): SEG_L=80 则把序列维改成 L，且**所有产物名带 _L80**。
#     SEG_L 不设时行为与以前**逐字相同**（不得改变部署路径）。
#     规则同 dit_seq_surgery.py：32->L，4128->4096+L，**4096 不动**。
SEG_L = int(os.environ.get("SEG_L", "32"))
TAG = "fp16" if SEG_L == 32 else "fp16_L%d" % SEG_L
if SEG_L != 32:
    def _rd(d):
        return ",".join(str(SEG_L) if x == "32" else
                        str(4096 + SEG_L) if x == "4128" else x
                        for x in d.split(","))
    DIMS = [(n, _rd(d)) for n, d in DIMS]
    print("[%s] L=%d 输入契约: %s" % (SEG, SEG_L, DIMS), flush=True)


TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[[^\]]*\]; "
                r"tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")
SYM = "_converted_unsigned_symmetric"


def sh(name, cmd, timeout):
    print("[%s] %s" % (name, time.strftime("%H:%M:%S")), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(W, "%s_%s.log" % (SEG, name)), "w", encoding="utf-8").write(out)
    print("[%s] rc=%d %.1f min" % (name, r.returncode, (time.time() - t0) / 60), flush=True)
    for line in out.splitlines():
        low = line.lower()
        if "processed" in low and "encoding" in low:
            print("    * " + line.strip(), flush=True)
        elif "ERROR" in line and "libcdsprpc" not in line and "Unsupported HTP Arch" not in line:
            print("    " + line.strip()[:170], flush=True)
    if r.returncode:
        sys.exit("FAIL %s" % name)
    return out


def enc1(e, sym):
    return dict(bitwidth=e["bitwidth"], min=e["min"], max=e["max"],
                scale=e["scale"], offset=e["offset"], is_symmetric=str(sym))


def main():
    white = set(json.load(open(os.path.join(W, "allseg_white.json"), encoding="utf-8"))[SEG])
    print("[%s] whitelist %d: %s" % (SEG, len(white), sorted(white)), flush=True)

    kind, ch, pt = {}, {}, {}
    with io.open(os.path.join(W, "%s_full_enc.csv" % SEG), encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), (m.group(2), m.group(3)))
            for m in CH.finditer(line):
                ch.setdefault(m.group(1), {})[int(m.group(2))] = dict(
                    bitwidth=int(m.group(3)), min=float(m.group(4)), max=float(m.group(5)),
                    scale=float(m.group(6)), offset=float(m.group(7)))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), dict(bitwidth=int(m.group(2)), min=float(m.group(3)),
                                               max=float(m.group(4)), scale=float(m.group(5)),
                                               offset=float(m.group(6))))
    static = set(k for k, v in kind.items() if v[1] == "STATIC")

    params, nch = {}, 0
    for w in sorted(static):
        if w.endswith("_bias") or kind[w][0] not in ("sFxp_8", "uFxp_8", "uFxp_16"):
            continue
        if w in ch:
            c = ch[w]
            assert all(c[i]["offset"] == 0 for i in range(len(c))), "%s per-axis offset != 0" % w
            params[w] = [enc1(c[i], True) for i in range(len(c))]
            nch += len(c)
        elif w in pt:
            params[w] = [enc1(pt[w], False)]

    acts, skipped = {}, []
    for a, e in pt.items():
        if a in static or e["bitwidth"] != 16:
            continue
        if a in white:
            skipped.append(a)
        else:
            acts[a] = [enc1(e, False)]

    nsym = 0
    for k, e in pt.items():
        if k.endswith(SYM):
            base = k[:-len(SYM)]
            if base in acts:
                acts[base] = [enc1(e, True)]
                nsym += 1

    print("  weights %d (per-ch %d / scales %d) | acts written %d, skipped %d | symfix %d"
          % (len(params), sum(1 for v in params.values() if len(v) > 1), nch,
             len(acts), len(skipped), nsym), flush=True)
    missing = [t for t in white if t not in skipped]
    if missing:
        print("  WARN whitelist entries not found among 16-bit acts: %s" % missing, flush=True)

    ov = os.path.join(W, "%s_%s_ovr.json" % (SEG, TAG))
    json.dump({"version": "0.6.1", "activation_encodings": acts, "param_encodings": params},
              open(ov, "w", encoding="utf-8"))
    if SEG_L != 32:
        # encoding 与序列长度无关（#139：n=59 同口径配对，\|a\|max 比值 0.97~1.02）
        # ⇒ 重生的 overrides 应与部署版**逐字节相同**。不同就是有别的东西变了，必须停。
        base_ov = os.path.join(W, "%s_fp16_ovr.json" % SEG)
        if os.path.exists(base_ov):
            a = open(base_ov, "rb").read()
            b = open(ov, "rb").read()
            print("  overrides vs 部署版: %s (%d vs %d bytes)"
                  % ("逐字节相同" if a == b else "🔴 不同", len(a), len(b)),
                  flush=True)
            if a != b:
                sys.exit("FAIL overrides 与部署版不同 ⇒ 不只是 L 变了，停")

    fdlc = os.path.join(W, "%s_%s_fp32.dlc" % (SEG, TAG))
    qdlc = os.path.join(W, "%s_%s_quantized.dlc" % (SEG, TAG))
    # 环境变量 SEG_ONNX_SUFFIX 指向改造过的 ONNX（如 `_clip`：已插入显式 Clip(±65504)，#111）
    suf = os.environ.get("SEG_ONNX_SUFFIX", "")
    src_onnx = os.path.join(D, ONNX + suf + ".onnx")
    if not os.path.exists(src_onnx):
        sys.exit("FAIL 源 ONNX 不存在: %s" % src_onnx)
    print("  源 ONNX: %s" % os.path.basename(src_onnx), flush=True)
    cmd = [PY, TOOL, "qairt-converter", "--input_network", src_onnx,
           "--output_path", fdlc, "--quantization_overrides", ov, "--float_bitwidth", "16"]
    for n, d in DIMS:
        cmd += ["--source_model_input_shape", n, d]
    sh("conv", cmd, 10800)
    sh("quant", [PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                 "--weights_bitwidth", "8", "--bias_bitwidth", "32",
                 "--param_quantizer_calibration", "min-max", "--use_per_row_quantization",
                 "--keep_weights_quantized", "--enable_float_fallback"], 14400)
    print("  quantized %.1f MB" % (os.path.getsize(qdlc) / 1e6), flush=True)

    out = os.path.join(W, "ctx_%s_%s" % (SEG, TAG))
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    det = os.path.join(out, "d.json")
    ext = os.path.join(out, "e.json")
    json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]}, open(det, "w"), indent=1)
    json.dump({"backend_extensions": {
        "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
        "config_file_path": det}}, open(ext, "w"), indent=1)
    o = sh("ctx", [os.path.join(BIN, "qnn-context-binary-generator.exe"), "--backend",
                   os.path.join(LIB, "QnnHtp.dll"), "--dlc_path", qdlc, "--binary_file",
                   "%s_%s" % (SEG, TAG), "--output_dir", out, "--htp_socs", "sm8750",
                   "--config_file", ext], 14400)
    if [l for l in o.splitlines() if "available PD" in l]:
        sys.exit("FAIL G2 PD red line")
    bins = [x for x in os.listdir(out) if x.endswith(".bin")]
    if not bins:
        sys.exit("FAIL no context produced")
    p = os.path.join(out, bins[0])
    print("  OK %s  %.1f MB" % (bins[0], os.path.getsize(p) / 1e6), flush=True)
    subprocess.run([PY, os.path.join("D:", os.sep, "LocalDreamZImage", "scripts",
                                     "check_ctx_identity.py"),
                    "--expect-arch", "79", "--expect-vtcm", "8", p])


if __name__ == "__main__":
    main()
