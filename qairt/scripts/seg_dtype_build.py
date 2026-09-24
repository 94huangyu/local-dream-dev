# 🔴 2026-08-27：本脚本此前产出的 part1a dtype 产物（part1a_dt_fp32.dlc /
# part1a_dt_quantized.dlc / ctx_part1a_dt/，共 6.5 GB）**已从盘上删除**，
# 依据是台账 #123「dtype 机制的产物在设备上不兑现收益，不再用它做画质实验」。
# 需要时重跑本脚本即可再生。见 docs/DISK_INVENTORY.md「2026-08-27 清理记录」。
"""用**官方混合精度机制**（encodings 里的 `dtype` 字段，指南 §三十）重建一段。

用法: python seg_dtype_build.py <part1a|part1b|part2a|part2b>

与旧做法（seg_fp16_build.py）的区别：
  旧：白名单张量**不给** encoding + `--enable_float_fallback`
      => 未文档化旁路；浮点沿输入锥无界扩散（白名单 8 => 产物 183 个浮点）
  新：**每个张量都显式声明 dtype**；走 `--input_list` 校准路径
      => 零扩散（实测两算子探针：1 个目标 + 2 个自动 Convert）
      => 后端图修正（`*_converted_unsigned_symmetric` 等）由校准路径**自动应用**，无需手工注入

坑（均已实测）：
  · `version` 只接受 "0.5.0" / "0.6.1"
  · 不给 `--float_bitwidth 16` 会落成 Float_32
  · 必须看 **quantizer 之后**的最终 DLC，converter 中间态会被纠正
"""
import os, re, io, sys, json, shutil, subprocess, time

sys.stdout.reconfigure(encoding="utf-8")
SEG = sys.argv[1]
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
DLCP = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "dlc_pipeline")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable

CFG = {
    "part1a": ("transformer_part1a",
               [("latents", "1,16,128,128"), ("timestep", "1"),
                ("caption", "1,32,2560"), ("cap_pad_mask", "1,32")],
               os.path.join(DLCP, "transformer_part1a", "input_list_raw.txt")),
    "part1b": ("transformer_part1b",
               [("add_138", "1,4128,3840"), ("add_131", "1,1,3840"), ("tanh_19", "1,1,3840"),
                ("select_45", "1,4128,1,64"), ("select_46", "1,4128,1,64"), ("adaln_input", "1,256")],
               os.path.join(DLCP, "transformer_part1b", "input_list_raw.txt")),
    "part2a": ("transformer_part2a_fixed",
               [("unified", "1,4128,3840"), ("unified_mask", "1,4128"),
                ("unified_freqs", "1,4128,64,2"), ("adaln_input", "1,256")],
               os.path.join(DLCP, "transformer_part2", "input_list_raw.txt")),
    "part2b": ("transformer_part2b_fixed",
               [("add_92", "1,4128,3840"), ("select", "1,4128,1,64"), ("select_1", "1,4128,1,64"),
                ("split_7_split_2", "1,1,3840"), ("split_7_split_3", "1,1,3840"),
                ("val_105", "1,1,1,4128"), ("adaln_input", "1,256")],
               os.path.join(P0, "p2split", "input_list_part2b_fixed.txt")),
}[SEG]
ONNX, DIMS, CALIB = CFG

TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[[^\]]*\]; "
                r"tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def sh(name, cmd, timeout):
    print("[%s] %s" % (name, time.strftime("%H:%M:%S")), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(W, "%s_dt_%s.log" % (SEG, name)), "w", encoding="utf-8").write(out)
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


def main():
    white = set(json.load(open(os.path.join(W, "allseg_white.json"), encoding="utf-8"))[SEG])
    print("[%s] whitelist(float16) %d: %s" % (SEG, len(white), sorted(white)), flush=True)
    if not os.path.exists(CALIB):
        sys.exit("FAIL 校准集不存在: %s" % CALIB)

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

    def ei(e, sym):
        d = dict(bitwidth=e["bitwidth"], dtype="int", min=e["min"], max=e["max"],
                 scale=e["scale"], offset=e["offset"])
        d["is_symmetric"] = str(sym)
        return d

    FLOAT16 = {"bitwidth": 16, "dtype": "float"}

    params, nch = {}, 0
    for w in sorted(static):
        if w.endswith("_bias") or kind[w][0] not in ("sFxp_8", "uFxp_8", "uFxp_16"):
            continue
        if w in ch:
            c = ch[w]
            assert all(c[i]["offset"] == 0 for i in range(len(c))), "%s per-axis offset != 0" % w
            params[w] = [ei(c[i], True) for i in range(len(c))]
            nch += len(c)
        elif w in pt:
            params[w] = [ei(pt[w], False)]

    # 🔴 双激活 MatMul 的输入**不要给 encoding**：让量化器自己校准并应用它的
    #    「转无符号对称」修正（部署 DLC 里可见 *_converted_unsigned_symmetric）。
    #    给了显式 encoding 反而使该修正被跳过 => node_MatMul_106 校验失败（2026-08-24 实测两次）。
    #    文档依据：未出现在 JSON 里的张量按定点处理（quantization.html §Quantized Mode）。
    # 🔴🔴 2026-08-24 两次失败后查明：**只要用了 --quantization_overrides，
    #    量化器的自动「转无符号对称」整体不生效**（部署那份的 converter 命令里 overrides 为空）。
    #    「把这些张量从 overrides 里去掉、交给量化器」**无效**（实测，代价 68 分钟）。
    #    唯一有效的是**手工把对称 encoding 注入基础张量名** —— 该做法在 fp16 版上四段全部成功。
    SYMSUF = "_converted_unsigned_symmetric"

    acts, nf = {}, 0
    for a, e in pt.items():
        if a in static or e["bitwidth"] != 16:
            continue
        if a in white:
            acts[a] = [dict(FLOAT16)]
            nf += 1
        else:
            acts[a] = [ei(e, False)]
    nsym = 0
    for k, e in pt.items():
        if k.endswith(SYMSUF):
            b = k[:-len(SYMSUF)]
            if b in acts:
                acts[b] = [ei(e, True)]     # offset -32768，对称
                nsym += 1
    print("  手工注入对称 encoding（MatMul）%d 个" % nsym, flush=True)
    print("  权重 %d（per-ch %d / scale %d）｜激活 int %d、**float %d**"
          % (len(params), sum(1 for v in params.values() if len(v) > 1), nch,
             len(acts) - nf, nf), flush=True)
    miss = [t for t in white if t not in acts]
    if miss:
        print("  WARN 白名单未落入激活表: %s" % miss, flush=True)

    ov = os.path.join(W, "%s_dt_ovr.json" % SEG)
    json.dump({"version": "0.6.1", "activation_encodings": acts, "param_encodings": params},
              open(ov, "w", encoding="utf-8"))

    fdlc = os.path.join(W, "%s_dt_fp32.dlc" % SEG)
    qdlc = os.path.join(W, "%s_dt_quantized.dlc" % SEG)
    cmd = [PY, TOOL, "qairt-converter", "--input_network", os.path.join(D, ONNX + ".onnx"),
           "--output_path", fdlc, "--quantization_overrides", ov, "--float_bitwidth", "16"]
    for n, d in DIMS:
        cmd += ["--source_model_input_shape", n, d]
    sh("conv", cmd, 10800)
    sh("quant", [PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                 "--input_list", CALIB, "--act_bitwidth", "16", "--weights_bitwidth", "8",
                 "--bias_bitwidth", "32", "--float_bitwidth", "16",
                 "--use_per_row_quantization"], 28800)
    print("  quantized %.1f MB" % (os.path.getsize(qdlc) / 1e6), flush=True)

    csv = os.path.join(W, "%s_dt_enc.csv" % SEG)
    subprocess.run([PY, TOOL, "snpe-dlc-info", "-i", qdlc, "-d", "-s", csv],
                   capture_output=True, text=True, timeout=7200)
    k2 = {}
    with io.open(csv, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                k2.setdefault(m.group(1), (m.group(2), m.group(3)))
    act2 = {a: v[0] for a, v in k2.items() if v[1] != "STATIC"}
    f16 = [a for a, v in act2.items() if v == "Float_16"]
    print("  [G1] Float_16 激活 %d 个（白名单 %d + 自动 Convert）" % (len(f16), nf), flush=True)
    bad = [t for t in white if act2.get(t) != "Float_16"]
    print("  [G1] 白名单全部落 Float_16: %s" % ("PASS" if not bad else "FAIL %s" % bad), flush=True)

    out = os.path.join(W, "ctx_%s_dt" % SEG)
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
                   "%s_dt" % SEG, "--output_dir", out, "--htp_socs", "sm8750",
                   "--config_file", ext], 14400)
    if [l for l in o.splitlines() if "available PD" in l]:
        sys.exit("FAIL G2 PD red line")
    bins = [x for x in os.listdir(out) if x.endswith(".bin")]
    if not bins:
        sys.exit("FAIL no context")
    p = os.path.join(out, bins[0])
    print("  OK %s  %.1f MB" % (bins[0], os.path.getsize(p) / 1e6), flush=True)
    subprocess.run([PY, os.path.join("D:", os.sep, "LocalDreamZImage", "scripts",
                                     "check_ctx_identity.py"),
                    "--expect-arch", "79", "--expect-vtcm", "8", p])


if __name__ == "__main__":
    main()
