# -*- coding: utf-8 -*-
"""#86 步 4~6：把手术后的 ONNX 转成 fp32 DLC -> 量化 -> 建 context。

配方**照抄 seg_fp16_build.py**（同一条已交付的链路），只换源 ONNX 与输入形状。
🔴 transformer 重量化**不需要校准数据**（#134 实测：encoding 全来自 overrides JSON，
   而该 JSON 只有 per-tensor scale/offset、无形状信息 => 换尺寸可直接沿用）。
🔴 建 context 必带 --config_file（#95），否则静默编成 dspArch 68 / vtcm 4MB。

用法: python aspect_build.py 1152x896 [--only part1a] [--skip-ctx]
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import canonical_sources as cs

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
PY = sys.executable
TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qairt_tool.py")
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
D = cs.ONNX_DIR

# 段 -> (手术产物主干, overrides 文件, 输入形状生成器)
SEGS = ["part1a", "part1b", "part2a", "part2b", "vae"]


def shapes(seg, gh, gw, lh, lw, cap, tok):
    uni = tok + cap
    if seg == "part1a":
        return [("latents", "1,16,%d,%d" % (lh, lw)), ("timestep", "1"),
                ("caption", "1,%d,2560" % cap), ("cap_pad_mask", "1,%d" % cap)]
    if seg == "part1b":
        return [("add_138", "1,%d,3840" % uni), ("add_131", "1,1,3840"),
                ("tanh_19", "1,1,3840"), ("select_45", "1,%d,1,64" % uni),
                ("select_46", "1,%d,1,64" % uni), ("adaln_input", "1,256")]
    if seg == "part2a":
        return [("unified", "1,%d,3840" % uni), ("unified_mask", "1,%d" % uni),
                ("unified_freqs", "1,%d,64,2" % uni), ("adaln_input", "1,256")]
    if seg == "part2b":
        return [("add_92", "1,%d,3840" % uni), ("select", "1,%d,1,64" % uni),
                ("select_1", "1,%d,1,64" % uni), ("split_7_split_2", "1,1,3840"),
                ("split_7_split_3", "1,1,3840"), ("val_105", "1,1,1,%d" % uni),
                ("adaln_input", "1,256")]
    return [("vae_latents", "1,16,%d,%d" % (lh, lw))]


def stem_for(seg, tag):
    # 🔴 2026-09-02：改为对接 dit_aspect_surgery.py 的产物命名。
    # 原来是路线 A 的 `<L32 主干>_ar_<tag>`，那套产物已随清理回收，且**源是 L=32**（已过期）。
    if seg == "vae":
        return "vae_decoder_%s" % tag
    return "%s_%s" % (os.path.splitext(os.path.basename(
        cs.onnx_for(seg, "deployed", L=cs.DEPLOYED_L)))[0], tag)


def sh(tag, cmd, timeout):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        err = [l for l in out.splitlines() if "ERROR" in l or "error" in l][:3]
        sys.exit("FAIL %s rc=%d (%.1f min)\n%s" % (tag, r.returncode,
                                                   (time.time() - t0) / 60,
                                                   "\n".join(err) or out[-700:]))
    print("     %s OK %.1f 分钟" % (tag, (time.time() - t0) / 60), flush=True)
    return out


def main():
    tag = sys.argv[1]
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    skip_ctx = "--skip-ctx" in sys.argv
    skip_build = "--skip-build" in sys.argv     # 只跳建图，仍做量化
    segs = SEGS
    if "--segs" in sys.argv:
        segs = sys.argv[sys.argv.index("--segs") + 1].split(",")
    w, h = [int(x) for x in tag.lower().split("x")]
    lh, lw = h // 8, w // 8
    gh, gw = lh // 2, lw // 2
    tok = gh * gw
    cap = cs.DEPLOYED_L                        # 🔴 2026-09-02：原写死 32（L=32 时代）
    os.makedirs(OUT, exist_ok=True)
    print("目标 %s  latent %dx%d  token %d  unified %d" % (tag, lh, lw, tok, tok + cap))

    for seg in segs:
        if only and seg != only:
            continue
        stem = stem_for(seg, tag)
        src = os.path.join(D, stem + ".onnx")
        if not os.path.isfile(src):
            sys.exit("FAIL 手术产物不存在: %s（先跑 aspect_surgery.py）" % src)
        print("\n--- %s  <- %s.onnx ---" % (seg, stem), flush=True)
        fdlc = os.path.join(OUT, "%s_%s_fp32.dlc" % (seg, tag))
        cmd = [PY, TOOL, "qairt-converter", "--input_network", src,
               "--output_path", fdlc, "--float_bitwidth", "16"]
        # 🔴 2026-09-02：原来指向 L=32 时代的 ovr（#148 形态：文档/脚本落后于交付）
        ov = os.path.join(P2, "%s_fp16_L%d_ovr.json" % (seg, cs.DEPLOYED_L))
        if seg != "vae" and os.path.isfile(ov):
            cmd += ["--quantization_overrides", ov]
            print("     overrides: %s" % os.path.basename(ov))
        for n, d in shapes(seg, gh, gw, lh, lw, cap, tok):
            cmd += ["--source_model_input_shape", n, d]
        if not os.path.isfile(fdlc):
            sh("转换", cmd, 10800)
        print("     fp32 DLC %.1f MB" % (os.path.getsize(fdlc) / 1e6), flush=True)
        if skip_ctx:
            continue

        # 量化：配方逐字照抄 seg_fp16_build.py（同一条已交付链路）
        qdlc = os.path.join(OUT, "%s_%s_quantized.dlc" % (seg, tag))
        if not os.path.isfile(qdlc):
            sh("量化", [PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc,
                        "--output_dlc", qdlc, "--weights_bitwidth", "8",
                        "--bias_bitwidth", "32",
                        "--param_quantizer_calibration", "min-max",
                        "--use_per_row_quantization", "--keep_weights_quantized",
                        "--enable_float_fallback"], 14400)
        print("     量化 DLC %.1f MB" % (os.path.getsize(qdlc) / 1e6), flush=True)
        if skip_build:
            continue

        # 建 context：🔴 必带 --config_file（#95），否则静默编成 dspArch 68 / vtcm 4MB
        od = os.path.join(OUT, "ctx_%s_%s" % (seg, tag))
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        det, ext = os.path.join(od, "d.json"), os.path.join(od, "e.json")
        json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]},
                  open(det, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": det}}, open(ext, "w"), indent=1)
        o = sh("建图", [os.path.join(BIN, "qnn-context-binary-generator.exe"),
                        "--backend", os.path.join(LIB, "QnnHtp.dll"),
                        "--dlc_path", qdlc, "--binary_file", "%s_%s" % (seg, tag),
                        "--output_dir", od, "--htp_socs", "sm8750",
                        "--config_file", ext], 21600)
        if "available PD" in o:
            sys.exit("FAIL %s 撞 PD 红线（#57）" % seg)
        b = [x for x in os.listdir(od) if x.endswith(".bin")]
        if not b:
            sys.exit("FAIL %s 未产出 .bin" % seg)
        print("     context %s  %.1f MB"
              % (b[0], os.path.getsize(os.path.join(od, b[0])) / 1e6), flush=True)


if __name__ == "__main__":
    main()
