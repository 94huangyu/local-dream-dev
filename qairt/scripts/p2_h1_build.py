# -*- coding: utf-8 -*-
"""D2-H1 · S2+S3：把手术后的 part1a ONNX 转成 fp32 DLC -> 量化 -> 建 1:1 单图 context。

🔴 命令逐字照抄部署那条链路（`seg_fp16_build.py` / `aspect_build.py`），只换源 ONNX：
   · 转换：`--float_bitwidth 16` + 部署同一份 overrides（`p2attr/part1a_fp16_L80_ovr.json`）
   · 量化：`--weights_bitwidth 8 --bias_bitwidth 32 --param_quantizer_calibration min-max
            --use_per_row_quantization --keep_weights_quantized --enable_float_fallback`
     （#134：transformer 重量化不需要校准数据，encoding 全来自 overrides ⇒ 确定性）
   · 建图：config **只有 devices 段**（部署那份就是这么建的，读回 vtcm = 8），`--htp_socs sm8750`
🔴 建完读回元数据与部署版逐项比对：dspArch / vtcm / O / dlbc 必须一致，
   只允许 constSize 下降（手术删掉的 STATIC 索引/填充张量）。

每步产物已存在就跳过（可断点续跑）。每步前查宿主可用内存（本项目被打穿过一次）。

用法: python scripts/p2_h1_build.py [--skip-ctx]
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "qairt_tool.py")
ONNX_DIR = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_" + ("noscat" if "--stem" not in sys.argv else sys.argv[sys.argv.index("--stem") + 1].split("_L80_")[-1]))
STEM = "transformer_part1a_clip_L80_noscat"
if "--stem" in sys.argv:
    STEM = sys.argv[sys.argv.index("--stem") + 1]
SRC = os.path.join(ONNX_DIR, STEM + ".onnx")
OVR = os.path.join(P2, "part1a_fp16_L80_ovr.json")
REF_CTX = os.path.join(P2, "ctx_part1a_fp16_L80", "part1a_fp16_L80.SM8750.bin")
SHAPES = [("latents", "1,16,128,128"), ("timestep", "1"),
          ("caption", "1,80,2560"), ("cap_pad_mask", "1,80")]
MIN_FREE_GB = 10.0


def free_gb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                       capture_output=True, text=True, timeout=120)
    try:
        return int(r.stdout.strip()) / 1024.0 / 1024.0
    except ValueError:
        return -1.0


def sh(tag, cmd, timeout):
    f = free_gb()
    print("  [%s] 开始（宿主可用内存 %.1f GB）" % (tag, f), flush=True)
    if 0 < f < MIN_FREE_GB:
        raise SystemExit("🔴 可用内存 %.1f GB < %.1f ⇒ 不启动 %s（本项目被打穿过一次）" % (f, MIN_FREE_GB, tag))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        err = [l for l in out.splitlines() if "ERROR" in l.upper()][:5]
        raise SystemExit("🔴 %s 失败 rc=%d（%.1f 分钟）\n%s" % (tag, r.returncode, (time.time() - t0) / 60,
                                                          "\n".join(err) or out[-1200:]))
    print("  [%s] OK %.1f 分钟" % (tag, (time.time() - t0) / 60), flush=True)
    return out


def ctx_meta(binpath, jf):
    subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                    "--context_binary", binpath, "--json_file", jf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    j = json.load(open(jf, encoding="utf-8", errors="replace"))
    g = j["info"]["graphs"][0]["info"]
    v1, v2 = g["graphBlobInfo"]["info"], g.get("graphBlobInfoV2", {})
    return {"dspArch": j["info"]["contextMetadata"]["info"].get("dspArch"),
            "graphName": g["graphName"], "vtcm": v1["vtcmSize"], "O": v1["optimizationLevel"],
            "dlbc": v1["htpDlbc"], "hvx": v1["numHvxThreads"], "spillFill": v1["spillFillBufferSize"],
            "constMiB": v2.get("constSize", 0) / 2**20, "opDataMiB": v2.get("opDataSize", 0) / 2**20,
            "ioMiB": v2.get("ioTensorSize", 0) / 2**20}


def main():
    for p in (SRC, OVR, REF_CTX):
        if not os.path.isfile(p):
            raise SystemExit("🔴 缺少 %s" % p)
    os.makedirs(OUT, exist_ok=True)
    fdlc = os.path.join(OUT, "fp32.dlc")
    qdlc = os.path.join(OUT, "quantized.dlc")

    if os.path.isfile(fdlc):
        print("  fp32 DLC 已存在，跳过转换（%.2f GB）" % (os.path.getsize(fdlc) / 1e9))
    else:
        cmd = [PY, TOOL, "qairt-converter", "--input_network", SRC, "--output_path", fdlc,
               "--float_bitwidth", "16", "--quantization_overrides", OVR]
        for n, d in SHAPES:
            cmd += ["--source_model_input_shape", n, d]
        sh("S2a 转换", cmd, 14400)
    print("  fp32 DLC %.2f GB" % (os.path.getsize(fdlc) / 1e9), flush=True)

    if os.path.isfile(qdlc):
        print("  量化 DLC 已存在，跳过量化（%.2f GB）" % (os.path.getsize(qdlc) / 1e9))
    else:
        sh("S2b 量化", [PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                        "--weights_bitwidth", "8", "--bias_bitwidth", "32",
                        "--param_quantizer_calibration", "min-max",
                        "--use_per_row_quantization", "--keep_weights_quantized",
                        "--enable_float_fallback"], 21600)
    print("  量化 DLC %.2f GB（部署版 %.2f GB）"
          % (os.path.getsize(qdlc) / 1e9,
             os.path.getsize(os.path.join(P2, "part1a_fp16_L80_quantized.dlc")) / 1e9), flush=True)
    if "--skip-ctx" in sys.argv:
        return 0

    od = os.path.join(OUT, "ctx")
    binp = os.path.join(od, "ctxbin.SM8750.bin")
    if not os.path.isfile(binp):
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        det, ext = os.path.join(od, "d.json"), os.path.join(od, "e.json")
        json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]}, open(det, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": det}}, open(ext, "w"), indent=1)
        o = sh("S3 建图", [os.path.join(BIN, "qnn-context-binary-generator.exe"),
                           "--backend", os.path.join(LIB, "QnnHtp.dll"), "--dlc_path", qdlc,
                           "--binary_file", "ctxbin", "--output_dir", od,
                           "--htp_socs", "sm8750", "--config_file", ext], 21600)
        if "available PD" in o:
            raise SystemExit("🔴 撞 PD 红线（#57）")
    if not os.path.isfile(binp):
        raise SystemExit("🔴 未产出 %s" % binp)

    new = ctx_meta(binp, os.path.join(OUT, "info_noscat.json"))
    ref = ctx_meta(REF_CTX, os.path.join(OUT, "info_deployed.json"))
    print("\n== 建图装置比对（部署版 vs 手术版）==")
    bad = []
    for k in ("dspArch", "vtcm", "O", "dlbc", "hvx"):
        same = ref[k] == new[k]
        print("  %-9s 部署 %-6s 手术 %-6s %s" % (k, ref[k], new[k], "✅" if same else "🔴 不一致"))
        if not same:
            bad.append(k)
    for k in ("constMiB", "opDataMiB", "ioMiB"):
        print("  %-9s 部署 %8.1f  手术 %8.1f  差 %+.1f MiB" % (k, ref[k], new[k], new[k] - ref[k]))
    print("  spillFill 部署 %d  手术 %d" % (ref["spillFill"], new["spillFill"]))
    print("  文件大小  部署 %.3f GB  手术 %.3f GB  差 %+.3f GB"
          % (os.path.getsize(REF_CTX) / 1e9, os.path.getsize(binp) / 1e9,
             (os.path.getsize(binp) - os.path.getsize(REF_CTX)) / 1e9))
    if bad:
        raise SystemExit("🔴 装置不一致（%s）⇒ A/B 就不是单变量，停" % ",".join(bad))
    if new["graphName"] != ref["graphName"]:
        print("  ⚠️ 图名不同：部署 %s / 手术 %s（qnn-net-run 用 --retrieve_context 不需要指定图名，"
              "但记录在案）" % (ref["graphName"], new["graphName"]))
    print("\n✅ S2+S3 完成。下一步 S4：设备离线 A/B（9 个输出逐字节比 sha256 + 加速器执行时间）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
