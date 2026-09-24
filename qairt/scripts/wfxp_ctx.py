"""EXP_PLAN_WFXP_ACTFP 阶段 1 收尾：建两臂 context 并过 V0 装置门。

🔴 配方照抄 scripts/o_dlbc_probe.py 的订正版（带 --config_file，detail 里有
   devices:[{soc_model:69,dsp_arch:"v79"}]）。**不照抄 p0b_build.py**（缺 --config_file = #95）。
🔴 两臂用**同一条命令**，只换 --dlc_path（#96：要对照就照抄同一配方）。
"""
import os, sys, json, shutil, subprocess, hashlib
sys.stdout.reconfigure(encoding="utf-8")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
OUT = os.path.join(P0B, "wfxp", "ctx")
ARMS = {"wfxp":   os.path.join(P0B, "wfxp", "kwq_f16_ffb", "fc99_quantized.dlc"),
        "perrow": os.path.join(P0B, "standalone_perrow", "fc99_quantized.dlc")}
GN = "fc99_fp32"     # #93：必须是 converter 产出的 fp32 DLC 文件名主干


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    made = {}
    for tag, dlc in ARMS.items():
        if not os.path.exists(dlc):
            print("  %-7s ❌ 缺 DLC %s" % (tag, dlc)); continue
        det = os.path.join(OUT, "d_%s.json" % tag)
        ext = os.path.join(OUT, "e_%s.json" % tag)
        json.dump({"graphs": [{"graph_names": [GN]}],
                   "devices": [{"soc_model": 69, "dsp_arch": "v79"}]},
                  open(det, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": det}}, open(ext, "w"), indent=1)
        od = os.path.join(OUT, tag)
        os.makedirs(od, exist_ok=True)
        r = subprocess.run([os.path.join(BIN, "qnn-context-binary-generator.exe"),
                            "--backend", os.path.join(LIB, "QnnHtp.dll"),
                            "--dlc_path", dlc, "--binary_file", "c_" + tag,
                            "--output_dir", od, "--htp_socs", "sm8750",
                            "--config_file", ext],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
        b = [f for f in os.listdir(od) if f.endswith(".bin")]
        if not b:
            err = [l for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l][:2]
            print("  %-7s ❌ 未产出 rc=%d %s" % (tag, r.returncode, err)); continue
        p = os.path.join(od, b[0])
        made[tag] = p
        print("  %-7s ✅ %s  %.2f MB" % (tag, b[0], os.path.getsize(p) / 1e6))
    json.dump(made, open(os.path.join(OUT, "made.json"), "w"), indent=1)
    print("\n产物:", json.dumps(made, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
