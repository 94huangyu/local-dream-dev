"""EXP_PLAN_ACTFP16 第 3 步：建 context（G2 容量门 + G3 装置门）。

🔴 配方照抄 o_dlbc_probe.py 的订正版：必带 --config_file，detail 里
   devices:[{soc_model:69,dsp_arch:"v79"}]（#95：不带会静默编成 dspArch 68 / vtcm 4MB）。
🔴 graph_names 必须是 **converter 产出的 fp32 DLC 文件名主干**（#93），填错整段落空且不报错。
🔴 G2 容量门：撞 `Failed to find available PD` ⇒ 按方案 §六 **直接关闭本形态**，不重试不调参。
"""
import os, sys, json, shutil, subprocess, time
sys.stdout.reconfigure(encoding="utf-8")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "actfp16")
DLC = os.path.join(W, "part1b_actfp16_quantized.dlc")
GN = "part1b_actfp16_fp32"          # = converter 产出的 fp32 DLC 文件名主干
OUT = os.path.join(W, "ctx")


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    det = os.path.join(OUT, "detail.json")
    ext = os.path.join(OUT, "ext.json")
    # 🔴 #96：照抄参照臂（perrow/cfg.json）的**同一形状** —— 只有 devices、没有 graphs 段。
    #    参照臂实测画像 dspArch=79 / vtcmSize=8 / spillFill=222822400。
    #    我们不设任何 per-graph 后端选项，所以 graphs 段无用；带上它就是"自己拼一条新命令"。
    json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]}, open(det, "w"), indent=1)
    json.dump({"backend_extensions": {
        "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
        "config_file_path": det}}, open(ext, "w"), indent=1)
    t0 = time.time()
    print("[ctxgen] 开始 %s" % time.strftime("%H:%M:%S"), flush=True)
    r = subprocess.run([os.path.join(BIN, "qnn-context-binary-generator.exe"),
                        "--backend", os.path.join(LIB, "QnnHtp.dll"),
                        "--dlc_path", DLC, "--binary_file", "part1b_actfp16",
                        "--output_dir", OUT, "--htp_socs", "sm8750",
                        "--config_file", ext],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=14400)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(W, "03_ctxgen.log"), "w", encoding="utf-8").write(out)
    print("[ctxgen] rc=%d  %.1f 分钟" % (r.returncode, (time.time() - t0) / 60), flush=True)
    # G2 容量门
    pd = [l for l in out.splitlines() if "available PD" in l or "context size estimate" in l]
    if pd:
        for l in pd[:3]:
            print("  🔴 G2 容量门: " + l.strip()[:180])
        sys.exit("❌ G2 撞 PD 红线 ⇒ 按方案 §六 直接关闭本形态，不重试")
    b = [f for f in os.listdir(OUT) if f.endswith(".bin")]
    if not b:
        for l in [x for x in out.splitlines() if "rror" in x][:5]:
            print("  " + l.strip()[:180])
        sys.exit("❌ 未产出 context")
    p = os.path.join(OUT, b[0])
    print("  ✅ %s  %.1f MB" % (b[0], os.path.getsize(p) / 1e6))
    print("     参照臂 perrow ctx = 1472.86 MB, spillFillBufferSize = 222822400 (222.8 MB)")
    print("\n下一步 G3 装置门：")
    print("  python scripts/check_ctx_identity.py --expect-arch 79 --expect-vtcm 8 %s" % p)


if __name__ == "__main__":
    main()
