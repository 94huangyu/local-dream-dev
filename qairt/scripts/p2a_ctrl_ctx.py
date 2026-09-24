"""part2a 手术式 FP16：建 context。🔴 必带 --config_file（#95）；照抄部署配方（只有 devices 段）。"""
import os, sys, json, shutil, subprocess, time
sys.stdout.reconfigure(encoding="utf-8")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN, LIB = os.path.join(SDK, "bin", "x86_64-windows-msvc"), os.path.join(SDK, "lib", "x86_64-windows-msvc")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
DLC, OUT = os.path.join(W, "part2a_ctrl_quantized.dlc"), os.path.join(W, "ctx_ctrl")
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
det, ext = os.path.join(OUT, "d.json"), os.path.join(OUT, "e.json")
json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]}, open(det, "w"), indent=1)
json.dump({"backend_extensions": {"shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
                                  "config_file_path": det}}, open(ext, "w"), indent=1)
t0 = time.time()
print("[ctxgen] %s" % time.strftime("%H:%M:%S"), flush=True)
r = subprocess.run([os.path.join(BIN, "qnn-context-binary-generator.exe"),
                    "--backend", os.path.join(LIB, "QnnHtp.dll"), "--dlc_path", DLC,
                    "--binary_file", "part2a_ctrl", "--output_dir", OUT,
                    "--htp_socs", "sm8750", "--config_file", ext],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=14400)
out = (r.stdout or "") + (r.stderr or "")
open(os.path.join(W, "03_ctx_p2a_ctrl.log"), "w", encoding="utf-8").write(out)
print("[ctxgen] rc=%d  %.1f 分钟" % (r.returncode, (time.time() - t0) / 60), flush=True)
pd = [l for l in out.splitlines() if "available PD" in l or "context size estimate" in l]
if pd:
    for l in pd[:3]:
        print("  🔴 G2 容量门: " + l.strip()[:180])
    sys.exit("❌ G2 撞 PD 红线 ⇒ 按方案直接关闭")
b = [x for x in os.listdir(OUT) if x.endswith(".bin")]
if not b:
    for l in [x for x in out.splitlines() if "ERROR" in x][:5]:
        print("  " + l.strip()[:180])
    sys.exit("❌ 未产出 context")
p = os.path.join(OUT, b[0])
print("  ✅ G2 PASS: %s  %.1f MB （参照臂 part2a_fixed_ctx = 1445.1 MB）"
      % (b[0], os.path.getsize(p) / 1e6))
subprocess.run([sys.executable, os.path.join("D:", os.sep, "LocalDreamZImage", "scripts",
                                             "check_ctx_identity.py"),
                "--expect-arch", "79", "--expect-vtcm", "8", p])
