# -*- coding: utf-8 -*-
"""EXP_PLAN_ODLBC_SPEED 阶段 1：建 part1b 的 CTRL / O3 / DLBC 三臂 context。

配方**照抄 seg_fp16_build.py 的建图段**（§7.2：要与部署对照就照抄部署的构建命令），
只在 detail json 里多一个 graphs 段。选项键名照抄 o_dlbc_probe.py（已验证生效）。

🔴 必带 --config_file，否则静默编成 dspArch 68 / vtcm 4MB（#95）。
🔴 graph_names 必须是 **converter 产出的 fp32 DLC 文件名主干**（#93），填错整段落空且不报错。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "odlbc_speed")
QDLC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr",
                    "part1b_fp16_L80_quantized.dlc")
GN = "part1b_fp16_L80_fp32"          # converter 产出的 fp32 DLC 主干名
DEPLOYED = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr",
                        "ctx_part1b_fp16_L80", "part1b_fp16_L80.SM8750.bin")

ARMS = {
    "CTRL": {"vtcm_mb": 8},
    "O3":   {"vtcm_mb": 8, "O": 3},
    "DLBC": {"vtcm_mb": 8, "dlbc": 1},
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert os.path.isfile(QDLC), "量化 DLC 不在: %s" % QDLC
    print("源 DLC   %s  (%.1f MB)" % (QDLC, os.path.getsize(QDLC) / 1e6), flush=True)
    print("graph_names = %r" % GN, flush=True)
    if os.path.isfile(DEPLOYED):
        print("部署版 part1b  %.1f MB  md5=%s" %
              (os.path.getsize(DEPLOYED) / 1e6, md5(DEPLOYED)), flush=True)
    os.makedirs(W, exist_ok=True)

    out = {}
    for tag, opts in ARMS.items():
        od = os.path.join(W, tag)
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        g = {"graph_names": [GN]}
        g.update(opts)
        det = os.path.join(od, "d.json")
        ext = os.path.join(od, "e.json")
        json.dump({"graphs": [g], "devices": [{"soc_model": 69, "dsp_arch": "v79"}]},
                  open(det, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": det}}, open(ext, "w"), indent=1)

        print("\n=== 建 %s  opts=%r ===" % (tag, opts), flush=True)
        cmd = [os.path.join(BIN, "qnn-context-binary-generator.exe"),
               "--backend", os.path.join(LIB, "QnnHtp.dll"),
               "--dlc_path", QDLC, "--binary_file", "part1b_%s" % tag,
               "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ext]
        log = os.path.join(od, "build.log")
        with open(log, "w", encoding="utf-8", errors="replace") as fo:
            # 🔴 不套管道、不吞返回码（§7.4）
            r = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, timeout=14400)
        txt = open(log, encoding="utf-8", errors="replace").read()
        if "available PD" in txt:
            print("  ❌ G4: 撞 PD 红线（#57）", flush=True)
            continue
        if r.returncode != 0:
            err = [l for l in txt.splitlines() if "ERROR" in l][:2]
            print("  ❌ rc=%d  %s" % (r.returncode, " | ".join(e[:100] for e in err)), flush=True)
            continue
        bins = [f for f in os.listdir(od) if f.endswith(".bin")]
        if not bins:
            print("  ❌ 未产出 .bin", flush=True)
            continue
        p = os.path.join(od, bins[0])
        out[tag] = p
        print("  OK %s  %.1f MB  md5=%s" %
              (bins[0], os.path.getsize(p) / 1e6, md5(p)), flush=True)
        subprocess.run([sys.executable,
                        os.path.join("D:", os.sep, "LocalDreamZImage", "scripts",
                                     "check_ctx_identity.py"),
                        "--expect-arch", "79", "--expect-vtcm", "8", p])

    # ---- V0 生效门 ----
    print("\n=== V0 生效门（md5 必须各不相同，#93：填错 graph_names 会静默落空）===", flush=True)
    if "CTRL" not in out:
        sys.exit("🔴 CTRL 未建成，无法判 V0")
    c = md5(out["CTRL"])
    ok = True
    for tag in ("O3", "DLBC"):
        if tag not in out:
            print("  %-5s 未建成 ⇒ 退出判定" % tag)
            ok = False
            continue
        m = md5(out[tag])
        good = m != c
        ok = ok and good
        print("  %-5s md5 %s CTRL  ⇒ %s" % (tag, "≠" if good else "==",
                                            "PASS" if good else "🔴 FAIL 选项未生效"))
    print("\nV0 %s" % ("PASS —— 可进入设备测速" if ok else
                       "🔴 FAIL —— 先修 graph_names，不得记为「无效果」（执行错≠方向错）"))
    print("产物目录 %s" % W)


if __name__ == "__main__":
    main()
