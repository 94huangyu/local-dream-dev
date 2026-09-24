# -*- coding: utf-8 -*-
"""EXP_PLAN_ODLBC_SPEED 阶段 2：四段全部用 `O:3` 重建 context。

只建 O3 一臂 —— CTRL 就是部署版本身（#147 实测 md5 逐字节相同），不必重建。
DLBC 已在阶段 1 出局（1.008×，无差异），不建。

🔴 必带 --config_file（#95），graph_names 必须是 converter 产出的 fp32 DLC 主干（#93）。
🔴 顺序建，不并行 —— 单臂实测 RSS 约 5.8 GB，并行会压死宿主（约束 9 清单第 7 条）。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "o3_all")
SEGS = ["part1a", "part1b", "part2a", "part2b"]
# 部署版（= CTRL）用于体积对照
DEPLOY = {s: os.path.join(SRC, "ctx_%s_fp16_L80" % s, "%s_fp16_L80.SM8750.bin" % s)
          for s in SEGS}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(W, exist_ok=True)
    print("=== O3 四段建图（顺序，预计 90~120 分钟）===", flush=True)
    t_all = time.time()
    for seg in SEGS:
        qdlc = os.path.join(SRC, "%s_fp16_L80_quantized.dlc" % seg)
        gn = "%s_fp16_L80_fp32" % seg
        assert os.path.isfile(qdlc), "缺 %s" % qdlc
        od = os.path.join(W, seg)
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        det = os.path.join(od, "d.json")
        ext = os.path.join(od, "e.json")
        json.dump({"graphs": [{"graph_names": [gn], "vtcm_mb": 8, "O": 3}],
                   "devices": [{"soc_model": 69, "dsp_arch": "v79"}]},
                  open(det, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": det}}, open(ext, "w"), indent=1)

        print("\n--- %s  graph_names=%r ---" % (seg, gn), flush=True)
        t0 = time.time()
        log = os.path.join(od, "build.log")
        with open(log, "w", encoding="utf-8", errors="replace") as fo:
            r = subprocess.run(
                [os.path.join(BIN, "qnn-context-binary-generator.exe"),
                 "--backend", os.path.join(LIB, "QnnHtp.dll"),
                 "--dlc_path", qdlc, "--binary_file", "%s_O3" % seg,
                 "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ext],
                stdout=fo, stderr=subprocess.STDOUT, timeout=21600)
        el = (time.time() - t0) / 60.0
        txt = open(log, encoding="utf-8", errors="replace").read()
        if "available PD" in txt:
            print("  ❌ 撞 PD 红线（#57）—— 该段 O3 不可用", flush=True)
            continue
        if r.returncode != 0:
            err = [l for l in txt.splitlines() if "ERROR" in l][:2]
            print("  ❌ rc=%d  %s" % (r.returncode, " | ".join(e[:110] for e in err)), flush=True)
            continue
        bins = [f for f in os.listdir(od) if f.endswith(".bin")]
        if not bins:
            print("  ❌ 未产出 .bin", flush=True)
            continue
        p = os.path.join(od, bins[0])
        sz = os.path.getsize(p)
        base = os.path.getsize(DEPLOY[seg]) if os.path.isfile(DEPLOY[seg]) else 0
        print("  OK %s  %.2f MB  用时 %.1f 分钟" % (bins[0], sz / 1e6, el), flush=True)
        if base:
            print("     部署版 %.2f MB ⇒ 体积 %+.2f%%（ION 会同比上升，直接影响 #145/#146）"
                  % (base / 1e6, 100.0 * (sz - base) / base), flush=True)
        print("     md5 %s" % md5(p), flush=True)
        subprocess.run([sys.executable,
                        os.path.join("D:", os.sep, "LocalDreamZImage", "scripts",
                                     "check_ctx_identity.py"),
                        "--expect-arch", "79", "--expect-vtcm", "8", p])

    print("\n=== 全部结束，总用时 %.1f 分钟 ===" % ((time.time() - t_all) / 60.0))
    print("产物 %s" % W)
    tot = 0
    for seg in SEGS:
        od = os.path.join(W, seg)
        b = [f for f in os.listdir(od) if f.endswith(".bin")] if os.path.isdir(od) else []
        if b:
            tot += os.path.getsize(os.path.join(od, b[0]))
    base = sum(os.path.getsize(DEPLOY[s]) for s in SEGS if os.path.isfile(DEPLOY[s]))
    if tot and base:
        print("四段合计 %.2f GB vs 部署 %.2f GB ⇒ %+.2f%%"
              % (tot / 1e9, base / 1e9, 100.0 * (tot - base) / base))


if __name__ == "__main__":
    main()
