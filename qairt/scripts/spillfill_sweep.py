# -*- coding: utf-8 -*-
"""扫 part2a 的 spill-fill：先试旋钮，再试几何。

背景（2026-09-02 实测）：part2a/2b 的 4:3 图 spillFillBufferSize 从现网的
276.9 / 255.3 MiB 跳到 **999 MiB**，导致设备 ION 超判据 1.38x / 1.44x。
part1a/1b 做同样的序列改写反而降了 => 不是序列长度本身。

🔴 先查旋钮而不是先改产品几何（改几何要改用户可见的输出尺寸，代价高得多）。
   `aspect_build.py` 与 `seg_fp16_build.py` 的建图配置都**只有 devices、没有 graphs 段**，
   而 #93 记过没有 `graphs` 段会让 `vtcm_mb` 等**静默失效**。

判据（事前锁定）：spillFillBufferSize < 400 MiB 视为回到正常区间
（现网 1:1 是 276.9 MiB；part1a 的 4:3 是 290.6 MiB，都在此线内）。

用法: python spillfill_sweep.py <臂名> [臂名...]
      臂名: novtcm | vtcm8 | vtcm4 | o2 | o3
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
AS = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "sfsweep")
SEG, TAG = "part2a", "1152x864"
GN = "%s_%s_fp32" % (SEG, TAG)
QDLC = os.path.join(AS, "%s_%s_quantized.dlc" % (SEG, TAG))

ARMS = {
    "novtcm": None,                                  # 无 graphs 段（= 现有装置）
    "vtcm8": {"graph_names": [GN], "vtcm_mb": 8},
    "vtcm4": {"graph_names": [GN], "vtcm_mb": 4},
    "o2": {"graph_names": [GN], "vtcm_mb": 8, "O": 2},
    "o3": {"graph_names": [GN], "vtcm_mb": 8, "O": 3},
}


def spillfill(binp, od):
    jf = os.path.join(od, "info.json")
    subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                    "--context_binary", binp, "--json_file", jf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(jf):
        return None
    import re
    t = open(jf, encoding="utf-8", errors="replace").read()
    m = re.search(r'"spillFillBufferSize"\s*:\s*(\d+)', t)
    return int(m.group(1)) if m else None


def main():
    arms = sys.argv[1:] or ["vtcm8"]
    assert os.path.isfile(QDLC), "缺 %s" % QDLC
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for arm in arms:
        assert arm in ARMS, "未知臂 %s" % arm
        od = os.path.join(OUT, arm)
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        det = {"devices": [{"soc_model": 69, "dsp_arch": "v79"}]}
        if ARMS[arm] is not None:
            det["graphs"] = [ARMS[arm]]
        dp, ep = os.path.join(od, "d.json"), os.path.join(od, "e.json")
        json.dump(det, open(dp, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": dp}}, open(ep, "w"), indent=1)
        print("\n--- 臂 %s  d.json=%s ---" % (arm, json.dumps(det, ensure_ascii=False)), flush=True)
        t0 = time.time()
        log = os.path.join(od, "build.log")
        with open(log, "w", encoding="utf-8", errors="replace") as fo:
            r = subprocess.run(
                [os.path.join(BIN, "qnn-context-binary-generator.exe"),
                 "--backend", os.path.join(LIB, "QnnHtp.dll"),
                 "--dlc_path", QDLC, "--binary_file", "%s_%s" % (SEG, arm),
                 "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ep],
                stdout=fo, stderr=subprocess.STDOUT, timeout=28800)
        el = (time.time() - t0) / 60.0
        if r.returncode != 0:
            txt = open(log, encoding="utf-8", errors="replace").read()
            print("    FAIL rc=%d\n%s" % (r.returncode, txt[-900:]), flush=True)
            continue
        b = [f for f in os.listdir(od) if f.endswith(".bin")]
        binp = os.path.join(od, b[0])
        sf = spillfill(binp, od)
        sz = os.path.getsize(binp) / 1048576.0
        res[arm] = sf
        print("    rc=0  %.1f 分钟  产物 %.1f MiB  spillFill = %s MiB  %s"
              % (el, sz, ("%.1f" % (sf / 1048576.0)) if sf else "?",
                 "🟢 回到正常区间" if (sf and sf / 1048576.0 < 400) else "🔴 仍然过大"),
              flush=True)
        json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("\n=== 汇总（判据：< 400 MiB 为正常；现网 1:1 = 276.9，part1a 4:3 = 290.6）===", flush=True)
    for a, v in res.items():
        print("  %-8s %s MiB" % (a, ("%.1f" % (v / 1048576.0)) if v else "?"), flush=True)
    return 0


sys.exit(main())
