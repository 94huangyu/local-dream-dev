# -*- coding: utf-8 -*-
"""几何扫描：找一个 spill-fill 正常的比例。

## 已知（2026-09-02 实测）
part2a **只看序列长度 N**（它没有网格概念）。两点：
  N=4176（现网 1:1）  spillFill **276.9 MiB**  ✅
  N=3968（1152x864）  spillFill **999.0 MiB**  ❌
旋钮已排除：`vtcm_mb` 8 / 4 / 不设 —— **三者产物 spillFill 逐字节相同（999.0）**
⇒ 与 #34「`.bin` 路径下 vtcm_mb 无效果」一致，该旋钮关闭。

## 宿主侧 ION 预测器（已验证，误差 6%）
`tools.html` 的公式：
  RAM = opDataSize + constSize + ddrTensorSize + spillFillBufferSize + ioTensorSize + vtcmSize
        （+ sharedWeightsSize，多图只算一次）
实测校验：part2a 1:1 算 1832 / 实测 ION 1948（1.06）；4:3 算 2602 / 实测 2749（1.06）。
⇒ 换几何**不必占设备**。

## 判据（执行前锁定）
spillFill < 400 MiB 且 预测 ION <= 1.15 x 现网单图（part2a 1990 MiB ⇒ <= 2289）。

用法: python geom_sweep.py <W>x<H> [<W>x<H> ...]
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
AS = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
BIN = os.path.join("D:", os.sep, "qairt", "2.48.0.260626", "bin", "x86_64-windows-msvc")
SEG = "part2a"
DEPLOYED_ION = 1990.0          # #145 实测 part2a 单图 ION (MiB)


def run(name, cmd, timeout=28800):
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace", timeout=timeout)
    print("    [%s] rc=%d %.1f 分钟" % (name, r.returncode, (time.time() - t0) / 60.0), flush=True)
    if r.returncode:
        print((r.stdout or "")[-1200:])
    return r.returncode


def meta(binp, od):
    jf = os.path.join(od, "info.json")
    subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                    "--context_binary", binp, "--json_file", jf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(jf):
        return {}
    t = open(jf, encoding="utf-8", errors="replace").read()
    out = {}
    for k in ("opDataSize", "constSize", "ddrTensorSize", "spillFillBufferSize",
              "ioTensorSize", "vtcmSize", "sharedWeightsSize"):
        m = re.search(r'"%s"\s*:\s*(\d+)' % k, t)
        if m:
            out[k] = int(m.group(1))
    return out


def main():
    rows = []
    for tag in sys.argv[1:]:
        w, h = [int(x) for x in tag.lower().split("x")]
        lh, lw = h // 8, w // 8
        gh, gw = lh // 2, lw // 2
        N = 80 + gh * gw
        print("\n" + "=" * 62, flush=True)
        print("候选 %s : latent %dx%d | 网格 %dx%d | tokens %d | N=%d | 比例 %.4f"
              % (tag, lh, lw, gh, gw, gh * gw, N, w / float(h)), flush=True)
        print("=" * 62, flush=True)
        if run("手术", [PY, os.path.join(HERE, "dit_aspect_surgery.py"), tag,
                        str(w), str(h), "--variant=deployed"], 3600):
            continue
        if run("转换+量化+建图", [PY, os.path.join(HERE, "aspect_build.py"), tag,
                                  "--segs", SEG]):
            continue
        od = os.path.join(AS, "ctx_%s_%s" % (SEG, tag))
        b = [f for f in os.listdir(od) if f.endswith(".bin")]
        if not b:
            print("    无产物"); continue
        m = meta(os.path.join(od, b[0]), od)
        sf = m.get("spillFillBufferSize", 0) / 1048576.0
        ram = sum(m.get(k, 0) for k in ("opDataSize", "constSize", "ddrTensorSize",
                                        "spillFillBufferSize", "ioTensorSize",
                                        "sharedWeightsSize")) / 1048576.0 + m.get("vtcmSize", 0)
        pred = ram * 1.06
        ok = sf < 400 and pred <= DEPLOYED_ION * 1.15
        print("    spillFill = %.1f MiB | opData %.1f | const %d | 预测 ION %.0f MiB (判据 <= %.0f)  %s"
              % (sf, m.get("opDataSize", 0) / 1048576.0, m.get("constSize", 0),
                 pred, DEPLOYED_ION * 1.15, "🟢 PASS" if ok else "🔴 FAIL"), flush=True)
        rows.append((tag, N, sf, pred, ok))
        json.dump([{"tag": t, "N": n, "spill_MiB": s, "pred_ion": p, "ok": o}
                   for t, n, s, p, o in rows],
                  open(os.path.join(AS, "geom_sweep.json"), "w"), indent=1)

    print("\n" + "=" * 62, flush=True)
    print("汇总（对照：N=4176 现网 spill 276.9 / ION 1990；N=3968 spill 999.0 / ION 2749）", flush=True)
    for t, n, s, p, o in rows:
        print("  %-10s N=%-6d spill %7.1f MiB  预测 ION %6.0f  %s"
              % (t, n, s, p, "PASS" if o else "FAIL"), flush=True)
    return 0


sys.exit(main())
