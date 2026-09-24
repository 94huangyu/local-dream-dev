# -*- coding: utf-8 -*-
"""查清 qnn-net-run 对 `.bin` context 的 I/O 类型契约（约束 3 的静默缩批陷阱就在这里）。

臂：
  I  part1b + **float32 尺寸**输入（无 native 标志）      -> 输出尺寸/耗时
  J  part1b + 原生输入 + 原生输出                          -> 输出尺寸/耗时
  K  part1a + **float32 尺寸**输入（无 native 标志）        -> 是否还 rc=17
判据（事前）：
  · 若 I 的输出 = 64,143,360（float32 全尺寸）且耗时 ≈ 2 × C 臂 ⇒ **C 臂确实只跑了一半**，
    「输出字节数 == 契约」这条检查被巧合骗过 ⇒ 波及 #147 的离线测速数字。
  · 若 I 输出仍是 32,071,680 ⇒ 另有机制，不得下缩批结论。
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\LocalDreamZImage\scripts")
import lab_dev
import p2_d1_profile as P

OUT = os.path.join(P.OUT, "iotype")
os.makedirs(OUT, exist_ok=True)


def make_float_inputs(seg):
    """按契约元素数造 float32 输入（值取该张量 encoding 量程内的随机数；只测尺寸/耗时，不看数值）。"""
    d = os.path.join(OUT, "fin_%s" % seg)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(20260918)
    names = []
    for t in P.contract_graph(seg)["inputs"]:
        q = t["quantization"]
        n_elem = t["exact_bytes"] // (2 if t["physical_dtype"] == "QNN_DATATYPE_UFIXED_POINT_16" else 1)
        if t["physical_dtype"] == "QNN_DATATYPE_BOOL_8":
            v = np.ones(n_elem, dtype=np.float32)
            v[:22] = 0.0
        else:
            lo = q["offset"] * q["scale"]
            hi = (65535 + q["offset"]) * q["scale"]
            v = rng.uniform(lo, hi, n_elem).astype(np.float32)
        p = os.path.join(d, t["name"] + ".raw")
        v.tofile(p)
        assert os.path.getsize(p) == n_elem * 4
        names.append((t["name"], n_elem * 4))
    return d, names


def run(tag, seg, indir, names, extra):
    od = "%s/o_%s" % (P.D, tag)
    line = " ".join("%s:=%s/%s/%s.raw" % (k, P.D, indir, k) for k, _ in names)
    lab_dev.sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' '%s' > %s/list.txt" % (od, od, line, od))
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context %s/%s.bin --backend libQnnHtp.so --input_list %s/list.txt "
           "--output_dir %s --log_level error --num_inferences 1 --perf_profile burst %s; echo RC=$?"
           % (P.T, P.T, P.T, P.D, seg, od, od, extra))
    t0 = time.time()
    o = lab_dev.sh(cmd, timeout=1800)
    el = time.time() - t0
    rc = re.search(r"RC=(\d+)", o)
    rc = rc.group(1) if rc else "?"
    ls = lab_dev.sh("for f in %s/Result_0/*; do stat -c '%%n %%s' $f; done 2>/dev/null; true" % od)
    got = {}
    for l in ls.strip().splitlines():
        p = l.split()
        if len(p) == 2 and p[0].endswith(".raw"):
            got[os.path.basename(p[0])[:-4]] = int(p[1])
    want = {t["name"]: t["exact_bytes"] for t in P.contract_graph(seg)["outputs"]}
    print("\n== %s seg=%s extra=%r rc=%s %.1fs" % (tag, seg, extra, rc, el))
    for n, w in sorted(want.items()):
        g = got.get(n)
        tagv = "原生" if g == w else ("float32(=契约x2)" if g == w * 2 else ("半尺寸!" if g == w // 2 else str(g)))
        print("   %-15s 契约 %10d 实际 %10s  %s" % (n, w, g, tagv))
    return {"tag": tag, "seg": seg, "extra": extra, "rc": rc, "sec": el, "out": got}


def main():
    lab_dev.require_online("iotype")
    print("MemAvailable %d MiB, NPU %.1f °C" % (lab_dev.mem_available_mb(), P.npu_c()))
    print("\n== G/H 输出目录到底有什么 ==")
    for tag in ("G_p1a_bothnative", "H_p1b_bothnative"):
        print(" %s: %s" % (tag, " ".join(lab_dev.sh("ls -R %s/o_%s 2>/dev/null | head -20; true" % (P.D, tag)).split())[:400]))
    res = []
    for seg in ("part1b", "part1a"):
        d, names = make_float_inputs(seg)
        rel = "fin_%s" % seg
        lab_dev.sh("mkdir -p %s/%s" % (P.D, rel))
        for n, nb in names:
            dev = "%s/%s/%s.raw" % (P.D, rel, n)
            have = lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % dev).strip()
            if have != str(nb):
                lab_dev.push(os.path.join(d, n + ".raw"), dev)
        print("  %s float32 输入已备：%s" % (seg, ", ".join("%s=%d" % x for x in names)))
    res.append(run("I_p1b_float", "part1b", "fin_part1b", make_float_inputs("part1b")[1], ""))
    res.append(run("J_p1b_native_both", "part1b", "in_part1b",
                   [(t["name"], t["exact_bytes"]) for t in P.contract_graph("part1b")["inputs"]],
                   "--use_native_input_files --use_native_output_files"))
    res.append(run("K_p1a_float", "part1a", "fin_part1a", make_float_inputs("part1a")[1], ""))
    json.dump(res, open(os.path.join(OUT, "iotype.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n== 汇总 ==")
    for r in res:
        print("  %-18s rc=%-3s %6.1fs" % (r["tag"], r["rc"], r["sec"]))


if __name__ == "__main__":
    main()
