# -*- coding: utf-8 -*-
"""EXP_PLAN_MULTIGRAPH 阶段 0 · 第 2 步：Windows 三臂 context 建图。

W1 = 单图 A          （体积基准）
W2 = A,B 双图 · weight_sharing_enabled=false （空白对照，R1 要求）
W3 = A,B 双图 · weight_sharing_enabled=true

唯一变量 = context.weight_sharing_enabled。
§7.2 硬规则：必带 --config_file，detail 里 devices 要有 soc_model/dsp_arch。
"""
import os, sys, json, time, shutil, subprocess

SDK = r"D:\qairt\2.48.0.260626"
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
W = r"D:\ZImage_Work\p0_experiments\ws_stage0"
A = os.path.join(W, "wsA_quant.dlc")
B = os.path.join(W, "wsB_quant.dlc")
ARMS = [("W1", [A], None), ("W2", [A, B], False), ("W3", [A, B], True)]


def build(tag, dlcs, ws):
    od = os.path.join(W, tag)
    if os.path.isdir(od):
        shutil.rmtree(od)
    os.makedirs(od)
    det = {"graphs": [{"graph_names": [os.path.basename(d).replace("_quant.dlc", "_fp32")],
                       "vtcm_mb": 8} for d in dlcs],
           "devices": [{"soc_model": 69, "dsp_arch": "v79"}]}
    if ws is not None:
        det["context"] = {"weight_sharing_enabled": ws}
    dp = os.path.join(od, "d.json")
    ep = os.path.join(od, "e.json")
    json.dump(det, open(dp, "w"), indent=1)
    json.dump({"backend_extensions": {
        "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
        "config_file_path": dp}}, open(ep, "w"), indent=1)

    print("\n--- %s  dlcs=%d  weight_sharing=%r ---" % (tag, len(dlcs), ws), flush=True)
    print("    d.json = %s" % json.dumps(det, ensure_ascii=False), flush=True)
    log = os.path.join(od, "build.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fo:
        r = subprocess.run(
            [os.path.join(BIN, "qnn-context-binary-generator.exe"),
             "--backend", os.path.join(LIB, "QnnHtp.dll"),
             "--dlc_path", ",".join(dlcs), "--binary_file", tag,
             "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ep],
            stdout=fo, stderr=subprocess.STDOUT, timeout=7200)
    el = time.time() - t0
    txt = open(log, encoding="utf-8", errors="replace").read()

    # G0-e：记录 weight sharing 相关的报错/告警（Windows 是否静默忽略）
    key = [l.strip() for l in txt.splitlines()
           if any(k in l.lower() for k in
                  ("weight", "shar", "error", "warn", "unsupport", "ignor", "not support"))]
    print("    rc=%d  用时 %.1f s" % (r.returncode, el), flush=True)
    for l in key[:15]:
        print("    | " + l[:200], flush=True)
    if r.returncode != 0:
        print("    ❌ 建图失败，日志尾部：\n%s" % txt[-1200:], flush=True)
        return None

    binp = os.path.join(od, "%s.SM8750.bin" % tag)
    if not os.path.isfile(binp):
        cand = [f for f in os.listdir(od) if f.endswith(".bin")]
        assert len(cand) == 1, "产物不唯一: %r" % cand
        binp = os.path.join(od, cand[0])
    sz = os.path.getsize(binp)
    print("    产物 %s = %d B (%.2f MB)" % (os.path.basename(binp), sz, sz / 1048576.0), flush=True)

    # G0-a：图确实编进去了几个
    jf = os.path.join(od, "info.json")
    r2 = subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                         "--context_binary", binp, "--json_file", jf],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding="utf-8", errors="replace")
    gnames, sfb = [], []
    if r2.returncode == 0 and os.path.isfile(jf):
        d = json.load(open(jf, encoding="utf-8", errors="replace"))
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "graphName" and isinstance(v, str):
                        gnames.append(v)
                    if k == "spillFillBufferSize":
                        sfb.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
    else:
        print("    ⚠️ utility rc=%d: %s" % (r2.returncode, r2.stdout[-300:]), flush=True)
    print("    图 = %r   spillFillBufferSize = %r" % (gnames, sfb), flush=True)
    return {"tag": tag, "bytes": sz, "graphs": gnames, "sfb": sfb, "sec": el}


def main():
    for f in (A, B):
        assert os.path.isfile(f), "缺 %s（先跑 ws_stage0_mkdlc.py）" % f
    res = {}
    for tag, dlcs, ws in ARMS:
        r = build(tag, dlcs, ws)
        if r:
            res[tag] = r
    json.dump(res, open(os.path.join(W, "stage0_win.json"), "w"), indent=1)

    print("\n" + "=" * 62)
    print("阶段 0 · Windows 三臂结果（判据见 EXP_PLAN_MULTIGRAPH §三 阶段 0）")
    print("=" * 62)
    for t in ("W1", "W2", "W3"):
        if t in res:
            print("  %s  %12d B  %6.2f MB  图%d %r" %
                  (t, res[t]["bytes"], res[t]["bytes"] / 1048576.0,
                   len(res[t]["graphs"]), res[t]["graphs"]))
    if all(t in res for t in ("W1", "W2", "W3")):
        s1, s2, s3 = (res[t]["bytes"] for t in ("W1", "W2", "W3"))
        print("\n  G0-a 双图编入   : W2 图数=%d  W3 图数=%d （需均为 2）" %
              (len(res["W2"]["graphs"]), len(res["W3"]["graphs"])))
        print("  G0-b 共享生效线 : (S(W3)-S(W1))/S(W1) = %.4f  < 0.30 ? %s" %
              ((s3 - s1) / s1, (s3 - s1) / s1 < 0.30))
        print("  G0-c 开关无效线 : S(W3)/S(W2) = %.4f  > 0.95 ? %s" %
              (s3 / s2, s3 / s2 > 0.95))
        print("  参考           : S(W2)/S(W1) = %.4f （无共享双图相对单图）" % (s2 / s1))
    print("=" * 62, flush=True)


main()
