# -*- coding: utf-8 -*-
"""EXP_PLAN_MULTIGRAPH 阶段 1' —— 真段多图 PD 探针（不需要先做 4:3 手术）。

问题：PD 红线（3.3 GB，#57）按【已启用的图】算，还是按【整个 context】算？
关键洞察：这个问题与图的形状无关 ⇒ 用同一段的 5 份【只改图名】的拷贝即可判定。

选 part2b：单段 PD 估算 1927 MB（§3.4），5 图若按整体算必然越过红线 3506~3535 MB
⇒ 两种答案可区分（约束 8：先确认判据能分辨）。

同时量出【真段、真实规模下每多一个图涨多少】—— 探针的 +13.0% 是 14 MB 单算子的最好情形。
"""
import os, sys, json, time, shutil, subprocess

SDK = r"D:\qairt\2.48.0.260626"
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
# 段可由 argv[1] 指定（默认 part2b）。🔴 part1a 才是最贴 PD 红线的那一段
# （context 2263 MiB vs part2b 1398 MiB；#147 实测 part1a 加 O:3 直接 0x3ea）。
SEG = sys.argv[1] if len(sys.argv) > 1 else "part2b"
SRC = r"D:\ZImage_Work\p0_experiments\p2attr\%s_fp16_L80_quantized.dlc" % SEG
W = r"D:\ZImage_Work\p0_experiments\pd_multigraph_%s" % SEG
BASE = "%s_fp16_L80_fp32" % SEG
VARIANTS = ["%s_fp16_L80_fpG%d" % (SEG, i) for i in (1, 2, 3, 4)]
PY, TOOL = sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "qairt_tool.py")


def sh(cmd, **kw):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace", **kw)
    return p.returncode, p.stdout


def make_copies():
    """等长改写图名。3 处明文，已实测。"""
    os.makedirs(W, exist_ok=True)
    out = [SRC]
    b = BASE.encode()
    for v in VARIANTS:
        assert len(v) == len(BASE), "图名长度必须相同：%s" % v
        dst = os.path.join(W, v + ".dlc")
        if os.path.isfile(dst):
            print("[skip] %s 已存在" % os.path.basename(dst), flush=True)
            out.append(dst); continue
        t0 = time.time()
        n = 0
        with open(SRC, "rb") as fi, open(dst, "wb") as fo:
            tail = b""
            while True:
                c = fi.read(1 << 24)
                if not c:
                    fo.write(tail); break
                s = tail + c
                keep = len(b) - 1
                body, tail = s[:len(s) - keep], s[len(s) - keep:]
                n += body.count(b)
                fo.write(body.replace(b, v.encode()))
            # 处理最后残留
        # 残留 tail 已写出但未替换；因 tail < len(b)，不可能含完整图名，安全
        print("[copy] %s  替换 %d 处  %.0fs" % (v, n, time.time() - t0), flush=True)
        out.append(dst)
    return out


def verify(dlcs):
    """门 P1：每个 DLC 恰好一个图，且图名与文件名主干一致（约束 8）。"""
    print("\n=== 门 P1 图名核验 ===", flush=True)
    names = []
    for d in dlcs:
        rc, o = sh([PY, TOOL, "snpe-dlc-info", "-i", d])
        assert rc == 0, "snpe-dlc-info rc=%d on %s\n%s" % (rc, d, o[-800:])
        gl = [l for l in o.splitlines() if l.startswith("Info of graph:")]
        assert len(gl) == 1, "%s 图数 != 1: %r" % (d, gl)
        g = gl[0].split(":", 1)[1].strip()
        names.append(g)
        print("   %-46s -> %s" % (os.path.basename(d), g), flush=True)
    assert len(set(names)) == len(names), "图名有重复: %r" % names
    return names


def build(tag, dlcs, names, ws):
    od = os.path.join(W, tag)
    if os.path.isdir(od):
        shutil.rmtree(od)
    os.makedirs(od)
    det = {"graphs": [{"graph_names": [n], "vtcm_mb": 8} for n in names],
           "devices": [{"soc_model": 69, "dsp_arch": "v79"}]}
    if ws is not None:
        det["context"] = {"weight_sharing_enabled": ws}
    dp, ep = os.path.join(od, "d.json"), os.path.join(od, "e.json")
    json.dump(det, open(dp, "w"), indent=1)
    json.dump({"backend_extensions": {
        "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
        "config_file_path": dp}}, open(ep, "w"), indent=1)

    print("\n--- %s  图数=%d  weight_sharing=%r ---" % (tag, len(dlcs), ws), flush=True)
    t0 = time.time()
    log = os.path.join(od, "build.log")
    with open(log, "w", encoding="utf-8", errors="replace") as fo:
        r = subprocess.run(
            [os.path.join(BIN, "qnn-context-binary-generator.exe"),
             "--backend", os.path.join(LIB, "QnnHtp.dll"),
             "--dlc_path", ",".join(dlcs), "--binary_file", tag,
             "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ep],
            stdout=fo, stderr=subprocess.STDOUT, timeout=21600)
    el = (time.time() - t0) / 60.0
    txt = open(log, encoding="utf-8", errors="replace").read()
    print("    rc=%d  用时 %.1f 分钟" % (r.returncode, el), flush=True)
    if "available PD" in txt:
        print("    🔴 建图期就撞 PD 红线（#57）", flush=True)
    if r.returncode != 0:
        print("    ❌ 失败，日志尾部:\n%s" % txt[-1500:], flush=True)
        return None
    cand = [f for f in os.listdir(od) if f.endswith(".bin")]
    assert len(cand) == 1, "产物不唯一 %r" % cand
    binp = os.path.join(od, cand[0])
    sz = os.path.getsize(binp)
    print("    产物 %s = %d B (%.1f MiB)" % (cand[0], sz, sz / 1048576.0), flush=True)
    jf = os.path.join(od, "info.json")
    r2 = subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                         "--context_binary", binp, "--json_file", jf],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        encoding="utf-8", errors="replace")
    gn, sfb = [], []
    if r2.returncode == 0 and os.path.isfile(jf):
        d = json.load(open(jf, encoding="utf-8", errors="replace"))
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "graphName" and isinstance(v, str): gn.append(v)
                    if k == "spillFillBufferSize": sfb.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o: walk(v)
        walk(d)
    print("    图 = %r" % gn, flush=True)
    print("    spillFillBufferSize = %r" % sfb, flush=True)
    return {"tag": tag, "bytes": sz, "graphs": gn, "sfb": sfb, "min": el, "bin": binp}


def main():
    assert os.path.isfile(SRC), "缺 %s" % SRC
    dlcs = make_copies()
    names = verify(dlcs)
    res = {}
    r = build("CTRL1", dlcs[:1], names[:1], None)
    if r: res["CTRL1"] = r
    r = build("SHARE5", dlcs, names, True)
    if r: res["SHARE5"] = r
    json.dump(res, open(os.path.join(W, "result.json"), "w"), indent=1)

    print("\n" + "=" * 66)
    print("阶段 1' 宿主结果")
    print("=" * 66)
    for t in ("CTRL1", "SHARE5"):
        if t in res:
            print("  %-7s %13d B  %8.1f MiB  图数=%d  %.1f 分钟" %
                  (t, res[t]["bytes"], res[t]["bytes"] / 1048576.0,
                   len(res[t]["graphs"]), res[t]["min"]))
    if "CTRL1" in res and "SHARE5" in res:
        s1, s5 = res["CTRL1"]["bytes"], res["SHARE5"]["bytes"]
        print("\n  真段每多一个图的增量 = (S5-S1)/4/S1 = %.2f%%" % (100.0 * (s5 - s1) / 4 / s1))
        print("  （对照：14 MB 单算子探针上是 +13.02%%/图）")
        print("  S5/S1 = %.4f   （不共享应约为 5.00）" % (s5 / s1))
    print("=" * 66, flush=True)
    print("\n⚠️ PD 生死线仍需设备：宿主只能建图，PD 检查在设备端 contextFinalize（#147）", flush=True)


main()
