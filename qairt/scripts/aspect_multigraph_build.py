# -*- coding: utf-8 -*-
"""把「现网 1:1」与「新比例」两个图编进同一个 context，权重共享。

这是 #86 路线 D 的**真正交付形态**（此前 `pd_multigraph_probe.py` 用的是同形状的改名
拷贝，只为回答 PD 记账问题；本脚本用的是**真实不同形状**的两个图）。

前置（均已实测）：
  · 权重共享在 Windows 上生效（阶段 0；官方文档说只支持 x86_Linux，实测以我们为准）
  · PD 红线按【已启用的图】算（阶段 2，part1a/part2b 两段独立实测）
  · 4:3 的量化 encoding 与 1:1 **逐项相同**（门 Q1a/Q1b）⇒ 权重可共享
  · 图名互异（`<seg>_fp16_L80_fp32` vs `<seg>_<tag>_fp32`）

🔴 建 context 必带 --config_file（#95），否则静默编成 dspArch 68 / vtcm 4MB。

用法: python aspect_multigraph_build.py <tag>[,<tag>...] [--segs part1a,part1b,...]
      例（五比例合成一个 context/段）:
        python aspect_multigraph_build.py 1184x896,896x1184,1280x720,720x1280
      🔴 1:1 基座图永远隐式在第一位，不要写进 tag 列表。
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
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
AS = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg")
SEGS = ["part1a", "part1b", "part2a", "part2b"]


def main():
    # 多比例：接受逗号分隔的 tag 列表；1:1 基座图隐式在第一位。
    # 🔴 两图 -> 五图不是自然外推：「PD 只计 enable 的图」这条只在**两图**下实测过，
    #    五图必须重新在设备上验（EXP_PLAN_ASPECT_REST 的 G-MG）。约束 4·补：
    #    把机制从 A 迁到 B 时，必须先验证迁移本身成立。
    tags = [x for x in sys.argv[1].split(",") if x]
    segs = SEGS
    if "--segs" in sys.argv:
        segs = sys.argv[sys.argv.index("--segs") + 1].split(",")
    os.makedirs(OUT, exist_ok=True)
    res = {}

    for seg in segs:
        d11 = os.path.join(P2, "%s_fp16_L80_quantized.dlc" % seg)
        dlcs = [d11] + [os.path.join(AS, "%s_%s_quantized.dlc" % (seg, tg)) for tg in tags]
        for f in dlcs:
            if not os.path.isfile(f):
                sys.exit("FAIL 缺 %s" % f)
        g11 = "%s_fp16_L80_fp32" % seg
        gnames = [g11] + ["%s_%s_fp32" % (seg, tg) for tg in tags]
        if len(set(gnames)) != len(gnames):
            sys.exit("FAIL 图名重复 %r —— 同一 context 内图名必须互异" % gnames)

        od = os.path.join(OUT, seg)
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        det = {"graphs": [{"graph_names": [g], "vtcm_mb": 8} for g in gnames],
               "devices": [{"soc_model": 69, "dsp_arch": "v79"}],
               "context": {"weight_sharing_enabled": True}}
        dp, ep = os.path.join(od, "d.json"), os.path.join(od, "e.json")
        json.dump(det, open(dp, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": dp}}, open(ep, "w"), indent=1)

        print("\n--- %s : %d 图 %s ---" % (seg, len(gnames), " + ".join(gnames)),
              flush=True)
        t0 = time.time()
        log = os.path.join(od, "build.log")
        with open(log, "w", encoding="utf-8", errors="replace") as fo:
            r = subprocess.run(
                [os.path.join(BIN, "qnn-context-binary-generator.exe"),
                 "--backend", os.path.join(LIB, "QnnHtp.dll"),
                 "--dlc_path", ",".join(dlcs),
                 "--binary_file", "%s_mg" % seg,
                 "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ep],
                stdout=fo, stderr=subprocess.STDOUT, timeout=28800)
        el = (time.time() - t0) / 60.0
        txt = open(log, encoding="utf-8", errors="replace").read()
        if "available PD" in txt:
            print("    🔴 建图期撞 PD 红线（#57）", flush=True)
        if r.returncode != 0:
            print("    FAIL rc=%d\n%s" % (r.returncode, txt[-1200:]), flush=True)
            return 1
        cand = [f for f in os.listdir(od) if f.endswith(".bin")]
        if len(cand) != 1:
            print("    FAIL 产物不唯一 %r" % cand, flush=True)
            return 1
        binp = os.path.join(od, cand[0])
        sz = os.path.getsize(binp)

        # 生效证据：读回图名（不得以「没报错」判定，§7.3）
        jf = os.path.join(od, "info.json")
        subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                        "--context_binary", binp, "--json_file", jf],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gn = []
        if os.path.isfile(jf):
            d = json.load(open(jf, encoding="utf-8", errors="replace"))

            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "graphName" and isinstance(v, str):
                            gn.append(v)
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(d)

        s1 = os.path.getsize(os.path.join(
            P2, "ctx_%s_fp16_L80" % seg, "%s_fp16_L80.SM8750.bin" % seg))
        print("    rc=0  %.1f 分钟  产物 %.1f MiB  单图现网 %.1f MiB  比值 %.4f"
              % (el, sz / 1048576.0, s1 / 1048576.0, sz / float(s1)), flush=True)
        print("    图 = %r" % gn, flush=True)
        # 🔴 硬失败而不是只打印一行警告：本项目已因「门是装饰性的」栽过（#156 的
        #    sha256 门只打印不比对）。图集合不符 = 产物不可用，必须让调用方看见非零退出。
        if sorted(gn) != sorted(gnames):
            print("    🔴 图集合不符，预期 %r 实得 %r" % (gnames, gn), flush=True)
            return 1
        res[seg] = {"bytes": sz, "single": s1, "graphs": gn, "min": el,
                    "per_graph_pct": (sz / float(s1) - 1.0) * 100.0 / max(1, len(gnames) - 1)}
        json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)

    print("\n" + "=" * 60, flush=True)
    # 🔴 这里原本是 `% tag` —— 两图推广到 N 图时改了 gnn->gnames、tag->tags，
    #    唯独漏了这一处，`NameError` 让整条编排在**四个产物已全部建成之后**
    #    才崩在一行汇总打印上（浪费 6.1 小时的调度，产物本身没事）。
    #    当时我 grep 了 `gnn` 确认为 0，却没 grep `tag` —— 约束 11·补 的
    #    「按 grep 结果逐个追，不按预期追」只做了一半。
    print("共享 context 汇总（1:1 + %d 个比例）" % len(tags), flush=True)
    tot_mg = tot_1 = 0
    for seg in segs:
        if seg in res:
            r = res[seg]
            tot_mg += r["bytes"]
            tot_1 += r["single"]
            print("  %-8s %8.1f MiB  (单图 %8.1f)  +%.2f%%  %.1f 分钟"
                  % (seg, r["bytes"] / 1048576.0, r["single"] / 1048576.0,
                     100.0 * (r["bytes"] - r["single"]) / r["single"], r["min"]))
    if tot_1:
        print("  %-8s %8.1f MiB  (单图 %8.1f)  +%.2f%%"
              % ("合计", tot_mg / 1048576.0, tot_1 / 1048576.0,
                 100.0 * (tot_mg - tot_1) / tot_1))
    print("=" * 60, flush=True)
    return 0


sys.exit(main())
