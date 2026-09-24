# -*- coding: utf-8 -*-
"""EXP_PLAN_MEMSPEED 实验 3 的装置 —— 把**两个不同段**编进同一个 context（无权重共享）。

## 为什么不用现成的 aspect_mg 五图 context（方案原文写的就是它）
`aspect_mg` 的五个图是**同一段的五个比例、权重共享**，#161 实测每多一个图只多约
**115 MiB** 常驻 ⇒ 在它上面切换，搬动的是 ~115 MiB 的非共享部分。
而 E 方案要切的是**四个不同段**、彼此没有共享权重、每个 1.8~2.9 GB ——
**两者差一个数量级**。拿前者的 t_switch 去判后者，正是约束 4·补 第 2 条点名的
「把机制从 A 迁到 B 却没验证迁移本身成立」。

## 这个装置为什么够用（三条）
1. **它就是 E 的最小真实形态**：不同段 + 无共享权重，不需要任何迁移假设。
2. **自带二值生效门**：两段公式合计 1857+1828 = **3685 MiB** > PD 红线 3506~3535 MiB（#57）
   ⇒ 不开图切换时**应当** `0x3ea` 失败，开了**应当**成功。
   这正好补上「传了参数不报错 ≠ 生效」这个坑（#59：同一工具的 spill_fill/weights_buffer
   在 2.48 上解析得了却用不了）。
3. 建图成本比合并四段小一个量级；判负就不必再投入「每比例合并四段」（估 10~20 小时宿主）。

🔴 建 context 必带 --config_file（#95），否则静默编成 dspArch 68 / vtcm 4MB。
🔴 建图成功 ≠ 装得上：PD 检查发生在设备端 contextFinalize（#60/#147）⇒ 产物必须上设备试装。

用法: python scripts/memspeed_probe_build.py [--segs part1b,part2a]
"""
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "memspeed_probe")


def main():
    segs = ["part1b", "part2a"]
    if "--segs" in sys.argv:
        segs = sys.argv[sys.argv.index("--segs") + 1].split(",")
    if len(segs) < 2:
        sys.exit("🔴 至少两个段，否则没有「切换」可言")

    dlcs, gnames = [], []
    for s in segs:
        d = os.path.join(P2, "%s_fp16_L80_quantized.dlc" % s)
        if not os.path.isfile(d):
            sys.exit("🔴 缺源 DLC: %s" % d)
        dlcs.append(d)
        gnames.append("%s_fp16_L80_fp32" % s)
    if len(set(gnames)) != len(gnames):
        sys.exit("🔴 图名重复 %r —— 同一 context 内图名必须互异" % gnames)

    os.makedirs(OUT, exist_ok=True)
    # 🔴 weight_sharing 显式关掉：两段之间本就没有可共享的权重，
    #    开着只会引入一个与 E 无关的变量（约束 3.6：归因模式必须单变量）。
    det = {"graphs": [{"graph_names": [g], "vtcm_mb": 8} for g in gnames],
           "devices": [{"soc_model": 69, "dsp_arch": "v79"}],
           "context": {"weight_sharing_enabled": False}}
    dp = os.path.join(OUT, "d.json")
    ep = os.path.join(OUT, "e.json")
    with open(dp, "w", encoding="utf-8") as f:
        json.dump(det, f, indent=1)
    with open(ep, "w", encoding="utf-8") as f:
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": dp}}, f, indent=1)

    name = "probe_%s" % "_".join(segs)
    log = os.path.join(OUT, "build.log")
    print("=== 建 %d 图 context: %s ===" % (len(gnames), " + ".join(gnames)), flush=True)
    print("    源 DLC 合计 %.1f MiB" % (sum(os.path.getsize(d) for d in dlcs) / 1048576.0),
          flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fo:
        r = subprocess.run(
            [os.path.join(BIN, "qnn-context-binary-generator.exe"),
             "--backend", os.path.join(LIB, "QnnHtp.dll"),
             "--dlc_path", ",".join(dlcs),
             "--binary_file", name,
             "--output_dir", OUT, "--htp_socs", "sm8750", "--config_file", ep],
            stdout=fo, stderr=subprocess.STDOUT, timeout=28800)
    el = (time.time() - t0) / 60.0
    txt = open(log, encoding="utf-8", errors="replace").read()
    # 🔴 透传：报错会说谎，先把工具自己说了什么打出来（约束 11·再补）
    if "available PD" in txt:
        print("    ⚠️ 建图期日志出现 'available PD' —— 注意 PD 检查其实在设备端（#60）", flush=True)
    if r.returncode != 0:
        print("    🔴 FAIL rc=%d\n%s" % (r.returncode, txt[-2000:]), flush=True)
        return 1

    cand = [f for f in os.listdir(OUT) if f.endswith(".bin")]
    if len(cand) != 1:
        print("    🔴 FAIL 产物不唯一 %r" % cand, flush=True)
        return 1
    binp = os.path.join(OUT, cand[0])

    # 生效证据：读回图名与各段内存字段（不得以「没报错」判定，§7.3）
    jf = os.path.join(OUT, "info.json")
    subprocess.run([os.path.join(BIN, "qnn-context-binary-utility.exe"),
                    "--context_binary", binp, "--json_file", jf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(jf):
        print("    🔴 FAIL 读不回元数据", flush=True)
        return 1
    meta = json.load(open(jf, encoding="utf-8", errors="replace"))
    got, rows = [], []
    for g in meta["info"]["graphs"]:
        gi = g["info"]
        got.append(gi["graphName"])
        b1 = (gi.get("graphBlobInfo") or {}).get("info") or {}
        b2 = gi.get("graphBlobInfoV2") or {}
        tot = (int(b2.get("opDataSize", 0)) + int(b2.get("constSize", 0))
               + int(b2.get("ddrTensorSize", 0)) + int(b1.get("spillFillBufferSize", 0))
               + int(b2.get("ioTensorSize", 0)) + int(b1.get("vtcmSize", 0)) * 1048576)
        rows.append((gi["graphName"], int(b2.get("constSize", 0)) / 1048576.0,
                     int(b2.get("sharedWeightsSize", 0)) / 1048576.0, tot / 1048576.0))

    print("    rc=0  %.1f 分钟  产物 %.1f MiB" % (el, os.path.getsize(binp) / 1048576.0),
          flush=True)
    print("    图 = %r" % got, flush=True)
    for n, c, sw, t in rows:
        print("      %-28s const %8.1f  shared %7.1f  公式合计 %8.1f MiB" % (n, c, sw, t),
              flush=True)
    total = sum(t for _, _, _, t in rows)
    print("    两图公式合计 %.1f MiB   vs PD 红线 3506~3535 MiB（#57）⇒ %s"
          % (total, "超线（生效门可判别）" if total > 3535 else
             "🔴 未超线 ⇒ V0-a 的二值判别失效，须换更大的段"), flush=True)

    if sorted(got) != sorted(gnames):
        print("    🔴 图集合不符，预期 %r 实得 %r" % (gnames, got), flush=True)
        return 1
    shared_total = sum(sw for _, _, sw, _ in rows)
    if shared_total > 1.0:
        print("    ⚠️ sharedWeightsSize 非零（%.1f MiB）—— 本装置本应无共享权重，"
              "先查清再解读 t_switch" % shared_total, flush=True)

    json.dump({"bin": binp, "graphs": got, "minutes": el,
               "bytes": os.path.getsize(binp),
               "formula_mib": {n: t for n, _, _, t in rows}},
              open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("✅ 完成: %s" % binp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
