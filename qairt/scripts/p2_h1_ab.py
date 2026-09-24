# -*- coding: utf-8 -*-
"""D2-H1 · S4：离线 A/B —— 手术版 vs 部署版 part1a，同一份输入，比**数值**与**耗时**。

判据（`scripts/EXP_PLAN_P2_SPEED.md` §6.3 事前锁定，**不得事后改**）：
  P1 精度（硬门）：9 个输出张量的 sha256 **逐个相同**。任一不同 ⇒ 停，不得以「差异很小」放行。
  S1 速度门：手术版加速器执行时间 ≤ 部署版 **0.80 倍**（③预期 0.71）。< 0.90 倍则不值得做 S5/S6。

做法与纪律：
  · 两臂同一份**原生**输入（`--use_native_input_files`），逐个输出核字节数（#176 的静默砍半陷阱）。
  · sha256 **在设备上算**（输出共约 35 MB/次，不必拉回宿主）。
  · 交替跑 A→B→B→A 两趟，抵消热漂移（#144：热会让耗时单调漂移）。
  · 设备只写 /data/local/tmp/p2ab，结束即删；不碰 app 的 files/models，不装 APK。

用法: python scripts/p2_h1_ab.py            # 全流程（需插线，约 12 分钟）
      python scripts/p2_h1_ab.py --keep     # 结束不删设备临时目录
"""
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import lab_dev  # noqa: E402
import p2_d1_profile as P  # noqa: E402

NEW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_noscat",
                   "ctx_part1a_noscat", "part1a_noscat.SM8750.bin")
ARMS = {"A_deployed": P.HOST_CTX["part1a"], "B_noscat": NEW}
D = "/data/local/tmp/p2ab"
OUT = os.path.join(P.REPO, "logs", "p2_20260917", "h1_ab")
SEG = "part1a"


def prep():
    lab_dev.require_online("S4 prep")
    print("设备可用内存 %d MiB，NPU %.1f °C" % (lab_dev.mem_available_mb(), P.npu_c()))
    lab_dev.sh("am force-stop %s" % P.PKG)
    lab_dev.sh("mkdir -p %s/in" % D)
    hd, names = P.make_inputs(SEG)
    for n in names:
        dev = "%s/in/%s.raw" % (D, n)
        want = os.path.getsize(os.path.join(hd, n + ".raw"))
        if lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % dev).strip() != str(want):
            lab_dev.push(os.path.join(hd, n + ".raw"), dev)
    for arm, host in ARMS.items():
        dev = "%s/%s.bin" % (D, arm)
        want = os.path.getsize(host)
        if lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % dev).strip() != str(want):
            print("  push %s（%.2f GiB）…" % (arm, want / 2**30), flush=True)
            lab_dev.push(host, dev)
        got = lab_dev.sh("sha256sum %s" % dev).split()[0]
        exp = P.sha256(host)
        if got != exp:
            raise SystemExit("🔴 G0 %s 设备侧 sha256 与宿主不一致" % arm)
        print("  ✅ %s 设备侧 sha256 == 宿主" % arm)
    return names


def run(arm, names, trip):
    tag = "%s_t%d" % (arm, trip)
    od = "%s/o_%s" % (D, tag)
    line = " ".join("%s:=%s/in/%s.raw" % (k, D, k) for k in names)
    lab_dev.sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' '%s' > %s/list.txt" % (od, od, line, od))
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context %s/%s.bin --backend libQnnHtp.so --input_list %s/list.txt "
           "--output_dir %s --perf_profile burst --use_native_input_files --use_native_output_files "
           "--profiling_level basic --num_inferences 3 --log_level error; echo RC=$?"
           % (P.T, P.T, P.T, D, arm, od, od))
    t0c, t0 = P.npu_c(), time.time()
    o = lab_dev.sh(cmd, timeout=1800)
    wall = time.time() - t0
    if "RC=0" not in o:
        raise SystemExit("🔴 %s 运行失败（执行错误）：%s" % (tag, o[-600:]))
    # 逐个输出核字节数 + 设备侧 sha256
    want = {t["name"]: t["exact_bytes"] for t in P.contract_graph(SEG)["outputs"]}
    lines = lab_dev.sh("for f in %s/Result_0/*.raw; do stat -c '%%n %%s' $f; sha256sum $f; done; true" % od)
    sizes, hashes, cur = {}, {}, None
    for l in lines.strip().splitlines():
        p = l.split()
        if len(p) == 2 and p[0].endswith(".raw") and p[1].isdigit():
            cur = os.path.basename(p[0])[:-4].replace("_native", "")
            sizes[cur] = int(p[1])
        elif len(p) == 2 and len(p[0]) == 64:
            hashes[os.path.basename(p[1])[:-4].replace("_native", "")] = p[0]
    bad = [(k, v, sizes.get(k)) for k, v in want.items() if sizes.get(k) != v]
    if bad:
        raise SystemExit("🔴 %s 输出字节数与契约不符 ⇒ 数据作废：%s" % (tag, bad))
    hd = os.path.join(OUT, tag)
    os.makedirs(hd, exist_ok=True)
    logs = [x for x in lab_dev.sh("ls %s; true" % od).split() if x.startswith("qnn-profiling-data")]
    for lg in logs:
        P.pull_robust("%s/%s" % (od, lg), os.path.join(hd, lg))
    txt = "".join(P.view(os.path.join(hd, lg)) for lg in logs)
    accel = P.nums_after(txt, "Accelerator (execute) time")
    print("  %s 墙钟 %.1f s（含装载）NPU前 %.1f °C｜加速器执行 %s us｜9 个输出字节数全部 == 契约 ✅"
          % (tag, wall, t0c, accel[:3]))
    lab_dev.sh("rm -rf %s/Result_0; true" % od)      # 省设备空间，哈希已取
    return {"tag": tag, "arm": arm, "trip": trip, "wall_s": wall, "npu_c": t0c,
            "accel_us": accel, "sha": hashes}


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isfile(NEW):
        raise SystemExit("🔴 手术版 context 不存在，先跑 p2_h1_build.py")
    names = prep()
    res = []
    for trip, order in ((1, ("A_deployed", "B_noscat")), (2, ("B_noscat", "A_deployed"))):
        for arm in order:
            res.append(run(arm, names, trip))
    json.dump(res, open(os.path.join(OUT, "ab.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n== P1 精度门（硬）：9 个输出逐个比 sha256 ==")
    sa = {}
    for r in res:
        sa.setdefault(r["arm"], []).append(r["sha"])
    names_out = [t["name"] for t in P.contract_graph(SEG)["outputs"]]
    same_all, diff = True, []
    for n in names_out:
        ha = {h[n] for h in sa["A_deployed"] if n in h}
        hb = {h[n] for h in sa["B_noscat"] if n in h}
        if len(ha) != 1 or len(hb) != 1:
            print("  ⚠️ %s：同臂两趟的哈希不一致（A %d 种 / B %d 种）⇒ 存在非确定性，先查这个" % (n, len(ha), len(hb)))
            same_all = False
            continue
        ok = ha == hb
        print("  %s %-15s A %s… B %s…" % ("✅" if ok else "🔴", n, list(ha)[0][:12], list(hb)[0][:12]))
        same_all &= ok
        if not ok:
            diff.append(n)
    print("  ⇒ P1 %s" % ("✅ 全部逐字节相同" if same_all else "🔴 不通过：%s" % diff))

    def med(arm):
        v = [x for r in res if r["arm"] == arm for x in r["accel_us"]]
        return statistics.median(v) if v else None
    a, b = med("A_deployed"), med("B_noscat")
    print("\n== S1 速度门 ==")
    if a and b:
        print("  部署版 %.3f s/次｜手术版 %.3f s/次｜比值 **%.3f**（门 ≤0.80，③预期 0.71）⇒ %s"
              % (a / 1e6, b / 1e6, b / a, "✅ 通过" if b / a <= 0.80 else
                 ("🟡 有收益但未达门" if b / a < 0.95 else "🔴 无收益")))
    else:
        print("  🟡 读不到加速器执行时间，人工读 logs/p2_20260917/h1_ab/*/**.viewer.txt")
    print("\n判决：%s" % ("🟢 P1+S1 均过 ⇒ 可以做 S5（建 mg 五图）"
                        if same_all and a and b and b / a <= 0.80 else
                        "🔴 未全过 ⇒ 按 §6.3 失败归因处理，不得进 S5"))
    if "--keep" not in sys.argv and lab_dev.online():
        lab_dev.sh("rm -rf %s; true" % D)
        print("\n已删设备临时目录 %s —— **可以拔线**" % D)
    return 0


if __name__ == "__main__":
    sys.exit(main())
