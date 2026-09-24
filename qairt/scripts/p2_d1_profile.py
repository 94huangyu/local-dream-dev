# -*- coding: utf-8 -*-
"""P2 · D1 设备诊断（`scripts/EXP_PLAN_P2_SPEED.md` §3）：qnn-net-run 逐算子 profiling，**零 APK 改动**。

三个子命令，刻意分开：
  preflight  只读：设备可达、工具与库在、版本、空间、内存、NPU 温度；宿主 context sha256 == single 契约（G0a）
  collect    占设备：停 app → 推 context/输入 → R1~R5 → 拉回 profiling 日志 → 删设备临时目录
  analyze    纯宿主：qnn-profile-viewer 解析 → 按 §3.3 事前锁定的判据给结论
理由：HTP profiling reader 的输出格式在宿主上无法预先验证（只能读设备产物），
      解析写错时**只需重跑 analyze，不必再占一次手机**。

🔴 口径
  · 输入：uint16 张量用固定种子随机值、BOOL 掩码用真实形态（前 22 个 0、其余 1）。
    ③假设：HTP 定点算子的耗时与数据无关（#147 同做法）。**本诊断只看耗时/cycles，不看数值。**
  · 这是 `qnn-net-run` **离线单段**；能否代表 app 由 G1 代表性门判定，不过门不下归因结论。
  · 设备只写 /data/local/tmp/p2diag，结束即删；不碰 files/models、不装 APK。

用法:
  python scripts/p2_d1_profile.py preflight
  python scripts/p2_d1_profile.py collect
  python scripts/p2_d1_profile.py analyze
"""
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab_dev  # noqa: E402

REPO = os.path.join("D:", os.sep, "LocalDreamZImage")
OUT = os.path.join(REPO, "logs", "p2_20260917", "d1")
CONTRACT = os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.single.json")
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
HOST_CTX = {
    "part1a": os.path.join(P2, "ctx_part1a_fp16_L80", "part1a_fp16_L80.SM8750.bin"),
    "part1b": os.path.join(P2, "ctx_part1b_fp16_L80", "part1b_fp16_L80.SM8750.bin"),
}
FC99_DLC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b", "actcal", "q_minmax.dlc")
QAIRT = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
VIEWER = os.path.join(QAIRT, "bin", "x86_64-windows-msvc", "qnn-profile-viewer.exe")
READER = os.path.join(QAIRT, "lib", "x86_64-windows-msvc", "QnnHtpProfilingReader.dll")
ANDROID_LIB = os.path.join(QAIRT, "lib", "aarch64-android")
PKG = "io.github.xororz.localdream.zimage"
T = "/data/local/tmp/htpcmp"          # 既有 qnn-net-run 与 HTP 库（#147 在用）
D = "/data/local/tmp/p2diag"
NEED = ["qnn-net-run", "libQnnHtp.so", "libQnnHtpV79Stub.so", "libQnnHtpV79Skel.so", "libQnnSystem.so"]
NEED_R5 = ["libQnnModelDlc.so", "libQnnHtpPrepare.so"]
MAIN_OUT = {"part1a": "add_138", "part1b": "unified"}
MIN_MEMAVAIL_MB = 3000


def contract_graph(name):
    for m in json.load(open(CONTRACT, encoding="utf-8"))["models"]:
        if m["internal_graph_name"] == "transformer_" + name:
            return m
    raise SystemExit("🔴 single 契约里没有 transformer_%s" % name)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def npu_c():
    o = lab_dev.sh("M=0; for z in /sys/class/thermal/thermal_zone*; do "
                   "case \"$(cat $z/type)\" in nsph*) V=$(cat $z/temp); "
                   "[ \"$V\" -gt \"$M\" ] && M=$V;; esac; done; echo $M")
    try:
        return int(o.strip()) / 1000.0
    except ValueError:
        return -1.0


# ---------------------------------------------------------------- preflight
def host_gate():
    """G0a：宿主 context 与 single 契约逐字节一致 ⇒ 诊断的就是部署同款 1:1 图。"""
    ok = True
    for seg, p in HOST_CTX.items():
        c = contract_graph(seg)
        if not os.path.isfile(p):
            print("  🔴 缺宿主 context：%s" % p)
            ok = False
            continue
        size, h = os.path.getsize(p), sha256(p)
        good = size == c["size_bytes"] and h == c["sha256"]
        print("  %s %s  %d B  sha %s…  %s" % ("✅" if good else "🔴", seg, size, h[:12],
                                             "== 契约" if good else "≠ 契约 %s…" % c["sha256"][:12]))
        ok &= good
    return ok


def preflight():
    print("== G0a 宿主 ==")
    ok = host_gate()
    print("== 设备（只读）==")
    lab_dev.require_online("preflight")
    have = lab_dev.sh("ls %s 2>/dev/null; true" % T).split()
    for n in NEED + NEED_R5:
        tag = "✅" if n in have else ("⚠️ 缺（R5 将从 SDK 推送）" if n in NEED_R5 else "🔴 缺")
        print("  %s %s/%s" % (tag, T, n))
        ok &= n in have or n in NEED_R5
    ver = lab_dev.sh("cd %s && LD_LIBRARY_PATH=%s ./qnn-net-run --version 2>&1; true" % (T, T))
    vok = "2.48" in ver
    print("  %s qnn-net-run 版本：%s" % ("✅" if vok else "🔴", " ".join(ver.split())[:120]))
    ok &= vok
    free = lab_dev.sh("df -k /data/local/tmp | tail -1").split()
    free_gb = int(free[3]) / 2**20 if len(free) > 3 and free[3].isdigit() else -1
    print("  %s /data/local/tmp 可用 %.1f GiB（需约 4 GiB）" % ("✅" if free_gb > 6 else "🔴", free_gb))
    ok &= free_gb > 6
    mem = lab_dev.mem_available_mb()
    print("  %s MemAvailable %d MiB（下限 %d）" % ("✅" if mem >= MIN_MEMAVAIL_MB else "🔴", mem, MIN_MEMAVAIL_MB))
    ok &= mem >= MIN_MEMAVAIL_MB
    print("  NPU 温度 %.1f °C" % npu_c())
    print("\n⇒ preflight %s" % ("✅ 通过" if ok else "🔴 未通过（先修再 collect）"))
    return 0 if ok else 1


# ---------------------------------------------------------------- collect
def make_inputs(seg):
    import numpy as np
    d = os.path.join(OUT, "in_%s" % seg)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(20260917)
    names = []
    for t in contract_graph(seg)["inputs"]:
        p = os.path.join(d, t["name"] + ".raw")
        nb = t["exact_bytes"]
        if t["physical_dtype"] == "QNN_DATATYPE_BOOL_8":
            m = np.ones(nb, dtype=np.uint8)
            m[:22] = 0                                    # 真实形态：22 个真实 token、其余为 padding（#164 实测极性）
            m.tofile(p)
        elif t["physical_dtype"] == "QNN_DATATYPE_UFIXED_POINT_16":
            rng.integers(0, 65536, size=nb // 2, dtype=np.uint16).tofile(p)
        else:
            raise SystemExit("🔴 未预期的输入类型 %s/%s" % (t["name"], t["physical_dtype"]))
        if os.path.getsize(p) != nb:
            raise SystemExit("🔴 %s 字节数 %d ≠ 契约 %d" % (p, os.path.getsize(p), nb))
        names.append(t["name"])
    return d, names


def pull_robust(dev, host):
    """拉设备文件。🔴 2026-09-18 实测：qnn-net-run 写出的 profiling 日志 `adb pull` 会 Permission denied，
    先放宽权限重试；仍失败则用 `adb exec-out cat`（二进制安全）兜底。**三条路都失败才抛**。"""
    try:
        lab_dev.pull(dev, host)
        return
    except lab_dev.AdbFailed:
        pass
    lab_dev.sh("chmod -R a+rX %s; true" % D)
    try:
        lab_dev.pull(dev, host)
        return
    except lab_dev.AdbFailed:
        pass
    r = subprocess.run([lab_dev.ADB, "-s", lab_dev.SERIAL, "exec-out", "cat", dev],
                       capture_output=True, timeout=600, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    if r.returncode != 0 or not r.stdout:
        raise SystemExit("🔴 三种方式都拉不回 %s：rc=%d stderr=%s" % (dev, r.returncode, (r.stderr or b"")[-300:]))
    open(host, "wb").write(r.stdout)


def run_netrun(tag, seg, level, names, inferences=3):
    od = "%s/o_%s" % (D, tag)
    line = " ".join("%s:=%s/in_%s/%s.raw" % (k, D, seg, k) for k in names)
    lab_dev.sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' '%s' > %s/list.txt" % (od, od, line, od))
    # 🔴 `--use_native_input_files` 必须带（2026-09-18 实测）：不带时 qnn-net-run 按 float32 解析输入，
    #    喂原生 uint16 字节 ⇒ 它按「字节数 ÷ 期望字节数 = 0.5」**静默把张量砍一半**，
    #    而输出按 float32 写出来恰好等于契约的原生字节数 ⇒ 字节数守卫被巧合骗过（约束 3 的变体）。
    #    `--use_native_output_files` 一并带：输出直接是原生字节，可逐个与契约核对（文件名加 `_native` 后缀）。
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context %s/%s.bin --backend libQnnHtp.so "
           "--input_list %s/list.txt --output_dir %s --perf_profile burst "
           "--use_native_input_files --use_native_output_files "
           "--profiling_level %s --num_inferences %d --log_level error; echo RC=$?"
           % (T, T, T, D, seg, od, od, level, inferences))
    t0c, t0 = npu_c(), time.time()
    o = lab_dev.sh(cmd, timeout=1800)
    el = time.time() - t0
    rc = re.search(r"RC=(\d+)", o)
    if not rc or rc.group(1) != "0":
        raise SystemExit("🔴 %s qnn-net-run 失败（执行错误，数据作废）:\n%s" % (tag, o[-800:]))
    # 约束 3 式自证：**逐个**输出张量核对字节数（只核一个会被「恰好等于一半」的巧合骗过）
    ls = lab_dev.sh("for f in %s/Result_0/*.raw; do stat -c '%%n %%s' $f; done 2>/dev/null; true" % od)
    got = {}
    for line_ in ls.strip().splitlines():
        parts = line_.split()
        if len(parts) == 2 and parts[0].endswith(".raw"):
            got[os.path.basename(parts[0])[:-4].replace("_native", "")] = int(parts[1])
    bad = [(x["name"], x["exact_bytes"], got.get(x["name"]))
           for x in contract_graph(seg)["outputs"] if got.get(x["name"]) != x["exact_bytes"]]
    if bad:
        raise SystemExit("🔴 %s 输出字节数与契约不符（输入未被正确接受，数据作废）：%s" % (tag, bad))
    logs = [x for x in lab_dev.sh("ls %s; true" % od).split() if x.startswith("qnn-profiling-data")]
    if not logs:
        raise SystemExit("🔴 %s 没有 profiling 日志" % tag)
    hd = os.path.join(OUT, tag)
    os.makedirs(hd, exist_ok=True)
    for lg in logs:
        pull_robust("%s/%s" % (od, lg), os.path.join(hd, lg))
    rec = {"tag": tag, "seg": seg, "level": level, "wall_s": el, "npu_c_before": t0c, "logs": logs}
    print("  ✅ %-3s %s %-8s 墙钟 %.1f s（含装载）NPU前 %.1f °C  输出字节 == 契约" % (tag, seg, level, el, t0c))
    return rec


def run_r5():
    """在线建图单算子，读 SoC 最大 HVX 线程（文档：在线建图不设则用最大值）。失败只记录，不中断。"""
    import numpy as np
    try:
        have = lab_dev.sh("ls %s 2>/dev/null; true" % T).split()
        for n in NEED_R5:
            if n not in have:
                lab_dev.push(os.path.join(ANDROID_LIB, n), "%s/%s" % (D, n))
        lab_dev.push(FC99_DLC, "%s/fc99.dlc" % D)
        x = os.path.join(OUT, "in_fc99_x.raw")
        np.random.default_rng(1).standard_normal(4128 * 3840, dtype=np.float32).tofile(x)
        lab_dev.push(x, "%s/in_fc99_x.raw" % D)
        od = "%s/o_R5" % D
        lab_dev.sh("rm -rf %s && mkdir -p %s && echo 'x:=%s/in_fc99_x.raw' > %s/list.txt" % (od, od, D, od))
        o = lab_dev.sh("cd %s && export LD_LIBRARY_PATH=%s:%s && export ADSP_LIBRARY_PATH=%s && "
                       "./qnn-net-run --dlc_path %s/fc99.dlc --model libQnnModelDlc.so --backend libQnnHtp.so "
                       "--input_list %s/list.txt --output_dir %s --perf_profile burst --profiling_level basic "
                       "--log_level error; echo RC=$?" % (T, D, T, T, D, od, od), timeout=900)
        if "RC=0" not in o:
            print("  ⚠️ R5 失败（H3 判 🟡，D2 直接 A/B）：%s" % " ".join(o.split())[-300:])
            return {"tag": "R5", "ok": False, "err": o[-800:]}
        logs = [x for x in lab_dev.sh("ls %s; true" % od).split() if x.startswith("qnn-profiling-data")]
        hd = os.path.join(OUT, "R5")
        os.makedirs(hd, exist_ok=True)
        for lg in logs:
            lab_dev.pull("%s/%s" % (od, lg), os.path.join(hd, lg))
        print("  ✅ R5 在线建图单算子，日志 %d 个" % len(logs))
        return {"tag": "R5", "ok": True, "logs": logs}
    except (lab_dev.AdbFailed, OSError) as e:
        print("  ⚠️ R5 异常（H3 判 🟡）：%s" % str(e)[-300:])
        return {"tag": "R5", "ok": False, "err": str(e)[-800:]}


def collect():
    os.makedirs(OUT, exist_ok=True)
    if preflight() != 0:
        return 1
    print("\n== 停 app（避免后端占着 DSP 会话与 ION）==")
    lab_dev.sh("am force-stop %s" % PKG)
    runs = []
    try:
        lab_dev.sh("mkdir -p %s" % D)
        print("== 推送 ==")
        for seg in ("part1a", "part1b"):
            hd, names = make_inputs(seg)
            lab_dev.sh("mkdir -p %s/in_%s" % (D, seg))
            for n in names:
                lab_dev.push(os.path.join(hd, n + ".raw"), "%s/in_%s/%s.raw" % (D, seg, n))
            have = lab_dev.sh("stat -c %%s %s/%s.bin 2>/dev/null; true" % (D, seg)).strip()
            if have == str(os.path.getsize(HOST_CTX[seg])):
                print("  %s 已在设备且字节数吻合，跳过推送（sha256 仍会核）" % seg)
            else:
                print("  push %s context（%.2f GiB）…" % (seg, os.path.getsize(HOST_CTX[seg]) / 2**30), flush=True)
                lab_dev.push(HOST_CTX[seg], "%s/%s.bin" % (D, seg))
            dev_sha = lab_dev.sh("sha256sum %s/%s.bin" % (D, seg)).split()[0]
            if dev_sha != contract_graph(seg)["sha256"]:
                raise SystemExit("🔴 G0 设备侧 %s sha256 %s… ≠ 契约" % (seg, dev_sha[:12]))
            print("  ✅ 设备侧 %s sha256 == 契约" % seg)
        names = {s: [t["name"] for t in contract_graph(s)["inputs"]] for s in ("part1a", "part1b")}
        print("== 运行（part1a/part1b 交替，抵消热漂移）==")
        for tag, seg, lvl in (("R1", "part1a", "basic"), ("R2", "part1b", "basic"),
                              ("R3", "part1a", "detailed"), ("R4", "part1b", "detailed")):
            if lab_dev.mem_available_mb() < MIN_MEMAVAIL_MB:
                raise SystemExit("🔴 MemAvailable 低于 %d MiB，停止（不自动杀进程）" % MIN_MEMAVAIL_MB)
            runs.append(run_netrun(tag, seg, lvl, names[seg]))
        runs.append(run_r5())
    finally:
        # 🔴 只在全部成功时清理：失败时保留现场（2026-09-18 教训：R1 失败后把刚推的 3.6 GB 一起删了，
        #    再排查得重推一次）。失败时由 `--cleanup` 显式清理。
        ok_all = len(runs) >= 4 and all(r.get("ok", True) for r in runs)
        if lab_dev.online() and (ok_all or "--cleanup" in sys.argv):
            lab_dev.sh("rm -rf %s; true" % D)
            print("== 已删设备临时目录 %s ==" % D)
        elif not ok_all:
            print("== ⚠️ 未清理 %s（保留现场供排查；确认无用后加 --cleanup 再跑一次）==" % D)
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": runs},
                  open(os.path.join(OUT, "collect.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n✅ collect 完成。设备已不再占用 —— **可以拔线**。下一步：python scripts/p2_d1_profile.py analyze")
    return 0


# ---------------------------------------------------------------- analyze
def view(logpath):
    txt = logpath + ".viewer.txt"
    csv = logpath + ".viewer.csv"
    r = subprocess.run([VIEWER, "--input_log", logpath, "--reader", READER, "--output", csv],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    open(txt, "w", encoding="utf-8").write((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        print("  ⚠️ viewer rc=%d：%s" % (r.returncode, logpath))
    return r.stdout or ""


def nums_after(text, key):
    return [int(m.group(1)) for m in re.finditer(re.escape(key) + r"[^0-9\n]*?(\d+)\s*us", text)]


def node_cycles(text):
    """逐算子 cycles。

    🔴 只认带 `:OpId_<n>` 的行（2026-09-18 教训）：viewer 的格式是
        `Accelerator (execute) time (cycles) : 15120154894  cycles`   <- **整段总量**
        `    node_xxx:OpId_75 (cycles) : 1961385256  cycles`          <- 逐算子
    两者都带 `cycles`；把总量行当算子会让「单算子占比」直接减半（我第一版就错在这里，
    靠「出现一个占 49% 的算子叫 Accelerator (execute) time」才发现）。
    多次推理各出一段，按算子名取最大值 = 单次口径。
    """
    per = {}
    for m in re.finditer(r"^\s*(\S+:OpId_\d+)\s*\(cycles\)\s*[:：]\s*(\d+)\s*cycles\s*$", text, re.M):
        per[m.group(1)] = max(per.get(m.group(1), 0), int(m.group(2)))
    return per


def total_cycles(text):
    """整段 Accelerator (execute) time（cycles），多次推理取最大值。"""
    v = [int(m.group(1)) for m in re.finditer(r"Accelerator \(execute\) time \(cycles\)\s*[:：]\s*(\d+)", text)]
    return max(v) if v else 0


def analyze():
    cj = json.load(open(os.path.join(OUT, "collect.json"), encoding="utf-8"))
    res = {}
    for r in cj["runs"]:
        if r.get("ok") is False:
            res[r["tag"]] = {"ok": False}
            continue
        texts = [view(os.path.join(OUT, r["tag"], lg)) for lg in r["logs"]]
        t = "\n".join(texts)
        res[r["tag"]] = {
            "accel_us": nums_after(t, "Accelerator (execute) time"),
            "qnn_us": nums_after(t, "QNN (execute) time"),
            # 实测格式：`Number of HVX threads used : 6  count`
            "hvx": [int(x) for x in re.findall(r"Number of HVX threads used\s*[:：]\s*(\d+)", t)],
            "nodes": node_cycles(t) if r.get("level") == "detailed" else {},
        }
    print("# D1 判据（EXP_PLAN_P2_SPEED §3.3，事前锁定）\n")
    verdict = {}

    def med(xs):
        return statistics.median(xs) if xs else None

    a1a, a1b = med(res.get("R1", {}).get("accel_us", [])), med(res.get("R2", {}).get("accel_us", []))
    print("R1 part1a Accelerator execute（us）：%s" % res.get("R1", {}).get("accel_us"))
    print("R2 part1b Accelerator execute（us）：%s" % res.get("R2", {}).get("accel_us"))
    if a1a and a1b:
        g1 = a1a / a1b
        verdict["G1"] = 1.55 <= g1 <= 2.05
        print("**G1 代表性门**：part1a/part1b = %.3f（app 内 1.72~1.83，门 [1.55, 2.05]）⇒ %s"
              % (g1, "✅" if verdict["G1"] else "🔴 离线装置不代表 app，D1 不下归因结论"))
    else:
        verdict["G1"] = None
        print("**G1**：🟡 读不到 Accelerator execute 时间 ⇒ 人工读 `*.viewer.txt` 后再判（不得猜）")

    n3, n4 = res.get("R3", {}).get("nodes", {}), res.get("R4", {}).get("nodes", {})
    c1a, c1b = sum(n3.values()), sum(n4.values())
    print("\nR3 part1a 逐算子条目 %d、总 cycles %d；R4 part1b 条目 %d、总 cycles %d" % (len(n3), c1a, len(n4), c1b))
    if c1a and c1b:
        g1c = c1a / c1b
        verdict["G1c"] = 1.45 <= g1c <= 2.15
        print("**G1c cycles 忠实性门**：%.3f（门 [1.45, 2.15]）⇒ %s" % (g1c, "✅" if verdict["G1c"] else "🔴 cycles 不代表耗时，H1 判据作废，跑 linting"))
        s_nodes = {k: v for k, v in n3.items() if re.search(r"select_scatter(_4)?(\b|$|[^_0-9])", k)}
        s = sum(s_nodes.values())
        print("\nS（两个全尺寸 ScatterElements）命中：%s" % s_nodes)
        if verdict["G1"] and verdict["G1c"]:
            share, rest = s / c1a, (c1a - s) / c1b
            if share >= 0.15 and rest <= 1.40:
                verdict["H1"] = "🟢 支持"
            elif share <= 0.05:
                verdict["H1"] = "🔴 证否"
            else:
                verdict["H1"] = "🟡 不下结论"
            print("**H1**：S/C1a = %.1f%%，(C1a−S)/C1b = %.3f ⇒ %s" % (100 * share, rest, verdict["H1"]))
        top = sorted(((v, k) for k, v in n3.items() if k not in n4), reverse=True)[:10]
        print("\npart1a 独有算子 cycles 前 10：")
        for v, k in top:
            print("  %12d  %5.1f%%  %s" % (v, 100.0 * v / c1a, k))
    else:
        print("🟡 逐算子 cycles 未解析出来 ⇒ 人工读 R3/R4 的 `*.viewer.txt` / `.csv`，修解析器后重跑 analyze（不占设备）")

    hvx = res.get("R1", {}).get("hvx", []) + res.get("R2", {}).get("hvx", [])
    hmax = res.get("R5", {}).get("hvx", [])
    print("\nHVX 线程：离线 context %s；在线建图（SoC 最大）%s" % (hvx or "读不到", hmax or "读不到"))
    if hvx and hmax:
        verdict["H3"] = "进 D2" if max(hvx) < max(hmax) else "🔴 关闭（已用满）"
    else:
        verdict["H3"] = "🟡 D2 直接 A/B"
    print("**H3** ⇒ %s" % verdict["H3"])
    json.dump({"verdict": verdict, "parsed": {k: {kk: vv for kk, vv in v.items() if kk != "nodes"} for k, v in res.items()}},
              open(os.path.join(OUT, "analyze.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preflight":
        sys.exit(preflight())
    if cmd == "collect":
        sys.exit(collect())
    if cmd == "analyze":
        sys.exit(analyze())
    raise SystemExit(__doc__)
