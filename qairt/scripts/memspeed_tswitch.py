# -*- coding: utf-8 -*-
"""EXP_PLAN_MEMSPEED 实验 3 —— 测 **t_switch**（一次图切换的耗时）。

## 装置（为什么不是方案原文写的 aspect_mg）
见 `memspeed_probe_build.py` 的模块说明：aspect_mg 的五个图**权重共享**，
切换只搬 ~115 MiB；E 要切的是**不同段、无共享、1.4~2.9 GB**，差一个数量级。
本脚本用 `part1b + part2a` 合成的**双段单 context**（无共享权重）。

## 判据（`EXP_PLAN_MEMSPEED.md` §6 事前锁定，不得改）
  t_switch <= 2 s 🟢 进实验 4 ／ 2~4 s 🟡 报给用户取舍 ／ > 4 s 🔴 E/F 出局
🔴 **解读时必须同时对照 D 的实测代价 43.4 s**（#145/F）：
   E 要赢 D，需 32 次切换总代价 < 43.4 s，即 **t_switch < 1.36 s/次**。

## V0 生效门（先跑；不过就不解读任何时间数字）
两段公式合计 3685 MiB > PD 红线 3506~3535 MiB（#57）⇒ 三臂应当可区分：
  V0-a 两图都 enable、**不开**切换      期望 🔴 0x3ea（PD 上限）
  V0-b 两图都 enable、开切换            期望 🟢 成功，ION ≈ 单段
  V0-c 只 enable part1b（对照）         期望 🟢 成功，ION ≈ 单段
V0-a 若也成功 ⇒ PD 记账与理解不符，**t_switch 的解读全部作废**，先回去查。

## t_switch 怎么取（单变量，斜率法；不依赖 profiling 的可用性）
两臂**context 配置完全相同**（两图都 enable + 切换开），唯一差别是**执不执行图2**：
  臂 X（执行 图1×N 后 图2×N）: T_x(N) = init + N*e1 + t_switch + N*e2
  臂 Y（只执行 图1×N，图2 用 "__" 跳过）: T_y(N) = init + N*e1
  ⇒ T_x(N) − T_y(N) = t_switch + N*e2 ⇒ 对 N 做线性拟合，**截距就是 t_switch**
N 取 1/3/5。#147 用过同样的 N 斜率法。

## 设备占用
推 context 约 2.7 GiB（约 45 s）+ 6 次运行 ⇒ **约 15~25 分钟**，全程需要插着线。

用法:
  python scripts/memspeed_tswitch.py --prep     # 纯宿主：读规格、生成输入、自检（可反复跑）
  python scripts/memspeed_tswitch.py --run      # 需要设备
"""
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
UTIL = os.path.join(BIN, "qnn-context-binary-utility.exe")
PROBE = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "memspeed_probe")
WORK = os.path.join(PROBE, "run")
T = "/data/local/tmp/htpcmp"          # 设备上已有 qnn-net-run 与 QNN 库
DEV = T + "/memspeed"
NS = [1, 3, 5]


def meta(binpath):
    """从产物读回图规格。🔴 规格一律由产物导出，不从 ONNX/契约推导（约束 3 的同一条纪律）。"""
    out = os.path.join(WORK, "info.json")
    os.makedirs(WORK, exist_ok=True)
    r = subprocess.run([UTIL, "--context_binary", binpath, "--json_file", out],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not os.path.isfile(out):
        sys.stderr.write((r.stdout or "") + "\n" + (r.stderr or "") + "\n")
        raise SystemExit("🔴 读不回元数据 (rc=%d)" % r.returncode)
    d = json.load(open(out, encoding="utf-8"))
    graphs = []
    for g in d["info"]["graphs"]:
        gi = g["info"]
        ins = [(t["info"]["name"], list(t["info"]["dimensions"]), t["info"]["dataType"])
               for t in gi.get("graphInputs", [])]
        outs = [(t["info"]["name"], list(t["info"]["dimensions"]), t["info"]["dataType"])
                for t in gi.get("graphOutputs", [])]
        graphs.append({"name": gi["graphName"], "inputs": ins, "outputs": outs})
    return graphs


def prep():
    b = [f for f in os.listdir(PROBE) if f.endswith(".bin")] if os.path.isdir(PROBE) else []
    if len(b) != 1:
        raise SystemExit("🔴 probe context 还没建好（%r）—— 先跑 memspeed_probe_build.py" % b)
    binp = os.path.join(PROBE, b[0])
    gs = meta(binp)
    if len(gs) != 2:
        raise SystemExit("🔴 期望 2 个图，实得 %d 个：%r" % (len(gs), [g["name"] for g in gs]))
    os.makedirs(WORK, exist_ok=True)
    print("context: %s (%.1f MiB)" % (binp, os.path.getsize(binp) / 1048576.0))
    plan = {"bin": binp, "graphs": []}
    for g in gs:
        print("\n图 %s" % g["name"])
        files = []
        for name, dims, dt in g["inputs"]:
            n = 1
            for x in dims:
                n *= int(x)
            # 宿主接口一律 float32（约束 3：与图内部声明的定点类型无关）
            fp = os.path.join(WORK, "%s__%s.raw" % (g["name"], name.replace("/", "_")))
            if not os.path.isfile(fp) or os.path.getsize(fp) != n * 4:
                import numpy as np
                rng = np.random.default_rng(42)
                rng.standard_normal(n, dtype=np.float32).tofile(fp)
            print("   输入 %-24s %-22s %-28s %10d 元素 -> %d 字节"
                  % (name, "x".join(str(x) for x in dims), dt, n, n * 4))
            files.append((name, fp))
        outs = []
        for name, dims, dt in g["outputs"]:
            n = 1
            for x in dims:
                n *= int(x)
            outs.append((name, n * 4))
            print("   输出 %-24s %-22s %-28s 期望 %d 字节（float32 宿主口径）"
                  % (name, "x".join(str(x) for x in dims), dt, n * 4))
        plan["graphs"].append({"name": g["name"], "inputs": files, "outputs": outs})
    json.dump({"bin": binp,
               "graphs": [{"name": g["name"],
                           "inputs": [[n, os.path.basename(p)] for n, p in g["inputs"]],
                           "outputs": g["outputs"]} for g in plan["graphs"]]},
              open(os.path.join(WORK, "plan.json"), "w"), indent=1)
    print("\n✅ 宿主准备完成 -> %s" % os.path.join(WORK, "plan.json"))
    print("   下一步需要设备：python scripts/memspeed_tswitch.py --run（约 15~25 分钟，全程插线）")
    return plan


def cfg_json(path, graphs, switching, enable):
    """写一份 qnn-net-run 的 --config_file。

    🔴 `context_configs` 里的键来自 tools.html 的模板；**能不能真生效由 V0 门判**，
       不得因为「传了没报错」就当生效（#59：同一工具的 spill_fill/weights_buffer
       在 2.48 上解析得了却用不了）。
    """
    cc = {"enable_graphs": enable}
    if switching:
        cc["is_persistent_binary"] = True
        cc["memory_limit_hint"] = 1
    body = {"backend_extensions": {
        "shared_library_path": "%s/libQnnHtpNetRunExtensions.so" % T,
        "config_file_path": "%s/det.json" % DEV},
        "context_configs": cc}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=1)
    return body


def main():
    if "--prep" in sys.argv or len(sys.argv) == 1:
        prep()
        return 0
    if "--run" not in sys.argv:
        raise SystemExit("用法: --prep | --run")
    import lab_dev as dev
    plan = json.load(open(os.path.join(WORK, "plan.json"), encoding="utf-8"))
    dev.require_online("memspeed_tswitch")
    print("设备就绪。⚠️ 本脚本全程占用设备，结束前请勿拔线。", flush=True)
    # 装置自检：设备上得有 qnn-net-run 与扩展库
    for f in ("qnn-net-run", "libQnnHtp.so", "libQnnHtpNetRunExtensions.so"):
        out = dev.sh("ls -l %s/%s 2>&1 || true" % (T, f))
        print("   %-32s %s" % (f, out.strip()[:100]), flush=True)
        if "No such file" in out:
            raise SystemExit("🔴 设备缺 %s —— 先补齐 QNN 运行时再跑" % f)

    dev.sh("mkdir -p %s" % DEV)
    free = int(dev.sh("df -k /data | awk 'NR==2{print $4}'").strip() or "0") // 1024
    print("   /data 可用 %d MiB" % free, flush=True)
    if free < 6000:
        raise SystemExit("🔴 设备空间不足 6 GiB，先清理再跑")

    # ---- 推产物：每一次都核字节数（adb push 失败仍会打印成功行，见 RUNBOOK）----
    def push_checked(hp, dp):
        want = os.path.getsize(hp)
        dev.push(hp, dp)
        got = int(dev.sh("stat -c %%s %s" % dp).strip())
        if got != want:
            raise SystemExit("🔴 推送字节不符 %s: %d != %d" % (dp, got, want))
        return got

    t0 = time.time()
    # 产物已在设备上时可跳过重推 2.75 GiB（--skip-push）。
    # 🔴 但仍然核对字节数：跳过推送不等于跳过校验。
    if "--skip-push" in sys.argv:
        want = os.path.getsize(plan["bin"])
        got = int(dev.sh("stat -c %%s %s/probe.bin 2>/dev/null || echo 0" % DEV).strip())
        if got != want:
            raise SystemExit("🔴 --skip-push 但设备上的 probe.bin 字节不符 %d != %d" % (got, want))
        print("跳过推送（设备上 probe.bin 字节数已核对：%d）" % got, flush=True)
    else:
        print("推 context %.1f MiB …" % (os.path.getsize(plan["bin"]) / 1048576.0), flush=True)
        push_checked(plan["bin"], "%s/probe.bin" % DEV)
        for g in plan["graphs"]:
            for name, base in g["inputs"]:
                push_checked(os.path.join(WORK, base), "%s/%s" % (DEV, base))
        print("推送完成，%.0f s" % (time.time() - t0), flush=True)

    # ---- 配置文件：det 给 soc/dsp_arch；三份 cfg 分别对应 V0-a/b/c ----
    det = os.path.join(WORK, "det.json")
    json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]},
              open(det, "w"), indent=1)
    push_checked(det, "%s/det.json" % DEV)
    names = [g["name"] for g in plan["graphs"]]
    arms = {"V0a_both_noswitch": (False, names),
            "V0b_both_switch": (True, names),
            "V0c_one_only": (False, names[:1])}
    for tag, (sw, en) in arms.items():
        hp = os.path.join(WORK, "cfg_%s.json" % tag)
        cfg_json(hp, names, sw, en)
        push_checked(hp, "%s/cfg_%s.json" % (DEV, tag))

    # ---- 输入清单：每图 N 行；"__" 表示跳过该图 ----
    def make_lists(n):
        paths = []
        for g in plan["graphs"]:
            line = " ".join("%s:=%s/%s" % (nm, DEV, base) for nm, base in g["inputs"])
            hp = os.path.join(WORK, "list_%s_%d.txt" % (g["name"], n))
            with open(hp, "w", encoding="utf-8", newline=chr(10)) as f:
                f.write(chr(10).join([line] * n) + chr(10))
            dp = "%s/list_%s_%d.txt" % (DEV, g["name"], n)
            push_checked(hp, dp)
            paths.append(dp)
        return paths

    def run_once(tag, cfg, lists, label):
        """跑一次 qnn-net-run，返回 (rc, 墙钟秒, 输出文本, ION 峰值 MiB)。

        🔴 ION 每秒采一次：抽稀采样会漏掉真峰值（#161 实测漏成 7557 vs 真值 9486）。
        """
        od = "%s/out_%s" % (DEV, label)
        dev.sh("rm -rf %s && mkdir -p %s && rm -f %s/ion.txt %s/avail.txt %s/stop"
               % (od, od, DEV, DEV, DEV))
        # 🔴 PD 失败的原文只出现在 **logcat**（`QnnDsp <E> Failed to find available PD …
        #    with context size estimate N`），qnn-net-run 的 stdout 只给 RC=16
        #    （文档：16 = failure during create from binary，只说在哪一步，不说为什么）。
        #    第一版只翻 stdout ⇒ 把一次如期而至的 PD 失败判成「其它失败」。
        dev.sh("logcat -c || true")
        # 两个量分别落两个文件：ION 取最大、MemAvailable 取最小。
        # 合成一行再解析会让「取第一个数字字段」在两个量之间串味（第一版就写错了）。
        sampler = ("sh -c 'while [ ! -f %s/stop ]; do "
                   "grep IonTotalUsed /proc/meminfo >> %s/ion.txt; "
                   "grep MemAvailable /proc/meminfo >> %s/avail.txt; "
                   "sleep 1; done' >/dev/null 2>&1 &" % (DEV, DEV, DEV))
        dev.sh(sampler)
        cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
               "./qnn-net-run --retrieve_context %s/probe.bin --backend libQnnHtp.so "
               "--config_file %s --input_list %s --output_dir %s --log_level error; echo RC=$?"
               % (T, T, T, DEV, cfg, ",".join(lists), od))
        t = time.time()
        out = dev.sh(cmd, timeout=3600)
        el = time.time() - t
        dev.sh("touch %s/stop" % DEV)
        def series(fn):
            txt = dev.sh("cat %s/%s 2>/dev/null || true" % (DEV, fn))
            v = []
            for ln in txt.splitlines():
                f = [x for x in ln.replace("kB", "").split() if x.isdigit()]
                if f:
                    v.append(int(f[0]) // 1024)
            return v
        ions, avails = series("ion.txt"), series("avail.txt")
        if len(ions) < 3:
            print("     ⚠️ ION 采样点只有 %d 个 —— 采样器可能没起来，"
                  "内存数字不得引用" % len(ions), flush=True)
        peak = max(ions) if ions else 0
        low = min(avails) if avails else 0
        rc = 0 if "RC=0" in out else 1
        if rc != 0:
            pdlog = dev.sh("logcat -d 2>/dev/null | grep -i 'available PD' || true")
            if pdlog.strip():
                out = out + chr(10) + "[logcat] " + pdlog.strip()
        return rc, el, out, peak, low

    print("")
    print("=== V0 生效门（N=1）===", flush=True)
    lists1 = make_lists(1)
    v0 = {}
    for tag, (_, en) in arms.items():
        ls = lists1 if len(en) == 2 else [lists1[0], "__"]
        rc, el, out, peak, low = run_once(tag, "%s/cfg_%s.json" % (DEV, tag), ls, tag)
        # 🔴 完整输出一律落盘。第一版只在内存里留着并且**打印开头 300 字符**，
        #    而 qnn-net-run 的开头是横幅+输入清单，真正的报错在末尾
        #    ⇒ 把一次 PD 失败判成了「其它失败」。判读装置自己会说谎（约束 11·再补）。
        with io.open(os.path.join(WORK, "out_%s.txt" % tag), "w",
                     encoding="utf-8", errors="replace") as f:
            f.write(out)
        low_out = out.lower()
        pd = any(k in low_out for k in
                 ("0x3ea", "available pd", "context size estimate", "1002"))
        v0[tag] = {"rc": rc, "sec": el, "pd_fail": pd,
                   "ion_peak_mib": peak, "memavail_low_mib": low}
        print("  %-20s rc=%d  %6.1f s  ION峰值 %5d  MemAvail 最低 %5d MiB  %s"
              % (tag, rc, el, peak, low,
                 "🔴 PD 上限失败" if pd else ("✅ 成功" if rc == 0 else "⚠️ 其它失败")),
              flush=True)
        if rc != 0:
            # 打**末尾**：报错在那里
            print("     尾部: " + out.strip().replace(chr(10), " | ")[-500:], flush=True)

    ok = (v0["V0a_both_noswitch"]["pd_fail"] and v0["V0b_both_switch"]["rc"] == 0
          and v0["V0c_one_only"]["rc"] == 0)
    json.dump(v0, open(os.path.join(WORK, "v0.json"), "w"), indent=1)
    if not ok:
        print("")
        print("🔴 V0 门未按预期分辨（a 应失败、b/c 应成功）⇒ **不解读任何时间数字**。", flush=True)
        print("   PD 记账与理解不符时，先回 EXP_PLAN_MEMSPEED §1.3 的③重新查。", flush=True)
        return 2

    print("")
    print("=== t_switch（斜率法：X 执行两图，Y 只执行图1，配置完全相同）===", flush=True)
    rows = []
    for n in NS:
        ls = make_lists(n)
        rcx, tx, ox, px, lx = run_once("x", "%s/cfg_V0b_both_switch.json" % DEV, ls, "x%d" % n)
        rcy, ty, oy, py, ly = run_once("y", "%s/cfg_V0b_both_switch.json" % DEV,
                                       [ls[0], "__"], "y%d" % n)
        if rcx or rcy:
            raise SystemExit("🔴 N=%d 的臂失败 rc=%d/%d" % (n, rcx, rcy))
        rows.append((n, tx, ty, tx - ty, px, py))
        print("  N=%d  T_x=%7.2f  T_y=%7.2f  差=%6.2f s   ION 峰值 x/y = %d/%d MiB"
              % (n, tx, ty, tx - ty, px, py), flush=True)

    xs = [r[0] for r in rows]
    ys = [r[3] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    inter = my - slope * mx
    print("")
    print("  拟合 T_x−T_y = %.3f + %.3f·N  ⇒ **t_switch = %.2f s**，图2 单次执行 e2 = %.2f s"
          % (inter, slope, inter, slope), flush=True)
    verdict = ("🟢 <=2 s：进实验 4" if inter <= 2 else
               ("🟡 2~4 s：报给用户取舍" if inter <= 4 else "🔴 >4 s：E/F 出局"))
    print("  判据（方案 §6 事前锁定）: %s" % verdict, flush=True)
    print("  🔴 对照 D 的实测代价 43.4 s（#145/F）：E 要赢需 t_switch < 1.36 s/次；"
          "本次 %.2f s ⇒ %s" % (inter, "E 赢 D" if inter < 1.36 else "**D 更划算**"), flush=True)
    json.dump({"rows": rows, "t_switch_s": inter, "e2_s": slope, "verdict": verdict},
              open(os.path.join(WORK, "tswitch.json"), "w"), indent=1)
    dev.sh("rm -rf %s/out_* %s/stop" % (DEV, DEV))
    print("")
    print("✅ 测量结束。**设备已空闲，可以拔线。**", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
