"""EXP_PLAN_FP16_SURGICAL 上机：两臂各跑一次，取 part2a 输出 add_92。

🔴 两臂用**完全相同**的方式：--retrieve_context 加载现成 context（不做在线建图 ⇒ 不压内存）。
🔴 输入用今天已推到设备上的那一份（逐字节相同）。
约束 3：输出字节数必须是 4128*3840*4 = 63,406,080，防静默缩批。
"""
import os, sys, json, time, subprocess
sys.stdout.reconfigure(encoding="utf-8")
os.environ["MSYS_NO_PATHCONV"] = "1"
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
SER = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
ARMS = {"ctrl": os.path.join(W, "ctx_ctrl", "part2a_ctrl.SM8750.bin"),
        "fp16": os.path.join(W, "ctx_fp16", "part2a_fp16.SM8750.bin")}
EXPECT = 4128 * 3840 * 4


def adb(a, t=1800):
    r = subprocess.run([ADB, "-s", SER] + a, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=t)
    if r.returncode:                       # 🔴 透传，不吞（#65）
        raise RuntimeError("adb %s rc=%d\n%s\n%s" % (a[:2], r.returncode, r.stdout[-600:], r.stderr[-600:]))
    return r.stdout


def sh(c, t=1800):
    return adb(["shell", c], t)


def main():
    print("设备:", sh("getprop ro.product.model").strip())
    print("内存:", sh("grep MemAvailable /proc/meminfo").strip())
    res = {}
    for tag, p in ARMS.items():
        print("\n=== 臂 %s ===" % tag, flush=True)
        if not os.path.exists(p):
            print("  ❌ 缺 %s" % p); continue
        t0 = time.time()
        adb(["push", p.replace(os.sep, "/"), "%s/p2a_%s.bin" % (T, tag)])
        print("  推送 %.0f MB  %.0f s" % (os.path.getsize(p) / 1e6, time.time() - t0), flush=True)
        od = "%s/p2a_%s_out" % (T, tag)
        sh("rm -rf %s && mkdir -p %s" % (od, od))
        t1 = time.time()
        o = sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
               "./qnn-net-run --retrieve_context %s/p2a_%s.bin --backend libQnnHtp.so "
               "--input_list %s/p2attr/in/list.txt --output_dir %s --log_level error 2>&1 | tail -4"
               % (T, T, T, T, tag, T, od))
        ok = "Finished Executing Graphs" in o
        print("  执行 %s  %.0f s" % ("✅" if ok else "❌\n" + o, time.time() - t1), flush=True)
        if not ok:
            res[tag] = {"error": o}; continue
        rp = "%s/Result_0/add_92.raw" % od
        nb = int(sh("stat -c %%s %s" % rp).strip())
        print("  输出字节 %d（期望 %d）%s" % (nb, EXPECT, "✅" if nb == EXPECT else "❌ 静默缩批"))
        if nb != EXPECT:
            res[tag] = {"error": "bytes %d" % nb}; continue
        loc = os.path.join(W, "htp", "add_92_%s.raw" % tag)
        adb(["pull", rp, loc.replace(os.sep, "/")])
        res[tag] = {"bytes": nb, "wall": time.time() - t1, "local": loc}
    json.dump(res, open(os.path.join(W, "device_2arm.json"), "w"), indent=1)
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))
    print("\n✅ 设备部分完成 —— 可以拔线")


if __name__ == "__main__":
    main()
