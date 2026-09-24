"""EXP_PLAN_WFXP_ACTFP 阶段 2：上机跑两臂，取输出与耗时。

判据见方案 §四（事前定稿）。本脚本只取数，不解释。

🔴 约束 3 / EXECUTION_MODEL：**不假设**输出 dtype。wfxp 臂图内输出是 Float_16，
   宿主侧到底拿到 float32 还是 float16 由**字节数门**判定，不得想当然。
🔴 约束 11·再补：辅助函数必须透传 stderr 与返回码，否则掉线会伪装成模型失败（#65/#83）。
"""
import os, sys, json, time, subprocess
sys.stdout.reconfigure(encoding="utf-8")
os.environ["MSYS_NO_PATHCONV"] = "1"          # #83②：否则 /sdcard、/data 被转成 Windows 路径
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
CTXD = os.path.join(P0B, "wfxp", "ctx")
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
T = "/data/local/tmp/htpcmp"
M, N = 4128, 3840
SER = "3B1F65EA9BBUMSHZ"
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"


def adb(args, timeout=1800):
    r = subprocess.run([ADB, "-s", SER] + args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:                      # 🔴 透传，不吞
        raise RuntimeError("adb %s 失败 rc=%d\nSTDOUT:%s\nSTDERR:%s"
                           % (" ".join(args[:2]), r.returncode, r.stdout[-800:], r.stderr[-800:]))
    return r.stdout


def sh(cmd, timeout=1800):
    return adb(["shell", cmd], timeout)


def main():
    print("设备:", sh("getprop ro.product.model").strip(), "|", sh("getprop ro.board.platform").strip())
    made = json.load(open(os.path.join(CTXD, "made.json"), encoding="utf-8"))
    print("[推送] 输入 %.1f MB" % (os.path.getsize(X_RAW) / 1e6), flush=True)
    adb(["push", X_RAW, "%s/wfxp_x.raw" % T])
    res = {}
    for tag, p in made.items():
        print("\n=== 臂 %s ===" % tag, flush=True)
        adb(["push", p, "%s/w_%s.bin" % (T, tag)])
        od = "%s/wfxp_%s" % (T, tag)
        sh("mkdir -p %s && rm -rf %s/Result_0" % (od, od))
        sh("printf '%%s\n' 'x:=%s/wfxp_x.raw' > %s/list.txt" % (T, od))
        t0 = time.time()
        o = sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
               "./qnn-net-run --retrieve_context w_%s.bin --backend libQnnHtp.so "
               "--input_list %s/list.txt --output_dir %s --log_level error 2>&1 | tail -4"
               % (T, T, T, tag, od, od))
        dt = time.time() - t0
        ok = "Finished Executing Graphs" in o
        print("  执行 %s  wall=%.1f s" % ("✅" if ok else "❌\n" + o, dt))
        if not ok:
            res[tag] = dict(error=o); continue
        rp = "%s/Result_0/linear_99_fc.raw" % od
        nb = int(sh("stat -c %%s %s" % rp).strip())
        exp32, exp16 = M * N * 4, M * N * 2
        dtype = "float32" if nb == exp32 else ("float16" if nb == exp16 else "??")
        print("  输出字节 %d ⇒ dtype=%s （float32 期望 %d / float16 期望 %d）"
              % (nb, dtype, exp32, exp16))
        if dtype == "??":
            res[tag] = dict(error="字节数 %d 不符任何期望" % nb); continue
        loc = os.path.join(P0B, "wfxp", "out_%s.raw" % tag)
        adb(["pull", rp, loc])
        res[tag] = dict(bytes=nb, dtype=dtype, wall=dt, local=loc)
    json.dump(res, open(os.path.join(P0B, "wfxp", "device.json"), "w"), indent=1)
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
