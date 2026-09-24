# -*- coding: utf-8 -*-
"""D1 排障：qnn-net-run RC=17（执行阶段失败）的单变量二分。

做法（约束 8：先用已知样本验证装置）：
  A  part1b + #147 原样参数（无 perf_profile / 无 profiling）   <- 已知样本，应当成功
  B  A + --perf_profile burst
  C  B + --profiling_level basic
  D  part1a + 与成功臂相同的参数
  E  若 D 失败：加 --use_native_input_files 再试（文档说默认按 float32 解析输入）
每臂都抓 logcat（指南 §43.4：create/execute 失败的真实原因在 logcat，不在 stdout）。
"""
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\LocalDreamZImage\scripts")
import lab_dev
import p2_d1_profile as P

D = P.D
T = P.T
OUT = os.path.join(P.OUT, "bisect")


def push_all():
    lab_dev.sh("mkdir -p %s" % D)
    for seg in ("part1b", "part1a"):
        have = lab_dev.sh("stat -c %%s %s/%s.bin 2>/dev/null; true" % (D, seg)).strip()
        want = str(os.path.getsize(P.HOST_CTX[seg]))
        if have != want:
            print("  push %s（%.2f GiB）…" % (seg, int(want) / 2**30), flush=True)
            lab_dev.push(P.HOST_CTX[seg], "%s/%s.bin" % (D, seg))
        else:
            print("  %s 已在设备且字节数吻合，跳过推送" % seg)
        hd, names = P.make_inputs(seg)
        lab_dev.sh("mkdir -p %s/in_%s" % (D, seg))
        for n in names:
            dev = "%s/in_%s/%s.raw" % (D, seg, n)
            have = lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % dev).strip()
            if have != str(os.path.getsize(os.path.join(hd, n + ".raw"))):
                lab_dev.push(os.path.join(hd, n + ".raw"), dev)


def arm(tag, seg, extra, native=False):
    names = [t["name"] for t in P.contract_graph(seg)["inputs"]]
    od = "%s/o_%s" % (D, tag)
    line = " ".join("%s:=%s/in_%s/%s.raw" % (k, D, seg, k) for k in names)
    lab_dev.sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' '%s' > %s/list.txt" % (od, od, line, od))
    lab_dev.sh("logcat -c; true")
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context %s/%s.bin --backend libQnnHtp.so "
           "--input_list %s/list.txt --output_dir %s --log_level info %s%s; echo RC=$?"
           % (T, T, T, D, seg, od, od, extra, " --use_native_input_files" if native else ""))
    t0 = time.time()
    o = lab_dev.sh(cmd, timeout=1800)
    el = time.time() - t0
    rc = re.search(r"RC=(\d+)", o)
    rc = rc.group(1) if rc else "?"
    lg = lab_dev.sh("logcat -d -v brief 2>/dev/null | grep -iE 'qnn|dsp|adsp|fastrpc|error' | tail -40; true")
    outs = lab_dev.sh("ls -l %s/Result_0 2>/dev/null; true" % od)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "%s.txt" % tag), "w", encoding="utf-8").write(
        "CMD:\n%s\n\nSTDOUT:\n%s\n\nLOGCAT:\n%s\n\nOUTPUTS:\n%s\n" % (cmd, o, lg, outs))
    print("\n=== %s  seg=%s extra=%r native=%s  rc=%s  %.1fs" % (tag, seg, extra, native, rc, el))
    print("  输出文件: %s" % " ".join(outs.split()[-6:]) if outs.strip() else "  输出文件: 无")
    if rc != "0":
        keep = [l for l in lg.splitlines() if re.search(r"(?i)error|fail|E/", l)]
        print("  logcat 关键行:")
        for l in keep[-8:]:
            print("   ", l[:200])
    return {"tag": tag, "seg": seg, "extra": extra, "native": native, "rc": rc, "sec": el}


def main():
    lab_dev.require_online("bisect")
    print("MemAvailable %d MiB, NPU %.1f °C" % (lab_dev.mem_available_mb(), P.npu_c()))
    print("== 推送（已在设备且字节数吻合的跳过）==")
    push_all()
    res = []
    res.append(arm("A_p1b_plain", "part1b", " --num_inferences 1"))
    if res[-1]["rc"] == "0":
        res.append(arm("B_p1b_burst", "part1b", " --num_inferences 1 --perf_profile burst"))
        res.append(arm("C_p1b_basic", "part1b", " --num_inferences 1 --perf_profile burst --profiling_level basic"))
    else:
        res.append(arm("A2_p1b_native", "part1b", " --num_inferences 1", native=True))
    ok_extra = " --num_inferences 1 --perf_profile burst --profiling_level basic"
    res.append(arm("D_p1a", "part1a", ok_extra))
    if res[-1]["rc"] != "0":
        res.append(arm("E_p1a_native", "part1a", ok_extra, native=True))
        res.append(arm("F_p1a_plain", "part1a", " --num_inferences 1"))
    json.dump(res, open(os.path.join(OUT, "bisect.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n== 汇总 ==")
    for r in res:
        print("  %-14s %-7s rc=%-3s %5.1fs  extra=%s native=%s" % (r["tag"], r["seg"], r["rc"], r["sec"], r["extra"], r["native"]))
    print("\n设备临时目录保留在 %s（排查完再清）" % D)


if __name__ == "__main__":
    main()
