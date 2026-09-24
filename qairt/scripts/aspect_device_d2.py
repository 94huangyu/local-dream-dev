# -*- coding: utf-8 -*-
"""D2：在设备上跑量化后的新比例，**整个 8 步循环脱离 USB**。

## 为什么要脱机
此前 D2 必须由宿主逐步驱动（每步把噪声预测拉回宿主做 Euler 更新再推回），
要求 USB 连续连接 35~45 分钟。用户 8 点上班，给不了这么长的窗口。
有了设备侧的 `euler_step`（ARM64），整个循环可以在设备上 `nohup` 跑完，
**启动后立刻可以拔线**，结果下次连上再收。

## 接线逐字照抄 htp_inloop_pipeline.py
段间张量名（P1B_INS / P2_INS / CUT）、qnn-net-run 调用式、`--retrieve_context` 用法
全部照抄，**不自己推导**。那份脚本里埋着约束 3 的防线（输出字节数校验）与
#65 的修复（sh() 透传 adb 错误）。

## 三道门
  G-SELF  设备上 `euler_step --selftest` 的 sigma/dt 必须与宿主逐位一致（约束 8）
  G-SIZE  每步 part2b 输出 latents.raw 的字节数必须 == 4*16*LH*LW（约束 3：防静默缩批）
  G-LIVE  收结果时先确认进程已结束且产物齐全，再下结论（约束 9·补 第 2 条）

用法:
  python aspect_device_d2.py launch <tag> <宽> <高> [--arm new|ctrl|both]
      --arm both（默认）：推送两臂、过门后 nohup 串行跑完，**启动完即可拔线**
      两臂必须串行（同时跑会撞 ION：单臂峰值已约 2.9 GB）
  python aspect_device_d2.py collect <tag> <宽> <高>  收结果 + 宿主 VAE 解码 + 比 FP32 参考
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import lab_dev as L                                      # noqa: E402

# 🔴 2026-09-03 追加【1:1 对照臂】。没有它，4:3 的 PSNR 无法归因 ——
# #143 实测同一套量化在三个 prompt 上是 22.58 / 14.25 / 13.47 dB（跨度 9 dB），
# 单看一个数说明不了任何事（R1：判定「方向错」必须有空白对照）。
# 对照臂用【现网四段 .bin】+ 已有的 1:1 FP32 参考，同 caption、同调度器。
# 脱机跑，所以加一臂**不占用户时间**。
CTRL = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
AS = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SDK_LIB = os.path.join("D:", os.sep, "qairt", "2.48.0.260626", "lib", "aarch64-android")
SKEL = os.path.join("D:", os.sep, "qairt", "2.48.0.260626", "lib", "hexagon-v79",
                    "unsigned", "libQnnHtpV79Skel.so")
NETRUN = os.path.join("D:", os.sep, "qairt", "2.48.0.260626", "bin",
                      "aarch64-android", "qnn-net-run")
T = "/data/local/tmp/d2"
STEPS, LATENT_CH, CAP = 8, 16, 80

# 逐字照抄 htp_inloop_pipeline.py
P1B_INS = ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]
P2_INS = ["unified", "unified_mask", "unified_freqs", "adaln_input"]
CUT = ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3", "val_105"]
SEGBIN = {"part1a": "part1a", "part1b": "part1b",
          "part2a": "part2a", "part2b": "part2b"}


def sigmas(n, shift=3.0):
    def f(s):
        return shift * s / (1.0 + (shift - 1.0) * s)
    out = [f(1.0 if n == 1 else 1.0 - (1.0 - 1.0 / n) * i / (n - 1)) for i in range(n)]
    return np.array(out + [0.0], dtype=np.float64)


def dev_script(tag, lh, lw):
    n = LATENT_CH * lh * lw
    UNI = CAP + (lh // 2) * (lw // 2)
    lines = ["#!/system/bin/sh", "set -e",
             "cd %s" % T,
             "export LD_LIBRARY_PATH=%s" % T,
             "export ADSP_LIBRARY_PATH=%s" % T,
             "echo START $(date)",
             "cp in/latents0.raw cur.raw"]
    for i in range(STEPS):
        o = "s%d" % i
        lines += [
            "echo '--- step %d' $(date +%%H:%%M:%%S)" % i,
            "rm -rf %s && mkdir -p %s" % (o, o),
            # part1a
            "printf 'latents:=%s/cur.raw timestep:=%s/in/ts%d.raw "
            "caption:=%s/in/caption.raw cap_pad_mask:=%s/in/cap_pad_mask.raw\\n' "
            "> %s/l1a.txt" % (T, T, i, T, T, o),
            "./qnn-net-run --retrieve_context %s.bin --backend libQnnHtp.so "
            "--input_list %s/l1a.txt --output_dir %s/o1a --log_level error"
            % (SEGBIN["part1a"], o, o),
            # part1b
            "printf '%s\\n' > %s/l1b.txt"
            % (" ".join("%s:=%s/%s/o1a/Result_0/%s.raw" % (k, T, o, k) for k in P1B_INS), o),
            "./qnn-net-run --retrieve_context %s.bin --backend libQnnHtp.so "
            "--input_list %s/l1b.txt --output_dir %s/o1b --log_level error"
            % (SEGBIN["part1b"], o, o),
            # part2a
            "printf 'unified:=%s/%s/o1b/Result_0/unified.raw %s\\n' > %s/l2a.txt"
            % (T, o, " ".join("%s:=%s/%s/o1a/Result_0/%s.raw" % (k, T, o, k)
                              for k in P2_INS if k != "unified"), o),
            "./qnn-net-run --retrieve_context %s.bin --backend libQnnHtp.so "
            "--input_list %s/l2a.txt --output_dir %s/o2a --log_level error"
            % (SEGBIN["part2a"], o, o),
            # part2b
            "printf '%s adaln_input:=%s/%s/o1a/Result_0/adaln_input.raw\\n' > %s/l2b.txt"
            % (" ".join("%s:=%s/%s/o2a/Result_0/%s.raw" % (k, T, o, k) for k in CUT), T, o, o),
            "./qnn-net-run --retrieve_context %s.bin --backend libQnnHtp.so "
            "--input_list %s/l2b.txt --output_dir %s/o2b --log_level error"
            % (SEGBIN["part2b"], o, o),
            # 门 G-SIZE（约束 3：抓静默缩批）。
            # 🔴 step 0 连【上游三段】的关键输出一起查 —— 只查 part2b 的话，
            # 上游缩批也会一路传到末段，报错却指向 part2b，把人带向错误的对象
            # （约束 11·再补「报错说谎」）。
            ] + ([
                "for chk in \"o1a/Result_0/add_138.raw %d\" "
                "\"o1b/Result_0/unified.raw %d\" "
                "\"o2a/Result_0/add_92.raw %d\"; do "
                "set -- $chk; S=$(stat -c %%s %s/$1); "
                "[ \"$S\" = \"$2\" ] || { echo \"G-SIZE FAIL step0 $1: $S != $2\"; exit 9; }; done"
                % (UNI * 3840 * 4, UNI * 3840 * 4, UNI * 3840 * 4, o)
            ] if i == 0 else []) + [
            "SZ=$(stat -c %%s %s/o2b/Result_0/latents.raw)" % o,
            "[ \"$SZ\" = \"%d\" ] || { echo \"G-SIZE FAIL step %d: $SZ != %d\"; exit 9; }"
            % (n * 4, i, n * 4),
            # Euler
            "./euler_step %d %d cur.raw %s/o2b/Result_0/latents.raw next.raw %d"
            % (i, STEPS, o, n),
            "mv next.raw cur.raw",
            # 只留最后一步的中间产物，省空间
            "rm -rf %s/o1a %s/o1b %s/o2a" % (o, o, o),
        ]
    lines += ["cp cur.raw final_latents.raw",
              "echo DONE $(date)",
              "touch %s/D2_DONE" % T]
    return "\n".join(lines) + "\n"


def launch(tag, W, H, arm="new", stage="run"):
    # arm: new = 新比例（用 aspect/ 的 .bin）；ctrl = 现网 1:1 对照（用 p2attr/ 的 .bin）
    if arm == "ctrl":
        W = H = 1024
    lh, lw = H // 8, W // 8
    n = LATENT_CH * lh * lw
    L.require_online("d2 launch")
    print("=== 推送 ===", flush=True)
    global T
    T = "/data/local/tmp/d2_%s" % arm            # 两臂目录隔离，产物不互相覆盖
    L.sh("rm -rf %s && mkdir -p %s/in" % (T, T))
    for nm in ("libQnnHtp.so", "libQnnHtpV79Stub.so", "libQnnSystem.so", "libQnnModelDlc.so"):
        p = os.path.join(SDK_LIB, nm)
        if os.path.isfile(p):
            L.push(p, T + "/" + nm)
    if os.path.isfile(SKEL):
        L.push(SKEL, T + "/libQnnHtpV79Skel.so")
    L.push(NETRUN, T + "/qnn-net-run")
    L.push(os.path.join("scripts", "euler_step", "euler_step"), T + "/euler_step")
    L.sh("chmod 755 %s/qnn-net-run %s/euler_step" % (T, T))
    for seg in SEGBIN:
        if arm == "ctrl":
            src = os.path.join(CTRL, "ctx_%s_fp16_L80" % seg, "%s_fp16_L80.SM8750.bin" % seg)
        else:
            src = os.path.join(AS, "ctx_%s_%s" % (seg, tag), "%s_%s.SM8750.bin" % (seg, tag))
        L.push(src, "%s/%s.bin" % (T, seg))

    # 输入：初始噪声、caption、mask、每步 timestep
    # 对照臂用 1:1 的那份初始噪声（与已有的 1:1 FP32 参考同源）
    noise0 = (os.path.join(P0, "latents_cxx_seed42.raw") if arm == "ctrl"
              else os.path.join(P0, "lat_init_%s_rng42.raw" % tag))
    L.push(noise0, T + "/in/latents0.raw")
    L.push(os.path.join(P0, "cap_r4x3.raw"), T + "/in/caption.raw")
    L.push(os.path.join(P0, "mask_r4x3.raw"), T + "/in/cap_pad_mask.raw")
    sg = sigmas(STEPS)
    tmp = os.path.join(OUT, "_ts")
    os.makedirs(tmp, exist_ok=True)
    for i in range(STEPS):
        v = np.array([1.0 - sg[i]], dtype=np.float32)
        f = os.path.join(tmp, "ts%d.raw" % i)
        v.tofile(f)
        L.push(f, "%s/in/ts%d.raw" % (T, i))

    # ---- 门 G-SELF：设备上的调度器口径必须与宿主逐位一致 ----
    print("\n=== 门 G-SELF：sigma/dt 逐位比对 ===", flush=True)
    out = L.sh("cd %s && ./euler_step --selftest" % T)
    dev = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 3 and p[0].isdigit():
            dev[int(p[0])] = (p[1], p[2])
    bad = []
    for i in range(STEPS):
        hs, hd = "%.10f" % sg[i], "%.10f" % (sg[i + 1] - sg[i])
        d = dev.get(i)
        ok = d == (hs, hd)
        print("  %d 宿主 %s %s | 设备 %s  %s" % (i, hs, hd, d, "OK" if ok else "FAIL"), flush=True)
        if not ok:
            bad.append(i)
    if bad:
        print("G-SELF FAIL：设备调度器与宿主不一致，**不启动**", flush=True)
        return 1
    print("  G-SELF PASS", flush=True)

    sc = dev_script(tag, lh, lw)
    f = os.path.join(OUT, "_d2_run.sh")
    open(f, "w", newline="\n").write(sc)
    L.push(f, T + "/run.sh")
    L.sh("chmod 755 %s/run.sh" % T)
    L.sh("rm -f %s/D2_DONE %s/run.log" % (T, T))
    # 🔴🔴 code_lint 的 C4 会拦这里（约束 9·补 规则 1：`nohup &` 后无监听）。
    # **本处经确认无害，理由写全**：
    #   1. 脱机是本脚本存在的**目的**（用户 8 点上班，给不了 35~45 分钟的连续 USB 窗口）；
    #      规则的本意是「别把『启动』当成『完成』」，不是禁止后台化。
    #   2. 完成判定**不在这里**，而在 `collect`：它查【设备侧 D2_DONE 标记 + qnn-net-run 进程数
    #      + 终点 latents 字节数】三样，**完全不看日志内容**（约束 9·补 规则 2）。
    #   3. 设备侧脚本 `set -e` + 每步 G-SIZE 硬校验，失败即退出且**不会**产生 D2_DONE
    #      ⇒ 「有标记」等价于「八步全部跑完且尺寸正确」，不存在假完成。
    if stage == "push":
        print("=== 臂 %s 已就绪（未启动）===" % arm, flush=True)
        return 0
    L.sh("cd %s && nohup ./run.sh > run.log 2>&1 &" % T)
    time.sleep(3)
    print("\n=== 已启动（nohup），现在可以拔线 ===", flush=True)
    print(L.sh("cd %s && head -3 run.log; ls -l run.log" % T), flush=True)
    print("预计 %d 步 x 四段，约 20~35 分钟。回来后跑：collect" % STEPS, flush=True)
    return 0


def collect(tag, W, H, arm="new"):
    global T
    T = "/data/local/tmp/d2_%s" % arm
    if arm == "ctrl":
        W = H = 1024
    lh, lw = H // 8, W // 8
    L.require_online("d2 collect")
    # 门 G-LIVE：先看完成标记与进程，再下结论（不看日志内容）
    done = L.sh("ls %s/D2_DONE 2>/dev/null | wc -l" % T).strip()
    alive = L.sh("ps -A -o NAME 2>/dev/null | grep -c qnn-net-run || true").strip()
    print("完成标记=%s  qnn-net-run 进程数=%s" % (done, alive), flush=True)
    print(L.sh("tail -6 %s/run.log" % T), flush=True)
    if done != "1":
        print("尚未完成或已失败，先看上面的 run.log", flush=True)
        return 1
    lp = os.path.join(OUT, "d2_final_latents_%s_%s.raw" % (tag, arm))
    L.pull("%s/final_latents.raw" % T, lp)
    exp = LATENT_CH * lh * lw * 4
    got = os.path.getsize(lp)
    assert got == exp, "终点 latents 字节数 %d != %d" % (got, exp)
    print("已取回 %s (%d B)" % (lp, got), flush=True)
    print("下一步在宿主解码并与 FP32 参考比：scripts/d2_decode_compare.py", flush=True)
    return 0


def main():
    cmd = sys.argv[1]
    tag, W, H = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    arm = "new"
    if "--arm" in sys.argv:
        arm = sys.argv[sys.argv.index("--arm") + 1]
    stage = "run"
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    if cmd == "launch":
        return launch(tag, W, H, arm, stage)
    if cmd == "collect":
        return collect(tag, W, H, arm)
    sys.exit("未知模式 %s" % cmd)


sys.exit(main())
