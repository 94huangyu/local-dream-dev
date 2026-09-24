# -*- coding: utf-8 -*-
"""D1：真实双比例共享 context 在设备上的加载 / PD / ION。

与阶段 2 的区别：阶段 2 用的是**同形状的改名拷贝**（只为回答 PD 记账问题）；
本轮是**真实不同形状**（现网 1:1 图 + 新 4:3 图，权重共享）。

判据（事前锁定）：
  T1  不设 ENABLE_GRAPHS（全解 2 图）   —— 预期撞 0x3ea（2 图约 2x 单图，超 3.3GB 红线）
  T2  ENABLE_GRAPHS = 4:3 那个图        —— 必须成功，且 ION <= 1.15x 现网单图
  T3  ENABLE_GRAPHS = 1:1 那个图        —— 必须成功（证明两个图都能单独解开）
  装置门：T2/T3 里逐个 graphRetrieve 的结果一并打印（⚠️ 已知它对被禁用的图也返回 OK，
         **不得用它判定可用性**，仅作记录）

用法: python aspect_device_d1.py 1152x864 [--segs part1a,...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import lab_dev as L                                    # noqa: E402

MG = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg")
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
SDK_LIB = os.path.join("D:", os.sep, "qairt", "2.48.0.260626", "lib", "aarch64-android")
SKEL = os.path.join("D:", os.sep, "qairt", "2.48.0.260626", "lib", "hexagon-v79",
                    "unsigned", "libQnnHtpV79Skel.so")
DEV = "/data/local/tmp/aspmg"
SEGS = ["part1a", "part1b", "part2a", "part2b"]


def main():
    tag = sys.argv[1]
    segs = SEGS
    if "--segs" in sys.argv:
        segs = sys.argv[sys.argv.index("--segs") + 1].split(",")
    L.require_online("aspect_device_d1")

    print("=== 推送 ===", flush=True)
    L.sh("mkdir -p %s" % DEV)
    t0 = time.time()
    for n in ("libQnnHtp.so", "libQnnHtpV79Stub.so", "libQnnSystem.so"):
        p = os.path.join(SDK_LIB, n)
        if os.path.isfile(p):
            L.push(p, DEV + "/" + n)
    if os.path.isfile(SKEL):
        L.push(SKEL, DEV + "/libQnnHtpV79Skel.so")
    L.push(os.path.join("scripts", "quadctx_probe", "quadctx_probe_v2"), DEV + "/probe")
    L.sh("chmod 755 %s/probe" % DEV)
    for seg in segs:
        L.push(os.path.join(MG, seg, "%s_mg.SM8750.bin" % seg), DEV + "/%s_mg.bin" % seg)
    print("推送用时 %.1f 分钟" % ((time.time() - t0) / 60.0), flush=True)

    env = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && " % (DEV, DEV, DEV))
    for seg in segs:
        g11 = "%s_fp16_L80_fp32" % seg
        gnn = "%s_%s_fp32" % (seg, tag)
        single = os.path.getsize(os.path.join(
            P2, "ctx_%s_fp16_L80" % seg, "%s_fp16_L80.SM8750.bin" % seg)) / 1048576.0
        print("\n" + "=" * 66, flush=True)
        print("### %s  （现网单图 context %.1f MiB）" % (seg, single), flush=True)
        print("=" * 66, flush=True)
        for tname, want in (("T1 全解 2 图", "ALL"), ("T2 只解 4:3", gnn), ("T3 只解 1:1", g11)):
            cmd = "./probe ./libQnnHtp.so enable:%s ./%s_mg.bin %s %s" % (want, seg, g11, gnn)
            out = L.sh(env + cmd + '; echo "PROBE_RC=$?"')
            for line in out.splitlines():
                s = line.strip()
                if not s or s.startswith("backend 已加载") or s.startswith("rpcmem"):
                    continue
                print("  [%s] %s" % (tname, s), flush=True)
            L.sh("sleep 2")
    print("\n完成。判据见本文件头。", flush=True)


sys.exit(main())
