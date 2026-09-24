# -*- coding: utf-8 -*-
"""EXP_PLAN_MULTIGRAPH 阶段 2 —— 生死线：PD 红线按【已启用的图】还是【整个 context】算。

判据（事前锁定，见 EXP_PLAN_MULTIGRAPH §三 阶段 2 的 G2-PD / G2-ION / G2-装置）：
  T1 SHARE5 + ENABLE_GRAPHS=ALL      全解 5 图
  T2 SHARE5 + ENABLE_GRAPHS=1 个图   只解 1 图
  T3 CTRL1  (单图 context)           ION 基线

  T2 成功 且 T1 失败(0x3ea)  => PD 按【已启用的图】算 => 路线 D 成立，可扩到 5 比例
  T1、T2 都失败(0x3ea)       => PD 按【整个 context】算 => 路线 D 被封顶
  T1、T2 都成功              => 5 图整体也没超红线；需加图数再逼（本段单图 PD 估算 1927 MB）
  装置门：T2 里被禁用的 4 个图必须 graphRetrieve 失败 —— 取得到就说明 ENABLE_GRAPHS 没生效，
         该轮结论作废（§7.3：不得以「没报错」判定生效）
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab_dev as L

# 段由 argv[1] 指定（默认 part2b）。part1a 是最贴 PD 红线的一段。
SEG = sys.argv[1] if len(sys.argv) > 1 else "part2b"
W = r"D:\ZImage_Work\p0_experiments\pd_multigraph_%s" % SEG
if SEG == "part2b" and not os.path.isdir(W):
    W = r"D:\ZImage_Work\p0_experiments\pd_multigraph"
DEV = "/data/local/tmp/pdmg"
LIB = "/data/local/tmp/pdmg/libQnnHtp.so"
SDK_LIB = r"D:\qairt\2.48.0.260626\lib\aarch64-android"
NAMES = ["%s_fp16_L80_fp32" % SEG] + ["%s_fp16_L80_fpG%d" % (SEG, i) for i in (1, 2, 3, 4)]
SHARE5 = os.path.join(W, "SHARE5", "SHARE5.SM8750.bin")
CTRL1 = os.path.join(W, "CTRL1", "CTRL1.SM8750.bin")


def main():
    L.require_online("pd_device_test 开始")
    for f in (SHARE5, CTRL1):
        assert os.path.isfile(f), "缺 %s" % f

    print("=== 段 %s：推送两个 context（最慢的一步）===" % SEG, flush=True)
    L.sh("mkdir -p %s" % DEV)
    t0 = time.time()
    for name in ("libQnnHtp.so", "libQnnHtpV79Stub.so", "libQnnSystem.so"):
        p = os.path.join(SDK_LIB, name)
        if os.path.isfile(p):
            L.push(p, DEV + "/" + name)
    sk = r"D:\qairt\2.48.0.260626\lib\hexagon-v79\unsigned\libQnnHtpV79Skel.so"
    if os.path.isfile(sk):
        L.push(sk, DEV + "/libQnnHtpV79Skel.so")
    L.push(os.path.join("scripts", "quadctx_probe", "quadctx_probe_v2"), DEV + "/probe")
    L.sh("chmod 755 %s/probe" % DEV)
    L.push(SHARE5, DEV + "/SHARE5.bin")
    L.push(CTRL1, DEV + "/CTRL1.bin")
    print("推送用时 %.1f 分钟" % ((time.time() - t0) / 60.0), flush=True)

    env = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && " % (DEV, DEV, DEV))
    tests = [
        ("T1  SHARE5 全解 5 图",
         "./probe ./libQnnHtp.so enable:ALL ./SHARE5.bin " + " ".join(NAMES)),
        ("T2  SHARE5 只解 1 图",
         "./probe ./libQnnHtp.so enable:%s ./SHARE5.bin %s" % (NAMES[0], " ".join(NAMES))),
        ("T3  CTRL1 单图基线",
         "./probe ./libQnnHtp.so enable:ALL ./CTRL1.bin " + NAMES[0]),
    ]
    for tag, cmd in tests:
        print("\n" + "=" * 66, flush=True)
        print("### %s" % tag, flush=True)
        print("=" * 66, flush=True)
        # 🔴 探针在【预期的失败】(PD 撞线) 时返回非零，而 lab_dev.sh 非零即抛。
        # 所以让 adb shell 自身永远 rc=0，把探针的返回码显式回传（不吞、不伪装，约束 11·再补）。
        out = L.sh(env + cmd + '; echo "PROBE_RC=$?"')
        print(out, flush=True)
        prc = [l for l in out.splitlines() if l.startswith("PROBE_RC=")]
        print("[探针 rc = %s]" % (prc[0].split("=")[1] if prc else "未回传!"), flush=True)
        L.sh("sleep 3")

    print("\n完成。判据见本文件头。", flush=True)


main()
