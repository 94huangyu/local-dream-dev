# -*- coding: utf-8 -*-
"""EXP_PLAN_MULTIGRAPH 阶段 0 · 第 1 步：造 A/B 两个图名不同、权重相同的量化 DLC。

样本：p0b/actcal/fc99.onnx（单 FullyConnected，权重 val_1800 [3840,3840] uFxp_8
      = 14,745,600 B，占 DLC 14.07 MB 的近 100%）⇒ 权重是否共享会直接体现在 context 体积上。
配方照抄 scripts/p0b_build.py 的 actcal 臂（min-max / w8 / a16 / bias32，无 overrides），
与 q_minmax.dlc 里记录的 Quantizer command 一致。
"""
import os, sys, subprocess, hashlib, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY, TOOL = sys.executable, os.path.join(HERE, "qairt_tool.py")
SRC = r"D:\ZImage_Work\p0_experiments\p0b\actcal\fc99.onnx"
XRAW = r"D:\ZImage_Work\p0_experiments\fc_inputs\cpu\out\Result_0\node_linear_99_pre_reshape.raw"
W = r"D:\ZImage_Work\p0_experiments\ws_stage0"


def sh(cmd):
    """透传 stderr 与返回码（约束 11·再补 / code_lint C5）。"""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout


def main():
    for f in (SRC, XRAW):
        assert os.path.isfile(f), "缺少源文件 %s" % f
    os.makedirs(W, exist_ok=True)
    lst = os.path.join(W, "input_list.txt")
    open(lst, "w").write("x:=" + XRAW + "\n")

    out = {}
    for tag in ("wsA", "wsB"):
        fdlc = os.path.join(W, "%s_fp32.dlc" % tag)
        qdlc = os.path.join(W, "%s_quant.dlc" % tag)
        if os.path.isfile(qdlc):
            print("[skip] %s 已存在" % qdlc, flush=True)
            out[tag] = qdlc
            continue

        t0 = time.time()
        print("[%s 1/2] converter -> %s" % (tag, os.path.basename(fdlc)), flush=True)
        rc, o = sh([PY, TOOL, "qairt-converter", "--input_network", SRC,
                    "--output_path", fdlc, "--float_bitwidth", "32"])
        if rc != 0:
            sys.exit("converter rc=%d\n%s" % (rc, o[-2000:]))

        print("[%s 2/2] quantizer -> %s" % (tag, os.path.basename(qdlc)), flush=True)
        rc, o = sh([PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                    "--input_list", lst, "--weights_bitwidth", "8", "--act_bitwidth", "16",
                    "--bias_bitwidth", "32", "--act_quantizer_calibration", "min-max",
                    "--param_quantizer_calibration", "min-max"])
        if rc != 0:
            sys.exit("quantizer rc=%d\n%s" % (rc, o[-2000:]))
        print("    用时 %.1f 分钟" % ((time.time() - t0) / 60.0), flush=True)
        out[tag] = qdlc

    # ---- 门 M1：图名确实是 fp32 DLC 主干，且 A/B 不同（约束 8：先验证操作化） ----
    print("\n=== 门 M1 图名 ===", flush=True)
    names = {}
    for tag, qdlc in out.items():
        rc, o = sh([PY, TOOL, "snpe-dlc-info", "-i", qdlc])
        assert rc == 0, "snpe-dlc-info rc=%d" % rc
        gl = [l for l in o.splitlines() if l.startswith("Info of graph:")]
        assert len(gl) == 1, "%s 图数 != 1: %r" % (tag, gl)
        names[tag] = gl[0].split(":", 1)[1].strip()
        print("  %s -> graph_name = %r" % (tag, names[tag]), flush=True)
    assert names["wsA"] == "wsA_fp32" and names["wsB"] == "wsB_fp32", \
        "图名不等于 fp32 DLC 主干，本方案的 graph_names 假设不成立: %r" % names
    assert names["wsA"] != names["wsB"], "A/B 图名相同"

    # ---- 门 M2：A/B 权重确实相同（否则测不出共享） ----
    print("\n=== 门 M2 权重同一性 ===", flush=True)
    a, b = [open(out[t], "rb").read() for t in ("wsA", "wsB")]
    print("  size A=%d  B=%d  相等=%s" % (len(a), len(b), len(a) == len(b)), flush=True)
    if len(a) == len(b):
        diff = sum(1 for x, y in zip(a, b) if x != y)
        print("  逐字节不同的字节数 = %d（%.6f%%）" % (diff, 100.0 * diff / len(a)), flush=True)
        print("  ⇒ 判读：只差图名字符串则应远小于权重体量 14,745,600 B", flush=True)
    print("\n  md5 A=%s\n  md5 B=%s" % (hashlib.md5(a).hexdigest(), hashlib.md5(b).hexdigest()),
          flush=True)
    print("\n✅ 阶段 0 第 1 步完成：%s" % W, flush=True)


main()
