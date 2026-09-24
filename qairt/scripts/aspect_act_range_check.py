# -*- coding: utf-8 -*-
"""预检：新比例的激活是否超出【部署 encoding 的量程】。

## 为什么这是 D2 最可能的真实失败原因
新比例的量化**复用了 1:1 的 encoding**（overrides 原样套用，门 Q1a/Q1b 已证权重逐字节相同）。
权重没问题，但**激活量程是按 1:1 的分布定的**。若新比例的激活超出该量程 =>
被钳到 encoding 的 min/max => 画质掉。这正是 #53 / #67 / #110 那一族的失效模式。

## 做法
跑**一步** FP32（step 0）拿段间张量，量它们的 |max|，与部署 encoding 的 max 比。
只跑一步，约 4~5 分钟，零设备。

## 判据（执行前锁定）
  比值 = 新比例实测 |max| / 部署 encoding 的 max
    <= 1.00  🟢 完全在量程内
    <= 1.10  🟡 略超（8/16-bit 下顶端 10% 的钳位影响有限，但要记）
    >  1.10  🔴 明显超量程 => D2 若画质差，这是首要嫌疑
⚠️ 本检查**只能否决不能放行**（指南 §41）：不超量程**不等于**画质没问题。

用法: python aspect_act_range_check.py <tag> <宽> <高>
"""
import io
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
CAP = 80

# 段间切口（逐字照抄 htp_inloop_pipeline.py 的定义）
BOUNDARY = {
    "part1a": ["add_138", "add_131", "tanh_19", "adaln_input", "select_45", "select_46"],
    "part1b": ["unified"],
    "part2a": ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3"],
}
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def deployed_ranges():
    """从部署的量化 DLC 读出每个切口张量的 encoding 量程。"""
    import subprocess
    out = {}
    for seg in ("part1a", "part1b", "part2a"):
        dlc = os.path.join(P2, "%s_fp16_L80_quantized.dlc" % seg)
        csv = os.path.join(P0, "actrange_%s.csv" % seg)
        if not os.path.isfile(csv):
            # 🔴 必须查返回码并透传错误（C5 / #65）。且 qairt_tool 走**系统 python**，
            # 不能用 sys.executable —— 本脚本可能由 venv-official 的 python 启动。
            r = subprocess.run(["python",
                                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "qairt_tool.py"),
                                "snpe-dlc-info", "-i", dlc, "-d", "-s", csv],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0 or not os.path.isfile(csv):
                print("FAIL snpe-dlc-info rc=%d" % r.returncode)
                print((r.stdout or "")[-800:])
                sys.exit(2)
        for line in io.open(csv, encoding="utf-8", errors="replace"):
            for m in PT.finditer(line):
                nm = m.group(1)
                if nm in BOUNDARY.get(seg, []):
                    out.setdefault(nm, (float(m.group(3)), float(m.group(4))))
    return out


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    lh, lw = H // 8, W // 8
    uni = CAP + (lh // 2) * (lw // 2)
    print("预检 %s: latent %dx%d unified %d" % (tag, lh, lw, uni), flush=True)

    rng = deployed_ranges()
    print("部署 encoding 量程读到 %d 个切口张量" % len(rng), flush=True)
    if not rng:
        print("FAIL 读不到 encoding")
        return 1

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    cwd = os.getcwd()
    os.chdir(E)
    vals = {
        "latents": np.fromfile(os.path.join(P0, "lat_init_%s_rng42.raw" % tag),
                               np.float32).reshape(1, 16, lh, lw),
        "timestep": np.array([1.0 - 1.0], dtype=np.float32),   # step 0: sigma=1.0
        "caption": np.fromfile(os.path.join(P0, "cap_r4x3.raw"),
                               np.float32).reshape(1, CAP, 2560),
        "cap_pad_mask": np.fromfile(os.path.join(P0, "mask_r4x3.raw"),
                                    np.float32).reshape(1, CAP).astype(bool),
    }
    stems = {"part1a": "transformer_part1a_clip_L80_%s" % tag,
             "part1b": "transformer_part1b_clip_L80_%s" % tag,
             "part2a": "transformer_part2a_fixed_clip_L80_%s" % tag}
    got = {}
    try:
        for seg in ("part1a", "part1b", "part2a"):
            s = ort.InferenceSession(stems[seg] + ".onnx", so,
                                     providers=["CPUExecutionProvider"])
            names = [i.name for i in s.get_inputs()]
            miss = [nm for nm in names if nm not in vals]
            if miss:
                print("  %s 缺输入 %s" % (seg, miss))
                return 1
            outs = s.run(None, {nm: vals[nm] for nm in names})
            for o, v in zip(s.get_outputs(), outs):
                vals[o.name] = v
                if o.name in BOUNDARY[seg] and v.dtype.kind == "f":
                    got[o.name] = (float(np.nanmin(v)), float(np.nanmax(v)))
            del s
            print("  %s 跑完" % seg, flush=True)
    finally:
        os.chdir(cwd)

    print("\n张量                 部署量程[min,max]        新比例实测[min,max]      |max|比  判定")
    worst, rows = 0.0, []
    for nm in sorted(got):
        if nm not in rng:
            continue
        dmin, dmax = rng[nm]
        gmin, gmax = got[nm]
        span = max(abs(dmin), abs(dmax))
        got_span = max(abs(gmin), abs(gmax))
        r = got_span / span if span else 0.0
        worst = max(worst, r)
        flag = "🟢" if r <= 1.0 else ("🟡" if r <= 1.10 else "🔴")
        print("%-20s [%9.3f,%9.3f]  [%9.3f,%9.3f]  %6.3f  %s"
              % (nm, dmin, dmax, gmin, gmax, r, flag))
        rows.append((nm, r))
    print("\n最大比值 = %.3f" % worst)
    print("判定: %s" % ("🟢 全部在量程内" if worst <= 1.0 else
                        "🟡 略超（<=1.10），记录但不阻塞" if worst <= 1.10 else
                        "🔴 明显超量程 —— D2 若画质差，这是首要嫌疑"))
    print("⚠️ 本检查只能否决不能放行（指南 §41）：不超量程不等于画质没问题。")
    return 0


sys.exit(main())
