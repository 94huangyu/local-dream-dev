# -*- coding: utf-8 -*-
"""多步激活量程检查：新比例复用 1:1 的 encoding，是否在任何一步被钳？

## 为什么补这一个（这是我方案里的一个真空洞）
到目前为止，三个新比例**没有任何画质证据**——所有已过的门（spill-fill、W1/W2、
多图元数据等价）都是结构/可行性，不是画质。原计划把画质全押在设备上的 G-QUALITY，
可万一某个比例挂了，用户那次短插线就白费。
⇒ 先在宿主上把**最可能的失效机制**查掉：
   新比例的激活超出 1:1 标定的量程 => 被钳 => 画质掉（#53/#67/#110 那一族）。

## 与旧版 `aspect_act_range_check.py` 的差别
旧版只测 **step 0**。但扩散过程里激活分布随 sigma 变化，step 0 未必是最大的那一步。
本版直接复用已经算好的**逐步 FP32 latents**（`fp32_steps_L80_<tag>/lat_i.raw`），
在若干步上各测一次 —— 无需重跑扩散，成本约 3 分钟/步。

## 判据
🔴🔴 **2026-09-04 订正：第一版判据是错的，用了绝对阈值。**
第一版写的是「比值 = 实测 |max| / 部署 encoding max，>1.10 判 🔴」。
实跑三个新比例全部在 step 7 得 1.153 而判负 —— 但补做 **1:1 对照臂**后发现
**现网 1:1 在同一步也是 1.153**（add_138 = 1820.746，三个新比例 1820.5~1820.9）。
⇒ 超量程是**部署 encoding 本身的性质**，现网带着同样的钳位一直在正常出图，
   与新几何无关。绝对阈值只会把现网也一起判负，**没有决策价值**。
⇒ 本项目已为此栽过两次（#143 绝对 PSNR 跨度 9 dB；D2 因此改用两臂之差），
   我明知这条纪律却又写成绝对值。

**现判据（差值口径）**：先跑 `1x1` 对照臂，再比
    Δ = 该比例的比值 − 1:1 在同一步的比值
      |Δ| <= 0.02   🟢 与现网等价
      |Δ| <= 0.10   🟡 记录
      |Δ| >  0.10   🔴 新几何确实带来了额外钳位
绝对比值仍然打印（用于了解 encoding 的松紧），但**不作判据**。
⚠️ **只能否决不能放行**（指南 §41）：不超量程**不等于**画质没问题。
⚠️ 措辞纪律：本脚本给的是「宿主 FP32 前向下的激活范围」，**不是设备实测**。

用法: python aspect_range_multistep.py <tag> <宽> <高> [--steps 0,2,4,7]
"""
import io
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
CAP = 80
BOUNDARY = {
    "part1a": ["add_138", "add_131", "tanh_19", "adaln_input", "select_45", "select_46"],
    "part1b": ["unified"],
    "part2a": ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3"],
}
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")
# flow-match 调度：与 t2_fp32_ref.py / C++ 侧一致（shift=3.0）
SHIFT, STEPS = 3.0, 8


def sigmas():
    s = [1.0 - i / float(STEPS) for i in range(STEPS + 1)]
    return [SHIFT * x / (1.0 + (SHIFT - 1.0) * x) for x in s]


def deployed_ranges():
    out = {}
    for seg in ("part1a", "part1b", "part2a"):
        dlc = os.path.join(P2, "%s_fp16_L80_quantized.dlc" % seg)
        csv = os.path.join(P0, "actrange_%s.csv" % seg)
        if not os.path.isfile(csv):
            r = subprocess.run(["python", os.path.join(HERE, "qairt_tool.py"),
                                "snpe-dlc-info", "-i", dlc, "-d", "-s", csv],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0 or not os.path.isfile(csv):
                print("FAIL snpe-dlc-info rc=%d" % r.returncode)
                print((r.stdout or "")[-800:])
                sys.exit(2)
        for line in io.open(csv, encoding="utf-8", errors="replace"):
            for m in PT.finditer(line):
                if m.group(1) in BOUNDARY.get(seg, []):
                    out.setdefault(m.group(1), (float(m.group(3)), float(m.group(4))))
    return out


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    steps = [0, 2, 4, 7]
    if "--steps" in sys.argv:
        steps = [int(x) for x in sys.argv[sys.argv.index("--steps") + 1].split(",")]
    lh, lw = H // 8, W // 8
    # 🔴 对照臂：tag="1x1" 时用**现网**的段与 1:1 的逐步 latents。
    #    没有它，本检查只有「新比例 vs encoding」而没有「1:1 vs 同一个 encoding」——
    #    那样判不出超量程是新几何造成的，还是 encoding 本身就偏紧（缺对照臂 = 判不了因果）。
    is_base = tag == "1x1"
    work = os.path.join(P0, "fp32_steps_L80" if is_base else "fp32_steps_L80_%s" % tag)
    rng = deployed_ranges()
    if not rng:
        print("FAIL 读不到部署 encoding")
        return 1
    print("=== %s  latent %dx%d  步 %s ===" % (tag, lh, lw, steps), flush=True)
    print("部署 encoding 量程读到 %d 个切口张量" % len(rng), flush=True)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sg = sigmas()
    sfx = "" if is_base else ("_" + tag)
    stems = {"part1a": "transformer_part1a_clip_L80%s" % sfx,
             "part1b": "transformer_part1b_clip_L80%s" % sfx,
             "part2a": "transformer_part2a_fixed_clip_L80%s" % sfx}
    worst_all, rows = 0.0, []
    cwd = os.getcwd()
    os.chdir(E)
    try:
        for st in steps:
            lp = os.path.join(work, "lat_%d.raw" % st)
            if not os.path.isfile(lp):
                print("  🔴 缺逐步 latents %s —— 该步跳过（不猜）" % lp)
                continue
            n = os.path.getsize(lp) // 4
            if n != 16 * lh * lw:
                print("  🔴 %s 元素数 %d != %d —— 跳过" % (lp, n, 16 * lh * lw))
                continue
            vals = {
                "latents": np.fromfile(lp, np.float32).reshape(1, 16, lh, lw),
                "timestep": np.array([1.0 - sg[st]], dtype=np.float32),
                "caption": np.fromfile(os.path.join(P0, "cap_r4x3.raw"),
                                       np.float32).reshape(1, CAP, 2560),
                "cap_pad_mask": np.fromfile(os.path.join(P0, "mask_r4x3.raw"),
                                            np.float32).reshape(1, CAP).astype(bool),
            }
            worst = 0.0
            for seg in ("part1a", "part1b", "part2a"):
                s = ort.InferenceSession(stems[seg] + ".onnx", so,
                                         providers=["CPUExecutionProvider"])
                names = [i.name for i in s.get_inputs()]
                miss = [x for x in names if x not in vals]
                if miss:
                    print("  🔴 %s 缺输入 %s" % (seg, miss))
                    return 1
                outs = s.run(None, {x: vals[x] for x in names})
                for o, v in zip(s.get_outputs(), outs):
                    vals[o.name] = v
                    if o.name in BOUNDARY[seg] and v.dtype.kind == "f" and o.name in rng:
                        dmin, dmax = rng[o.name]
                        span = max(abs(dmin), abs(dmax))
                        got = max(abs(float(np.nanmin(v))), abs(float(np.nanmax(v))))
                        r = got / span if span else 0.0
                        rows.append((st, o.name, r, got, span))
                        worst = max(worst, r)
                del s
            worst_all = max(worst_all, worst)
            print("  step %d (sigma %.4f)  该步最大比值 %.3f  %s"
                  % (st, sg[st], worst,
                     "🟢" if worst <= 1.0 else ("🟡" if worst <= 1.10 else "🔴")), flush=True)
    finally:
        os.chdir(cwd)

    rows.sort(key=lambda x: -x[2])
    print("\n比值最高的 6 个（步/张量/比值/实测|max|/量程）:")
    for st, nm, r, got, span in rows[:6]:
        print("  step %d  %-20s %6.3f   %10.3f / %10.3f" % (st, nm, r, got, span))
    print("\n全局最大比值 = %.3f（**不作判据**，见文件头的订正）" % worst_all)
    # 差值口径：与 1:1 对照臂比。缺对照就明说判不了，不许拿绝对值硬判。
    ref = os.path.join(P0, "range_ref_1x1.json")
    cur = {"%d|%s" % (st, nm): r for st, nm, r, _, _ in rows}
    if is_base:
        io.open(ref, "w", encoding="utf-8").write(json.dumps(cur))
        print("已写出 1:1 对照基准 -> %s" % ref)
        return 0
    if not os.path.isfile(ref):
        print("🔴 缺 1:1 对照基准（先跑 `aspect_range_multistep.py 1x1 1024 1024`）")
        print("   **不下判定** —— 绝对比值判不出是新几何造成的还是 encoding 本身如此。")
        return 2
    base = json.loads(io.open(ref, encoding="utf-8").read())
    d = [(k, cur[k] - base[k]) for k in cur if k in base]
    d.sort(key=lambda x: -abs(x[1]))
    print("\n与 1:1 的差（Δ = 本比例比值 − 1:1 比值），前 5:")
    for k, v in d[:5]:
        st, nm = k.split("|")
        print("  step %-2s %-20s Δ=%+.4f" % (st, nm, v))
    md = max(abs(v) for _, v in d) if d else 0.0
    print("\n最大 |Δ| = %.4f" % md)
    print("判定: %s" % ("🟢 与现网等价（|Δ|<=0.02）" if md <= 0.02 else
                        "🟡 记录（|Δ|<=0.10）" if md <= 0.10 else
                        "🔴 新几何带来额外钳位（|Δ|>0.10）"))
    print("⚠️ 只能否决不能放行（指南 §41）；且这是宿主 FP32 前向的范围，不是设备实测。")
    return 0 if md <= 0.10 else 1


sys.exit(main())
