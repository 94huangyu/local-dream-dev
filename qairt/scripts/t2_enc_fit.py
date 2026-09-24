# -*- coding: utf-8 -*-
"""转换前的门：L=80 实测量程是否仍落在 L=32 导出的 encoding 区间内。

## 为什么需要
部署的 transformer 量化 **`input_list=None`**（实查四段 Quantizer command），
encoding 全部来自 `<seg>_fp16_ovr.json`，是在 **L=32** 下导出的。
张量名不变 ⇒ JSON 直接可用；但**取值范围是否仍适用必须查** ——
超出部分会被 encoding 硬钳（#76/#80 正是被「输入超校准量程 2.77 倍硬钳」毁掉的）。

## 🔴 第一版是错的（约束 8 的又一个例子）
第一版直接拿 `<seg>_L80_amax.json` 的键（**DLC 名**，如 `linear_23_fc`）
去查 overrides（**ONNX 名**，#45）。JSON 里确实有 81 个 `_fc` 键，但那是**死条目** ——
#45 已实测：用 DLC 名 `Processed 0 encodings`，**静默失效**。
于是第一版把白名单里的浮点张量当成了受 encoding 约束的定点张量，报出假 FAIL。
正确做法：先 DLC 名 -> ONNX 名，**查不到 ONNX 名 = 该张量是浮点（白名单），无 encoding 约束**。

## 判据（事前定稿）
对每个 **ONNX 名在 overrides 里、且是 16-bit 定点** 的激活张量：
  超出比 = `|a|max` / max(|lo|,|hi|)，其中 hi=(2^bw-1+off)*sc, lo=off*sc
  🔴 FAIL 只在「**L=80 超出而 L=32 不超出**」时成立 —— 那才是 Tier 2 引入的。
  「两个 L 都超出」= **既有配置本来就有的性质**，不是 Tier 2 的问题（须单独登记）。
⇒ 因此本脚本**强制要求对照臂**：part1b / part2a 有 L=32 实测，用它们判归因；
   part1a / part2b 没有 ⇒ 只报现象，**不得归因**。
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                                     # noqa: E402

W = r"D:\ZImage_Work\p0_experiments\p2attr"
SEGS = ("part1a", "part1b", "part2a", "part2b")
# 有同口径 L=32 实测的段（唯一能做归因的）
CTRL = {"part1b": "part1b_amax.json", "part2a": "risky_stats.json"}


def node_outputs(seg):
    import onnx
    m = onnx.load(canonical_sources.onnx_for(seg, "base"), load_external_data=False)
    return {o for n in m.graph.node for o in n.output}


def hi_of(e):
    e = e[0]
    if "scale" not in e or e.get("dtype") == "float":
        return None
    bw, sc, off = int(e["bitwidth"]), float(e["scale"]), float(e["offset"])
    if sc <= 0:
        return None
    return max(abs((2 ** bw - 1 + off) * sc), abs(off * sc))


def load_ctrl(seg):
    if seg not in CTRL:
        return None
    J = json.load(io.open(os.path.join(W, CTRL[seg]), encoding="utf-8"))
    d = J["tensors"] if "tensors" in J else J
    return {k: v["amax"] for k, v in d.items()}


def main():
    verdict, notes = "PASS", []
    for seg in SEGS:
        ovr = json.load(io.open(os.path.join(W, "%s_fp16_ovr.json" % seg),
                                encoding="utf-8"))["activation_encodings"]
        am = json.load(io.open(os.path.join(W, "%s_L80_amax.json" % seg),
                               encoding="utf-8"))["tensors"]
        nodes = node_outputs(seg)
        ctrl = load_ctrl(seg)

        def to_onnx(t):
            if t in nodes:
                return t
            if t.endswith("_fc") and t[:-3] in nodes:
                return t[:-3]
            return None

        fixed, floats, unmapped = [], 0, 0
        for t, v in am.items():
            o = to_onnx(t)
            if o is None:
                unmapped += 1
                continue
            e = ovr.get(o)
            if e is None:
                floats += 1                      # 白名单浮点：无 encoding 约束
                continue
            h = hi_of(e)
            if h is None:
                floats += 1
                continue
            fixed.append((v["amax"] / h, o, t, v["amax"], h))
        fixed.sort(reverse=True)
        over = [r for r in fixed if r[0] >= 1.05]

        print("=== %s ===  定点受约束 %d ｜浮点(白名单，无 encoding) %d ｜映射不到 %d"
              % (seg, len(fixed), floats, unmapped))
        # G0-known：被钳的张量必须全部落在「浮点」一侧
        clip = json.load(io.open(os.path.join(W, "%s_clipped.json" % seg), encoding="utf-8"))
        bad_clip = [c for c in clip if c in ovr]
        print("   G0-known：%d 个被钳张量的 ONNX 名都不该在 overrides 里 ⇒ %s"
              % (len(clip), "✅" if not bad_clip else "🔴 %s" % bad_clip))
        if fixed:
            r = fixed[0]
            print("   最高超出比 %.4f (%s: L80 %.0f vs encoding 上界 %.0f)" % (r[0], r[1], r[3], r[4]))
        print("   超出 >=1.05 的 %d 个" % len(over))

        if ctrl is None:
            if over:
                notes.append("%s 有 %d 个超出，但**无 L=32 实测对照 ⇒ 不得归因给 Tier 2**"
                             % (seg, len(over)))
            print("   ⚠️ 本段无 L=32 实测对照 ⇒ 只报现象，不归因")
        else:
            newly = []
            for r in over:
                c = ctrl.get(r[2], ctrl.get(r[1]))
                if c is None:
                    continue
                if c / r[4] < 1.05:              # L=32 不超，L=80 超 ⇒ Tier 2 引入
                    newly.append((r, c))
            print("   对照（L=32 实测）：可对照 %d 个；**L=32 不超而 L=80 超的 %d 个**"
                  % (sum(1 for r in over if (r[2] in ctrl or r[1] in ctrl)), len(newly)))
            for r, c in newly[:5]:
                print("      🔴 %-20s L32 %.0f / L80 %.0f / 上界 %.0f" % (r[1], c, r[3], r[4]))
            if newly:
                verdict = "FAIL"
        print()

    print("判决: %s" % {"PASS": "🟢 PASS —— 未发现「L=32 不超而 L=80 超」的张量",
                        "FAIL": "🔴 FAIL —— Tier 2 引入了新的 encoding 硬钳"}[verdict])
    for n in notes:
        print("  ⚠️ %s" % n)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
