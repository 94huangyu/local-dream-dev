# -*- coding: utf-8 -*-
"""Tier 2 前置门：seq 32 挑出的钳位集合在 seq 80 下还成立吗。

方案：`scripts/EXP_PLAN_T2_CLIPSET.md`（判据事前定稿，本脚本不得偏离）。

规则逐字复刻自 `insert_clip.py:80-89`（**唯一造出部署产物的脚本**）：
    候选 = <seg>_full_enc.csv 里 bitwidth==16 且 tensor type != STATIC 的张量
    |a|max = 实测（<seg>_amax.json）优先，无实测用标定 max(|(2^bw-1+off)*sc|, |off*sc|)
    入选 = |a|max > 65504 / 1.2

用法:
    python t2_clipset_gate.py g0            # G0-known：复现 seq 32 的集合，必须逐元素相同
    python t2_clipset_gate.py judge         # 用 <seg>_L80_amax.json 判 S80 vs S32
"""
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                                   # noqa: E402

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
FP16MAX = 65504.0
THR = FP16MAX / 1.2
SEGS = ("part1a", "part1b", "part2a", "part2b")

TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: "
                r"\[([^\]]*)\]; tensor type: ([A-Z]+)\)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def load_enc(seg):
    kind, pt = {}, {}
    with io.open(os.path.join(W, "%s_full_enc.csv" % seg), encoding="utf-8",
                 errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), m.group(4))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), (int(m.group(2)), float(m.group(5)),
                                           float(m.group(6))))
    return kind, pt


def load_amax(path):
    if not os.path.exists(path):
        return {}
    return {k: v["amax"] for k, v in
            json.load(io.open(path, encoding="utf-8"))["tensors"].items()}


def node_outputs(seg):
    """base ONNX 的节点输出名集合（源名由 canonical_sources 给出，禁止自己拼）。"""
    import onnx
    m = onnx.load(canonical_sources.onnx_for(seg, "base"), load_external_data=False)
    return {o for n in m.graph.node for o in n.output}


def select(seg, amax_path):
    """复刻 insert_clip.py 的选择规则。返回 (picked{onnx名:|a|max}, why, 全部候选值)."""
    kind, pt = load_enc(seg)
    meas = load_amax(amax_path)
    node_out = node_outputs(seg)

    def to_onnx(t):
        if t in node_out:
            return t
        if t.endswith("_fc") and t[:-3] in node_out:
            return t[:-3]
        return None

    picked, why, allv = {}, {}, {}
    for t, (bw, sc, off) in pt.items():
        if bw != 16 or kind.get(t) == "STATIC":
            continue
        o = to_onnx(t)
        if o is None:
            continue
        if t in meas:
            hi, src = meas[t], "实测"
        else:
            hi, src = max(abs((2 ** bw - 1 + off) * sc), abs(off * sc)), "标定"
        allv.setdefault(o, (hi, src))
        if o in picked:
            continue
        if hi > THR:
            picked[o] = hi
            why[o] = src
    return picked, why, allv


def g0():
    """G0-known：用 seq 32 的现有数据复现盘上的 <seg>_clipped.json，必须逐元素相同。"""
    print("G0-known —— 复现 seq 32 的钳位集合（阈值 %.1f）" % THR)
    ok = True
    for seg in SEGS:
        want = sorted(json.load(io.open(os.path.join(W, "%s_clipped.json" % seg),
                                        encoding="utf-8")))
        picked, why, _ = select(seg, os.path.join(W, "%s_amax.json" % seg))
        got = sorted(picked)
        same = (got == want)
        ok &= same
        print("  %-8s 盘上 %2d 个 / 复现 %2d 个  %s  （实测判定 %d / 标定判定 %d）"
              % (seg, len(want), len(got), "✅ 逐元素相同" if same else "🔴 不同",
                 sum(1 for k in picked if why[k] == "实测"),
                 sum(1 for k in picked if why[k] == "标定")))
        if not same:
            print("     只在盘上: %s" % sorted(set(want) - set(got)))
            print("     只在复现: %s" % sorted(set(got) - set(want)))
    print()
    print("G0-known: %s" % ("✅ PASS —— 规则复现正确，可用于 L=80"
                            if ok else "🔴 FAIL —— 规则复现错误，停止，后续比较无意义"))
    return 0 if ok else 1


def judge():
    """步 4：把规则套到 L=80 实测上，与 S32 比对。判据见方案 §五。"""
    verdict = "PASS"
    for seg in SEGS:
        p80 = os.path.join(W, "%s_L80_amax.json" % seg)
        if not os.path.exists(p80):
            print("  %-8s 🔴 缺 %s —— 步 3 未完成" % (seg, os.path.basename(p80)))
            return 2
        s32 = set(json.load(io.open(os.path.join(W, "%s_clipped.json" % seg),
                                    encoding="utf-8")))
        picked80, why80, all80 = select(seg, p80)
        s80 = set(picked80)
        new = sorted(s80 - s32)
        gone = sorted(s32 - s80)
        # 🔴 硬失败：L=80 实测 >65504 却不在 S32
        hard = sorted(o for o, (hi, src) in all80.items()
                      if src == "实测" and hi > FP16MAX and o not in s32)
        n_meas = sum(1 for _o, (_h, src) in all80.items() if src == "实测")
        n_cal = len(all80) - n_meas
        print("=== %s ===  S32=%d  S80=%d  新增=%d  离开=%d" %
              (seg, len(s32), len(s80), len(new), len(gone)))
        print("   覆盖率：候选 %d 个中 L=80 实测 %d，回落 L=32 标定 %d"
              % (len(all80), n_meas, n_cal))
        if n_cal:
            top = sorted(((h, o) for o, (h, src) in all80.items() if src == "标定"),
                         reverse=True)[:3]
            print("   ⚠️ 无 L=80 实测的最高 3 个（只有 L=32 标定，结论对它们不成立）：%s"
                  % ", ".join("%s=%.0f" % (o, h) for h, o in top))
        for o in new:
            print("   ➕ %-20s L80 |a|max=%10.0f (%s) %s"
                  % (o, picked80[o], why80[o], "🔴 已超 65504" if picked80[o] > FP16MAX else ""))
        for o in gone:
            print("   ➖ %-20s L80 |a|max=%10.0f" % (o, all80.get(o, (0, ""))[0]))
        if hard:
            verdict = "FAIL"
            print("   🔴 硬失败张量: %s" % hard)
        elif new and verdict == "PASS":
            verdict = "CHANGED"
    print()
    print("判决: %s" % {"PASS": "🟢 PASS —— 未发现新成员（n=1 prompt，不得写成「集合已验证稳定」）",
                        "CHANGED": "🟡 CHANGED —— 有新成员但未超 65504，按规则加入，L3 需重验",
                        "FAIL": "🔴 FAIL —— S32 在 L=80 下不成立"}[verdict])
    return 0


if __name__ == "__main__":
    sys.exit({"g0": g0, "judge": judge}[sys.argv[1]]())
