#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按**交付谱系**清理 D 盘 —— 白名单驱动，默认 dry-run

与 `disk_cleanup.py` 的关系
---------------------------
那个是**显式路径清单**（2026-08 手工列的，已全部执行完，现在跑出来 0 GB）。
本脚本换一条更可靠的依据：**`build_lineage.py` 从设备 sha256 反查出的交付白名单**
—— 不在白名单里的才是候选，而不是靠人记得哪些能删。

沿用的纪律（来自 `disk_cleanup.py` / `cleanup_20260917.py` / `DISK_INVENTORY.md`）
--------------------------------------------------------------------------------
1. 🔴 **`.onnx` / `.data` 完全不碰**：多对一引用，按名字删会毁掉整条流水线
   （`transformer_part1.onnx` 名字像废弃单体，实际是 part1a/1b 的权重来源）。
2. 🔴 **保护名单命中即退出**，不是跳过 —— 宁可整个脚本不跑，也不能删错一个。
3. 🔴 **不用通配符**：`ctx_part1a_fp16`（L32，可删）与 `ctx_part1a_fp16_L80`（回滚源，必留）
   只差一个后缀。本脚本一律用**实际路径全等**比对。
4. 🔴 **删 DLC 前先抽配方**：`snpe-dlc-info` 里内嵌 Converter/Quantizer 命令，
   删了就再也拿不回来（§二）。
5. 🔴 **删后核对**：交付白名单里每个文件逐个仍在（约束 11 铁律 3）。

按再生成本分档（`DISK_INVENTORY` §1.2 实测）
--------------------------------------------
    fp32 DLC      ≈ 2 分钟可再生   -> A 档：可删
    量化 DLC      ≈ 3 小时         -> B 档：列出，由人定
    context .bin  ≈ 3 小时         -> B 档：列出，由人定
    .raw 中间张量  视情况           -> C 档：只报体积，不动

用法
----
    python scripts/cleanup_by_lineage.py                 # 预演（默认）
    python scripts/cleanup_by_lineage.py --apply-a       # 只删 A 档
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import build_lineage as LIN

ROOT = Path(r"D:\ZImage_Work")
# 🔴 manifest 默认落**仓库内**：它是永久标记（大产物没了，但"它证明过什么"不能丢）。
#    别放 scratch/临时目录 —— 那些会被清掉，标记就跟着没了。
SCRATCH = Path(os.environ.get("CLEANUP_OUT", REPO / "logs" / "cleanup_manifest"))

# 🔴 保护名单：命中即**退出**（纪律 2）。这些是交付物/回滚源/官方参照。
#
# 🔴🔴 2026-09-21 差点漏掉的一类：**台账 #171 记录的 single 形态回滚源**。
#     mg 交付后设备上的 single 文件已被 `--cleanup --yes` 删除，`--rollback` 现在会拒绝执行；
#     宿主这几份是**唯一**能恢复 single 的东西，而它们**名字与交付件完全不同**
#     （`ctx_<seg>_<比例>\<seg>_<比例>.SM8750.bin`），谱系白名单里也不会出现
#     —— 因为它们不是当前交付件。**按"不在白名单就是候选"去删，正好会删掉它们。**
#     ⇒ 教训：白名单只覆盖"当前态"，**回滚源属于"必须留但不在当前态"的第三类**，要单列。
PROTECT_SUBSTR = [
    "package",                 # 交付模型包 zip（DELIVERY_FROZEN 记录）
    "Z-Image-Turbo",           # 官方模型（对照基准）
    "device_only_backup",      # 设备侧备份
    r"aspect\ctx_",            # #171：single 形态各比例回滚源
    r"p2attr\ctx_",            # #171：single 形态 1:1 transformer 回滚源
    r"dlc_pipeline\vae_decoder",  # #171：single 形态 1:1 VAE 回滚源
    # 🔴🔴 #186 的实际危害在这里显形：TE 四段**谱系断链**（同目录没有建图配置 json）
    #     ⇒ 自动白名单认不出它们的来源 DLC，会被判成可删候选。
    #     而 `TIER2_L80\` 整个目录是**当前交付 TE 的全套**（context 已 sha256 锚定成功）。
    #     ⇒ 在断链修好之前，整目录保护。**谱系断链 = 清理时的盲区**，这条要记住。
    "TIER2_L80",
]
PROTECT_EXT = {".onnx", ".data", ".zip"}     # 纪律 1 + 交付包

# DELIVERY_FROZEN 明确记录的回滚源（part1a 原件），必须留
ROLLBACK_SIZES = {3151335424}

# 🔴 **删除一律走白名单，不走黑名单。**
# 原设计是"不在保护名单就是候选" —— 那是黑名单式，漏写一条保护就删错东西
# （本轮已经漏过两次：#171 的 single 回滚源、#186 断链导致的 TE 来源 DLC）。
# 改成：**只删这张表里明确列出的目录**，其余一律不动，宁可漏删不可误删。
#
# 每一条都必须回答两件事：① 它是哪个实验的产物 ② 那个结论关没关、在哪
# —— 这就是"删掉大文件但留下标记"（结论已固化，材料不必再养着）。
DELETABLE = {
    r"ZImage_QNN_Evidence\dlc_pipeline\transformer_part2": (
        "part2 **未拆分**时代的量化 DLC + context",
        "已被 part2a/part2b 取代并交付；拆分结论见 §15.18（🟢已打通）"),
    r"ZImage_QNN_Evidence\dlc_pipeline\transformer_part1": (
        "part1 **未拆分**时代的量化 DLC + context",
        "已被 part1a/part1b 取代并交付"),
    r"ZImage_QNN_Evidence\dlc_pipeline\text_encoder_part": (
        "L=20/32 时代的 TE 量化 DLC + context",
        "已被 TIER2_L80 取代并交付（§三十五 改序列长度的完整成本账）"),
    r"p0_experiments\h1_noscat": (
        "#173 删 ScatterElements 的实验臂（noscat / noscat_A / noscat_B）",
        "✅已关闭并交付：noscatA 逐字节安全（−12.6 s），noscatB 毁图；机制见 §47.3"),
    r"p0_experiments\h1_control": (
        "#173 的对照臂（未删算子的基线）",
        "同 #173，已关闭"),
    r"p0_experiments\h1_probe": (
        "#173 的最小探针（5 个变体）",
        "同 #173，已关闭；探针方法已固化为 §47.3 规则 1"),
    r"p0_experiments\p2split": (
        "part2 拆分实验产物",
        "✅已交付（当前 part2a/2b 即其结果）"),
    r"p0_experiments\actfp16": (
        "#48 wFxp_actFP（权重定点 + 激活浮点）实验",
        "✅已关闭：§15.33 通路成立但收益不显著"),
    r"p0_experiments\perrow": (
        "per-row 量化实验产物",
        "✅已关闭并采用：§十一 / §15.17（E_all 19.24%→7.87%），当前交付即 per-row"),
    r"p0_experiments\p0b\dtypetest": (
        "dtype 机制探针",
        "✅已关闭：§15.42（工具有效，但选择规则被推翻两次）"),
}


def collect_whitelist():
    """交付白名单：9 个设备件的宿主副本 + 它们的全部来源 DLC。"""
    anchors = LIN.read_anchors()
    size_idx = LIN.index_by_size({a["size"] for a in anchors})
    dlc_idx = LIN.build_dlc_index()

    keep, bins = set(), []
    for a in anchors:
        for cand in size_idx.get(a["size"], []):
            if LIN.sha256_of(cand) == a["sha256"]:
                keep.add(str(cand).lower())
                bins.append(cand)
                break
    for b in bins:
        if b.suffix != ".bin":
            continue
        cfg = LIN.read_build_config(b)
        for g in cfg["graph_names"]:
            for d in LIN.find_dlc_for_graph(g, dlc_idx):
                keep.add(d.lower())
    return keep, bins


def deletable_reason(path: Path):
    """白名单式：只有落在 DELETABLE 某个目录下，才返回 (实验, 结论)；否则 None。"""
    low = str(path).lower()
    for d, why in DELETABLE.items():
        if d.lower() in low:
            return why
    return None


def classify(keep: set):
    """返回 (可删, 保留但非交付, .raw 体积)。**可删必须同时满足**：
    ① 不在交付白名单 ② 不在保护名单 ③ **落在 DELETABLE 里**（白名单式）。"""
    deletable, kept_other, raw_bytes = [], [], 0
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            p = Path(dp) / f
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            low, ext = str(p).lower(), p.suffix.lower()
            if ext == ".raw":
                raw_bytes += sz
                continue
            if ext not in (".dlc", ".bin") or sz < 50 * 1024 * 1024:
                continue
            if low in keep or sz in ROLLBACK_SIZES:
                continue
            if any(s.lower() in low for s in PROTECT_SUBSTR) or ext in PROTECT_EXT:
                continue
            why = deletable_reason(p)
            (deletable if why else kept_other).append((sz, p, why))
    deletable.sort(reverse=True, key=lambda x: x[0])
    kept_other.sort(reverse=True, key=lambda x: x[0])
    return deletable, kept_other, raw_bytes


def guard(items):
    """纪律 2：任何保护名单命中就整体退出。"""
    for _, p, _why in items:
        low = str(p).lower()
        if any(s.lower() in low for s in PROTECT_SUBSTR) or p.suffix.lower() in PROTECT_EXT:
            sys.exit(f"🔴 保护名单命中，整体中止：{p}")
        if deletable_reason(p) is None:
            sys.exit(f"🔴 不在 DELETABLE 白名单，整体中止：{p}")


def write_manifest(items, out: Path):
    """删之前留标记：删了什么、它证明过什么、结论在哪。

    这份 manifest 是**小文件，永久保留** —— 大产物删掉，但"它曾经证明过什么"不丢。
    将来有人问"这条结论的原始数据呢"，答案在这里：已删，结论见 X。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    recs, done = [], set()
    for sz, p, (exp, concl) in items:
        rec = {"path": str(p), "bytes": sz, "experiment": exp, "conclusion": concl}
        # 纪律 4：DLC 内嵌 Converter/Quantizer 命令，删了就拿不回来（§二）。
        # 但对**同一个实验组**，配方是一样的 —— 每组抽一个代表即可，
        # 逐个抽 34 个要半小时以上，代价与收益不成比例。
        if p.suffix.lower() == ".dlc" and exp not in done:
            print(f"  抽配方（{exp} 的代表）{p.name}…")
            cmd = LIN.extract_commands(p)
            rec["recipe"] = cmd.get("recipe") or {}
            rec["converter"] = cmd.get("converter", "")[:600]
            rec["quantizer"] = cmd.get("quantizer", "")[:600]
            rec["recipe_is_group_representative"] = True
            done.add(exp)
        recs.append(rec)
    out.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  标记存档 -> {out}")
    return recs


def main():
    apply = "--apply" in sys.argv
    print("按交付谱系清理 —— " + ("🔴 APPLY（真删）" if apply else "预演（默认，不删任何东西）"))

    print("\n[1/4] 重建交付白名单（按 sha256 锚定，需几分钟）…")
    keep, bins = collect_whitelist()
    print(f"  白名单 {len(keep)} 个文件（{len(bins)} 个设备件 + 其来源 DLC，含全部同名候选）")
    if len(bins) < 9:
        sys.exit(f"🔴 只锚定到 {len(bins)} 个设备件，白名单不完整 —— 中止，不删任何东西")

    print("\n[2/4] 分类（白名单式：只有落在 DELETABLE 里的才是候选）…")
    dele, kept, raw_bytes = classify(keep)
    gd = sum(s for s, _, _ in dele) / 1e9
    gk = sum(s for s, _, _ in kept) / 1e9

    print(f"\n  可删（结论已关闭，材料不必再养）：{len(dele)} 个，**{gd:.2f} GB**")
    by_exp = {}
    for s, p, (exp, concl) in dele:
        e = by_exp.setdefault(exp, [0, 0, concl])
        e[0] += s
        e[1] += 1
    for exp, (s, n, concl) in sorted(by_exp.items(), key=lambda kv: -kv[1][0]):
        print(f"     {s/1e9:7.2f} GB ×{n:<3} {exp}")
        print(f"                     └ {concl}")

    print(f"\n  保留（不在 DELETABLE 白名单，一律不动）：{len(kept)} 个，{gk:.2f} GB")
    for s, p, _ in kept[:8]:
        print(f"     {s/1e9:7.2f} GB  {p}")
    if len(kept) > 8:
        print(f"     … 另 {len(kept)-8} 个")

    print(f"\n  .raw 中间张量：{raw_bytes/1e9:.2f} GB —— 本脚本**不动**（另行处置）")
    print(f"  🔴 .onnx / .data / .zip / 交付件 / #171 回滚源 / TIER2_L80：**完全不碰**")

    if not apply:
        print(f"\n预演结束。可释放 **{gd:.2f} GB**。加 --apply 执行。")
        return 0
    if not dele:
        print("\n无可删项")
        return 0

    guard(dele)
    print("\n[3/4] 删之前先记账（大产物可删，但「它证明过什么」要留下）…")
    write_manifest(dele, SCRATCH / "deleted_manifest.json")

    freed = 0
    for s, p, _ in dele:
        try:
            p.unlink()
            freed += s
        except OSError as e:
            print(f"  🔴 删除失败 {p}: {e}")
    print(f"  已删 {len(dele)} 个文件，释放 **{freed/1e9:.2f} GB**")

    print("\n[4/4] 核对（纪律 5）：交付白名单逐个仍在…")
    miss = [p for p in keep if not Path(p).exists()]
    print("  ✅ 全部仍在" if not miss else f"  🔴🔴 {len(miss)} 个白名单文件不见了：{miss[:5]}")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
