#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where/Select 输出钳位扫描 —— 卡 B 门 B3（指南 §3.2.1）

找什么
------
`Where(mask, a, -BIG)` 这类加性掩码，量化器很容易只按**多数分支**定输出区间，
把 `-BIG` 那一侧**整个排除在表示范围之外** ⇒ 掩码在设备上被钳掉、语义悄悄变了。

本项目实测：part2 的 `node_Where_105` 掩码分支 min=-100、输出区间却是 [0, 1e-4]，
`(100/65535)/(1e-4/65535)` 正好 = 1e6，与编译日志 `requant qu16->qu16: 1000000.0` 逐位吻合。

🔴 **光查"有没有 ±FLT_MAX 张量"查不出这一类** —— 常量改成 -100 之后它照样成立。

判据（操作化）
--------------
    min(输出) <= min(所有数据分支输入)

比 §3.2.1 的原文（"min(输出) <= min(掩码分支)"）更严格，好处是**不需要判断哪个输入是掩码分支**
—— 语义识别不可靠，而"输出必须覆盖所有输入分支"是同样的保护且纯数值可判。

判据的已知样本验证（约束 8）
----------------------------
    part1a_dlcinfo.txt  node_where_1    -> PASS（输出 min -18.5425 <= 分支 min -1.6797）
    part2_dlcinfo.txt   node_Where_105  -> FAIL（输出 min 0 > 分支 min -100）
双向，且与 §3.2.1 表格逐位吻合。

用法
----
    python scripts/scan_where_clip.py <snpe-dlc-info 转储.txt> [更多...]
    python scripts/scan_where_clip.py --selftest

转储怎么来：`python scripts/qairt_tool.py snpe-dlc-info -i <model.dlc> > dump.txt`
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 🔴 自检样本是 2026-08-14 的**历史**转储，不是当前交付态（当前是 mg 多图 + part2a/2b + L=80）。
#    它验的是「判据的操作化能否复现已知数值」，**不是**「当前交付件有没有这个缺陷」。
#    后者要拿当前产物的转储另跑一次（见 --selftest 末尾提示）。
#
# 断言到**具体数值**而不只是 PASS/FAIL：只比结论会放过解析错误
#   —— 本脚本第一版就因为只比 PASS/FAIL，没发现 part1a 是被「正常分支」而非「掩码分支」判过的。
SELFTEST_DIR = Path(r"D:\ZImage_Work\p0_experiments")
SELFTEST_CASES = [
    # (转储, 节点, 期望PASS, {张量名: 期望min})  —— 期望值来自 §3.2.1，且与转储逐位核对过
    ("part1a_dlcinfo.txt", "node_where_1", True,
     {"cap_pad_token": -1.679710745811,              # 掩码分支（不是最负的那个）
      "split_with_sizes_3_split_0": -18.542541503906,  # 正常分支（最负）
      "constant_pad_nd_2": -18.542541503906}),        # 输出
    ("part2_dlcinfo.txt", "node_Where_105", False,
     {"val_104": -3.4028234663852886e+38,            # 掩码分支：未 maskfix，仍是 -FLT_MAX
      "val_105": 0.0}),                               # 输出：一个负数都表示不了
    # maskfix 版：常量已改成 -100，**同一个节点仍然 FAIL** —— 这正是 §3.2.1 的要点
    ("maskfix_part2_dlcinfo.txt", "node_Where_105", False,
     {"val_104": -100.0, "val_105": 0.0}),
]

# 🟢 独立交叉验证（不依赖 §3.2.1 那张表，来自**编译器自己的报错**）：
#    maskfix 版编译时报 `requant qu16->qu16: 1000000.000000`，
#    而 (100/65535)/(1e-4/65535) = 1e6。若本工具算出的量程比与它同量级，
#    说明工具读到的是编译器实际用的那组 encoding，而不是我挑出来凑数的。
XCHECK = {"file": "maskfix_part2_dlcinfo.txt", "node": "node_Where_105",
          "branch": "val_104", "out": "val_105",
          "expect_ratio": 1e6, "tol": 0.05,
          "source": "编译日志 linearclip.h:248 requant qu16->qu16: 1000000.000000"}
# ⚠️ 容差不能设太小：**转储精度陷阱**（§3.2.2）—— `snpe-dlc-info` 的 scale 只打印 12 位小数，
#    拿打印值做比值得到的是 **999936** 而不是整 1e6。要做等式检验得用精确值
#    （100/65535、1e-4/65535）重算。5% 的容差就是为这个留的，不是随手设的。

TERNARY_TYPES = ("Eltwise_Ternary",)          # snpe-dlc-info 里 Where/Select 的算子类型
# 🔴 张量名用 \S+ —— **必须允许含点**（`layers.0.weight` 这类）。
#    已废弃的 scan_requant_ratios.py 的盲区之一就是正则不许名字含点，静默漏报（§3.2.2）。
ENC_RE = re.compile(
    r"(\S+) encoding : bitwidth (\d+), min (-?[\d.e+-]+), max (-?[\d.e+-]+), scale ([\d.eE+-]+)")
TENSOR_RE = re.compile(r"(\S+) \(data type: (\w+); tensor dimension: \[[^\]]*\]; tensor type: (\w+)\)")
ROW_RE = re.compile(r"^\|\s*(\d*)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|")


def parse_dump(path: Path):
    """返回 (encodings: name->{bw,min,max,scale}, nodes: [{id,name,type,inputs,outputs}])。

    这个解析器同时供 `scan_requant_edges.py`（门 B2）使用 —— 两个门共用一份，
    免得两份解析器各自漂移（约束 11·补：同一份东西出现多处就改成单一数据源）。
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # encoding 在表格最后一列，按张量名索引；per-axis 的 channel_0 行也是这个格式
    enc = {}
    for m in ENC_RE.finditer(text):
        name = m.group(1)
        try:
            enc.setdefault(name, {"bw": int(m.group(2)), "min": float(m.group(3)),
                                  "max": float(m.group(4)), "scale": float(m.group(5))})
        except ValueError:
            pass                                 # 只保留第一次（per-axis 取 channel_0）

    nodes, cur = [], None
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        nid, name, op = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        ins, outs = m.group(4), m.group(5)
        if nid and name:                         # 节点块的第一行
            cur = {"id": nid, "name": name, "type": op, "inputs": [], "outputs": []}
            nodes.append(cur)
        if cur is None:
            continue
        for col, key in ((ins, "inputs"), (outs, "outputs")):
            for t in TENSOR_RE.finditer(col):
                cur[key].append({"name": t.group(1), "dtype": t.group(2), "ttype": t.group(3)})
    return enc, nodes


def check(path: Path):
    enc, nodes = parse_dump(path)
    results = []
    for n in nodes:
        if n["type"] not in TERNARY_TYPES:
            continue
        # 条件输入是 Bool_8；其余为数据分支
        branches = [t for t in n["inputs"] if t["dtype"] != "Bool_8"]
        branch_mins = {t["name"]: enc[t["name"]]["min"] for t in branches if t["name"] in enc}
        out_mins = {t["name"]: enc[t["name"]]["min"] for t in n["outputs"] if t["name"] in enc}
        if not branch_mins or not out_mins:
            results.append({"node": n["name"], "verdict": "SKIP",
                            "why": "缺 encoding（分支或输出未量化）",
                            "branch_min": None, "out_min": None})
            continue
        bmin = min(branch_mins.values())
        omin = min(out_mins.values())
        ok = omin <= bmin + 1e-12
        results.append({
            "node": n["name"], "verdict": "PASS" if ok else "FAIL",
            "branch_min": bmin, "out_min": omin,
            "branch_of": min(branch_mins, key=branch_mins.get),
            "out_of": min(out_mins, key=out_mins.get),
            "why": "" if ok else "输出表示范围覆盖不了最负的那个分支 ⇒ 该分支会被钳掉",
        })
    return results


def report(path: Path, results):
    fails = [r for r in results if r["verdict"] == "FAIL"]
    skips = [r for r in results if r["verdict"] == "SKIP"]
    print(f"\n=== {path.name} ===")
    print(f"  Where/Select 节点 {len(results)} 个："
          f"PASS {len(results) - len(fails) - len(skips)} / FAIL {len(fails)} / SKIP {len(skips)}")
    for r in fails:
        print(f"  🔴 {r['node']}")
        print(f"     分支最小 {r['branch_min']:.6f}（{r['branch_of']}）"
              f" > 输出最小 {r['out_min']:.6f}（{r['out_of']}）")
        print(f"     ⇒ {r['why']}")
        if r["branch_min"] and r["out_min"] is not None:
            span = abs(r["branch_min"]) / max(abs(r["out_min"]), 1e-12)
            print(f"     ⊕ 两者量程比约 {span:.3g} —— 编译期 requant 比值会是这个量级")
    if skips:
        print(f"  ⚪ SKIP {len(skips)} 个（分支或输出没有 encoding，多为浮点/未量化）")
    return fails


def selftest():
    print("=== 判据自检（已知样本）===")
    ok = True

    # ① 结论 + ② 具体数值：只比结论会放过解析错误
    for fname, node, expect_pass, expect_mins in SELFTEST_CASES:
        p = SELFTEST_DIR / fname
        if not p.exists():
            print(f"  ⚠️  跳过（样本不在）：{p}")
            ok = False
            continue
        enc, _ = parse_dump(p)
        hit = next((r for r in check(p) if r["node"] == node), None)
        if hit is None:
            print(f"  ❌ {fname}: 没找到节点 {node}")
            ok = False
            continue
        got_pass = hit["verdict"] == "PASS"
        good = got_pass == expect_pass
        print(f"  {'✅' if good else '❌'} {fname} :: {node} -> {hit['verdict']}"
              f"（期望 {'PASS' if expect_pass else 'FAIL'}）")
        ok &= good
        for tname, want in expect_mins.items():
            got = (enc.get(tname) or {}).get("min")
            hit_ok = got is not None and abs(got - want) <= max(abs(want) * 1e-9, 1e-9)
            print(f"      {'✅' if hit_ok else '❌'} {tname}.min = {got}（期望 {want}）")
            ok &= hit_ok

    # ③ 独立交叉验证：与编译器自己的报错对表，不依赖 §3.2.1 那张表
    p = SELFTEST_DIR / XCHECK["file"]
    if p.exists():
        enc, _ = parse_dump(p)
        # §3.2.1 的公式就是 **scale 之比**：(100/65535) / (1e-4/65535) = 1e6
        s_branch = (enc.get(XCHECK["branch"]) or {}).get("scale")
        s_out = (enc.get(XCHECK["out"]) or {}).get("scale")
        ratio = s_branch / s_out if s_branch and s_out else None
        good = ratio is not None and abs(ratio - XCHECK["expect_ratio"]) / XCHECK["expect_ratio"] <= XCHECK["tol"]
        print(f"  {'✅' if good else '❌'} 交叉验证：scale({XCHECK['branch']})/scale({XCHECK['out']})"
              f" = {ratio:.6g}（期望 ≈ {XCHECK['expect_ratio']:.0e}）")
        print(f"      证据源：{XCHECK['source']}")
        ok &= good
    else:
        print(f"  ⚠️  交叉验证跳过（样本不在）：{p}")
        ok = False

    print("\n判据自检：" + ("✅ 通过" if ok else "❌ 未通过 —— 结论不可用"))
    print("🔴 注意：样本是 2026-08-14 的**历史**转储（part2 尚未拆分）。"
          "本自检只证明**判据的操作化正确**，\n"
          "   **不**证明当前交付件（part2a/part2b、mg 多图、L=80）没有这个缺陷 ——"
          "那要拿当前产物的转储另跑。")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="扫 Where/Select 的输出是否钳掉了掩码分支")
    ap.add_argument("dumps", nargs="*", help="snpe-dlc-info 转储 .txt")
    ap.add_argument("--selftest", action="store_true", help="用已知样本验证判据本身")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.dumps:
        ap.error("需要转储文件，或用 --selftest")

    total = 0
    for d in a.dumps:
        p = Path(d)
        if not p.exists():
            print(f"❌ 不存在：{p}")
            continue
        total += len(report(p, check(p)))
    print(f"\n{'=' * 60}\n合计 FAIL {total} 个")
    if total:
        print("🔴 每个 FAIL 都要处理：① 让校准数据真的覆盖掩码分支，或 ② 用 "
              "--quantization_overrides 手工指定该输出张量的 encoding（§3.2.1）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
