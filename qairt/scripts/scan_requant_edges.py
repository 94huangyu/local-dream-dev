#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""requant 极端边扫描 —— 卡 B 门 B2（指南 §3.2、§3.2.2）

找什么
------
同一个算子内两个张量的 **scale 之比**过大 ⇒ HTP 编译期报
`scale too large for requant qu16->qu16: inf`，或虽不报错但精度被毁。

本项目实测：全流水线 4940 个张量里**只有一个**退化（注意力掩码的 -FLT_MAX 常量），
但它经 `Where` → 加性掩码，污染了 part2 **全部 15 个注意力块**（16 条极端边）。

🔴 为什么不能用 `scan_requant_ratios.py`（已废弃）
--------------------------------------------------
§3.2.2 记载它有两个**静默漏报**的盲区，本项目因此漏掉一整轮：

1. 正则不允许张量名含点 ⇒ `layers.*.weight` 这类全部漏掉；
2. **更致命**：只算「激活输入 → 输出」并显式排除 `STATIC`。
   而掩码常量正是 `STATIC`，`Eltwise` 两个操作数之间的对齐（input↔input）它也从不算
   ⇒ **它恰好扫不到这个坑**。

本工具：**算子内全部张量两两取比值，不排除 STATIC，张量名允许含点。**

⚠️ 转储精度陷阱（§3.2.2）
-------------------------
`snpe-dlc-info` 的 scale 只打印 12 位小数。拿打印值做比值会得到 **999936** 而不是整 1e6。
⇒ 本工具报的是**量级**，要做等式检验得用精确值（如 `100/65535`）重算。

判据的已知样本验证（约束 8，双向 + 独立证据源）
-----------------------------------------------
    part2_dlcinfo.txt        node_Where_105 -> inf   （5.19e33 / 1.53e-9 溢出 float32）
    maskfix_part2_dlcinfo.txt node_Where_105 -> ~1e6 （与编译日志 requant ... 1000000.000000 吻合）
第二条的期望值来自**编译器自己的报错**，不是本项目的分析表 —— 不是用文档验文档。

用法
----
    python scripts/scan_requant_edges.py <dump.txt> [--max-ratio 1e4]
    python scripts/scan_requant_edges.py --selftest

转储怎么来：`python scripts/qairt_tool.py snpe-dlc-info -i <model.dlc> > dump.txt`
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_where_clip import parse_dump          # 共用一份解析器，免得两份各自漂移

FLT_MAX = 3.4028234663852886e38
SELFTEST_DIR = Path(r"D:\ZImage_Work\p0_experiments")
SELFTEST_CASES = [
    ("part2_dlcinfo.txt", "node_Where_105", float("inf"),
     "原始：掩码常量是 -FLT_MAX，scale 5.19e33 / 1.53e-9 ⇒ float32 溢出"),
    ("maskfix_part2_dlcinfo.txt", "node_Where_105", 1e6,
     "maskfix：常量改成 -100 后仍有 1e6 —— **改输入常量不够**（§3.2.1）"),
]


def f32(x: float) -> float:
    """按 float32 复算：超过 FLT_MAX 即 inf，与 SDK 内部表示一致（§3.2.2）。"""
    return float("inf") if abs(x) > FLT_MAX else x


def scan(path: Path, max_ratio: float = 1e4):
    enc, nodes = parse_dump(path)
    hits = []
    for n in nodes:
        # 🔴 算子内**全部**张量，输入输出一起，**不排除 STATIC**
        names = [t["name"] for t in n["inputs"] + n["outputs"]]
        scales = {nm: enc[nm]["scale"] for nm in names
                  if nm in enc and enc[nm]["scale"] > 0}
        worst = None
        for a, b in itertools.combinations(scales, 2):
            r = f32(max(scales[a], scales[b]) / min(scales[a], scales[b]))
            if worst is None or r > worst[0]:
                worst = (r, a, b)
        if worst and worst[0] > max_ratio:
            hits.append({"node": n["name"], "id": n["id"], "op": n["type"],
                         "ratio": worst[0], "pair": (worst[1], worst[2]),
                         "scales": (scales[worst[1]], scales[worst[2]])})
    hits.sort(key=lambda h: (h["ratio"] == float("inf"), h["ratio"]), reverse=True)
    return hits


def report(path: Path, hits):
    print(f"\n=== {path.name} ===")
    if not hits:
        print("  ✅ 没有超过阈值的边")
        return 0
    infs = [h for h in hits if h["ratio"] == float("inf")]
    print(f"  🔴 极端边 {len(hits)} 条（其中 float32 溢出 {len(infs)} 条）")
    for h in hits[:20]:
        r = "inf" if h["ratio"] == float("inf") else f"{h['ratio']:.6g}"
        print(f"     · Id={h['id']:<5} {h['op']:<18} {h['node']}")
        print(f"       比值 {r}  ←  {h['pair'][0]}({h['scales'][0]:.6g}) / "
              f"{h['pair'][1]}({h['scales'][1]:.6g})")
    if len(hits) > 20:
        print(f"     … 另 {len(hits) - 20} 条")
    print("  ⇒ 处理见 §3.2：把掩码的 -inf 常量换成温和负值（**必要但不充分**），"
          "还要看 `Where` 的**输出** encoding（门 B3 / `scan_where_clip.py`）")
    return len(hits)


def selftest():
    print("=== 判据自检（已知样本，双向）===")
    ok = True
    for fname, node, expect, why in SELFTEST_CASES:
        p = SELFTEST_DIR / fname
        if not p.exists():
            print(f"  ⚠️  跳过（样本不在）：{p}")
            ok = False
            continue
        hit = next((h for h in scan(p, max_ratio=1.0) if h["node"] == node), None)
        if hit is None:
            print(f"  ❌ {fname}: 没找到 {node}")
            ok = False
            continue
        got = hit["ratio"]
        if expect == float("inf"):
            good = got == float("inf")
        else:
            good = got != float("inf") and abs(got - expect) / expect <= 0.05
        r = "inf" if got == float("inf") else f"{got:.6g}"
        print(f"  {'✅' if good else '❌'} {fname} :: {node} 比值 = {r}"
              f"（期望 {'inf' if expect == float('inf') else f'≈{expect:.0e}'}）")
        print(f"      {why}")
        ok &= good
    print("\n判据自检：" + ("✅ 通过" if ok else "❌ 未通过 —— 结论不可用"))
    print("🔴 样本是 2026-08-14 的**历史**转储（part2 尚未拆分）。"
          "只证明判据的操作化正确，\n   **不**证明当前交付件没有这个缺陷。")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="扫算子内 scale 比值过大的边")
    ap.add_argument("dumps", nargs="*", help="snpe-dlc-info 转储 .txt")
    ap.add_argument("--max-ratio", type=float, default=1e4, help="阈值，默认 1e4")
    ap.add_argument("--selftest", action="store_true")
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
        total += report(p, scan(p, a.max_ratio))
    print(f"\n{'=' * 60}\n合计极端边 {total} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
