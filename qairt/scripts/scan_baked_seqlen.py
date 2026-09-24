#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""烘焙序列长度扫描 —— 卡 A 门 A3-c（指南 §33、§34.1、§33.5）

为什么必须做
------------
`torch.onnx` 会把**导出时示例 prompt 的长度**烘焙成图里的常量。
⇒ **导出时用的示例 prompt 就是产品的硬上限**，而且它不报错、不警告，
   直到用户输入长一点的提示词才暴露。

改起来比重导便宜得多（§33.4：转换分钟级、量化约 20 秒/段、建 context 约 10 秒/段），
**而且张量名不变** ⇒ 按张量名钉死的配方（FP16 白名单、Clip 注入点、encoding overrides）全部继续适用。

🔴 这个工具只负责「找出候选」，**不负责判定哪个该改**
-------------------------------------------------------
§33.5 有两个**真实发生过**的歧义，本工具无法自动消解：

| 撞值 | 后果 |
|---|---|
| 目标 seq=32 与注意力**头数 32** 撞 | 按位置替换会把头数改坏 |
| `32×128 = 4096`，而 4096 又是**图像 token 数** | 同一字面量两种含义 |

本工具实测量化了这个风险：`text_encoder_part1.onnx`（真实 L=20）去扫 32，
**会命中 10 个，全部是撞值**。⇒ 所以每个命中都打印**消费者节点**，
供你按「它出现在形状的哪一维、上游是谁」逐个判定（§33.5：不许假设）。

**替换规则必须是「逐元素把 L_old 换成 L_new」，绝不能按位置或整条形状替换。**

用法
----
    python scripts/scan_baked_seqlen.py <model.onnx> --len 80
    python scripts/scan_baked_seqlen.py <onnx目录> --len 80 --quiet
    python scripts/scan_baked_seqlen.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("需要 onnx：pip install onnx")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

INT_TYPES = (onnx.TensorProto.INT64, onnx.TensorProto.INT32)
MAX_ELEMS = 16          # §33.3：只看"小整数常量"

# 已知样本（约束 8）。期望值是 2026-09-20 **实测**锁定的，不是从 §33.3 抄的。
# 三个版本各扫自己的 L 都恰好 22 个 —— 这是该图的结构常数，三向一致。
# 第 4 条是**假阳性对照**：真实 L=20 的图去扫 32，命中的 10 个全是撞值（头数等）。
SELFTEST_DIR = Path(r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx")
SELFTEST_CASES = [
    ("text_encoder_part1_L80.onnx", 80, 22, "L80 版扫自己的 L"),
    ("text_encoder_part1_s32.onnx", 32, 22, "s32 版扫自己的 L"),
    ("text_encoder_part1.onnx",     20, 22, "原始版（真实 L=20）扫自己的 L"),
    ("text_encoder_part1.onnx",     32, 10, "🔴 假阳性对照：L=20 的图扫 32，命中全是撞值"),
]


def scan(path: Path, L: int):
    # load_external_data=False：避免爆内存 + protobuf 2GB 上限（§3.1 坑 1）
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph

    consumers = {}
    for node in g.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    inits = []
    for t in g.initializer:
        if t.data_type not in INT_TYPES:
            continue
        try:
            a = numpy_helper.to_array(t)
        except Exception:
            continue
        vals = a.reshape(-1).tolist() if a.size else []
        if 0 < a.size <= MAX_ELEMS and L in vals:
            cons = consumers.get(t.name, [])
            inits.append({
                "name": t.name, "values": vals,
                "consumers": [{"name": n.name or "<anon>", "op": n.op_type,
                               "argpos": [i for i, x in enumerate(n.input) if x == t.name]}
                              for n in cons],
            })

    # §34.1：只改 initializer 是不够的 —— graph.value_info 也带着旧长度
    shapes = []
    for coll, where in ((g.input, "input"), (g.output, "output"), (g.value_info, "value_info")):
        for vi in coll:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if L in dims:
                shapes.append({"name": vi.name, "where": where, "dims": dims,
                               "axis": [i for i, d in enumerate(dims) if d == L]})
    return {"file": path.name, "L": L, "initializers": inits, "shapes": shapes}


def report(res, quiet=False):
    ni, ns = len(res["initializers"]), len(res["shapes"])
    print(f"\n=== {res['file']}  （找 L={res['L']}）===")
    print(f"  initializer 命中 {ni} 个；shape 维度命中 {ns} 处")
    if not quiet:
        for it in res["initializers"]:
            cons = "、".join(f"{c['op']}:{c['name']}(第{c['argpos']}位)" for c in it["consumers"]) or "🔴 无消费者"
            print(f"     · {it['name']} = {it['values']}")
            print(f"       消费者: {cons}")
        if ns:
            print(f"  --- shape 里带 L 的张量（§34.1：改 initializer 不够，这些也要改）---")
            for s in res["shapes"][:12]:
                print(f"     · [{s['where']}] {s['name']} {s['dims']}  轴 {s['axis']}")
            if ns > 12:
                print(f"     … 另 {ns - 12} 处")
    if ni or ns:
        print("  🔴 **逐个追消费者判定归属，不许按位置或整条形状替换**（§33.5，本项目栽过两次）")
    return ni


def selftest():
    print("=== 判据自检（已知样本，含假阳性对照）===")
    ok = True
    for fname, L, expect, why in SELFTEST_CASES:
        p = SELFTEST_DIR / fname
        if not p.exists():
            print(f"  ⚠️  跳过（样本不在）：{p}")
            ok = False
            continue
        got = len(scan(p, L)["initializers"])
        good = got == expect
        print(f"  {'✅' if good else '❌'} {fname} L={L}: 命中 {got}（期望 {expect}） — {why}")
        ok &= good
    print("\n判据自检：" + ("✅ 通过" if ok else "❌ 未通过 —— 结论不可用"))
    print("🔴 第 4 条是关键：它证明**本工具会报出撞值假阳性**（10 个），"
          "所以命中数不是判据，\n   **逐个追消费者才是**。工具只缩小范围，不下结论。")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="扫 ONNX 里烘焙的序列长度常量")
    ap.add_argument("target", nargs="?", help="ONNX 文件或目录")
    ap.add_argument("--len", type=int, help="要找的序列长度 L")
    ap.add_argument("--quiet", action="store_true", help="只报数量，不列明细")
    ap.add_argument("--selftest", action="store_true", help="用已知样本验证判据本身")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.target or a.len is None:
        ap.error("需要 target 与 --len，或用 --selftest")

    t = Path(a.target)
    files = sorted(t.glob("*.onnx")) if t.is_dir() else [t]
    if not files:
        sys.exit(f"没有找到 .onnx：{t}")

    total = 0
    for f in files:
        try:
            total += report(scan(f, a.len), a.quiet)
        except Exception as e:
            print(f"\n=== {f.name} ===\n  ❌ 解析失败：{e}")
    print(f"\n{'=' * 60}\n合计 initializer 命中 {total} 个（{len(files)} 个图）")
    print("⊕ 改法见 §33.4（不必从 PyTorch 重导，逐元素改常量即可，**张量名不变**）；"
          "别忘了 §33.6 的两类连带项与 §33.7 的下游契约。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
