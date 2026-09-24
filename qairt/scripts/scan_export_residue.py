#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出残留扫描器 —— 卡 A 门 A3-b（指南 §47.7 第 2 条，本项目最贵的一条导出坑）

找什么
------
"输出是全尺寸、但语义上是恒等/填充"的算子。它们是 PyTorch→ONNX 导出的副产品
（最典型：batch=1 的 `pad_sequence` 变成全尺寸 `ScatterElements`），
在 FP32 下数学恒等、看不出问题，但在 HTP 上**真的按全尺寸算**。

本项目实测代价：part1a 里两处这类算子占该段 **29.1% 的 cycles**；
等到量化建图之后再删，就是图级手术 —— 而且**同型的两处，删一处安全、删另一处毁图**（§47.3）。
⇒ **在导出阶段查出来改掉，代价是分钟级。**

判据的已知样本验证（约束 8）
----------------------------
本脚本的操作化用本项目的一对图验证过（`--selftest`）：
  transformer_part1a_clip_L80_896x1184.onnx          -> 必须扫出 2 个 ScatterElements
  transformer_part1a_clip_L80_896x1184_noscatA.onnx  -> 必须扫出 1 个（图像流那个已删）
同比例、单变量。两边数目不符即判脚本失效。

用法
----
    python scripts/scan_export_residue.py <onnx文件或目录>
    python scripts/scan_export_residue.py <onnx目录> --json residue.json
    python scripts/scan_export_residue.py --selftest        # 用已知样本验证判据本身
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("需要 onnx：pip install onnx")

# 已知样本（约束 8：判据的操作化必须先用已知样本验证）
#
# 这对图是同比例、单变量：noscatA 比原版少了图像流那个 `node_select_scatter`
# （[1,4144,3840]，交付时已删，#173）。⇒ 6 与 5，差正好 1 个。
#
# 🔴 期望值是 2026-09-20 **实测**锁定的，不是从文档描述推的。
# 第一版按指南 §47.3 的叙述写成 (2, 1)，自检当场失败 —— 两个原因：
#   ① §47.3 说的"两个"指的是**两个大的**（各 16M 元素、占 part1a 29.1% cycles），
#      实际全图有 6 个恒等 ScatterElements，另 4 个较小的从未被提及；
#   ② 原版图的 value_info 缺失 ⇒「沿大小为 1 的轴」这条规则静默失效，只扫出 2 个。
# 这两条正是本脚本要防的东西，别把期望值改回去迁就描述。
SELFTEST_DIR = Path(r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx")
SELFTEST_CASES = [
    ("transformer_part1a_clip_L80_896x1184.onnx", 6),
    ("transformer_part1a_clip_L80_896x1184_noscatA.onnx", 5),
]


def _const_map(graph):
    """名字 -> numpy 数组。含 initializer 与 Constant 节点。"""
    out = {}
    for init in graph.initializer:
        try:
            out[init.name] = numpy_helper.to_array(init)
        except Exception:
            pass
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    try:
                        out[node.output[0]] = numpy_helper.to_array(attr.t)
                    except Exception:
                        pass
    return out


def _shape_map(graph):
    """张量名 -> shape（list[int]，动态维记为 -1）。"""
    out = {}
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else -1)
            if dims:
                out[vi.name] = dims
    return out


def _numel(shape):
    if not shape or any(d < 0 for d in shape):
        return None
    n = 1
    for d in shape:
        n *= d
    return n


def _get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return a.i if a.type == a.INT else default
    return default


def classify(node, consts, shapes):
    """返回 (是否恒等, 理由, 置信度) 或 None（不是高危类型）。

    置信度：high = 常量可读且判定明确；low = 常量不可读，只能按类型标记待查。
    """
    op = node.op_type

    if op in ("ScatterElements", "ScatterND"):
        idx_name = node.input[1] if len(node.input) > 1 else None
        idx = consts.get(idx_name) if idx_name else None
        axis = _get_attr(node, "axis", 0)
        data_shape = shapes.get(node.input[0])
        # 沿大小为 1 的轴散射 => 只能写回原位
        if op == "ScatterElements" and data_shape and 0 <= axis < len(data_shape) and data_shape[axis] == 1:
            return True, f"ScatterElements 沿 axis={axis}，而该轴大小为 1 ⇒ 数学恒等", "high"
        if idx is not None and idx.size and (idx == 0).all():
            return True, f"indices 全为 0（{idx.size} 个）⇒ 写回原位，数学恒等", "high"
        return False, "散射算子：indices 非全 0 或不可读，需人工确认", "low"

    if op == "Pad":
        pads = consts.get(node.input[1]) if len(node.input) > 1 else None
        if pads is None:
            for a in node.attribute:          # opset<11 的 pads 在 attribute 里
                if a.name == "pads":
                    pads = list(a.ints)
        if pads is not None and len(pads) and not any(int(p) for p in pads):
            return True, "pads 全为 0 ⇒ 恒等", "high"
        return False, "Pad：pads 非全 0 或不可读", "low"

    if op == "Expand":
        tgt = consts.get(node.input[1]) if len(node.input) > 1 else None
        src = shapes.get(node.input[0])
        if tgt is not None and src and list(tgt.tolist()) == list(src):
            return True, "目标 shape == 输入 shape ⇒ 恒等", "high"
        return False, "Expand：目标 shape 与输入不同或不可读", "low"

    if op == "Tile":
        reps = consts.get(node.input[1]) if len(node.input) > 1 else None
        if reps is not None and reps.size and (reps == 1).all():
            return True, "repeats 全为 1 ⇒ 恒等", "high"
        return False, "Tile：repeats 非全 1 或不可读", "low"

    if op == "Concat" and len(node.input) == 1:
        return True, "Concat 只有一个输入 ⇒ 恒等", "high"

    return None


def scan(path: Path):
    # 🔴 load_external_data=False：大模型连权重加载会爆内存并撞 protobuf 2GB 上限（指南 §3.1 坑 1）
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph
    consts, shapes = _const_map(g), _shape_map(g)

    # 🔴🔴 判据依赖 shape：有一条规则是「沿大小为 1 的轴散射 ⇒ 恒等」。
    # 导出的图**不保证**带 value_info —— 缺了它这条规则会**静默失效**（只漏报、不报错）。
    # 本项目实测：同一对图，带 value_info 的扫出 5 个、不带的只扫出 2 个，
    # 差异全部来自这里，不是结构差异。⇒ 先补全，再判定。
    covered = sum(1 for n in g.node if n.output and n.output[0] in shapes)
    shape_complete = covered >= max(1, len(g.node)) * 0.9
    if not shape_complete:
        try:
            from onnx import shape_inference as si
            inferred = si.infer_shapes(model, strict_mode=False, data_prop=True)
            shapes = _shape_map(inferred.graph)
            covered = sum(1 for n in g.node if n.output and n.output[0] in shapes)
            shape_complete = covered >= max(1, len(g.node)) * 0.9
        except Exception:
            pass

    hits, by_type = [], {}
    for node in g.node:
        rec = by_type.setdefault(node.op_type, {"count": 0, "out_elems": 0})
        rec["count"] += 1
        n = _numel(shapes.get(node.output[0])) if node.output else None
        if n:
            rec["out_elems"] += n

        verdict = classify(node, consts, shapes)
        if verdict is None:
            continue
        is_identity, reason, conf = verdict
        hits.append({
            "name": node.name or "<anon>",
            "op_type": node.op_type,
            "output": node.output[0] if node.output else "",
            "out_shape": shapes.get(node.output[0] if node.output else ""),
            "out_elems": n,
            "identity": is_identity,
            "reason": reason,
            "confidence": conf,
        })

    hits.sort(key=lambda h: (h["identity"], h["out_elems"] or 0), reverse=True)
    return {"file": path.name, "path": str(path), "nodes": len(g.node),
            "shape_complete": shape_complete, "shape_covered": covered,
            "by_type": by_type, "hits": hits}


def report(res, verbose=True):
    ident = [h for h in res["hits"] if h["identity"]]
    susp = [h for h in res["hits"] if not h["identity"]]
    print(f"\n=== {res['file']}  （{res['nodes']} 节点）===")
    if not res.get("shape_complete", True):
        print(f"  ⚠️  shape 信息不完整（{res['shape_covered']}/{res['nodes']} 节点有 shape，"
              f"shape inference 也没能补全）")
        print(f"     ⇒ 「沿大小为 1 的轴」这条规则会失效，**本次结果可能漏报**。"
              f"先跑 §3.1 的固化流程补全 value_info 再扫。")
    if not ident and not susp:
        print("  ✅ 未发现高危类型算子")
        return ident

    if ident:
        print(f"  🔴 判定为恒等/填充的全尺寸算子：{len(ident)} 个  <- 这些应在导出阶段去掉")
        for h in ident:
            elems = f"{h['out_elems']:,}" if h["out_elems"] else "?"
            print(f"     · {h['op_type']:<16} {h['name']}")
            print(f"       输出 {h['out_shape']}（{elems} 元素）— {h['reason']}")
    if susp and verbose:
        print(f"  ⚠️  同类但未判定（需人工确认）：{len(susp)} 个")
        for h in susp[:10]:
            print(f"     · {h['op_type']:<16} {h['name']} — {h['reason']}")
        if len(susp) > 10:
            print(f"     … 另 {len(susp) - 10} 个")
    return ident


def selftest():
    """约束 8：先用已知样本验证判据的操作化本身成立。"""
    print("=== 判据自检（已知样本）===")
    ok = True
    for fname, expect in SELFTEST_CASES:
        p = SELFTEST_DIR / fname
        if not p.exists():
            print(f"  ⚠️  跳过（样本不在）：{p}")
            ok = False
            continue
        ident = [h for h in scan(p)["hits"] if h["identity"]]
        got = sum(1 for h in ident if h["op_type"].startswith("Scatter"))
        mark = "✅" if got == expect else "❌"
        print(f"  {mark} {fname}: 期望 {expect} 个 Scatter*，实得 {got}")
        if got != expect:
            ok = False
    print("\n判据自检：" + ("✅ 通过，可用于新模型" if ok else "❌ 未通过 —— 结论不可用"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="扫描 ONNX 里'全尺寸但语义恒等'的导出残留算子")
    ap.add_argument("target", nargs="?", help="ONNX 文件或目录")
    ap.add_argument("--json", help="把完整结果写到这个 JSON")
    ap.add_argument("--selftest", action="store_true", help="用已知样本验证判据本身")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.target:
        ap.error("需要 target，或用 --selftest")

    t = Path(args.target)
    files = sorted(t.glob("*.onnx")) if t.is_dir() else [t]
    if not files:
        sys.exit(f"没有找到 .onnx：{t}")

    all_res, total = [], 0
    for f in files:
        try:
            res = scan(f)
        except Exception as e:
            print(f"\n=== {f.name} ===\n  ❌ 解析失败：{e}")
            continue
        total += len(report(res))
        all_res.append(res)

    print(f"\n{'=' * 60}\n合计判定为恒等/填充的全尺寸算子：{total} 个（{len(all_res)} 个图）")
    if total:
        print("🔴 这些应在**导出阶段**去掉。事后删是图级手术，且同型结构不可外推 —— 见指南 §47.3。")

    if args.json:
        Path(args.json).write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"完整结果 -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
