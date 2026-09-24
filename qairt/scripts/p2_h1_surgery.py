# -*- coding: utf-8 -*-
"""D2-H1 · S1：删掉 part1a 里两处 `pad_sequence` 导出的全尺寸 ScatterElements。

## 为什么可以删（① 结构 + ① 编码，见 EXP_PLAN_P2_SPEED §1.3 / §6.1）
结构是 `ScatterElements(data=Expand(fill), indices=Expand(0), updates=Unsqueeze(Pad(x)), axis=0)`：
**沿大小为 1 的第 0 维、索引全 0、逐位置全覆盖 ⇒ 输出严格等于 updates**。
且部署 DLC 里这两个输出与各自 updates 的 **scale/offset 逐位相同** ⇒ 下游读到的量化值不变。

## 本脚本的自证（任一不过就退出，不产出文件）
  A1 两个目标节点存在且 op_type == ScatterElements、axis == 0
  A2 data / indices / updates / output 四者形状相同，且第 0 维 == 1
  A3 indices 由 Expand(常量) 生成，且该常量**每个元素都是 0**（不是"看起来像 0"，是全量检查）
  A4 手术后：两个节点消失、消费者改读 updates、图输入/输出名与形状**逐个不变**
  A5 手术后形状推理通过，且**每个保留节点的输入都能解析**（无悬空引用）
  A6 只删了预期的节点：删除集合 ⊆ {两个 ScatterElements + 它们专用的 Expand}

🔴 **不动** `Pad`（`constant_pad_nd*`，其 encoding 与下游不同，删它会改数值），
🔴 **不动**其余 4 个小 ScatterElements（合计仅 0.02% cycles，其中 `select_scatter_5` 还是图输出）。

用法: python scripts/p2_h1_surgery.py [--out-stem transformer_part1a_clip_L80_noscat]
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import canonical_sources as cs  # noqa: E402

TARGETS = ["node_select_scatter", "node_select_scatter_4"]


def die(msg):
    raise SystemExit("🔴 " + msg)


def const_value(model, name):
    """取常量张量的值：initializer 或 Constant 节点。取不到返回 None。"""
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def shapes_of(model):
    sh = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            sh[vi.name] = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]
    for init in model.graph.initializer:
        sh[init.name] = list(init.dims)
    return sh


def main():
    src = cs.onnx_for("part1a", "deployed", L=cs.DEPLOYED_L)
    if "--src" in sys.argv:                      # 交付版：对五个比例的 ONNX 逐个手术
        src = sys.argv[sys.argv.index("--src") + 1]
    # --only <节点名>：只删这一个（用于「单删」变体，判断是哪一个触发后端崩坏）
    targets = list(TARGETS)
    if "--only" in sys.argv:
        targets = [sys.argv[sys.argv.index("--only") + 1]]
        if targets[0] not in TARGETS:
            die("--only 只能是 %s" % TARGETS)
    stem = "transformer_part1a_clip_L80_noscat"
    if len(targets) == 1:
        stem += "_" + ("A" if targets[0] == TARGETS[0] else "B")
    if "--out-stem" in sys.argv:
        stem = sys.argv[sys.argv.index("--out-stem") + 1]
    elif "--src" in sys.argv:
        stem = os.path.splitext(os.path.basename(src))[0] + "_noscatA"
    dst = os.path.join(os.path.dirname(src), stem + ".onnx")
    print("源   : %s" % src)
    print("产物 : %s" % dst)
    if not os.path.isfile(src):
        die("源 ONNX 不存在")

    m = onnx.load(src, load_external_data=False)      # 外部权重保持引用，产物必须存回同目录
    m = shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=True)
    sh = shapes_of(m)
    producer = {o: n for n in m.graph.node for o in n.output}
    users = {}
    for n in m.graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    rewire = {}          # scatter 输出 -> updates
    drop = set()         # 待删节点名
    for tname in targets:
        node = next((n for n in m.graph.node if n.name == tname), None)
        if node is None or node.op_type != "ScatterElements":
            die("A1 找不到 %s（或 op_type 不是 ScatterElements）" % tname)
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        if axis != 0:
            die("A1 %s 的 axis = %d ≠ 0" % (tname, axis))
        data_n, idx_n, upd_n = node.input[0], node.input[1], node.input[2]
        out_n = node.output[0]
        s = [sh.get(x) for x in (data_n, idx_n, upd_n, out_n)]
        if any(x is None for x in s) or len({tuple(x) for x in s}) != 1:
            die("A2 %s 的 data/indices/updates/output 形状不一致：%s" % (tname, s))
        if s[0][0] != 1:
            die("A2 %s 的第 0 维 = %s ≠ 1 ⇒ 不是 batch=1 的 pad_sequence 形态" % (tname, s[0][0]))
        # A3：indices 必须来自 Expand(常量 0)
        ip = producer.get(idx_n)
        if ip is None or ip.op_type != "Expand":
            die("A3 %s 的 indices 不是 Expand 产出（实际 %s）" % (tname, ip.op_type if ip else "图输入/常量"))
        base = const_value(m, ip.input[0])
        if base is None:
            die("A3 %s 的 indices 源 %s 不是常量 ⇒ 不能断定全 0" % (tname, ip.input[0]))
        if not np.all(np.asarray(base) == 0):
            die("A3 %s 的 indices 源常量非全 0：%s" % (tname, np.unique(np.asarray(base))[:5]))
        dp = producer.get(data_n)
        print("  ✅ %s：形状 %s，axis=0，indices = Expand(%s = 全 0，%d 个元素)"
              % (tname, s[0], ip.input[0], np.asarray(base).size))
        rewire[out_n] = upd_n
        drop.add(node.name)
        for feeder, role in ((ip, "indices 的 Expand"), (dp, "data 的 Expand")):
            if feeder is not None and feeder.op_type == "Expand":
                # 只有当该 Expand 的输出除了这个 scatter 之外无人使用时才删
                if all(u.name == node.name for u in users.get(feeder.output[0], [])):
                    drop.add(feeder.name)
                    print("     ⊕ 连带删除 %s（%s，无其他消费者）" % (feeder.name, role))

    # A6：删除集合只能是目标 + Expand
    for n in m.graph.node:
        if n.name in drop and n.op_type not in ("ScatterElements", "Expand"):
            die("A6 意外要删 %s（%s）" % (n.name, n.op_type))

    before_out = [(o.name, sh.get(o.name)) for o in m.graph.output]
    before_in = [(i.name, sh.get(i.name)) for i in m.graph.input]
    n_before = len(m.graph.node)

    keep = [n for n in m.graph.node if n.name not in drop]
    for n in keep:
        for k, i in enumerate(n.input):
            if i in rewire:
                n.input[k] = rewire[i]
    del m.graph.node[:]
    m.graph.node.extend(keep)
    # 清掉被删张量的 value_info（否则形状推理会带着已不存在的名字）
    vis = [vi for vi in m.graph.value_info if vi.name not in rewire]
    del m.graph.value_info[:]
    m.graph.value_info.extend(vis)

    # A5：无悬空引用
    avail = {i.name for i in m.graph.input} | {i.name for i in m.graph.initializer}
    for n in m.graph.node:
        for i in n.input:
            if i and i not in avail and i not in {o for x in m.graph.node for o in x.output}:
                die("A5 悬空引用：%s 的输入 %s 无生产者" % (n.name, i))
    m2 = shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=True)
    sh2 = shapes_of(m2)
    after_out = [(o.name, sh2.get(o.name)) for o in m2.graph.output]
    after_in = [(i.name, sh2.get(i.name)) for i in m2.graph.input]
    if after_out != before_out or after_in != before_in:
        die("A4 图输入/输出发生变化\n  前 %s\n  后 %s" % (before_out, after_out))
    if any(n.op_type == "ScatterElements" and n.name in targets for n in m2.graph.node):
        die("A4 目标节点仍在图里")

    onnx.save(m, dst)   # 存回同目录 ⇒ 外部权重相对路径仍可解析
    print("\n删除节点 %d 个：%s" % (len(drop), sorted(drop)))
    print("节点数 %d -> %d；图输入/输出逐个不变 ✅" % (n_before, len(m2.graph.node)))
    print("剩余 ScatterElements：%s"
          % [n.name for n in m2.graph.node if n.op_type == "ScatterElements"])
    print("产物大小 %.2f MB" % (os.path.getsize(dst) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
