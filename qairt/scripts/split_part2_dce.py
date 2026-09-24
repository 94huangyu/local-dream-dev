"""part2 正确切分：**先死代码消除(DCE)，再切**。

## 为什么必须先 DCE（2026-08-16 实测，代价：一整轮返工）

`transformer_part2_fixed.onnx` 里有 **84 个死节点**
（IsNaN 15、Where 15、Gather 10、Reshape 14、Cast 6 …），
它们的输出不可达图输出，但仍在图里"消费" **30 个 `[1,30,4128,4128]` 张量**（合计 57.1 GiB）。

| 做法 | 切割集 |
|---|---|
| 不 DCE 直接切 | **14 个张量、16.4 GB**（8 个巨型注意力矩阵被死节点拽过切口） |
| **先 DCE 再切** | **6 个张量、65.6 MB** ✅ |

转换器对**完整图**会自己做 DCE（生产 part2 DLC 里 `IsNaN` 计数为 0），
所以完整图量化没问题；**一旦切分，死节点就被"复活"成子图的必需部分**，
量化时每个 1.9044 GiB 的巨型张量都要参与校准
——实测内存增长步长恰好 **1.90~1.91 GB**，与此吻合。

## 另外两个已踩的坑

1. 不能用 `onnx.utils.extract_model`：它内联外部权重，撞 protobuf 2 GB 上限
   （`EncodeError: Failed to serialize proto`）。用底层 `Extractor`，
   产物存到与 `.onnx.data` 同目录，外部引用照样解析且不复制 10.86 GB 权重。
2. **切割集不能按节点序号算**（约束 8：Id 顺序 ≠ 依赖顺序）。必须用反向可达性。

## 源文件必须是 `_fixed`

生产 part2 DLC 的 `--input_network` 是 `transformer_part2_fixed.onnx`（已从
`dlc_pipeline/transformer_part2/01_qairt_converter.log` 核实）。
我第一次误用了 `transformer_part2.onnx`（base），两者差 **108 个节点**，整轮作废。
"""
import os

import onnx
from onnx.utils import Extractor

E = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
SRC = os.path.join(E, "transformer_part2_fixed.onnx")
DCE = os.path.join(E, "transformer_part2_fixed_dce.onnx")
CUT = ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3", "val_105"]

m = onnx.load(SRC, load_external_data=False)
g = m.graph
init = {t.name for t in g.initializer}
gin = {i.name for i in g.input}
prod = {o: i for i, n in enumerate(g.node) for o in n.output}

# ---- DCE：从图输出反向可达 ----
live, stack = set(), [o.name for o in g.output]
while stack:
    t = stack.pop()
    k = prod.get(t)
    if k is None or k in live:
        continue
    live.add(k)
    for x in g.node[k].input:
        if x and x not in init and x not in gin:
            stack.append(x)
dead = set(range(len(g.node))) - live
print(f"DCE: 活节点 {len(live)}  死节点 {len(dead)} / 共 {len(g.node)}")

keep = [g.node[k] for k in sorted(live)]
used = {x for n in keep for x in n.input if x}
del g.node[:]
g.node.extend(keep)
kept_init = [t for t in g.initializer if t.name in used]
print(f"     权重 {len(g.initializer)} -> {len(kept_init)}")
del g.initializer[:]
g.initializer.extend(kept_init)
kept_in = [i for i in g.input if i.name in used]
print(f"     图输入 {[i.name for i in g.input]} -> {[i.name for i in kept_in]}")
del g.input[:]
g.input.extend(kept_in)
onnx.save(m, DCE)
print(f"写出 {DCE}")

# ---- 在 DCE 后的图上切 ----
m2 = onnx.load(DCE, load_external_data=False)
e = Extractor(m2)
P2A_IN = [i.name for i in m2.graph.input]
for tag, ins, outs in [("part2a", P2A_IN, CUT), ("part2b", CUT + ["adaln_input"], ["latents"])]:
    sub = e.extract_model(ins, outs)
    dst = os.path.join(E, f"transformer_{tag}_fixed.onnx")
    onnx.save(sub, dst)
    print(f"  {tag}: 节点 {len(sub.graph.node):>5}  权重 {len(sub.graph.initializer):>4}")
    print(f"        输入 {[i.name for i in sub.graph.input]}")
    print(f"        输出 {[o.name for o in sub.graph.output]}")

# ---- 自检 ----
a = onnx.load(os.path.join(E, "transformer_part2a_fixed.onnx"), load_external_data=False).graph
b = onnx.load(os.path.join(E, "transformer_part2b_fixed.onnx"), load_external_data=False).graph
print(f"\n自检: {len(a.node)} + {len(b.node)} = {len(a.node)+len(b.node)}  vs DCE 后 {len(live)}"
      f"   {'✅' if len(a.node)+len(b.node) == len(live) else '❌ 不一致'}")
for tag, gg in (("part2a", a), ("part2b", b)):
    n = sum(1 for x in gg.node if x.op_type in ("IsNaN",))
    print(f"     {tag} 残留 IsNaN: {n}  {'✅' if n == 0 else '❌'}")
