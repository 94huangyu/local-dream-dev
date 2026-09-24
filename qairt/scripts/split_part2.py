"""把 transformer_part2 在残差流 `add_92` 处切成两段，绕开 unsigned PD 的容量上限。

依据（①实测，台账 #57 / EXP_PLAN_P2FIT）：
  part2 单个 context 合计 3506 MB 已贴着 unsigned PD 上限，per-row 需 +30 MB ⇒ 装不下。

⚠️ 两个已踩的坑：
 1. 不能用 `onnx.utils.extract_model`——它会内联外部权重，撞 protobuf 2 GB 上限
    （实测 `EncodeError: Failed to serialize proto`）。改用底层 `Extractor`，
    产物存到与 `.onnx.data` 同目录，外部引用照样解析且不复制 10.9 GB 权重。
 2. **切割集不能按节点序号算**（约束 8：Id 顺序 ≠ 依赖顺序）。
    早期节点（如 node_unsqueeze_1 展开 freqs）的输出会被很后面的节点消费。
    首次按序号切，转换直接报 `Node node_unsqueeze_1: 'unified_freqs'`。
    正确做法是**反向可达性**：part2a = 生成切口张量所需的全部节点；
    切割集 = part2a 产出 ∩ part2b 消费。实测为 6 个张量、合计 65.6 MB。
"""
import os

import onnx
from onnx.utils import Extractor

SRC = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part2_fixed.onnx"
OUTDIR = os.path.dirname(SRC)
CUT = ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3", "val_105"]
P2A_IN = ["unified", "unified_mask", "unified_freqs", "adaln_input"]
P2B_IN = CUT + ["adaln_input", "latents_shape"]

m = onnx.load(SRC, load_external_data=False)
print(f"载入 {SRC}（未加载外部权重）nodes={len(m.graph.node)}")
e = Extractor(m)
for tag, ins, outs in [("part2a", P2A_IN, CUT), ("part2b", P2B_IN, ["latents"])]:
    sub = e.extract_model(ins, outs)
    dst = os.path.join(OUTDIR, f"transformer_{tag}_fixed.onnx")
    onnx.save(sub, dst)
    print(f"  {tag}: nodes={len(sub.graph.node):>5} inits={len(sub.graph.initializer):>4} -> {dst}")
    print(f"       输入 {[i.name for i in sub.graph.input]}")
    print(f"       输出 {[o.name for o in sub.graph.output]}")
