"""通用：为某一段筛出「量化分辨率被摧毁且 FP16 装得下」的张量。

用法: python seg_fp16_stats.py <part1a|part1b|part2b>
规则（EXP_PLAN_FP16_SURGICAL §三，事前锁定）：
  1) 量化级/元素 = median(|a|)/scale < 10
  2) **实测** |a|max × 2.0 <= 65504     （标定值会低估 1.8~1.95 倍，见指南 §28.10）
  3) 输入锥有界 => 残差流张量（add_*）不入选
"""
import os, re, io, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
SEG = sys.argv[1]
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
ONNX = {"part1a": "transformer_part1a", "part1b": "transformer_part1b",
        "part2b": "transformer_part2b_fixed"}[SEG]
FP16MAX, BATCH = 65504.0, 14
# 各段的输入契约（喂 FP32 前向用；与设备侧同一份数据）
SRC = {"part1a": (os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "htp_inloop"), None),
       "part1b": (os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "testB",
                               "s0_transformer_part1b"), None),
       "part2b": (None, None)}[SEG]


def main():
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort
    csv = os.path.join(W, "%s_full_enc.csv" % SEG)
    PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                    r"scale ([-\d.eE+]+), offset")
    TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: [A-Za-z_0-9]+; tensor dimension: \[[^\]]*\]; "
                    r"tensor type: ([A-Z]+)\)")
    enc, kind = {}, {}
    with io.open(csv, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), m.group(2))
            for m in PT.finditer(line):
                enc.setdefault(m.group(1), dict(bw=int(m.group(2)), max=float(m.group(4)),
                                                min=float(m.group(3)), scale=float(m.group(5))))
    p = os.path.join(D, ONNX + ".onnx")
    m = onnx.load(p, load_external_data=False)
    inits = {t.name for t in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    sig = {n.output[0] for n in m.graph.node if n.op_type == "Sigmoid"}
    cand = []
    for n in m.graph.node:
        if n.op_type != "Mul":
            continue
        a = [i for i in n.input if i not in inits]
        if len(a) != 2:
            continue
        for i in a:
            q = prod.get(i)
            if q is not None and q.op_type == "Mul" and any(z in sig for z in q.input):
                cand.append(n.output[0]); break
    cand = sorted(set(cand))
    print("[%s] SwiGLU 输出 %d 个：%s" % (SEG, len(cand), cand))
    json.dump(cand, open(os.path.join(W, "%s_swiglu.json" % SEG), "w"))


if __name__ == "__main__":
    main()
