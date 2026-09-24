"""实测被排除的 SwiGLU 张量的真实 |a|max，用「实测 x 2.0 <= 65504」重筛。

背景：四段建图时用的是保守代理「标定 |max| x 4.0 <= 65504」，
因为实测校准低估 1.83~1.95 倍（指南 §28.10）。
用实测值可能救回几个 —— 本脚本给出确定答案，不再靠倍数外推。

用法: python excluded_actual.py <part1a|part1b|part2a>
"""
import os, re, io, sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SEG = sys.argv[1]
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
FP16MAX = 65504.0

CFG = {
    "part1a": ("transformer_part1a",
               {"latents": (os.path.join(P0, "htp_inloop", "s0", "latents.raw"), (1, 16, 128, 128)),
                "timestep": (os.path.join(P0, "htp_inloop", "s0", "timestep.raw"), (1,)),
                "caption": (os.path.join(P0, "htp_inloop", "const", "caption.raw"), (1, 32, 2560)),
                "cap_pad_mask": (os.path.join(P0, "htp_inloop", "const", "cap_pad_mask.raw"), (1, 32))}),
    "part1b": ("transformer_part1b",
               {n: (os.path.join(P0, "testB", "s0_transformer_part1b", n + ".raw"), s)
                for n, s in [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
                             ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
                             ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))]}),
    "part2a": ("transformer_part2a_fixed",
               {n: (os.path.join(P0, "testB", "s0_transformer_part2", n + ".raw"), s)
                for n, s in [("unified", (1, 4128, 3840)), ("unified_mask", (1, 4128)),
                             ("unified_freqs", (1, 4128, 64, 2)), ("adaln_input", (1, 256))]}),
}[SEG]
ONNX, INS = CFG
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset")


def main():
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort

    enc = {}
    with io.open(os.path.join(W, "%s_full_enc.csv" % SEG), encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in PT.finditer(line):
                enc.setdefault(m.group(1), dict(bw=int(m.group(2)), mn=float(m.group(3)),
                                                mx=float(m.group(4)), scale=float(m.group(5))))
    white = set(json.load(open(os.path.join(W, "allseg_white.json"), encoding="utf-8"))[SEG])

    p = os.path.join(D, ONNX + ".onnx")
    m = onnx.load(p, load_external_data=False)
    inits = set(t.name for t in m.graph.initializer)
    prod = {o: n for n in m.graph.node for o in n.output}
    sig = set(n.output[0] for n in m.graph.node if n.op_type == "Sigmoid")
    sw = []
    for n in m.graph.node:
        if n.op_type != "Mul":
            continue
        a = [i for i in n.input if i not in inits]
        if len(a) != 2:
            continue
        for i in a:
            q = prod.get(i)
            if q is not None and q.op_type == "Mul" and any(z in sig for z in q.input):
                sw.append(n.output[0])
                break
    excluded = sorted(set(sw) - white)
    print("[%s] SwiGLU %d 个，已入选 %d，**被排除 %d**：%s"
          % (SEG, len(set(sw)), len(white), len(excluded), excluded), flush=True)
    if not excluded:
        return

    have = set(o.name for o in m.graph.output)
    for t in excluded:
        if t not in have:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    dst = os.path.join(D, ONNX + "_exc.onnx")
    onnx.save(m, dst)
    feeds = {}
    for n, (path, shape) in INS.items():
        feeds[n] = np.fromfile(path, np.float32).reshape(shape)
    for i in m.graph.input:
        if i.type.tensor_type.elem_type == TensorProto.BOOL:
            feeds[i.name] = feeds[i.name].astype(bool)
    print("  跑 FP32 前向 ...", flush=True)
    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs() if o.name in excluded]
    r = s.run(outs, feeds)
    print("\n%-12s %12s %12s %8s %11s %10s"
          % ("tensor", "标定|max|", "实测|a|max", "低估", "量化级/元素", "实测x2 可行?"))
    print("-" * 74)
    rescued = []
    for n, a in zip(outs, r):
        a = np.abs(a.astype(np.float64)).ravel()
        amax, med = float(a.max()), float(np.median(a))
        e = enc.get(n) or enc.get(n + "_fc")
        cmx = max(abs(e["mn"]), abs(e["mx"]))
        lv = med / e["scale"]
        ok = amax * 2.0 <= FP16MAX
        if ok:
            rescued.append(n)
        print("%-12s %12.1f %12.1f %7.2fx %11.2f %10s"
              % (n, cmx, amax, amax / max(cmx, 1e-9), lv, "✅ 可救" if ok else "🔴 仍超"))
    print("\n⇒ [%s] 可救回 **%d / %d** 个：%s" % (SEG, len(rescued), len(excluded), rescued))
    json.dump(rescued, open(os.path.join(W, "%s_rescued.json" % SEG), "w"))


if __name__ == "__main__":
    main()
