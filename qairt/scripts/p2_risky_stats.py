"""EXP_PLAN_FP16_SURGICAL 步骤 A：实测 part2a 高危张量的真实量级。

🔴 不用标定值：已实测 `linear_7` 标定 max 76492.65 但真实 |a|max = 77704.5（超 1.6%）
   ⇒ 「标定 max < 65504」不等于运行时不溢出。

只算标量（|a|max、主体中位），不落盘，分批跑控内存。
"""
import os, sys, json, re, io
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "testB", "s0_transformer_part2")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
BASE = "transformer_part2a_fixed"
IN_SHAPES = {"unified": (1, 4128, 3840), "unified_mask": (1, 4128),
             "unified_freqs": (1, 4128, 64, 2), "adaln_input": (1, 256)}
BATCH = 14


def main():
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort
    cand = json.load(open(os.path.join(OUT, "risky.json"), encoding="utf-8"))
    # DLC 里的 encoding（用于算「量化级/元素」）
    t = io.open(os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments",
                             "transformer_part2a_dlcinfo.txt"), encoding="utf-8", errors="replace").read()
    PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                    r"scale ([-\d.eE+]+), offset")
    enc = {}
    for m in PT.finditer(t):
        enc.setdefault(m.group(1), dict(bw=int(m.group(2)), min=float(m.group(3)),
                                        max=float(m.group(4)), scale=float(m.group(5))))
    feeds = {}
    for n, s in IN_SHAPES.items():
        feeds[n] = np.fromfile(os.path.join(SRC, n + ".raw"), np.float32).reshape(s)

    res = {}
    for b in range(0, len(cand), BATCH):
        grp = cand[b:b + BATCH]
        mo = onnx.load(os.path.join(D, BASE + ".onnx"), load_external_data=False)
        have = {o.name for o in mo.graph.output}
        for x in grp:
            if x not in have:
                mo.graph.output.append(helper.make_tensor_value_info(x, TensorProto.FLOAT, None))
        p = os.path.join(D, BASE + "_stats.onnx")
        onnx.save(mo, p)
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        f2 = dict(feeds)
        for i in mo.graph.input:
            if i.type.tensor_type.elem_type == TensorProto.BOOL:
                f2[i.name] = feeds[i.name].astype(bool)
        outs = [o.name for o in s.get_outputs() if o.name in grp]
        r = s.run(outs, f2)
        for n, a in zip(outs, r):
            a = np.abs(a.astype(np.float64)).ravel()
            res[n] = dict(amax=float(a.max()), med=float(np.median(a)))
        del s, r
        print("  批 %d/%d 完成（%d 个张量）" % (b // BATCH + 1, (len(cand) + BATCH - 1) // BATCH, len(outs)),
              flush=True)

    FP16MAX = 65504.0
    rows = []
    for n, v in res.items():
        e = enc.get(n) or enc.get(n + "_fc")
        if not e or e["bw"] != 16:
            continue
        lv = v["med"] / e["scale"]
        rows.append((n, e["scale"], v["med"], lv, v["amax"], e["max"],
                     v["amax"] * 2.0 <= FP16MAX))
    print("\n%-14s %10s %10s %11s %11s %11s %8s"
          % ("tensor", "步长", "主体中位", "量化级/元素", "实测|a|max", "标定max", "FP16(2x余量)"))
    print("-" * 84)
    for r in sorted(rows, key=lambda z: z[3]):
        print("%-14s %10.5f %10.4f %11.2f %11.1f %11.1f %8s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], "✅" if r[6] else "🔴"))
    bad = [r for r in rows if r[3] < 10]
    ok = [r for r in bad if r[6]]
    print("\n量化级/元素 < 10 的：%d / %d 个；其中满足 2 倍 FP16 余量的：**%d 个**"
          % (len(bad), len(rows), len(ok)))
    print("⇒ 手术名单：", [r[0] for r in ok])
    json.dump({r[0]: dict(scale=r[1], med=r[2], levels=r[3], amax=r[4], calib_max=r[5], fp16ok=r[6])
               for r in rows}, open(os.path.join(OUT, "risky_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
