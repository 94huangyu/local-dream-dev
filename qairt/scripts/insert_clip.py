"""在 ONNX 里为**溢出张量**插入显式 `Clip(±65504)`（台账 #111 的落地）。

为什么（今日 QDQ 模拟实测，`EXP_PLAN_SIM_CEILING`）：
  F（除溢出外全 FP16，3105 个）= 24.96%   —— 浮点张量数量几乎不重要
  G（全 FP16 + 显式 Clip，3141 个）= 14.98%  —— **那几十个 massive activation 独占 10 pp**
  H（old-26 集合 + Clip 溢出，466 个）= **18.18%**  ⇒ 推算设备 22.18%（当前最优 30.68%）
⇒ **Clip 是仅次于「用浮点」本身的第二大杠杆。**

🔴 阈值锁定 **65504，不得为保险多钳**（#111 实测：钳到 32752 全量误差跳到 15.49%，
   钳到 16376 跳到 43.85% —— 钳位代价超线性）。
🔴 危险的是**隐式**溢出（FP16 变 inf），**显式**钳位后超出部分变成 65504 ⇒ 1.0 倍余量是安全的。

做法：把生产者的输出改名为 `<T>_preclip`，插 `Clip` 产出 `<T>`
      => 下游与白名单**都不用改名**。

用法: python insert_clip.py <part1a|part1b|part2a|part2b>
产物: <base>_clip.onnx  与  <seg>_clipped.json（被钳张量的 ONNX 名，供白名单使用）
"""
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
ONNX = {"part1a": "transformer_part1a", "part1b": "transformer_part1b",
        "part2a": "transformer_part2a_fixed", "part2b": "transformer_part2b_fixed"}
FP16MAX = 65504.0
TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: "
                r"\[([^\]]*)\]; tensor type: ([A-Z]+)\)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def main():
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    seg = sys.argv[1]
    base = ONNX[seg]
    kind, pt = {}, {}
    with io.open(os.path.join(W, "%s_full_enc.csv" % seg), encoding="utf-8",
                 errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), m.group(4))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), (int(m.group(2)), float(m.group(5)),
                                           float(m.group(6))))
    meas = {}
    ap = os.path.join(W, "%s_amax.json" % seg)
    if os.path.exists(ap):
        meas = {k: v["amax"] for k, v in
                json.load(io.open(ap, encoding="utf-8"))["tensors"].items()}

    m = onnx.load(os.path.join(D, base + ".onnx"), load_external_data=False)
    g = m.graph
    node_out = {o for n in g.node for o in n.output}
    prod = {}
    for n in g.node:
        for k, o in enumerate(n.output):
            prod[o] = (n, k)

    def to_onnx(t):
        if t in node_out:
            return t
        if t.endswith("_fc") and t[:-3] in node_out:
            return t[:-3]
        return None

    # 选出溢出张量：实测优先，没实测的用标定（#110：标定不可靠，故阈值放宽到 /1.2）
    picked, why = {}, {}
    for t, (bw, sc, off) in pt.items():
        if bw != 16 or kind.get(t) == "STATIC":
            continue
        o = to_onnx(t)
        if o is None or o in picked:
            continue
        if t in meas:
            hi, src = meas[t], "实测"
        else:
            hi, src = max(abs((2 ** bw - 1 + off) * sc), abs(off * sc)), "标定"
        if hi > FP16MAX / 1.2:
            picked[o] = hi
            why[o] = src

    lo_n, hi_n = "_clip_lo", "_clip_hi"
    g.initializer.append(numpy_helper.from_array(np.array(-FP16MAX, np.float32), lo_n))
    g.initializer.append(numpy_helper.from_array(np.array(FP16MAX, np.float32), hi_n))
    added = []
    for o in sorted(picked):
        n0, k0 = prod[o]
        pre = o + "_preclip"
        n0.output[k0] = pre
        added.append(helper.make_node("Clip", [pre, lo_n, hi_n], [o],
                                      name="clip_" + o))
    g.node.extend(added)

    # 拓扑排序（新 Clip 必须排在生产者之后）
    ready = {t.name for t in g.initializer} | {i.name for i in g.input}
    order, pend, guard = [], list(g.node), 0
    while pend and guard < 200000:
        guard += 1
        nxt = []
        for n in pend:
            if all((i in ready) or i == "" for i in n.input):
                order.append(n)
                ready.update(n.output)
            else:
                nxt.append(n)
        if len(nxt) == len(pend):
            sys.exit("FAIL 拓扑排序卡住，剩 %d" % len(nxt))
        pend = nxt
    del g.node[:]
    g.node.extend(order)

    dst = os.path.join(D, base + "_clip.onnx")
    onnx.save(m, dst)
    json.dump(sorted(picked), io.open(os.path.join(W, "%s_clipped.json" % seg), "w",
                                      encoding="utf-8"))
    print("[%s] 插入 Clip(±65504) **%d 个**（实测判定 %d / 标定判定 %d）⇒ %s"
          % (seg, len(picked), sum(1 for k in picked if why[k] == "实测"),
             sum(1 for k in picked if why[k] == "标定"), os.path.basename(dst)), flush=True)
    for o in sorted(picked, key=lambda x: -picked[x])[:5]:
        print("    %-22s |a|max=%10.0f (%s)" % (o, picked[o], why[o]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
