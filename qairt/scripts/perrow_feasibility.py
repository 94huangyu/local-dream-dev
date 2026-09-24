"""#47 事前可行性测算：逐行量化能把 FC 权重的 scale 细化多少？

动机（台账 #52，①实测）：同一改动 scale 变**粗** 1.073 倍 ⇒ 设备主体余弦掉 0.0526
（CPU 只掉 0.0014，放大 37.6 倍）。⇒ scale 变**细**是当前最有依据的方向。

本脚本**不改任何东西**，只回答："值不值得花 70 分钟去重量化？"
若细化不明显（中位比值接近 1），就不该做——这是 MAINLINE 执行顺序里写死的前置条件。

口径：per-tensor 用基线 DLC 里 SDK 自己算的 scale；per-row 用同样的 min-max 规则
逐行算（`s_row = (max_row - min_row) / 255`），两个轴都算，因为"行"的方向
（K 还是 N）在方阵上无法由 dims 判断。

约束 7：同时报全量与主体口径的表示误差。

用法: python perrow_feasibility.py [张量数上限]
"""
import os
import sys

import numpy as np
import onnx

import map_opid as M
from repr_cost_sym import read_external

PARTS = {
    "part1a": (r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1a.onnx",
               r"D:\ZImage_Work\p0_experiments\transformer_part1a_dlcinfo.txt"),
    "part1b": (r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1b.onnx",
               r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"),
    "part2":  (r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part2.onnx",
               r"D:\ZImage_Work\p0_experiments\transformer_part2_dlcinfo.txt"),
}
_PART = sys.argv[2] if len(sys.argv) > 2 else "part1b"
ONNX_PATH, BASE = PARTS[_PART]


def fc_weights(path):
    ops = M.parse(path)
    out = {}
    for op in ops.values():
        if op["type"] != "FullyConnected":
            continue
        for t in op["in"]:
            if t["ttype"] != "STATIC":
                continue
            e = op["enc"].get(t["name"])
            if e and e["bw"] == 8:
                out[t["name"]] = (e, tuple(int(x) for x in t["dims"].split(",")))
    return out


def deq_err(w, s, off):
    q = np.clip(np.rint(w / s) - off, 0, 255)
    return s * (q + off)


def bulk_rel(a, b):
    a = a.ravel()
    b = b.ravel()
    thr = np.percentile(np.abs(a), 99.0)
    m = np.abs(a) <= thr
    return (np.linalg.norm(b - a) / np.linalg.norm(a),
            np.linalg.norm(b[m] - a[m]) / np.linalg.norm(a[m]))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    print(f"=== 分段: {_PART} ===\n  ONNX: {ONNX_PATH}\n  DLC info: {BASE}")
    base = fc_weights(BASE)
    model = onnx.load(ONNX_PATH, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    base_dir = os.path.dirname(ONNX_PATH)

    rows = []
    for i, dlc_name in enumerate(sorted(base)):
        if i >= limit:
            break
        e, dims = base[dlc_name]
        onnx_name = (dlc_name[:-len("_permute")]
                     if dlc_name.endswith("_permute") else dlc_name)
        if onnx_name not in inits:
            continue
        w = read_external(inits[onnx_name], base_dir).reshape(dims)
        st = e["scale"]
        pt_all, pt_bulk = bulk_rel(w, deq_err(w, st, e["offset"]))

        best = {}
        for axis, tag in ((1, "按行(轴0=K)"), (0, "按列(轴1=N)")):
            mn = w.min(axis=axis, keepdims=True)
            mx = w.max(axis=axis, keepdims=True)
            s = np.maximum((mx - mn) / 255.0, 1e-12)
            off = np.rint(mn / s)
            q = np.clip(np.rint(w / s) - off, 0, 255)
            wd = s * (q + off)
            a, b = bulk_rel(w, wd)
            best[tag] = (a, b, float(np.median(s)) / st)
        rows.append((dlc_name, w.size, pt_all, pt_bulk, best))
        if i < 8:
            k = "按行(轴0=K)"
            c = "按列(轴1=N)"
            print(f"  {dlc_name[:26]:26} per-tensor {pt_bulk*100:6.3f}%  "
                  f"{k} {best[k][1]*100:6.3f}% (s中位 x{best[k][2]:.3f})  "
                  f"{c} {best[c][1]*100:6.3f}% (s中位 x{best[c][2]:.3f})")

    print("\n" + "=" * 78)
    print(f"张量数 {len(rows)}")
    pt = np.array([r[3] for r in rows]) * 100
    print(f"per-tensor 主体表示误差: 中位 {np.median(pt):.4f}%  最大 {pt.max():.4f}%")
    for tag in ("按行(轴0=K)", "按列(轴1=N)"):
        v = np.array([r[4][tag][1] for r in rows]) * 100
        sr = np.array([r[4][tag][2] for r in rows])
        print(f"{tag}: 主体误差 中位 {np.median(v):.4f}%  最大 {v.max():.4f}%  |  "
              f"相对 per-tensor 降到 {np.median(v/pt):.4f} 倍  |  "
              f"scale 中位变细 x{np.median(sr):.4f}")
    print("\n判读：scale 中位比值明显 < 1 且表示误差明显下降 ⇒ 值得做；"
          "接近 1 ⇒ 不值得，别花 70 分钟。")


if __name__ == "__main__":
    main()
