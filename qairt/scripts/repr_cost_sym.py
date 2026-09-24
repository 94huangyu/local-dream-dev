"""对称化 FC 权重在【表示层面】要付多少代价？（宿主侧，零设备成本）

为什么要测（对两个分支都有用）：
  · 若设备侧 HTP 变好 ⇒ 需要知道这里付出的表示代价有多小，才能说净收益；
  · 若设备侧没变化 ⇒ 至少能说清"对称化本身没把权重表示改坏"，
    从而把失败归因锁定在 HTP 执行侧而不是量化表示侧。

口径：只看**权重张量自身的量化表示误差**，不涉及下游。
  非对称 uFxp_8 : q = clip(round(w/s) - off, 0, 255),  w' = s*(q + off)
  对称   sFxp_8 : q = clip(round(w/s), -128, 127),     w' = s*q
两侧的 s/off 都取 SDK 自己算出来的（分别来自基线 DLC 与 symw DLC），不自己推。

约束 7：同时报告能量集中度与主体口径，不允许只报全量相对 L2。

用法: python repr_cost_sym.py [张量数上限]
"""
import os
import sys

import numpy as np
import onnx

import map_opid as M

ONNX_PATH = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1b.onnx"
BASE = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"
SYMW = r"D:\ZImage_Work\p0_experiments\symw_part1b_dlcinfo.txt"


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
                out[t["name"]] = e
    return out


def read_external(t, base_dir):
    """直接按 offset/length 读外部权重。

    不用 onnx 的 loader：本项目的 `.onnx.data` 是 13.7 GB，
    `load_external_data_for_tensor` 在它上面报 "not regular file"（①实测）；
    而 `load_external_data=True` 会把整段权重全载进内存（GB 级），与并行任务抢内存。
    """
    d = {e.key: e.value for e in t.external_data}
    assert t.data_type == onnx.TensorProto.FLOAT, f"{t.name} 不是 float32"
    with open(os.path.join(base_dir, d["location"]), "rb") as f:
        f.seek(int(d["offset"]))
        buf = f.read(int(d["length"]))
    a = np.frombuffer(buf, dtype="<f4")
    n = int(np.prod(list(t.dims)))
    assert a.size == n, f"{t.name}: 读到 {a.size} 个元素，dims 要求 {n}"
    return a.astype(np.float64)


def rel(a, b, tag):
    """(全量相对L2, 主体相对L2, 前1%能量占比)"""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    e_all = np.linalg.norm(b - a) / np.linalg.norm(a)
    sq = a ** 2
    k = max(1, int(round(a.size * 0.01)))
    conc = np.partition(sq, -k)[-k:].sum() / sq.sum()
    thr = np.percentile(np.abs(a), 99.0)
    m = np.abs(a) <= thr
    e_bulk = np.linalg.norm(b[m] - a[m]) / np.linalg.norm(a[m])
    return e_all, e_bulk, conc


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    base, sym = fc_weights(BASE), fc_weights(SYMW)
    model = onnx.load(ONNX_PATH, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    base_dir = os.path.dirname(ONNX_PATH)

    rows = []
    for i, dlc_name in enumerate(sorted(base)):
        if i >= limit:
            break
        onnx_name = (dlc_name[:-len("_permute")]
                     if dlc_name.endswith("_permute") else dlc_name)
        if onnx_name not in inits:
            print(f"  跳过 {dlc_name}（ONNX 里没有 {onnx_name}）")
            continue
        w = read_external(inits[onnx_name], base_dir)

        eb, es = base[dlc_name], sym[dlc_name]
        q = np.clip(np.rint(w / eb["scale"]) - eb["offset"], 0, 255)
        w_asym = eb["scale"] * (q + eb["offset"])
        q = np.clip(np.rint(w / es["scale"]), -128, 127)
        w_sym = es["scale"] * q

        a_all, a_bulk, conc = rel(w, w_asym, "asym")
        s_all, s_bulk, _ = rel(w, w_sym, "sym")
        rows.append((dlc_name, w.size, conc, a_all, s_all, a_bulk, s_bulk,
                     eb["scale"], es["scale"]))
        print(f"  {dlc_name[:34]:34} n={w.size:>9} "
              f"asym {a_bulk*100:6.3f}%  sym {s_bulk*100:6.3f}%  "
              f"(主体口径; scale {eb['scale']:.6f} -> {es['scale']:.6f})")

    print("\n" + "=" * 78)
    n = len(rows)
    ca = np.mean([r[2] for r in rows])
    print(f"张量数 {n}；权重的能量集中度（前1%占 ||w||²）均值 {ca*100:.2f}%"
          f"  ⇒ {'必须看主体口径' if ca > 0.5 else '两口径可互相印证'}")
    for tag, ia, isym in (("全量口径", 3, 4), ("主体口径", 5, 6)):
        va = np.array([r[ia] for r in rows]) * 100
        vs = np.array([r[isym] for r in rows]) * 100
        print(f"{tag}: 非对称 中位 {np.median(va):.4f}% 最大 {va.max():.4f}%  |  "
              f"对称 中位 {np.median(vs):.4f}% 最大 {vs.max():.4f}%  |  "
              f"对称/非对称 中位比 {np.median(vs/va):.4f}")
    sc = np.array([r[8] / r[7] for r in rows])
    print(f"scale 变粗倍数：中位 {np.median(sc):.4f}  最大 {sc.max():.4f}  最小 {sc.min():.4f}")


if __name__ == "__main__":
    main()
