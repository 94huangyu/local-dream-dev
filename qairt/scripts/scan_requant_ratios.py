"""
检验"val_104 是唯一 HTP 隐患"这个判断的稳健性。

HTP 报的是 `scale too large for requant qu16->qu16: inf`，即某条边两端的
scale 比值溢出。前一次扫描只看了【单个张量的绝对量级】，这次看【图上每条边的比值】——
这才是重量化真正关心的量。

零内存开销：ONNX 只加载图结构（不载外部权重），encoding 从 dlc-info 转储解析。

用法: python scan_requant_ratios.py <part> [阈值]
"""
import re
import sys

import onnx

ONNX_DIR = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
DUMP_DIR = r"D:\ZImage_Work\p0_experiments"

ENC = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*) encoding : bitwidth (\d+), min ([-0-9.e+]+), "
    r"max ([-0-9.e+]+), scale ([-0-9.e+]+), offset ([-0-9.e+]+)")

ONNX_OF = {
    "transformer_part2": "transformer_part2_fixed.onnx",
    "transformer_part1a": "transformer_part1a.onnx",
    "transformer_part1b": "transformer_part1b.onnx",
}


def load_enc(part):
    txt = open(f"{DUMP_DIR}\\{part}_dlcinfo.txt", encoding="utf-8", errors="replace").read()
    out = {}
    for n, bw, lo, hi, sc, off in ENC.findall(txt):
        out.setdefault(n, {"bw": int(bw), "min": float(lo), "max": float(hi),
                           "scale": float(sc), "offset": float(off)})
    return out


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "transformer_part2"
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6

    enc = load_enc(part)
    m = onnx.load(f"{ONNX_DIR}\\{ONNX_OF[part]}", load_external_data=False)
    g = m.graph

    print(f"{part}: 图中 {len(g.node)} 个算子，转储里 {len(enc)} 个张量 encoding")
    print(f"扫描每条边（算子的每个输入 -> 该算子的输出）的 scale 比值，阈值 {thresh:g}")
    print("=" * 104)

    rows = []
    for n in g.node:
        for o in n.output:
            eo = enc.get(o)
            if not eo or eo["scale"] <= 0:
                continue
            for i in n.input:
                ei = enc.get(i)
                if not ei or ei["scale"] <= 0:
                    continue
                r = ei["scale"] / eo["scale"]
                ratio = max(r, 1.0 / r)
                if ratio >= thresh:
                    rows.append((ratio, n.op_type, n.name, i, ei, o, eo))

    rows.sort(reverse=True, key=lambda x: x[0])
    print(f"超过阈值的边：{len(rows)}")
    print()
    for ratio, op, name, i, ei, o, eo in rows[:25]:
        print(f"  比值 {ratio:.3e}   {op:<18} {name}")
        print(f"      输入 {i:<22} scale={ei['scale']:.6e}  range=[{ei['min']:.4g}, {ei['max']:.4g}]")
        print(f"      输出 {o:<22} scale={eo['scale']:.6e}  range=[{eo['min']:.4g}, {eo['max']:.4g}]")
    if not rows:
        print("  （无）")

    # 顺带给出全图 scale 的分布，便于判断"离群"是否只有一个
    scales = sorted(v["scale"] for v in enc.values() if v["scale"] > 0)
    if scales:
        import statistics
        print()
        print("全图非零 scale 分布：")
        print(f"  最小 {scales[0]:.3e}   中位 {statistics.median(scales):.3e}   最大 {scales[-1]:.3e}")
        print(f"  最大/中位 = {scales[-1]/statistics.median(scales):.3e}")


if __name__ == "__main__":
    main()
