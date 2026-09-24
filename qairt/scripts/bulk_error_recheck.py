"""重算误差：相对 L2 被离群值主导，对"量程撑大数千倍"的张量完全失真。

发现（2026-08-15）：`mul_481` 量程被撑大 43322 倍，相对 L2 只有 5.09%，
而它的输入 `silu_19` 是 75.24% —— 误差"变小"是假象：
||a|| 被极少数巨大元素主导，HTP 把那几个算对就显得误差小，哪怕主体全错。

本脚本对每个张量同时给出：
  全量相对 L2（旧口径）
  主体相对 L2（只算 |a| <= p99 的元素，剔除离群主导）
  逐元素相对误差的中位数（完全不受量级分布影响）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_opid as M

DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"
PAIRS = [
    (r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu\out\Result_0",
     r"D:\ZImage_Work\p0_experiments\layer_probe2\htp"),
    (r"D:\ZImage_Work\p0_experiments\layer_probe\cpu\out\Result_0",
     r"D:\ZImage_Work\p0_experiments\layer_probe\htp"),
]


def load(name):
    for c, h in PAIRS:
        pc, ph = os.path.join(c, f"{name}.raw"), os.path.join(h, f"{name}.raw")
        if os.path.exists(pc) and os.path.exists(ph):
            a = np.fromfile(pc, np.float32).astype(np.float64)
            b = np.fromfile(ph, np.float32).astype(np.float64)
            if a.size == b.size:
                return a, b
    return None, None


def main():
    ops = M.parse(DUMP)
    prod = {}
    for o in ops.values():
        for t in o["out"]:
            prod[t["name"]] = o

    rows = []
    for name, op in prod.items():
        a, b = load(name)
        if a is None:
            continue
        full = 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)
        p99 = np.percentile(np.abs(a), 99)
        m = np.abs(a) <= p99
        bulk = 100.0 * np.linalg.norm((b - a)[m]) / np.linalg.norm(a[m])
        # 逐元素相对误差中位数（只在幅值不太小的元素上算，避免除零放大）
        sig = np.abs(a) > 0.1 * a.std()
        med = 100.0 * np.median(np.abs(b - a)[sig] / np.abs(a)[sig]) if sig.sum() else float("nan")
        rows.append((op["id"], op["type"], name, full, bulk, med))

    rows.sort(key=lambda r: r[0])
    print(f"{'Id':>5}  {'算子':<18}{'张量':<32}{'全量L2%':>10}{'主体L2%':>10}{'逐元素中位%':>12}{'倍差':>8}")
    print("=" * 100)
    for r in rows:
        ratio = r[4] / r[3] if r[3] > 1e-9 else float("nan")
        flag = "  <<< 全量口径严重低估" if ratio > 3 else ""
        print(f"{r[0]:>5}  {r[1]:<18}{r[2]:<32}{r[3]:>10.2f}{r[4]:>10.2f}{r[5]:>12.2f}{ratio:>8.1f}{flag}")

    print()
    print("=" * 100)
    print("结论检查")
    print("=" * 100)
    bad = [r for r in rows if r[3] > 1e-9 and r[4] / r[3] > 3]
    print(f"  全量口径被离群值掩盖（主体误差 > 3 倍全量误差）的张量：{len(bad)}/{len(rows)}")
    for r in bad:
        print(f"    Id={r[0]:<5} {r[1]:<18}{r[2]:<32} 全量 {r[3]:.2f}% -> 主体 {r[4]:.2f}%")


if __name__ == "__main__":
    main()
