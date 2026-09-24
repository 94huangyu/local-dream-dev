"""离线验证：RmsNorm 输出的【encoding 余量（有效位宽）】与【HTP 输出误差】是否真的相关。

背景（EXP_PLAN_LAYER_LOCALIZE 第二轮）：3 个 RmsNorm 上看到方向一致的关系，
但 3 个样本不足以下结论，且本轮已推翻过两个同样"看起来合理"的假设。

本脚本把两轮探针里【所有】有两侧数据的张量都纳入，做全样本相关性检验，
并按算子类型分组——若关系只在 RmsNorm 上成立、在别的算子上不成立，
那说明它不是普适规律，需要重新解释。

零成本：全部用已拉回的本地数据，不跑设备、不重量化。
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
        e = op["enc"].get(name)
        if not e or e["scale"] <= 0:
            continue
        rel = 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)
        k = float(b @ a / (a @ a))
        half = max(abs(e["min"]), abs(e["max"]))
        p999 = np.percentile(np.abs(a), 99.9)
        stretch = half / max(p999, 1e-12)          # 量程被离群值撑大的倍数
        eff = np.log2(max(a.std() / e["scale"], 1))  # 主体有效位宽
        rows.append((op["id"], op["type"], name, eff, stretch, rel, k))

    rows.sort(key=lambda r: r[3])
    print(f"{'Id':>5}  {'算子':<18}{'张量':<32}{'有效bit':>8}{'量程撑大':>9}{'误差%':>9}{'增益k':>8}")
    print("=" * 96)
    for r in rows:
        print(f"{r[0]:>5}  {r[1]:<18}{r[2]:<32}{r[3]:>8.2f}{r[4]:>9.1f}{r[5]:>9.2f}{r[6]:>8.4f}")

    eff = np.array([r[3] for r in rows])
    err = np.array([r[5] for r in rows])
    n = len(rows)
    print()
    print("=" * 96)
    print(f"全样本相关性（n={n}）")
    print("=" * 96)
    if n >= 4:
        c = np.corrcoef(eff, err)[0, 1]
        # 秩相关（不假设线性）
        re_, rr = eff.argsort().argsort(), err.argsort().argsort()
        cs = np.corrcoef(re_, rr)[0, 1]
        print(f"  有效位宽 vs 误差   Pearson r = {c:+.3f}   Spearman ρ = {cs:+.3f}")
        print(f"  期望：若假设成立，应为【显著负相关】（位宽越低误差越大）")
        verdict = ("支持" if cs <= -0.5 else "不支持" if cs > -0.2 else "弱")
        print(f"  => {verdict}")

    print()
    print("按算子类型分组（看是否只在 RmsNorm 上成立）")
    print("-" * 60)
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[r[1]].append(r)
    for ty, rs in sorted(g.items(), key=lambda x: -len(x[1])):
        e2 = np.array([x[3] for x in rs]); r2 = np.array([x[5] for x in rs])
        s = ""
        if len(rs) >= 3:
            s = f"  Spearman ρ = {np.corrcoef(e2.argsort().argsort(), r2.argsort().argsort())[0,1]:+.3f}"
        print(f"  {ty:<18} n={len(rs):<3} 有效bit {e2.min():.2f}~{e2.max():.2f} "
              f" 误差 {r2.min():.1f}~{r2.max():.1f}%{s}")


if __name__ == "__main__":
    main()
