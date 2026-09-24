"""EXP_PLAN_FC_VS_REST 判据 1：按算子类型归因主体余弦的下降量。

判据（事前定稿）：
  FullyConnected 的 Δcos 总和 >= 2 x 其余合计  -> H-fc 成立，选择性 FP16 不可行
  其余合计 >= 2 x FullyConnected                -> H-rest 成立，选择性 FP16 可行且近乎零成本
  介于两者之间                                   -> 两类都有实质贡献，报比例，不宣称可行

判据 3（防误判）：所有输入都不在探针集内的算子，不参与归因统计。
度量按约束 7：主体（|a| <= p99）余弦；Reshape 视为透明。
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_opid as M
from fc_vs_rest_cpu import PROBES

CPU = r"D:\ZImage_Work\p0_experiments\fc_probe\cpu\out\Result_0"
HTP = r"D:\ZImage_Work\p0_experiments\fc_probe\htp"
DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"


def load(n):
    pa, pb = os.path.join(CPU, n + ".raw"), os.path.join(HTP, n + ".raw")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None, None
    a = np.fromfile(pa, np.float32).astype(np.float64)
    b = np.fromfile(pb, np.float32).astype(np.float64)
    return (a, b) if a.size == b.size else (None, None)


def metrics(a, b):
    p99 = np.percentile(np.abs(a), 99)
    m = np.abs(a) <= p99
    ca, cb = a[m], b[m]
    cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))
    bulk = 100.0 * np.linalg.norm((b - a)[m]) / np.linalg.norm(a[m])
    return bulk, cos


def main():
    ops = M.parse(DUMP)
    prod = {}
    for o in ops.values():
        for t in o["out"]:
            prod[t["name"]] = o

    def transparent_src(name, depth=6):
        """Reshape 视为透明：向上找到第一个非 Reshape 的产出张量名。"""
        cur = name
        for _ in range(depth):
            op = prod.get(cur)
            if op is None or op["type"] != "Reshape":
                return cur
            ins = [i["name"] for i in op["in"] if i["ttype"] != "STATIC"]
            if not ins:
                return cur
            cur = ins[0]
        return cur

    data = {}
    print(f"{'Id':>5}  {'算子':<18}{'张量':<34}{'主体L2%':>10}{'主体余弦':>11}{'状态':>8}")
    print("=" * 92)
    for n, sh, ty, oid in PROBES:
        a, b = load(n)
        if a is None:
            print(f"{oid:>5}  {ty:<18}{n:<34}(缺文件/尺寸不符)")
            continue
        bulk, cos = metrics(a, b)
        data[n] = (oid, ty, bulk, cos)
        st = "已毁" if cos < 0.3 else ("劣化" if cos < 0.9 else "健康")
        print(f"{oid:>5}  {ty:<18}{n:<34}{bulk:>10.2f}{cos:>11.4f}{st:>8}")

    if not data:
        raise SystemExit("无数据")

    print()
    print("=" * 92)
    print("判据 1：按算子类型归因主体余弦下降量 Δcos")
    print("=" * 92)
    agg = defaultdict(list)
    rows = []
    for n, (oid, ty, bulk, cos) in data.items():
        op = prod.get(n)
        if not op:
            continue
        ins = []
        for i in op["in"]:
            if i["ttype"] == "STATIC":
                continue
            src = transparent_src(i["name"])
            if src in data:
                ins.append((src, data[src][3]))
        if not ins:          # 判据 3：无已知输入，不参与统计
            continue
        cin = max(c for _, c in ins)
        d = cin - cos
        agg[ty].append(d)
        rows.append((oid, ty, n, cin, cos, d, [x[0] for x in ins]))

    rows.sort()
    print(f"{'Id':>5}  {'算子':<18}{'张量':<30}{'输入余弦':>10}{'输出余弦':>10}{'Δcos':>9}")
    print("-" * 92)
    for oid, ty, n, cin, cos, d, srcs in rows:
        print(f"{oid:>5}  {ty:<18}{n:<30}{cin:>10.4f}{cos:>10.4f}{d:>+9.4f}")

    print()
    print(f"{'算子类型':<20}{'参与统计数':>10}{'Δcos 总和':>12}{'Δcos 中位':>12}")
    print("-" * 56)
    tot = {}
    for ty, ds in sorted(agg.items(), key=lambda x: -sum(x[1])):
        tot[ty] = sum(ds)
        print(f"{ty:<20}{len(ds):>10}{sum(ds):>+12.4f}{np.median(ds):>+12.4f}")

    fc = tot.get("FullyConnected", 0.0)
    rest = sum(v for k, v in tot.items() if k != "FullyConnected")
    print()
    print(f"  FullyConnected 合计 Δcos = {fc:+.4f}")
    print(f"  其余算子       合计 Δcos = {rest:+.4f}")
    print()
    if fc >= 2 * max(rest, 1e-9) and fc > 0:
        print("  => 【H-fc 成立】误差主要由 FullyConnected 产生。")
        print("     FC 占 99.99% 权重，浮点化必撞体积红线 ⇒ **选择性 FP16 不可行**。")
    elif rest >= 2 * max(fc, 1e-9) and rest > 0:
        print("  => 【H-rest 成立】误差主要由 FullyConnected 以外的算子产生。")
        print("     浮点化它们体积仅增 ~1 MB ⇒ **选择性 FP16 可行且近乎零成本**。")
    else:
        r = fc / (fc + rest) if (fc + rest) > 0 else float("nan")
        print(f"  => 两类都有实质贡献：FC 占 {100*r:.1f}%，其余占 {100*(1-r):.1f}%。")
        print("     不得直接宣称可行，需评估折中方案。")


if __name__ == "__main__":
    main()
