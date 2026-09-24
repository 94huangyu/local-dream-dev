"""第二轮定位比对：沿【真实依赖链】看误差在哪个算子产生。

与第一轮的关键区别：第一轮按 Id 顺序排列并要求误差单调（判据 3），
那个前提是错的——Id 顺序 ≠ 依赖顺序，且相对误差按各张量自身范数归一，
并行分支/残差/归一化都会让它非单调。第一轮判据 1/3 已作废。

本轮判据（重新定稿，见 EXP_PLAN_LAYER_LOCALIZE 修订）：
  对每个算子 X，比较【X 的输出误差】与【X 的各输入误差】：
  若 out_err 显著大于 max(in_err)（≥2 倍且绝对增量 ≥5 个百分点），
  则 X 是一个【误差放大点】。列出全部放大点并按增量排序。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_opid as M
from probes2_lastblock import PROBES2

CPU = r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu\out\Result_0"
HTP = r"D:\ZImage_Work\p0_experiments\layer_probe2\htp"
# 第一轮已有的两个，可一并纳入
CPU1 = r"D:\ZImage_Work\p0_experiments\layer_probe\cpu\out\Result_0"
HTP1 = r"D:\ZImage_Work\p0_experiments\layer_probe\htp"
DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"


def load_pair(name):
    for c, h in ((CPU, HTP), (CPU1, HTP1)):
        pc, ph = os.path.join(c, f"{name}.raw"), os.path.join(h, f"{name}.raw")
        if os.path.exists(pc) and os.path.exists(ph):
            return (np.fromfile(pc, np.float32).astype(np.float64),
                    np.fromfile(ph, np.float32).astype(np.float64))
    return None, None


def stats(a, b):
    na = np.linalg.norm(a)
    rel = 100.0 * np.linalg.norm(b - a) / na
    k = float(b @ a / (a @ a))
    resid = 100.0 * np.linalg.norm(b - k * a) / na
    return rel, k, resid


def main():
    ops = M.parse(DUMP)
    producer = {}
    for o in ops.values():
        for t in o["out"]:
            producer[t["name"]] = o

    names = [p[0] for p in PROBES2] + ["unified"]
    err = {}
    print(f"{'Id':>5}  {'张量':<34}{'算子':<18}{'误差%':>9}{'增益k':>9}{'残差%':>9}")
    print("=" * 88)
    for n in names:
        a, b = load_pair(n)
        if a is None:
            print(f"{'':>5}  {n:<34}(缺文件)")
            continue
        if a.size != b.size:
            print(f"{'':>5}  {n:<34}元素数不一致 {a.size} vs {b.size}")
            continue
        rel, k, resid = stats(a, b)
        op = producer.get(n)
        err[n] = rel
        print(f"{op['id'] if op else -1:>5}  {n:<34}"
              f"{(op['type'] if op else '?'):<18}{rel:>9.3f}{k:>9.4f}{resid:>9.3f}")

    print()
    print("=" * 88)
    print("误差放大点（out_err vs 该算子各输入的 err）")
    print("=" * 88)
    amps = []
    for n, e_out in err.items():
        op = producer.get(n)
        if not op:
            continue
        ins = [(i["name"], err.get(i["name"])) for i in op["in"]
               if i["ttype"] != "STATIC"]
        known = [(nm, v) for nm, v in ins if v is not None]
        if not known:
            continue
        mx_name, mx = max(known, key=lambda x: x[1])
        gain = e_out / mx if mx > 1e-9 else float("inf")
        delta = e_out - mx
        amps.append((delta, gain, n, op, mx_name, mx, e_out, known))

    amps.sort(reverse=True)
    for delta, gain, n, op, mx_name, mx, e_out, known in amps:
        flag = "  <<< 放大点" if (gain >= 2.0 and delta >= 5.0) else ""
        print(f"  Id={op['id']:<5} {op['type']:<18} {n:<30}{flag}")
        print(f"      输入误差: " + ", ".join(f"{nm}={v:.2f}%" for nm, v in known))
        print(f"      输出误差: {e_out:.2f}%   增量 {delta:+.2f}pp   倍数 {gain:.2f}x")

    print()
    strong = [a for a in amps if a[1] >= 2.0 and a[0] >= 5.0]
    if strong:
        print(f"=> 找到 {len(strong)} 个误差放大点，按增量排序：")
        for delta, gain, n, op, *_ in strong:
            print(f"   {op['type']:<18} {n:<30} 增量 {delta:+.2f}pp ({gain:.2f}x)")
    else:
        print("=> 未找到显著放大点：误差在链路上均匀累积，无单一引入算子。")


if __name__ == "__main__":
    main()
