"""EXP_PLAN_LAYER_LOCALIZE 的比对：沿图逐层看 HTP vs CPU 参考的相对误差，
找【第一个】误差跳变的算子。

判据 1（事前定稿）：找第一个满足「自身输出相对误差 >= 5%，而其所有输入误差 < 2%」的算子 X。
判据 3（自检）：逐层误差序列应单调不减或接近单调；大幅回落 => 张量配错，本次无效。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_opid as M

CPU_DIR = r"D:\ZImage_Work\p0_experiments\layer_probe\cpu\out\Result_0"
HTP_DIR = r"D:\ZImage_Work\p0_experiments\layer_probe\htp"
DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"

# 与 layer_probe_cpu.py 的 PROBES 保持一致（含图内 Id，用于排序）
PROBES = [
    ("mul_304", 0), ("stack_28", 50), ("add_153", 103), ("view_125", 153),
    ("linear_115_fc", 203), ("linear_118_fc", 256), ("mul_394", 306),
    ("node_linear_129_pre_reshape", 359), ("mul_424", 409), ("mul_436", 459),
    ("select_160_pre_reshape", 562), ("mul_484", 624), ("unified", 625),
]


def stats(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    na = np.linalg.norm(a)
    if na == 0:
        return float("nan"), float("nan"), float("nan")
    rel = 100.0 * np.linalg.norm(b - a) / na
    k = float(b @ a / (a @ a))                  # 增益斜率
    resid = 100.0 * np.linalg.norm(b - k * a) / na   # 去掉增益后的残差
    return rel, k, resid


def main():
    ops = M.parse(DUMP)
    # 张量名 -> 产出它的算子
    producer = {}
    for o in ops.values():
        for t in o["out"]:
            producer[t["name"]] = o

    rows = []
    print(f"{'Id':>5}  {'张量':<28}{'算子类型':<18}{'相对误差%':>11}{'增益k':>9}{'去增益残差%':>13}")
    print("=" * 92)
    for name, oid in PROBES:
        pc = os.path.join(CPU_DIR, f"{name}.raw")
        ph = os.path.join(HTP_DIR, f"{name}.raw")
        if not (os.path.exists(pc) and os.path.exists(ph)):
            print(f"{oid:>5}  {name:<28}{'(缺文件)':<18}"
                  f"{'cpu' if not os.path.exists(pc) else ''}"
                  f"{' htp' if not os.path.exists(ph) else ''}")
            continue
        a = np.fromfile(pc, np.float32)
        b = np.fromfile(ph, np.float32)
        if a.size != b.size:
            print(f"{oid:>5}  {name:<28}元素数不一致 {a.size} vs {b.size} —— 判据3失败")
            continue
        rel, k, resid = stats(a, b)
        ty = producer.get(name, {}).get("type", "?")
        rows.append((oid, name, ty, rel, k, resid))
        print(f"{oid:>5}  {name:<28}{ty:<18}{rel:>11.4f}{k:>9.4f}{resid:>13.4f}")

    if not rows:
        print("\n没有可比对的张量。")
        return

    rows.sort(key=lambda r: r[0])
    print()
    print("=" * 92)
    print("判据 3（自检）：逐层误差是否单调不减")
    print("=" * 92)
    bad = [(rows[i - 1], rows[i]) for i in range(1, len(rows))
           if rows[i][3] < rows[i - 1][3] * 0.5]
    if bad:
        print("  [FAIL] 出现大幅回落，疑似张量配错：")
        for p, c in bad:
            print(f"    Id {p[0]}({p[1]}) {p[3]:.3f}%  ->  Id {c[0]}({c[1]}) {c[3]:.3f}%")
        print("  按判据 3，本次比对无效，需先查清张量对应关系。")
    else:
        print("  [OK] 未出现大幅回落")

    print()
    print("=" * 92)
    print("判据 1（定位）：第一个误差跳变点")
    print("=" * 92)
    jump = None
    for i, r in enumerate(rows):
        prev = rows[i - 1][3] if i > 0 else 0.0
        if r[3] >= 5.0 and prev < 2.0:
            jump = (rows[i - 1] if i > 0 else None, r)
            break
    if jump:
        p, c = jump
        print(f"  找到跳变：")
        if p:
            print(f"    前一个探针 Id {p[0]:<5} {p[1]:<28} 误差 {p[3]:.4f}%")
        print(f"    跳变探针   Id {c[0]:<5} {c[1]:<28} 误差 {c[3]:.4f}%  "
              f"(增益 k={c[4]:.4f}, 去增益残差 {c[5]:.4f}%)")
        lo = p[0] if p else 0
        print(f"  => 引入点位于算子 Id 区间 ({lo}, {c[0]}]，共 {c[0]-lo} 个算子。")
        print(f"     下一步：在该区间内继续二分取探针。")
        print()
        print("  判据 2（形态）：", end="")
        if abs(1 - c[4]) > 0.05:
            print(f"增益型（k={c[4]:.4f}，偏离 1 达 {abs(1-c[4])*100:.1f}%）")
        else:
            print(f"噪声型（k={c[4]:.4f} 接近 1，误差主要在残差 {c[5]:.4f}%）")
    else:
        first = rows[0]
        print(f"  未找到符合判据的跳变点。")
        print(f"  最早的探针 Id {first[0]} ({first[1]}) 误差已是 {first[3]:.4f}%")
        if first[3] >= 5.0:
            print("  => 偏差在第一个探针处就已存在，需在 Id 0 之前/之内再细分。")
        else:
            print("  => 误差沿图【均匀累积】，不存在单一引入点。")
            print("     按判据 1 第三分支：这是全局性的实现差异，")
            print("     此时才可以回到『改全局配置』这条路，且结论须写明无单一引入点。")


if __name__ == "__main__":
    main()
