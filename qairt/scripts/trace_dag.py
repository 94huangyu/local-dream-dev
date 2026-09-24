"""沿【真实数据依赖图】回溯，而不是沿 Id 顺序。

教训（2026-08-15）：EXP_PLAN_LAYER_LOCALIZE 的判据 1/3 都默认"Id 顺序 = 依赖顺序"，
这是错的——图上有并行分支、残差连接、归一化算子，相对误差沿 Id 既不单调也不可比。
两条判据因此一并作废，改用本脚本沿 DAG 回溯。

用法: python trace_dag.py <张量名> [回溯深度]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_opid as M

DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"


def build(dump=DUMP):
    ops = M.parse(dump)
    producer = {}          # 张量名 -> 产出它的算子
    for o in ops.values():
        for t in o["out"]:
            producer[t["name"]] = o
    return ops, producer


def trace(name, producer, depth=6, seen=None, indent=0):
    """从某张量向上游回溯，打印依赖链。"""
    seen = seen if seen is not None else set()
    pad = "  " * indent
    op = producer.get(name)
    if op is None:
        print(f"{pad}{name}  <- 图输入/常量")
        return
    enc = op["enc"].get(name)
    e = f" scale={enc['scale']:.3e}" if enc else ""
    print(f"{pad}{name}  <= Id={op['id']} {op['type']}({op['name']}){e}")
    if depth <= 0:
        print(f"{pad}  ... （深度截止）")
        return
    for i in op["in"]:
        tag = "STATIC" if i["ttype"] == "STATIC" else ""
        if i["name"] in seen:
            print(f"{'  '*(indent+1)}{i['name']}  （已展开）")
            continue
        seen.add(i["name"])
        if tag:
            ie = op["enc"].get(i["name"])
            s = (f" bw={ie['bw']} scale={ie['scale']:.3e} "
                 f"range=[{ie['min']:.4g},{ie['max']:.4g}]" if ie else "")
            print(f"{'  '*(indent+1)}{i['name']}  [STATIC]{s}")
        else:
            trace(i["name"], producer, depth - 1, seen, indent + 1)


def main():
    ops, producer = build()
    names = sys.argv[1:2] or ["mul_484"]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    for n in names:
        print("=" * 78)
        print(f"回溯 {n}（深度 {depth}）")
        print("=" * 78)
        trace(n, producer, depth)
        print()
        # 该张量的消费者
        cons = [o for o in ops.values() if any(i["name"] == n for i in o["in"])]
        print(f"{n} 的消费者：")
        for o in cons:
            print(f"  Id={o['id']:<5} {o['type']:<18} {o['name']} -> "
                  f"{', '.join(t['name'] for t in o['out'])}")


if __name__ == "__main__":
    main()
