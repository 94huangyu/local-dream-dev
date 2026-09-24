"""
交叉验证：把 add_138 标成 float 之后，是否让额外的算子也退化成了浮点？
（如果 fallback 列表完全相同 => 这个改动只影响 add_138 自己的存储，没有连带效应）
"""
import re

def get_fallback_ops(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Following OPs fallback to float:\s*(.*?)\.\s*\n", txt, re.S)
    if not m:
        return None
    return set(x.strip() for x in m.group(1).replace("\n", " ").split(",") if x.strip())

p04 = get_fallback_ops(r"D:/ZImage_Work/p0_experiments/p0_4_full061_unmodified.log")
p05 = get_fallback_ops(r"D:/ZImage_Work/p0_experiments/p0_5_add138_floatfallback.log")

print("p0_4 (原样回灌，未改 add_138)      fallback 算子数:", len(p04) if p04 else "解析失败")
print("p0_5 (add_138 标成 float)          fallback 算子数:", len(p05) if p05 else "解析失败")
print()
if p04 is not None and p05 is not None:
    only05 = p05 - p04
    only04 = p04 - p05
    print("只在 p0_5 出现的算子 (= add_138 改动带来的额外 float 化):", sorted(only05) if only05 else "无")
    print("只在 p0_4 出现的算子:", sorted(only04) if only04 else "无")
    print()
    if not only05 and not only04:
        print(">>> 两份完全相同：把 add_138 标成 float 没有让任何额外算子转入浮点计算，")
        print(">>> 它只改变了 add_138 这一个张量自身的存储格式。")
