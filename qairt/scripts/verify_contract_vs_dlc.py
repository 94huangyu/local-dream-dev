"""
实验 B 的核心检查：C++ 侧自己做量化/反量化，scale/offset 全部取自
final_qnn_contract.json。若其中任何一个与 DLC 内部真实 encoding 不符，
数据就会被系统性错误缩放 —— 这正是"语义丢失但颜色/构图尚存"的典型表现。

本脚本把契约 JSON 里每个张量的 scale/offset，与 snpe-dlc-info 转储里
DLC 声明的 encoding 逐一比对。

用法: python verify_contract_vs_dlc.py <contract.json>
"""
import json
import re
import sys

DUMP_DIR = r"D:\ZImage_Work\p0_experiments"

# dlc-info 里的形式：
#   latents encoding : bitwidth 16, min -5.473064899445, max 4.722713947296,
#                      scale 0.000155577611, offset -35179.000000000000
ENC = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*) encoding : bitwidth (\d+), min ([-0-9.e+]+), "
    r"max ([-0-9.e+]+), scale ([-0-9.e+]+), offset ([-0-9.e+]+)")


def dlc_encodings(part):
    """从转储里取每个张量的 encoding（同名多次出现时取第一次）"""
    path = f"{DUMP_DIR}\\{part}_dlcinfo.txt"
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    out = {}
    for name, bw, lo, hi, sc, off in ENC.findall(txt):
        out.setdefault(name, {"bitwidth": int(bw), "scale": float(sc), "offset": float(off)})
    return out


def close(a, b, rtol=1e-4):
    if a is None or b is None:
        return a is None and b is None
    if b == 0:
        return abs(a) < 1e-12
    return abs(a - b) / abs(b) < rtol


contract_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\ZImage_Work\phone_final_qnn_contract.json"
raw = open(contract_path, "rb").read()
enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
contract = json.loads(raw.decode(enc))

print(f"契约文件: {contract_path}  (编码 {enc})")
print("=" * 104)

total = mismatch = missing = skipped = 0
problems = []

for model in contract["models"]:
    part = model["internal_graph_name"]
    decl = dlc_encodings(part)
    if decl is None:
        print(f"[跳过] {part}: 没有 dlc-info 转储")
        continue
    print(f"\n--- {part} ---")
    for kind in ("inputs", "outputs"):
        for t in model.get(kind, []):
            name = t["name"]
            q = t.get("quantization") or {}
            c_scale, c_off = q.get("scale"), q.get("offset")
            if c_scale is None:          # Bool_8 之类没有 scale
                skipped += 1
                continue
            total += 1
            d = decl.get(name)
            if d is None:
                missing += 1
                problems.append((part, kind, name, "DLC 转储里找不到该张量的 encoding", None, None))
                continue
            ok_s = close(c_scale, d["scale"])
            ok_o = close(float(c_off), d["offset"], rtol=1e-9) or float(c_off) == d["offset"]
            if not (ok_s and ok_o):
                mismatch += 1
                problems.append((part, kind, name,
                                 "scale/offset 不符",
                                 f"契约 scale={c_scale!r} offset={c_off!r}",
                                 f"DLC  scale={d['scale']!r} offset={d['offset']!r}"))
                print(f"  [不符] {kind[:-1]:<6} {name:<20} "
                      f"契约({c_scale:.12g}, {c_off}) vs DLC({d['scale']:.12g}, {d['offset']:.0f})")
    print(f"  已比对该图的量化张量")

print()
print("=" * 104)
print(f"总计比对 {total} 个量化张量；不符 {mismatch}；DLC 里找不到 {missing}；"
      f"无 scale 跳过 {skipped}")
print("=" * 104)
if problems:
    print("\n问题清单：")
    for p in problems:
        print(f"  {p[0]} / {p[1]} / {p[2]}: {p[3]}")
        if p[4]:
            print(f"      {p[4]}")
            print(f"      {p[5]}")
else:
    print("\n>>> 全部一致：契约里的 scale/offset 与 DLC 声明相符。")
    print(">>> 这【排除】了'C++ 用错量化参数'这一条，但【不排除】：")
    print("    - C++ 的 quantize/dequantize 舍入规则与转换器不同")
    print("    - 张量的内存布局/维度顺序不同")
    print("    - 图与图之间 requantize 的实现有误")
