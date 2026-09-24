"""
把 HTP 报错的 OpID 映射到具体算子/张量。

背景（实测，见 EXP_PLAN_OPID_MAPPING.md 第 0 节）：
  linearclip.h:248::ERROR:scale too large for requant qu16->qu16: <V>; OpID: 0x6994000000a2
  原始 DLC   V = inf
  maskfix DLC V = 1000000.000000     <- 有限、可离线复算

两条独立的定位路线：
  H1  OpID 低 32 位 == dlc-info 表格里的 Id 序号（0xa2 = 162）
  H2  出问题的边，其 scale_in/scale_out 在原始侧为 inf、在 maskfix 侧约 1e6

解析对象是 `snpe-dlc-info` 转储（契约的唯一合法来源，禁止从 ONNX 推导）。

用法: python map_opid.py
"""
import re
import struct
import sys

DUMPS = {
    "原始":     r"D:\ZImage_Work\p0_experiments\transformer_part2_dlcinfo.txt",
    "maskfix": r"D:\ZImage_Work\p0_experiments\maskfix_part2_dlcinfo.txt",
}

TARGET_ID = 0xA2  # OpID 0x6994000000a2 的低位

# 张量名允许含点（既有 scan_requant_ratios.py 的正则不允许，会静默漏掉所有权重张量）
ENC = re.compile(
    r"([\w.]+) encoding : bitwidth (\d+), min (\S+?), max (\S+?), "
    r"scale (\S+?), offset (\S+)")
# "name (data type: uFxp_16; tensor dimension: [1,4128,3840]; tensor type: NATIVE)"
TENSOR = re.compile(r"([\w.]+) \(data type: (\w+); tensor dimension: \[([\d,]*)\]; "
                    r"tensor type: (\w+)\)")


def f(s):
    """容忍 inf/nan 的 float 解析。"""
    try:
        return float(s.rstrip(",").rstrip(";"))
    except ValueError:
        return float("nan")


def parse(path):
    """解析转储表格，返回 {id: record}。带续行合并。"""
    ops, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) < 8:
                continue
            if cols[0] == "Id":          # 表头
                continue
            if cols[0]:                  # 新算子
                if not cols[0].isdigit():
                    continue
                cur = {"id": int(cols[0]), "name": cols[1], "type": cols[2],
                       "in": [], "out": [], "enc": {}}
                ops[cur["id"]] = cur
            if cur is None:
                continue
            for slot, col in (("in", 3), ("out", 4)):
                for name, dt, dims, tt in TENSOR.findall(cols[col]):
                    cur[slot].append({"name": name, "dtype": dt, "dims": dims, "ttype": tt})
            for name, bw, lo, hi, sc, off in ENC.findall(cols[7]):
                cur["enc"][name] = {"bw": int(bw), "min": f(lo), "max": f(hi),
                                    "scale": f(sc), "offset": f(off)}
    return ops


def f32(x):
    """按 float32 复算——SDK 内部即为 float32，溢出会变 inf。这正是报错里 inf 的来源。"""
    return float(struct.unpack("f", struct.pack("f", min(x, 3.5e38)))[0]) if x < 3.5e38 \
        else float("inf")


def ratios(op):
    """该算子上所有 (张量 A, 张量 B) 的 scale 比值。

    第一版只算了「激活输入 -> 输出」并排除 STATIC，那是执行方法错误：
      1. Eltwise_Binary 的两个**操作数之间**必须先对齐到同一 scale，这是 input<->input
         的比值，第一版根本没算；
      2. 掩码常量 val_104 是 STATIC，被第一版直接排除掉了。
    重量化发生在任意两个需要对齐的张量之间，所以这里对全部张量两两取比值。
    """
    tens = [(t, op["enc"][t["name"]]) for t in op["in"] + op["out"]
            if t["name"] in op["enc"]]
    out = []
    for a in range(len(tens)):
        for b in range(a + 1, len(tens)):
            (ta, ea), (tb, eb) = tens[a], tens[b]
            sa, sb = ea["scale"], eb["scale"]
            if sa <= 0 or sb <= 0:
                if sa <= 0 and sb <= 0:
                    continue
                r = float("inf")
            else:
                r = f32(max(sa / sb, sb / sa))
            out.append((r, ta, ea, tb, eb))
    return out


def show(tag, op):
    print(f"  [{tag}] Id={op['id']}  {op['type']:<18} {op['name']}")
    for i in op["in"]:
        e = op["enc"].get(i["name"])
        s = (f"scale={e['scale']:.6e} range=[{e['min']:.6g}, {e['max']:.6g}] bw={e['bw']}"
             if e else "（转储中无 encoding）")
        print(f"        输入  {i['name']:<34} {i['dtype']:<8} {i['ttype']:<10} {s}")
    for o in op["out"]:
        e = op["enc"].get(o["name"])
        s = (f"scale={e['scale']:.6e} range=[{e['min']:.6g}, {e['max']:.6g}] bw={e['bw']}"
             if e else "（转储中无 encoding）")
        print(f"        输出  {o['name']:<34} {o['dtype']:<8} {o['ttype']:<10} {s}")
    rs = ratios(op)
    if rs:
        print(f"        比值  max = {max(r for r, *_ in rs):.6g}")


def main():
    parsed = {}
    for tag, path in DUMPS.items():
        parsed[tag] = parse(path)
        print(f"{tag:>8}: 解析出 {len(parsed[tag])} 个算子   <- {path}")
    print()

    print("=" * 100)
    print(f"判据 1（H1）：OpID 0x6994000000a2 低位 0xa2 = {TARGET_ID} -> Id={TARGET_ID}")
    print("=" * 100)
    for tag in DUMPS:
        op = parsed[tag].get(TARGET_ID)
        if op is None:
            print(f"  [{tag}] 不存在 Id={TARGET_ID}")
        else:
            show(tag, op)
        print()

    print("=" * 100)
    print("判据 2（H2）：全图 scale 比值扫描，独立于 H1")
    print("=" * 100)
    tops = {}
    for tag in DUMPS:
        rows = []
        for op in parsed[tag].values():
            rs = ratios(op)
            if rs:
                r, i, ei, o, eo = max(rs, key=lambda x: x[0])
                rows.append((r, op, i, ei, o, eo))
        rows.sort(reverse=True, key=lambda x: x[0])
        tops[tag] = rows
        print(f"\n---- {tag}：比值最大的 12 个算子 ----")
        for r, op, i, ei, o, eo in rows[:12]:
            mark = "  <<< Id==162" if op["id"] == TARGET_ID else ""
            print(f"  {r:>12.6g}   Id={op['id']:<5} {op['type']:<18} {op['name']}{mark}")
            print(f"                 {i['name']} (scale={ei['scale']:.6e}) -> "
                  f"{o['name']} (scale={eo['scale']:.6e})")

    print()
    print("=" * 100)
    print("判据 2 具体检查：maskfix 侧比值落在 [1e5, 1e7] 的算子")
    print("=" * 100)
    band = [(r, op) for r, op, *_ in tops["maskfix"] if 1e5 <= r <= 1e7]
    if not band:
        print("  （无）-> 判据 2 不通过，按计划转 fallback（verbose 日志 / 二分裁剪）")
    for r, op in band:
        orig = parsed["原始"].get(op["id"])
        ro = max((x[0] for x in ratios(orig)), default=float("nan")) if orig else float("nan")
        print(f"  maskfix 比值 {r:.6g}   Id={op['id']}  {op['type']}  {op['name']}")
        print(f"     同 Id 在原始侧的比值 = {ro:.6g}   "
              f"（判据 2 要求这里是 inf 或除零）")


if __name__ == "__main__":
    sys.exit(main())
