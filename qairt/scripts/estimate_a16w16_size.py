"""评估 A16W16（权重 8bit -> 16bit）对模型体积与设备内存的影响。

必须先算清楚再决定要不要做——上一个 agent 的判断是"体积过大手机跑不了"，
本脚本用 DLC 里的实际张量规模核算，而不是估。

关键约束（来自 C++ 源码注释 PipelineZImage.hpp:128-137，属①实测记录）：
  - 三个 transformer context 一次全载：实测常驻 ~7.3 GB，**被低内存杀手干掉**
  - part1a+part1b 一组（~3.84 GB）：实测安全
  - part2 单独（~2.93 GB）：实测安全
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map_opid as M

DUMPS = {
    "transformer_part1a": r"D:\ZImage_Work\p0_experiments\transformer_part1a_dlcinfo.txt",
    "transformer_part1b": r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt",
    "transformer_part2":  r"D:\ZImage_Work\p0_experiments\transformer_part2_dlcinfo.txt",
}
BIN_MB = {"transformer_part1a": 2367.3, "transformer_part1b": 1471.5,
          "transformer_part2": 2928.7}
# A16W16 官方列出的适用算子（htp_backend.html "List of Convolution type of
# operations supported for A16W16"）
A16W16_OPS = {"Conv2d", "DepthConv2d", "TransposeConv2D",
              "FullyConnected", "MatMul", "Batchnorm", "LayerNorm"}


def dims_of(op, name):
    for t in op["in"] + op["out"]:
        if t["name"] == name and t["dims"]:
            n = 1
            for d in t["dims"].split(","):
                if d:
                    n *= int(d)
            return n
    return 0


def main():
    print(f"{'图':<20}{'ctx .bin':>11}{'8bit权重':>11}{'其中可A16W16':>14}"
          f"{'全量16bit增':>13}{'仅适用算子增':>14}")
    print("=" * 84)
    tot = {"bin": 0.0, "all": 0.0, "sel": 0.0}
    for part, path in DUMPS.items():
        ops = M.parse(path)
        w_all = w_sel = 0
        for o in ops.values():
            for t in o["in"]:
                if t["ttype"] != "STATIC":
                    continue
                e = o["enc"].get(t["name"])
                if not e or e["bw"] != 8:
                    continue          # 只看 8-bit 权重（bias 是 32bit，不动）
                n = dims_of(o, t["name"])
                w_all += n
                if o["type"] in A16W16_OPS:
                    w_sel += n
        mb8_all, mb8_sel = w_all / 1e6, w_sel / 1e6      # 8bit: 1 字节/元素
        b = BIN_MB[part]
        tot["bin"] += b
        tot["all"] += mb8_all
        tot["sel"] += mb8_sel
        print(f"{part:<20}{b:>10.0f}M{mb8_all:>10.0f}M{mb8_sel:>13.0f}M"
              f"{mb8_all:>12.0f}M{mb8_sel:>13.0f}M")
    print("-" * 84)
    print(f"{'合计':<20}{tot['bin']:>10.0f}M{tot['all']:>10.0f}M{tot['sel']:>13.0f}M"
          f"{tot['all']:>12.0f}M{tot['sel']:>13.0f}M")

    print()
    print("=" * 84)
    print("对设备内存的影响（8bit->16bit，权重字节数翻倍，增量 = 原 8bit 权重字节数）")
    print("=" * 84)
    print(f"{'配置':<26}{'part1a+1b':>14}{'part2':>12}{'三段合计':>12}")
    print("-" * 66)
    a, b_, c = BIN_MB["transformer_part1a"], BIN_MB["transformer_part1b"], BIN_MB["transformer_part2"]
    print(f"{'现状 (A16W8)':<26}{a+b_:>13.0f}M{c:>11.0f}M{a+b_+c:>11.0f}M")

    # 逐图算增量
    inc_all, inc_sel = {}, {}
    for part, path in DUMPS.items():
        ops = M.parse(path)
        wa = ws = 0
        for o in ops.values():
            for t in o["in"]:
                if t["ttype"] != "STATIC":
                    continue
                e = o["enc"].get(t["name"])
                if not e or e["bw"] != 8:
                    continue
                n = dims_of(o, t["name"])
                wa += n
                if o["type"] in A16W16_OPS:
                    ws += n
        inc_all[part] = wa / 1e6
        inc_sel[part] = ws / 1e6

    for tag, inc in [("全部权重 16bit", inc_all), ("仅 A16W16 适用算子", inc_sel)]:
        na = a + inc["transformer_part1a"]
        nb = b_ + inc["transformer_part1b"]
        nc = c + inc["transformer_part2"]
        print(f"{tag:<26}{na+nb:>13.0f}M{nc:>11.0f}M{na+nb+nc:>11.0f}M")

    print()
    print("对照【实测】的设备内存红线（C++ 源码注释记录）：")
    print("  三段全载 ~7300M -> 被低内存杀手干掉")
    print("  part1a+1b ~3840M -> 安全")
    print("  part2 ~2930M     -> 安全")


if __name__ == "__main__":
    main()
