"""核查：本项目的权重张量到底是不是对称量化的？

官方依据（QAIRT 2.48.0 SDK 文档 docs/QAIRT-Docs/QNN/general/htp/htp_backend.html）：
  "All weights/filters need to be symmetrically quantized. For Matmul, Input A must be
   asymmetrically quantized, Input B must be symmetrically quantized."
  "It is recommended to always use symmetrical quantization of weights when quantizing
   the model to obtain best accuracy on HTP based targets."

对称量化的判据：offset == -2^(bw-1)（8bit 即 -128，16bit 即 -32768），
即 0.0 恰好落在量化格点上、且区间关于 0 对称。
非对称量化的 offset 是任意值。

用法: python check_weight_symmetry.py
"""
import re
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import map_opid as M

DUMPS = {
    "transformer_part1a": r"D:\ZImage_Work\p0_experiments\transformer_part1a_dlcinfo.txt",
    "transformer_part1b": r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt",
    "transformer_part2": r"D:\ZImage_Work\p0_experiments\transformer_part2_dlcinfo.txt",
}

# 静态权重张量（参与 MatMul/FullyConnected/Conv 的 B 输入）
WEIGHT_RX = re.compile(r"\.weight|_weight|weight_permute")


def main():
    grand = {"sym": 0, "asym": 0}
    for part, path in DUMPS.items():
        ops = M.parse(path)
        sym = asym = 0
        examples = []
        for op in ops.values():
            for t in op["in"]:
                if t["ttype"] != "STATIC":
                    continue
                if not WEIGHT_RX.search(t["name"]):
                    continue
                e = op["enc"].get(t["name"])
                if not e or e["scale"] <= 0:
                    continue
                ideal = -(2 ** (e["bw"] - 1))
                if abs(e["offset"] - ideal) < 0.5:
                    sym += 1
                else:
                    asym += 1
                    if len(examples) < 4:
                        examples.append((t["name"], e))
        grand["sym"] += sym
        grand["asym"] += asym
        tot = sym + asym
        print(f"=== {part} ===")
        print(f"  权重张量 {tot} 个：对称 {sym}，**非对称 {asym}** "
              f"（非对称占 {100.0*asym/max(tot,1):.1f}%）")
        for n, e in examples:
            ideal = -(2 ** (e["bw"] - 1))
            print(f"    {n[:60]:<60} bw={e['bw']} offset={e['offset']:.1f} "
                  f"(对称应为 {ideal})  range=[{e['min']:.4g}, {e['max']:.4g}]")
        print()

    tot = grand["sym"] + grand["asym"]
    print("=" * 78)
    print(f"三段合计：权重张量 {tot} 个 —— 对称 {grand['sym']}，"
          f"**非对称 {grand['asym']}**（{100.0*grand['asym']/max(tot,1):.1f}%）")
    print("=" * 78)
    if grand["asym"] > 0:
        print("=> 违反 HTP 官方要求：'All weights/filters need to be symmetrically quantized.'")
        print("   量化命令里的 param_quantizer_schema=asymmetric 与此一致。")
    else:
        print("=> 权重已是对称量化，该方向不成立。")


if __name__ == "__main__":
    main()
