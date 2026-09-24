"""G3 / G4 门禁核验：selective symmetric FC 的量化 DLC 是否只改了该改的。

方案 `scripts/EXP_PLAN_SYM_FC_OVR.md` 第 3 节。判据事前定稿，本脚本只做核验不做解释。

G3（类型确实变了、且只变了该变的）：
  1. 59 个 FC 权重      : sFxp_8 且 offset == 0
     （约束 8：**有符号表示下对称是 offset==0，不是 -128**；-128 是 AIMET JSON 的写法）
  2. 44 个 RmsNorm gamma: 仍是 uFxp_8 且 offset != 0（未被牵连）
  3. 59 个 FC bias      : 仍是 sFxp_32
G4（上游未被扰动）：6 个图输入的 encoding 必须与基线逐字段相同。
  ⚠️ 原写法是"全部激活 scale 漂移 >1% 的张量数为 0"，用已知样本（symw 全局对称 DLC）
  预跑发现 **42 个激活漂移 >1%、最大 6.98%** —— 那是被测效应本身（权重变→校准值变），
  不是混淆。已在结果产生前按约束 8 修正，激活漂移改为**观测项**。

用法: python check_sym_fc.py <new_dlcinfo.txt> [<base_dlcinfo.txt>]
"""
import sys

import map_opid as M

BASE = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"
INPUTS = ("add_138", "add_131", "tanh_19", "select_45", "select_46", "adaln_input")


def axis_quant_summary(path):
    """扫 per-axis(逐行)量化的转储行。见 EXP_PLAN_PERROW G1。"""
    import collections
    import re
    txt = open(path, encoding="utf-8", errors="replace").read()
    axes = re.findall(r"axis-quant: axis: (\d+), num_elements: (\d+)", txt)
    enc8 = re.findall(r"encoding for channel_0: bitwidth 8,.*?offset ([-0-9.]+)", txt)
    return {
        "n_tensors": len(axes),
        "axes": dict(collections.Counter(a for a, _ in axes)),
        "sizes": dict(collections.Counter(int(n) for _, n in axes)),
        "n_enc8": len(enc8),
        "n_off0": sum(1 for o in enc8 if abs(float(o)) < 1e-9),
    }


def index(ops):
    """返回 (fc_w, fc_bias, rms_gamma, act) 四张 {name: (enc, dtype)} 表"""
    fc_w, fc_b, rms, act = {}, {}, {}, {}
    for op in ops.values():
        for t in op["in"] + op["out"]:
            e = op["enc"].get(t["name"])
            if e is None:
                continue
            rec = (e, t["dtype"])
            if t["ttype"] != "STATIC":
                act[t["name"]] = rec
            elif op["type"] == "FullyConnected":
                (fc_b if e["bw"] == 32 else fc_w)[t["name"]] = rec
            elif op["type"] == "RmsNorm":
                if e["bw"] == 8:
                    rms[t["name"]] = rec
    return fc_w, fc_b, rms, act


def main():
    new_path = sys.argv[1]
    base_path = sys.argv[2] if len(sys.argv) > 2 else BASE
    nw, nb, nr, na = index(M.parse(new_path))
    bw_, bb, br, ba = index(M.parse(base_path))

    ok = True
    print("=" * 78)
    # 逐行(per-axis)量化的 DLC 里，权重不再打印 "name encoding : ..."，
    # 而是 "name encoding for channel_0: ..." + 单独一行 "axis-quant: axis: A, num_elements: N"。
    # map_opid 的 ENC 正则匹配不到它们 ⇒ 下面的 G3-1/G3-3 会显示 0 个，那是**工具口径不适用**，
    # 不是门禁失败。此时适用的是 EXP_PLAN_PERROW 的 G1（见这一段）。
    ax = axis_quant_summary(new_path)
    if ax["n_tensors"]:
        print(f"[检测到 per-axis 量化 DLC] axis-quant 张量 {ax['n_tensors']} 个；"
              f"轴分布 {ax['axes']}；num_elements 分布 {ax['sizes']}")
        print(f"  channel_0 的 8-bit encoding {ax['n_enc8']} 个，"
              f"其中 offset==0 的 {ax['n_off0']} 个")
        print("  ⇒ 下面的 G3-1 / G3-3 计数对本 DLC 不适用（正则口径不同），"
              "请以 EXP_PLAN_PERROW 的 G1 为准")
        print("-" * 78)
    print("G3-1  FullyConnected 权重必须是 sFxp_8 / offset==0")
    bad = [(n, d, e["offset"]) for n, (e, d) in nw.items()
           if not d.startswith("sFxp_8") or abs(e["offset"]) > 1e-9]
    print(f"  FC 权重张量 {len(nw)} 个（基线 {len(bw_)} 个），不合格 {len(bad)} 个")
    for n, d, off in bad[:5]:
        print(f"    ✗ {n:44} dtype={d} offset={off}")
    ok &= (len(nw) == len(bw_) and not bad)

    # ⚠️ 判据操作化订正 #2（2026-08-15 20:20，设备实测【之前】）：
    #    原写法 "uFxp_8 且 offset != 0"。实测**基线自身**就有 14 个 gamma 是 offset==0
    #    （norm_q / norm_k 的值全为正，min=0.0 ⇒ 非对称 min-max 天然给出 offset 0）。
    #    ⇒ 原写法会把"与基线完全一致"误判为不合格。改为直接比对基线（严格更强）。
    print("G3-2  RmsNorm gamma 必须与基线逐字段相同（未被牵连）")
    bad = [(n, br.get(n), nr[n]) for n in nr if br.get(n) != nr[n]]
    print(f"  RmsNorm 8-bit 静态张量 {len(nr)} 个（基线 {len(br)} 个），"
          f"与基线不同的 {len(bad)} 个")
    for n, b_, n_ in bad[:5]:
        print(f"    ✗ {n:44}\n        基线 {b_}\n        新   {n_}")
    ok &= (len(nr) == len(br) and not bad)

    print("G3-3  FullyConnected bias 必须仍是 sFxp_32")
    bad = [(n, d) for n, (e, d) in nb.items() if not d.startswith("sFxp_32")]
    print(f"  FC bias 张量 {len(nb)} 个（基线 {len(bb)} 个），不合格 {len(bad)} 个")
    ok &= (len(nb) == len(bb) and not bad)

    print("=" * 78)
    print("G4  图输入 encoding 必须与基线逐字段相同（判据）")
    bad = []
    for n in sorted(INPUTS):
        ea, eb = na.get(n, (None,))[0], ba.get(n, (None,))[0]
        same = ea == eb
        print(f"  {n:14} {'IDENTICAL' if same else '✗ DIFFER'}")
        if not same:
            bad.append((n, eb, ea))
            print(f"      基线 {eb}\n      新   {ea}")
    ok &= not bad

    print("-" * 78)
    print("观测项（不作判据）：激活 encoding 漂移面")
    common = set(na) & set(ba)
    drift = []
    for n in common:
        s_new, s_base = na[n][0]["scale"], ba[n][0]["scale"]
        if s_base <= 0:
            continue
        r = abs(s_new - s_base) / s_base
        if r > 0.01:
            drift.append((r, n, s_base, s_new))
    drift.sort(reverse=True)
    print(f"  共同激活张量 {len(common)} 个（新 {len(na)} / 基线 {len(ba)}），"
          f"scale 漂移 >1% 的 {len(drift)} 个"
          + (f"，最大 {drift[0][0]*100:.2f}%（{drift[0][1]}）" if drift else ""))
    for r, n, sb, sn in drift[:8]:
        print(f"    · {n:40} {sb:.6e} -> {sn:.6e}  ({r*100:.2f}%)")

    print("=" * 78)
    print("门禁结论：", "✅ 全部通过，可以解读实验结果" if ok else "❌ 未通过，本次无效，不得解读")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
