"""EXP_PLAN_ACTFP16 第 1 步：生成 overrides —— 权重逐字钉死，激活只钉 6 个溢出张量。

单变量纪律：**唯一变量 = 激活数据类型**。所以权重（FC 的 sFxp_8 per-row）与
RmsNorm gamma（uFxp_8）都必须**逐字取自现有 per-row DLC** 并注入，不许重新校准。

bias（sFxp_32）**不注入**：官方 `--keep_weights_quantized` 说明写明
*"Bias will be converted to floating point as per the output of the op"*。

🔴 #45：overrides 必须用 **ONNX 张量名**，且必查 converter 日志的 `Processed N encodings`。
🔴 #46：必须显式 `--float_bitwidth`，否则单变量被破坏。
"""
import re, io, os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
CSV = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b", "part1b_perrow_encodings.csv")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "actfp16")
BF16 = ("--bf16" in sys.argv)   # BF16 模式：范围足够 => 不钉任何激活张量
FP16MAX = 65504.0

TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: (\[[^\]]*\]); "
                r"tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def main():
    os.makedirs(OUT, exist_ok=True)
    kind, ch, pt = {}, {}, {}
    with io.open(CSV, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind[m.group(1)] = (m.group(2), m.group(4))
            for m in CH.finditer(line):
                ch.setdefault(m.group(1), {})[int(m.group(2))] = dict(
                    bitwidth=int(m.group(3)), min=float(m.group(4)), max=float(m.group(5)),
                    scale=float(m.group(6)), offset=float(m.group(7)))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), dict(
                    bitwidth=int(m.group(2)), min=float(m.group(3)), max=float(m.group(4)),
                    scale=float(m.group(5)), offset=float(m.group(6))))

    static = {k for k, v in kind.items() if v[1] == "STATIC"}
    # 权重 = STATIC 且不是 bias 且不是整型常量
    wnames = sorted(k for k in static
                    if not k.endswith("_bias") and kind[k][0] in ("sFxp_8", "uFxp_8", "uFxp_16"))
    # 溢出张量 = 非 STATIC、16-bit、|a|max > FP16 上限
    ov = sorted(k for k, v in pt.items()
                if k not in static and v["bitwidth"] == 16
                and max(abs(v["min"]), abs(v["max"])) > FP16MAX)
    # 🔴 只钉输出不够（2026-08-22 实测 G1 FAIL）：
    #    OpDef 的 FullyConnected 只有两行 —— `FP16 激活+SFxp8 权重` 或 `uFxp_16 激活+SFxp8 权重`，
    #    **没有「FP16 输入 + 定点输出」**。只钉输出时转换器在 FC 后插 Convert，
    #    FC 本身仍是 FP16 ⇒ 内部张量 `linear_NNN_fc` 落成 Float_16 并溢出（max 175619 > 65504）。
    #    ⇒ 必须把这些 FC 的**激活输入**一起钉成定点，整个算子才落到 INT16 行。
    #    输入名取自 ONNX：MatMul(mul_NNN, val_NNNN) -> linear_NNN
    FC_IN = ["mul_331", "mul_381", "mul_406", "mul_431", "mul_456", "mul_481"]
    ov = sorted(set(ov) | {k for k in FC_IN if k in pt and k not in static})
    if BF16:
        ov = []          # BF16 上限 3.4e38，x^2 最大 5.68e10 => 无需钉任何激活

    def enc1(e, sym):
        return dict(bitwidth=e["bitwidth"], min=e["min"], max=e["max"],
                    scale=e["scale"], offset=e["offset"], is_symmetric=str(sym))
    # 🔴 约束 8 的坑，本项目第二次踩：**有符号表示下「对称」是 offset == 0，不是 -128**。
    #    实查 CSV：517120 个 per-channel offset **全部为 0**。
    #    转换器强制 "Axis quantization is required to be symmetric" ⇒ per-axis 必须 True。
    #    per-tensor 保持 asymmetric（与 DLC 记录的 param_quantizer_schema=asymmetric 一致）。

    params, nch = {}, 0
    for w in wnames:
        if w in ch:
            c = ch[w]
            assert all(c[i]["offset"] == 0 for i in range(len(c))), "per-axis offset 非 0，与对称假设不符"
            lst = [enc1(c[i], True) for i in range(len(c))]
            nch += len(lst)
        elif w in pt:
            lst = [enc1(pt[w], False)]
        else:
            continue
        params[w] = lst
    acts = {a: [enc1(pt[a], False)] for a in ov}

    doc = {"version": "0.6.1", "activation_encodings": acts, "param_encodings": params}
    p = os.path.join(OUT, "part1b_actbf16_overrides.json" if BF16 else "part1b_actfp16_overrides.json")
    json.dump(doc, open(p, "w", encoding="utf-8"))
    print("权重张量 %d 个（其中 per-channel %d 个，scale 合计 %d 条）"
          % (len(params), sum(1 for v in params.values() if len(v) > 1), nch))
    print("钉死的激活张量（|a|max > %.0f，保持定点）%d 个:" % (FP16MAX, len(acts)))
    for a in ov:
        print("   %-28s min=%12.2f max=%12.2f" % (a, pt[a]["min"], pt[a]["max"]))
    print("其余 %d 个 16-bit 激活张量将缺 encoding ⇒ 落成 FP16"
          % (sum(1 for k, v in pt.items() if k not in static and v["bitwidth"] == 16) - len(acts)))
    print("\n写出 %s  (%.1f MB)" % (p, os.path.getsize(p) / 1e6))


if __name__ == "__main__":
    main()
