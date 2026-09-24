"""EXP_PLAN_FP16_SURGICAL：生成 part2a 手术式 FP16 的 overrides。

🔴 指南 §28.9 规则 B：**要把某张量转浮点，只需让其生产者算子跑浮点** ——
   即**不给白名单张量写 encoding**，其余全部写。转换器会在下游定点算子边界自动插 Convert。

白名单 = 「量化级/元素 < 10」且「实测 |a|max × 2.0 <= 65504」（方案 §三，事前锁定）。
"""
import re, io, os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
P = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
CSV = os.path.join(P, "part2a_full_enc.csv")
OUT = os.path.join(P, "part2a_fp16_overrides.json")
TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: (\[[^\]]*\]); "
                r"tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def main():
    stats = json.load(open(os.path.join(P, "risky_stats.json"), encoding="utf-8"))
    # 🔴 第 4 条（2026-08-23 追加）：输入锥必须有界 => 残差流张量不得入选。
    #    实测：含 15 个残差 Add 时 Float_16 达 183 个（铺满全图）并导致 MatMul 校验失败。
    WHITE = sorted(k for k, v in stats.items()
                   if v["levels"] < 10 and v["fp16ok"] and not k.startswith("add_"))
    print("白名单（转 FP16）%d 个：%s" % (len(WHITE), WHITE))

    kind, ch, pt = {}, {}, {}
    with io.open(CSV, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), (m.group(2), m.group(4)))
            for m in CH.finditer(line):
                ch.setdefault(m.group(1), {})[int(m.group(2))] = dict(
                    bitwidth=int(m.group(3)), min=float(m.group(4)), max=float(m.group(5)),
                    scale=float(m.group(6)), offset=float(m.group(7)))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), dict(bitwidth=int(m.group(2)), min=float(m.group(3)),
                                               max=float(m.group(4)), scale=float(m.group(5)),
                                               offset=float(m.group(6))))
    static = {k for k, v in kind.items() if v[1] == "STATIC"}

    def enc1(e, sym):
        return dict(bitwidth=e["bitwidth"], min=e["min"], max=e["max"],
                    scale=e["scale"], offset=e["offset"], is_symmetric=str(sym))

    params, nch = {}, 0
    for w in sorted(static):
        if w.endswith("_bias") or kind[w][0] not in ("sFxp_8", "uFxp_8", "uFxp_16"):
            continue
        if w in ch:
            c = ch[w]
            # 约束 8：per-axis 必须对称，且「对称」= offset==0（不是 -128）
            assert all(c[i]["offset"] == 0 for i in range(len(c))), "%s per-axis offset 非 0" % w
            params[w] = [enc1(c[i], True) for i in range(len(c))]
            nch += len(c)
        elif w in pt:
            params[w] = [enc1(pt[w], False)]

    acts, skipped = {}, []
    for a, e in pt.items():
        if a in static or e["bitwidth"] != 16:
            continue
        if a in WHITE:
            skipped.append(a); continue          # 🔴 故意不写 => 生产者跑浮点 => 落 FP16
        acts[a] = [enc1(e, False)]


    # 🔴🔴 2026-08-23 实测修正：量化器在**校准路径**上会自动为双激活 MatMul 的输入
    #   插一个「转无符号对称」的 Convert（部署 DLC 里可见 31 个 `*_converted_unsigned_symmetric`，
    #   offset 全为 -32768）。用 --enable_float_fallback（无校准）注入 encoding 时**这一步被跳过**，
    #   导致 `QnnBackend_validateOpConfig failed` / `expected equal to -32768`。
    #   ⇒ 把部署 DLC 里那份对称 encoding **直接注入到基础张量名上**，复现量化器本该做的事。
    SYMSUF = "_converted_unsigned_symmetric"
    nsym = 0
    for k, e in list(pt.items()):
        if not k.endswith(SYMSUF):
            continue
        base = k[:-len(SYMSUF)]
        if base in acts:
            acts[base] = [enc1(e, True)]
            nsym += 1
    print("[对称修正] 为 %d 个双激活 MatMul 输入注入对称 encoding（offset -32768）" % nsym)

    doc = {"version": "0.6.1", "activation_encodings": acts, "param_encodings": params}
    json.dump(doc, open(OUT, "w", encoding="utf-8"))
    print("\n权重张量 %d 个（per-channel %d 个 / scale 合计 %d）" %
          (len(params), sum(1 for v in params.values() if len(v) > 1), nch))
    print("激活张量：写 encoding %d 个 ；**故意留空（将落 FP16）** %d 个" % (len(acts), len(skipped)))
    print("留空的：", sorted(skipped))
    miss = [w for w in WHITE if w not in skipped]
    if miss:
        print("⚠️ 白名单里这些在 DLC 的 16-bit 激活表里没找到，未生效：", miss)
    print("\n写出 %s (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
