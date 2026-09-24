"""生成「只对 FullyConnected 权重施加对称量化」的 --quantization_overrides JSON。

方案见 `scripts/EXP_PLAN_SYM_FC_OVR.md`。

对称 encoding **不自己推公式**，直接复用 SDK 自己算出来的：
上一轮 `--param_quantizer_schema symmetric` 产出的 symw DLC 里，
这些张量已是 `sFxp_8 / offset=0 / min=-128s / max=127s`，把 scale 原样搬过来。
（约束 8：量化器的对称取整规则实测 != max|w|/127，猜公式必错。）

⚠️ 两个已实测的陷阱（HANDOVER 15.15.3 / 台账 #45）：
  · 必须用 **ONNX 张量名**（`val_1800`），不是 DLC 名（Gemm 权重 DLC 侧多个 `_permute` 后缀）
  · 名字错了转换器不报错、EXIT=0，只是 `Processed 0` —— 必须每次 grep 那一行

用法:
    python make_sym_fc_overrides.py <out.json> [--limit N]
"""
import json
import sys

import onnx

import map_opid as M

SYMW_INFO = r"D:\ZImage_Work\p0_experiments\symw_part1b_dlcinfo.txt"
BASE_INFO = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"
ONNX_PATH = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1b.onnx"


def collect_fc_weights(info_path):
    """返回 {dlc_tensor_name: encoding}，只取 FullyConnected 的 8-bit 静态权重"""
    ops = M.parse(info_path)
    out = {}
    for op in ops.values():
        if op["type"] != "FullyConnected":
            continue
        for t in op["in"]:
            if t["ttype"] != "STATIC":
                continue
            e = op["enc"].get(t["name"])
            if not e or e["bw"] != 8:
                continue  # 跳过 sFxp_32 的 bias
            out[t["name"]] = dict(e, dtype=t.get("dtype", "?"), op=op["name"])
    return out


def main():
    out_path = sys.argv[1]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    sym = collect_fc_weights(SYMW_INFO)
    base = collect_fc_weights(BASE_INFO)
    print(f"symw DLC 里的 FC 8-bit 权重: {len(sym)} 个")
    print(f"基线 DLC 里的 FC 8-bit 权重: {len(base)} 个")
    assert set(sym) == set(base), "两份 DLC 的 FC 权重张量集合不一致，拒绝继续"

    onnx_inits = {i.name for i in onnx.load(ONNX_PATH, load_external_data=False).graph.initializer}

    enc = {}
    bad = []
    for dlc_name in sorted(sym):
        e = sym[dlc_name]
        # 有效性：symw 侧必须真的是对称（有符号表示下 offset==0）
        if abs(e["offset"]) > 1e-9 or not e["dtype"].startswith("sFxp_8"):
            bad.append((dlc_name, "symw 侧不是 sFxp_8/offset=0", e))
            continue
        s = e["scale"]
        # 基线侧必须真的是非对称（否则这个张量本来就对称，改它没意义）
        if abs(base[dlc_name]["offset"] + 128) < 0.5:
            bad.append((dlc_name, "基线侧本来就对称", base[dlc_name]))

        onnx_name = dlc_name[:-len("_permute")] if dlc_name.endswith("_permute") else dlc_name
        if onnx_name not in onnx_inits:
            bad.append((dlc_name, f"ONNX 里找不到 initializer `{onnx_name}`", e))
            continue
        enc[onnx_name] = [{
            "bitwidth": 8,
            "is_symmetric": "True",
            "min": -128.0 * s,
            "max": 127.0 * s,
            "offset": -128,
            "scale": s,
        }]

    if bad:
        print("\n!! 有问题的张量：")
        for n, why, e in bad:
            print(f"   {n:50} {why}  {e}")
        if any("找不到" in w or "不是 sFxp_8" in w for _, w, _ in bad):
            sys.exit("存在致命问题，拒绝生成 JSON")

    names = sorted(enc)
    if limit is not None:
        names = names[:limit]
        enc = {n: enc[n] for n in names}

    doc = {"activation_encodings": {}, "param_encodings": enc, "version": "0.6.1"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"\n写出 {out_path}")
    print(f"点名张量数 = {len(enc)}  <-- 转换日志里的 `Processed N quantization encodings` 必须等于它")
    for n in names[:5]:
        print(f"   {n:20} scale={enc[n][0]['scale']:.12f} "
              f"min={enc[n][0]['min']:.9f} max={enc[n][0]['max']:.9f}")
    if len(names) > 5:
        print(f"   ... 共 {len(names)} 个")


if __name__ == "__main__":
    main()
