"""从 `.bin` 产物自动生成 app 用的 `final_qnn_contract.json`（四段 transformer 版）。

## 纪律：禁止手写 manifest
输入输出规格、dtype、scale/offset 一律从 `qnn-context-binary-utility` 的元数据 dump 取，
sha256 从文件本身算。**不得从 ONNX 推导、不得照抄旧 manifest**
（同约束 3 对 `dlc_contracts.json` 的要求：机器可读契约必须由工具转储自动生成）。

## 本次交付与 app 现装版本的差异（全部实测，不是推测）

| 图 | app 现装 | 本次 |
|---|---|---|
| transformer_part1a | SM8550 基线 2,367,122,432 B | **per-row SM8750** |
| transformer_part1b | SM8550 基线 1,470,838,792 B | **per-row SM8750** |
| transformer_part2 | SM8550 基线 2,925,528,280 B | **拆成 part2a + part2b，per-row SM8750** |
| text_encoder ×4 / vae | SM8550 | **SM8750**（与设备 v79 一致） |

⚠️ **未验证项（必须随交付一起说明）**：出猫那次实验用的是
**宿主 FP32 文本编码 + FP32 VAE**，量化版 text_encoder / VAE **不在该验证范围内**。
app 端到端是否同样出猫，**只能装机实测**（约束 1：必须自己打开图看）。

用法: python gen_zimage_manifest.py <输出目录>
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

SDK = r"D:\qairt\2.48.0.260626"
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
EV = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline"
P0 = r"D:\ZImage_Work\p0_experiments"

# (internal_graph_name, 源 .bin 绝对路径, 交付文件名)
GRAPHS = [
    ("text_encoder_part1", f"{EV}\\text_encoder_part1\\text_encoder_part1_ctx_sm8750.SM8750.bin",
     "text_encoder_part1_ctx.SM8750.bin"),
    ("text_encoder_part2", f"{EV}\\text_encoder_part2\\text_encoder_part2_ctx_sm8750.SM8750.bin",
     "text_encoder_part2_ctx.SM8750.bin"),
    ("text_encoder_part3", f"{EV}\\text_encoder_part3\\text_encoder_part3_ctx_sm8750.SM8750.bin",
     "text_encoder_part3_ctx.SM8750.bin"),
    ("text_encoder_part4", f"{EV}\\text_encoder_part4\\text_encoder_part4_ctx_sm8750.SM8750.bin",
     "text_encoder_part4_ctx.SM8750.bin"),
    ("transformer_part1a", f"{P0}\\perrow_p1a\\ctx\\part1a_perrow_ctx.SM8750.bin",
     "transformer_part1a_ctx.SM8750.bin"),
    ("transformer_part1b", f"{P0}\\perrow\\ctx\\part1b_perrow_ctx.SM8750.bin",
     "transformer_part1b_ctx.SM8750.bin"),
    ("transformer_part2a", f"{P0}\\p2split\\ctx\\part2a_fixed_ctx.SM8750.bin",
     "transformer_part2a_ctx.SM8750.bin"),
    ("transformer_part2b", f"{P0}\\p2split\\ctx\\part2b_fixed_ctx.SM8750.bin",
     "transformer_part2b_ctx.SM8750.bin"),
    ("vae_decoder", f"{EV}\\vae_decoder\\vae_decoder_ctx_sm8750.SM8750.bin",
     "vae_decoder_ctx.SM8750.bin"),
]

DT_BYTES = {"QNN_DATATYPE_UFIXED_POINT_16": 2, "QNN_DATATYPE_BOOL_8": 1,
            "QNN_DATATYPE_INT_32": 4}


def meta(binpath):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "m.json")
        r = subprocess.run([UTIL, "--context_binary", binpath, "--json_file", out],
                           capture_output=True, text=True)
        if not os.path.isfile(out):
            raise RuntimeError(f"元数据 dump 失败: {binpath}\n{r.stdout}\n{r.stderr}")
        with open(out, encoding="utf-8") as f:
            return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def tensors(lst, graph):
    out = []
    for t in lst:
        i = t["info"]
        dt = i["dataType"]
        if dt not in DT_BYTES:
            raise RuntimeError(f"{graph}: 未支持的 dtype {dt}（app 侧 normalizeFinalDtype 会拒绝）")
        n = 1
        for d in i["dimensions"]:
            n *= d
        q = i.get("quantizeParams", {})
        so = q.get("scaleOffset") if q.get("definition") == "QNN_DEFINITION_DEFINED" else None
        out.append({
            "name": i["name"],
            "physical_dtype": dt,
            "fixed_shape": list(i["dimensions"]),
            "layout": i.get("dataFormat", "QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER"),
            "exact_bytes": n * DT_BYTES[dt],
            "quantization": {
                "storage_dtype": dt,
                "scale": so["scale"] if so else None,
                "offset": so["offset"] if so else None,
                "per_tensor_or_axis": "per-tensor" if so else "none",
                "axis": "NONE",
            },
        })
    return out


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else r"D:\ZImage_Work\p0_experiments\app_bundle"
    os.makedirs(os.path.join(dst, "models"), exist_ok=True)
    models = []
    for name, src, fname in GRAPHS:
        if not os.path.isfile(src):
            raise SystemExit(f"缺少 .bin: {src}")
        m = meta(src)["info"]
        g = m["graphs"][0]["info"]
        size = os.path.getsize(src)
        print(f"  {name:<20} {g['graphName']:<26} in={g['numGraphInputs']} "
              f"out={g['numGraphOutputs']} {size/1048576:8.1f} MB", flush=True)
        models.append({
            "actual_filename": fname,
            "internal_graph_name": name,
            "qnn_graph_name": g["graphName"],
            "context_binary": f"models/{fname}",
            "sha256": sha256(src),
            "size_bytes": size,
            "quantization_type": "W8A16",
            "inputs": tensors(g["graphInputs"], name),
            "outputs": tensors(g["graphOutputs"], name),
        })

    contract = {
        "schema_note": "四段 transformer 交付（part2 已拆为 part2a/part2b）。"
                       "由 scripts/gen_zimage_manifest.py 从 .bin 元数据自动生成，禁止手工编辑。",
        "generated_by": "gen_zimage_manifest.py",
        "transformer_segments": 4,
        "models": models,
    }
    p = os.path.join(dst, "final_qnn_contract.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=1, ensure_ascii=False)
    print(f"\n写出 {p}")

    # 自检：切口张量在 part2a 输出与 part2b 输入两侧必须一一对应
    a = next(m for m in models if m["internal_graph_name"] == "transformer_part2a")
    b = next(m for m in models if m["internal_graph_name"] == "transformer_part2b")
    ao = {t["name"] for t in a["outputs"]}
    bi = {t["name"] for t in b["inputs"]}
    missing = ao - bi
    extra = bi - ao
    print(f"自检 切口: part2a 输出 {sorted(ao)}")
    print(f"          part2b 输入 {sorted(bi)}")
    print(f"          part2a 输出未被 part2b 消费: {missing or '无 ✅'}")
    print(f"          part2b 需要但非来自 part2a: {sorted(extra)}（应为 adaln_input，来自 part1a）")
    if missing:
        raise SystemExit("❌ 切口不闭合，交付无效")
    print(f"\n下一步: 把 9 个 .bin 复制到 {dst}\\models\\ 并推到设备")


if __name__ == "__main__":
    main()
