# -*- coding: utf-8 -*-
"""Tier 1 修补：把 final_qnn_contract.json 里 4 个 text_encoder 条目按**新 .bin 的元数据**重生成。

## 为什么需要这个脚本（2026-08-27 实测事故）

交付 seq-32 模型后，app 在真机上报：

    Z-Image manifest input byte mismatch: input_ids

根因：我更新契约时**只改了 `sha256` 与 `size_bytes`，漏了张量规格**。
契约里每个模型条目还带 `inputs`/`outputs` 的 `fixed_shape` 与 `exact_bytes`
（`input_ids` 仍写着 `[1,20]` / 80 B），而 app 现在送 128 B ⇒
`PipelineZImage.hpp:408` 的 `valueFor()` 直接抛错。
⚠️ **app 的校验是对的、拦得也干净**——错的是我给的契约。

## 纪律

`gen_zimage_manifest.py` 的开头写着「禁止手写 manifest，规格一律从
`qnn-context-binary-utility` 的元数据 dump 取」。本脚本遵守同一条：
**规格全部来自工具转储，不从 ONNX 推导、不照抄旧值。**

## 为什么不直接跑 gen_zimage_manifest.py 整份重生成

它的 `GRAPHS` 清单里 transformer 那几项仍指向 **per-row** 版本的 `.bin`，
而设备上跑的是 **Clip+FP16** 版（#111）。整份重生成会把 transformer 写成错的版本。
⇒ 只替换 4 个 text_encoder 条目，其余**逐字保留**，并在末尾断言未被改动。
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

SDK = r"D:\qairt\2.48.0.260626"
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
NEW_ROOT = r"D:\ZImage_Work\TIER1_S32"
SRC = r"D:\LocalDreamZImage\logs\tier1_20260827\contract.json"
DST = r"D:\LocalDreamZImage\logs\tier1_20260827\final_qnn_contract.s32.json"
DT_BYTES = {
    "QNN_DATATYPE_INT_32": 4,
    "QNN_DATATYPE_UFIXED_POINT_16": 2,
    "QNN_DATATYPE_BOOL_8": 1,
    "QNN_DATATYPE_FLOAT_32": 4,
}
TE = ["text_encoder_part%d" % i for i in (1, 2, 3, 4)]


def sh(cmd):
    """透传返回码与 stderr（约束 11·再补：包装子进程不得吞错）。"""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("cmd failed rc=%d\n%s\n%s" % (
            r.returncode, cmd, (r.stderr or b"")[-500:].decode("utf-8", "replace")))
    return r.stdout


def meta(binpath):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "m.json")
        sh([UTIL, "--context_binary", binpath, "--json_file", out])
        return json.load(open(out, encoding="utf-8"))


def tensors(lst, graph):
    out = []
    for t in lst:
        i = t["info"]
        dt = i["dataType"]
        if dt not in DT_BYTES:
            raise RuntimeError("%s: 未支持的 dtype %s" % (graph, dt))
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


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    before = {m["internal_graph_name"]: json.dumps(m, sort_keys=True) for m in d["models"]}

    for m in d["models"]:
        name = m["internal_graph_name"]
        if name not in TE:
            continue
        P = name[-1]
        b = os.path.join(NEW_ROOT, "text_encoder_part%s" % P,
                         "text_encoder_part%s_ctx_sm8750.SM8750.bin" % P)
        if not os.path.isfile(b):
            print("FAIL: 缺少 %s" % b)
            return 1
        g = meta(b)["info"]["graphs"][0]["info"]
        m["sha256"] = sha256(b)
        m["size_bytes"] = os.path.getsize(b)
        m["inputs"] = tensors(g["graphInputs"], name)
        m["outputs"] = tensors(g["graphOutputs"], name)
        m["text_seq_len"] = 32
        print("%-20s graphName=%-26s in=%d out=%d %.1f MB"
              % (name, g["graphName"], g["numGraphInputs"], g["numGraphOutputs"],
                 m["size_bytes"] / 1048576))
        for t in m["inputs"] + m["outputs"]:
            print("      %-16s %-14s %s B" % (t["name"], t["fixed_shape"], t["exact_bytes"]))

    # ---- 断言 1：transformer / vae 条目一个字节都不许变 ----
    for m in d["models"]:
        n = m["internal_graph_name"]
        if n in TE:
            continue
        if json.dumps(m, sort_keys=True) != before[n]:
            print("FAIL: 非 text_encoder 条目 %s 被改动了" % n)
            return 1
    print("\n✅ transformer / vae 条目逐字未变")

    # ---- 断言 2：本次事故的直接判据 ----
    m1 = [m for m in d["models"] if m["internal_graph_name"] == "text_encoder_part1"][0]
    ids = [t for t in m1["inputs"] if t["name"] == "input_ids"][0]
    if ids["fixed_shape"] != [1, 32] or ids["exact_bytes"] != 128:
        print("FAIL: input_ids 仍是 %s / %d B" % (ids["fixed_shape"], ids["exact_bytes"]))
        return 1
    print("✅ input_ids = [1,32] / 128 B（app 送的正是 32*4=128）")

    # ---- 断言 3：段间衔接的字节数必须首尾相接 ----
    chain = [("text_encoder_part1", "add_2452", "text_encoder_part2"),
             ("text_encoder_part2", "add_4828", "text_encoder_part3"),
             ("text_encoder_part3", "add_7204", "text_encoder_part4")]
    byname = {m["internal_graph_name"]: m for m in d["models"]}
    for a, tname, b in chain:
        o = [t for t in byname[a]["outputs"] if t["name"] == tname]
        i = [t for t in byname[b]["inputs"] if t["name"] == tname]
        if not o or not i or o[0]["exact_bytes"] != i[0]["exact_bytes"]:
            print("FAIL: %s -> %s 的 %s 字节数不接" % (a, b, tname))
            return 1
    print("✅ 段间 add_2452 / add_4828 / add_7204 字节数首尾相接")

    # ---- 断言 4：caption 必须仍是 32 槽，且与 transformer 的输入一致 ----
    cap = [t for t in byname["text_encoder_part4"]["outputs"] if t["name"] == "caption"][0]
    tcap = [t for t in byname["transformer_part1a"]["inputs"] if t["name"] == "caption"][0]
    if cap["fixed_shape"] != tcap["fixed_shape"] or cap["exact_bytes"] != tcap["exact_bytes"]:
        print("FAIL: caption 与 transformer_part1a 的输入不符：%s vs %s"
              % (cap["fixed_shape"], tcap["fixed_shape"]))
        return 1
    print("✅ caption %s / %d B，与 transformer_part1a 的输入完全一致"
          % (cap["fixed_shape"], cap["exact_bytes"]))

    d.setdefault("provenance", {})
    if isinstance(d["provenance"], dict):
        d["provenance"]["tier1_s32"] = (
            "2026-08-27: text_encoder 四段重转至 seq 32；本文件的 4 个 text_encoder 条目"
            "由 scripts/te_s32_contract.py 从新 .bin 的 qnn-context-binary-utility 元数据"
            "重新生成（含 inputs/outputs 规格）；transformer/vae 条目逐字未变。")
    json.dump(d, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出 %s" % DST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
