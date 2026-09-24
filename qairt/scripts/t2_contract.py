# -*- coding: utf-8 -*-
"""Tier 2: rebuild all EIGHT graph entries of final_qnn_contract.json from the new
L=80 .bin metadata.

## 纪律（与 gen_zimage_manifest.py / te_s32_contract.py 同一条）
**禁止手写 manifest**：规格一律来自 `qnn-context-binary-utility` 的元数据 dump，
不从 ONNX 推导、不照抄旧值。sha256 从文件本身算。

## 为什么不跑 gen_zimage_manifest.py 整份重生成
它的 GRAPHS 清单仍指向 per-row 版 .bin（写于 per-row 部署时期），
而设备上跑的是 Clip+FP16（#111）。整份重生成会把 transformer 写成错的版本。

## 本次要改什么（#137 第一条：只改 sha256/size 会让 app 报 manifest input byte mismatch）
八个图全部重生成 sha256 / size_bytes / inputs / outputs；
TE 的 text_seq_len 32 -> 80；transformer 的 qnn_graph_name 记为新图名。
**vae_decoder 一个字节不许变**（断言）。

## qnn_graph_name 变了要紧吗 —— 不要紧，但仍如实记录
新 bin 的内部图名是 part1a_fp16_L80_fp32（部署版是 part1a_fp16_fp32）。
已 grep 全库确认：app 用 internal_graph_name 匹配条目、按索引取图
（QnnModel.hpp:1122 用 (*m_graphsInfo)[graphIdx].graphName），
契约里的 qnn_graph_name **app 不读**。=> 无害，但按 #73 的教训仍要写对，不留假信息。
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
SDK = r"D:\qairt\2.48.0.260626"
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
SRC = r"D:\LocalDreamZImage\logs\tier1_20260827\final_qnn_contract.s32.json"
OUTDIR = r"D:\LocalDreamZImage\logs\tier2_20260829"
DST = os.path.join(OUTDIR, "final_qnn_contract.L80.json")
TE_ROOT = r"D:\ZImage_Work\TIER2_L80"
TR_ROOT = r"D:\ZImage_Work\p0_experiments\p2attr"
L = 80
UNIFIED = 4096 + L

DT_BYTES = {"QNN_DATATYPE_INT_32": 4, "QNN_DATATYPE_UFIXED_POINT_16": 2,
            "QNN_DATATYPE_BOOL_8": 1, "QNN_DATATYPE_FLOAT_32": 4}

NEWBIN = {}
for _i in (1, 2, 3, 4):
    NEWBIN["text_encoder_part%d" % _i] = os.path.join(
        TE_ROOT, "text_encoder_part%d" % _i,
        "text_encoder_part%d_ctx_sm8750.SM8750.bin" % _i)
for _s in ("part1a", "part1b", "part2a", "part2b"):
    NEWBIN["transformer_%s" % _s] = os.path.join(
        TR_ROOT, "ctx_%s_fp16_L80" % _s, "%s_fp16_L80.SM8750.bin" % _s)
TE = ["text_encoder_part%d" % i for i in (1, 2, 3, 4)]
TR = ["transformer_%s" % s for s in ("part1a", "part1b", "part2a", "part2b")]


def sh(cmd):
    """透传返回码与 stderr（约束 11 再补：包装子进程不得吞错）。"""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("cmd failed rc=%d\n%s\n%s"
                           % (r.returncode, cmd,
                              (r.stderr or b"")[-500:].decode("utf-8", "replace")))
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
            raise RuntimeError("%s: unsupported dtype %s" % (graph, dt))
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
    os.makedirs(OUTDIR, exist_ok=True)
    d = json.load(open(SRC, encoding="utf-8"))
    before = {m["internal_graph_name"]: json.dumps(m, sort_keys=True) for m in d["models"]}

    for m in d["models"]:
        name = m["internal_graph_name"]
        if name not in NEWBIN:
            continue
        b = NEWBIN[name]
        if not os.path.isfile(b):
            print("FAIL missing %s" % b)
            return 1
        g = meta(b)["info"]["graphs"][0]["info"]
        m["sha256"] = sha256(b)
        m["size_bytes"] = os.path.getsize(b)
        m["inputs"] = tensors(g["graphInputs"], name)
        m["outputs"] = tensors(g["graphOutputs"], name)
        if name in TE:
            m["text_seq_len"] = L
        if name in TR:
            m["qnn_graph_name"] = g["graphName"]
        # 🔴 非破坏性交付（用户要求：别破坏之前的成果）：
        #    L=80 用**新文件名**下发，不覆盖设备上的 L=32 产物。
        #    设备本来就并存多代（_ctx_ / _perrow_ctx_ / _clip_ctx_，实查 24 GB），
        #    且 /data 剩 529 GB => 多放 10.3 GB 无压力。
        #    回滚因此只需推回 40 KB 的旧契约，**不需要恢复 10 GB 模型**。
        #    app 用 `context_binary` 定位文件（ZImageQnnContract.hpp:152），故改它即可。
        old_fn = m["actual_filename"]
        assert old_fn.endswith("_ctx_sm8750.SM8750.bin"), old_fn
        new_fn = old_fn.replace("_ctx_sm8750.SM8750.bin",
                                "_L%d_ctx_sm8750.SM8750.bin" % L)
        m["actual_filename"] = new_fn
        m["context_binary"] = "models/" + new_fn
        print("%-22s %-26s in=%d out=%d %8.1f MB"
              % (name, g["graphName"], g["numGraphInputs"], g["numGraphOutputs"],
                 m["size_bytes"] / 1048576))

    byname = {m["internal_graph_name"]: m for m in d["models"]}

    def T(g, kind, tn):
        r = [t for t in byname[g][kind] if t["name"] == tn]
        if not r:
            raise RuntimeError("%s has no %s named %s" % (g, kind, tn))
        return r[0]

    fails = []

    if json.dumps(byname["vae_decoder"], sort_keys=True) != before["vae_decoder"]:
        fails.append("vae_decoder entry was modified")
    else:
        print("\nOK vae_decoder 条目逐字未变")

    ids = T("text_encoder_part1", "inputs", "input_ids")
    if ids["fixed_shape"] != [1, L] or ids["exact_bytes"] != L * 4:
        fails.append("input_ids = %s / %d B" % (ids["fixed_shape"], ids["exact_bytes"]))
    else:
        print("OK input_ids = [1,%d] / %d B" % (L, L * 4))

    for a, tn, b2 in (("text_encoder_part1", "add_2452", "text_encoder_part2"),
                      ("text_encoder_part2", "add_4828", "text_encoder_part3"),
                      ("text_encoder_part3", "add_7204", "text_encoder_part4")):
        if T(a, "outputs", tn)["exact_bytes"] != T(b2, "inputs", tn)["exact_bytes"]:
            fails.append("%s -> %s %s byte mismatch" % (a, b2, tn))
    print("OK TE 段间 add_2452/4828/7204 字节首尾相接")

    cap = T("text_encoder_part4", "outputs", "caption")
    tcap = T("transformer_part1a", "inputs", "caption")
    if cap["fixed_shape"] != [1, L, 2560]:
        fails.append("caption = %s" % cap["fixed_shape"])
    if cap["fixed_shape"] != tcap["fixed_shape"] or cap["exact_bytes"] != tcap["exact_bytes"]:
        fails.append("caption TE-out %s vs transformer-in %s"
                     % (cap["fixed_shape"], tcap["fixed_shape"]))
    else:
        print("OK caption %s / %d B，TE 出口与 transformer_part1a 入口一致"
              % (cap["fixed_shape"], cap["exact_bytes"]))

    cpm = T("transformer_part1a", "inputs", "cap_pad_mask")
    if cpm["fixed_shape"] != [1, L]:
        fails.append("cap_pad_mask = %s" % cpm["fixed_shape"])
    else:
        print("OK cap_pad_mask = [1,%d] / %d B" % (L, cpm["exact_bytes"]))

    CHAIN = [("transformer_part1a", "transformer_part1b",
              ["add_138", "add_131", "tanh_19", "select_45", "select_46", "adaln_input"]),
             ("transformer_part1b", "transformer_part2a", ["unified"]),
             ("transformer_part1a", "transformer_part2a",
              ["unified_mask", "unified_freqs", "adaln_input"]),
             ("transformer_part2a", "transformer_part2b",
              ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3",
               "val_105"])]
    for a, b2, names in CHAIN:
        for tn in names:
            o, i = T(a, "outputs", tn), T(b2, "inputs", tn)
            if o["exact_bytes"] != i["exact_bytes"] or o["fixed_shape"] != i["fixed_shape"]:
                fails.append("%s -> %s %s: %s vs %s"
                             % (a, b2, tn, o["fixed_shape"], i["fixed_shape"]))
    print("OK transformer 段间 %d 个张量字节首尾相接"
          % sum(len(x[2]) for x in CHAIN))

    uni = T("transformer_part1b", "outputs", "unified")
    if uni["fixed_shape"] != [1, UNIFIED, 3840]:
        fails.append("unified = %s, want [1,%d,3840]" % (uni["fixed_shape"], UNIFIED))
    else:
        print("OK unified = [1,%d,3840]（= 图像 4096 + caption %d）" % (UNIFIED, L))

    lat = T("transformer_part1a", "inputs", "latents")
    if lat["fixed_shape"] != [1, 16, 128, 128]:
        fails.append("latents = %s (image stream must not change)" % lat["fixed_shape"])
    else:
        print("OK latents = [1,16,128,128]（图像流未动）")

    # 断言 9：非破坏性 —— 八段文件名必须都带 _L80，vae 必须没带
    for n in TE + TR:
        if "_L%d_" % L not in byname[n]["context_binary"]:
            fails.append("%s 的 context_binary 没带 _L%d：%s"
                         % (n, L, byname[n]["context_binary"]))
    if "_L%d_" % L in byname["vae_decoder"]["context_binary"]:
        fails.append("vae_decoder 的文件名被改了")
    if not fails:
        print("OK 非破坏性：八段指向新文件名 *_L%d_ctx_*，vae 未动，"
              "设备上的 L=32 产物不被覆盖" % L)

    if fails:
        print("\nFAIL %d 条：" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 1

    d.setdefault("provenance", {})
    if isinstance(d["provenance"], dict):
        d["provenance"]["tier2_L80"] = (
            "2026-08-29: caption 槽 32->80（可用正文 24->72 token）。八段图全部重切"
            "（caption 32->80、unified 4128->4176、图像 4096 未动），未从 PyTorch 重导，"
            "故张量名不变、Clip+FP16 白名单沿用，overrides JSON 在 L=80 下重生逐字节相同。"
            "安全性前置门见台账 #139：四段 0 新增 FP16 溢出；n=59 同口径配对显示序列长度"
            "不改变激活量程（比值 0.9706~1.0179）。本文件八个图条目由 scripts/t2_contract.py "
            "从 qnn-context-binary-utility 元数据重生成（含 inputs/outputs 规格），"
            "vae_decoder 逐字未变。")
    json.dump(d, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出 %s（%.1f KB）" % (DST, os.path.getsize(DST) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
