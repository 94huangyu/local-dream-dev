"""把设备上正在运行的 Z-Image 交付升级为【四段 transformer】版。按正式交付标准执行。

## 权威来源（都是查证过的，不是约定）

1. **契约 schema 用 `models`，不是 `graphs`** —— 从 C++ 源码读死：
   `ZImageQnnContract::load()` 在没有 `models` 键时走 legacy 分支，
   那条分支只登记 4 个图名（`text_encoder` / `transformer_part1` /
   `transformer_part2` / `vae_decoder`），而 `PipelineZImage` 请求的是
   `text_encoder_part1..4` / `transformer_part1a` / `part1b` / ...
   ⇒ **第一次 `loadGraph("text_encoder_part1")` 必然抛 `Unknown Z-Image graph`。**
   设备现场佐证：app 私有目录里 `final_qnn_contract.json` 用 `models`（在跑），
   而 `final_qnn_contract.imported-graphs.json` 是 `graphs` 版遗留（带 BOM，未被使用）。

2. **基准是 app 私有目录里实际装的那份**（`files/models/ZIMAGE/`），
   不是 `/storage/.../Download/` 里的 staging 副本。
   ⚠️ 实测教训：staging 副本是 SM8550 且 schema 为 `graphs`，
   据它推断会得出"app 装的是 SM8550"这个**错误**结论——
   实际装的是 **SM8750**（字节数与宿主基线产物逐一对上）。

## 变更范围（单变量：只动 transformer）

| 图 | 变更 |
|---|---|
| text_encoder ×4 / vae_decoder | **不动**（已是 SM8750，逐字保留条目与文件） |
| transformer_part1a / part1b | 基线 per-tensor → **per-row**（同为 SM8750） |
| transformer_part2 | **移除**（拆分） |
| transformer_part2a / part2b | **新增**，per-row |

⚠️ **未验证项，必须随交付说明**：出猫那次（`EXP_PLAN_INLOOP4`）用的是
**宿主 FP32 文本编码 + FP32 VAE**，量化版 text_encoder / VAE **不在验证范围内**；
且 app 是 **C++ 实现**，与 Python 流水线是两套实现（已知出过 2 个 bug）。
⇒ **app 端到端是否同样出猫，只能装机实测**（约束 1：必须自己打开图看）。

用法: python build_app_bundle.py
"""
import hashlib
import json
import os
import subprocess
import tempfile

SDK = r"D:\qairt\2.48.0.260626"
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
OUT = r"D:\ZImage_Work\p0_experiments\app_bundle"
LIVE = os.path.join(OUT, "live_contract.json")
PKG = "io.github.xororz.localdream.zimage"
APPDIR = "files/models/ZIMAGE"

# internal_graph_name -> (宿主 .bin, 设备上同一份文件的现成路径, 交付文件名)
# 设备路径下的四个文件就是 EXP_PLAN_INLOOP4 实测出猫时用的**那几个文件本身**。
# 文件名带 perrow/split，避免与基线同名不同内容（正式交付要自述）。
TRANSFORMER = {
    "transformer_part1a": (
        r"D:\ZImage_Work\p0_experiments\perrow_p1a\ctx\part1a_perrow_ctx.SM8750.bin",
        "/data/local/tmp/htpcmp/part1a.bin",
        "transformer_part1a_perrow_ctx_sm8750.SM8750.bin"),
    "transformer_part1b": (
        r"D:\ZImage_Work\p0_experiments\perrow\ctx\part1b_perrow_ctx.SM8750.bin",
        "/data/local/tmp/htpcmp/part1b.bin",
        "transformer_part1b_perrow_ctx_sm8750.SM8750.bin"),
    "transformer_part2a": (
        r"D:\ZImage_Work\p0_experiments\p2split\ctx\part2a_fixed_ctx.SM8750.bin",
        "/data/local/tmp/htpcmp/part2a_fixed.bin",
        "transformer_part2a_perrow_ctx_sm8750.SM8750.bin"),
    "transformer_part2b": (
        r"D:\ZImage_Work\p0_experiments\p2split\ctx\part2b_fixed_ctx.SM8750.bin",
        "/data/local/tmp/htpcmp/part2b_fixed.bin",
        "transformer_part2b_perrow_ctx_sm8750.SM8750.bin"),
}
# ---------------------------------------------------------------------------
# 2026-08-25：变体选择器。`BUNDLE_VARIANT=clip` 交付 #111 的 Clip+FP16 四段
# （L3 18.67 dB，比 per-row 的 11.93 dB 高 6.74 dB）。
# 🔴 交付文件名带 `clip`，**不与旧的 `_perrow_` 同名** => 旧文件原样保留、可回滚
#    （约束 11 铁律 2：能不覆盖就不覆盖）。
# 设备源就是今天推上去的那四个文件本身 —— 端到端 18.67 dB 就是它们跑出来的。
_P2A = r"D:\ZImage_Work\p0_experiments\p2attr"
if os.environ.get("BUNDLE_VARIANT", "") == "clip":
    TRANSFORMER = {
        "transformer_part1a": (
            _P2A + r"\ctx_part1a_fp16\part1a_fp16.SM8750.bin",
            "/data/local/tmp/htpcmp/part1a_fp16.bin",
            "transformer_part1a_clip_ctx_sm8750.SM8750.bin"),
        "transformer_part1b": (
            _P2A + r"\ctx_part1b_fp16\part1b_fp16.SM8750.bin",
            "/data/local/tmp/htpcmp/part1b_fp16.bin",
            "transformer_part1b_clip_ctx_sm8750.SM8750.bin"),
        "transformer_part2a": (
            _P2A + r"\ctx_part2a_fp16\part2a_fp16.SM8750.bin",
            "/data/local/tmp/htpcmp/part2a_fixed_fp16.bin",
            "transformer_part2a_clip_ctx_sm8750.SM8750.bin"),
        "transformer_part2b": (
            _P2A + r"\ctx_part2b_fp16\part2b_fp16.SM8750.bin",
            "/data/local/tmp/htpcmp/part2b_fixed_fp16.bin",
            "transformer_part2b_clip_ctx_sm8750.SM8750.bin"),
    }
    _QGRAN = ("per-row + 显式 Clip(±65504) + 选择性 FP16"
              "（白名单 = old-26 的 26 个 + 18 个被钳的 massive activation）")
    _PROV = ("#111 Clip+FP16：设备实测 L3 18.67 dB / L1 24.37% / 主体余弦 0.9140"
             "（per-row 基线为 11.93 dB）。配方见台账 #111 与 scripts/insert_clip.py")
else:
    _QGRAN = "per-row (--use_per_row_quantization)"
    _PROV = "EXP_PLAN_INLOOP4 实测出猫的那批 .bin（per-row，part2 已拆为 2a/2b）"

KEEP = ["text_encoder_part1", "text_encoder_part2", "text_encoder_part3",
        "text_encoder_part4", "vae_decoder"]
DT_BYTES = {"QNN_DATATYPE_UFIXED_POINT_16": 2, "QNN_DATATYPE_BOOL_8": 1,
            "QNN_DATATYPE_INT_32": 4}


def meta(binpath):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "m.json")
        r = subprocess.run([UTIL, "--context_binary", binpath, "--json_file", out],
                           capture_output=True, text=True)
        if not os.path.isfile(out):
            raise RuntimeError(f"元数据 dump 失败: {binpath}\n{r.stdout}{r.stderr}")
        with open(out, encoding="utf-8") as f:
            return json.load(f)["info"]["graphs"][0]["info"]


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
            raise RuntimeError(f"{graph}: app 的 normalizeFinalDtype 不支持 {dt}")
        n = 1
        for d in i["dimensions"]:
            n *= d
        q = i.get("quantizeParams", {})
        so = q.get("scaleOffset") if q.get("definition") == "QNN_DEFINITION_DEFINED" else None
        out.append({
            "name": i["name"], "physical_dtype": dt,
            "fixed_shape": list(i["dimensions"]),
            "layout": i.get("dataFormat", "QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER"),
            "exact_bytes": n * DT_BYTES[dt],
            "quantization": {"storage_dtype": dt,
                             "scale": so["scale"] if so else None,
                             "offset": so["offset"] if so else None,
                             "per_tensor_or_axis": "per-tensor" if so else "none",
                             "axis": "NONE"},
        })
    return out


def main():
    with open(LIVE, encoding="utf-8-sig") as f:
        live = json.load(f)
    if "models" not in live:
        raise SystemExit("基准契约没有 `models` 键——schema 不对，拒绝继续")
    by_name = {m["internal_graph_name"]: m for m in live["models"]}
    print("设备上【正在运行】的交付（app 私有目录，非 Download staging）:")
    for m in live["models"]:
        print(f"  {m['internal_graph_name']:<20} {m['size_bytes']:>12}  {m['context_binary']}")

    models = [json.loads(json.dumps(by_name[n])) for n in KEEP]
    print(f"\n保留（条目与文件均逐字未改）: {', '.join(KEEP)}")
    # 保留项的张量键名必须已经是 `name`（C++ parseFinalTensors 读的就是它）
    for m in models:
        for kind in ("inputs", "outputs"):
            for t in m[kind]:
                if "name" not in t:
                    raise SystemExit(f"{m['internal_graph_name']} 的 {kind} 缺 `name` 键，"
                                     f"C++ parseFinalTensors 会抛异常")

    print("\n重建 transformer 四段（规格取自 .bin 元数据，sha256 取自文件）:")
    for name, (host, devsrc, fname) in TRANSFORMER.items():
        if not os.path.isfile(host):
            raise SystemExit(f"缺少 .bin: {host}")
        g = meta(host)
        models.append({
            "actual_filename": fname,
            "internal_graph_name": name,
            "qnn_graph_name": g["graphName"],
            "context_binary": f"models/{fname}",
            "device_source": devsrc,
            "sha256": sha256(host),
            "size_bytes": os.path.getsize(host),
            "quantization_type": "W8A16",
            "quantization_granularity": _QGRAN,
            "inputs": tensors(g["graphInputs"], name),
            "outputs": tensors(g["graphOutputs"], name),
        })
        print(f"  {name:<20} {os.path.getsize(host):>12}  {fname}")

    contract = dict(live)
    contract["models"] = models
    contract["transformer_segments"] = 4
    contract["generated_by"] = "scripts/build_app_bundle.py"
    contract["provenance"] = {
        "transformer": _PROV,
        "text_encoder_and_vae": "沿用设备原有交付，文件与条目均未改动（单变量）",
        "unverified": "出猫实验用的是宿主 FP32 文本编码与 FP32 VAE；"
                      "量化版 text_encoder/VAE 与 C++ 实现均不在该验证范围内，需装机实测",
    }

    # ---------- 自检 ----------
    names = {m["internal_graph_name"] for m in models}
    need = set(KEEP) | set(TRANSFORMER)
    assert names == need, f"图集合不符: 多 {names - need} 缺 {need - names}"
    assert "transformer_part2" not in names, "旧单段 part2 必须移除，否则布局判定歧义"
    a = next(m for m in models if m["internal_graph_name"] == "transformer_part2a")
    b = next(m for m in models if m["internal_graph_name"] == "transformer_part2b")
    ao = {t["name"] for t in a["outputs"]}
    bi = {t["name"] for t in b["inputs"]}
    assert not (ao - bi), f"切口不闭合: part2a 输出未被消费 {ao - bi}"
    p1a = next(m for m in models if m["internal_graph_name"] == "transformer_part1a")
    p1a_out = {t["name"] for t in p1a["outputs"]}
    assert (bi - ao) <= p1a_out, f"part2b 有无来源输入: {(bi - ao) - p1a_out}"
    p1b = next(m for m in models if m["internal_graph_name"] == "transformer_part1b")
    p2a_in = {t["name"] for t in a["inputs"]}
    avail = p1a_out | {t["name"] for t in p1b["outputs"]}
    assert p2a_in <= avail, f"part2a 有无来源输入: {p2a_in - avail}"
    print(f"\n自检 ✅ 图集合正确 | 切口闭合 {len(ao)} 张量 | "
          f"part2a 输入全部有来源 | part2b 额外输入 {sorted(bi - ao)} 来自 part1a")

    cp = os.path.join(OUT, "final_qnn_contract.json")
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=1, ensure_ascii=False)
    print(f"写出 {cp}")

    # ---------- 部署脚本（设备侧就地复制，不走 USB 传 6.7 GB） ----------
    D = f"{APPDIR}/models"
    sh = [
        "#!/bin/sh",
        "# 由 scripts/build_app_bundle.py 生成。设备侧就地复制 + 逐个校验 sha256。",
        "# 旧文件一律保留（可回滚），只增不删。",
        "set -e",
        f'PKG={PKG}',
        'RA="run-as $PKG"',
        "echo '--- 备份现有契约（可回滚）---'",
        f'$RA sh -c "cp -n {APPDIR}/final_qnn_contract.json '
        f'{APPDIR}/final_qnn_contract.3seg-backup.json" || true',
        "echo '--- 复制四段 transformer（shell 读 -> run-as 写，全程在设备本地）---'",
    ]
    for name, (host, devsrc, fname) in TRANSFORMER.items():
        sh.append(f'echo "  {fname}"')
        sh.append(f'cat {devsrc} | $RA sh -c "cat > {D}/{fname}"')
    sh += ["echo '--- 校验 sha256（必须与契约逐一相同）---'"]
    for name, (host, devsrc, fname) in TRANSFORMER.items():
        sh.append(f'echo "{sha256(host)}  expected {fname}"')
        sh.append(f'$RA sha256sum {D}/{fname}')
    sh += ["echo '--- 部署新契约 ---'",
           f'cat "$(dirname "$0")/final_qnn_contract.json" | '
           f'$RA sh -c "cat > {APPDIR}/final_qnn_contract.json"',
           f'$RA ls -la {D}',
           "echo '完成。旧的 transformer_part2*.bin 仍在，可回滚。'"]
    sp = os.path.join(OUT, "deploy_4seg.sh")
    with open(sp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sh) + "\n")
    print(f"写出 {sp}")
    total = sum(os.path.getsize(h) for h, _, _ in TRANSFORMER.values())
    print(f"\n设备侧需复制 {total/2**30:.2f} GiB（本地 cp，不经 USB）")


if __name__ == "__main__":
    main()
