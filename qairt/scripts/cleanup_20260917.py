# -*- coding: utf-8 -*-
"""2026-09-17 磁盘清理（用户逐项勾选批准，见 docs/DISK_INVENTORY.md「2026-09-17 清理记录」）。

纪律（沿用 disk_cleanup.py / DISK_INVENTORY）：
  · **显式路径清单，不用通配符**：p2attr 里 `ctx_part1a_fp16`（L32，删）与 `ctx_part1a_fp16_L80`（现网回滚，留）
    只差一个后缀，通配符一错就删掉回滚路径。
  · 保护名单命中即退出（不是跳过）：宁可整个脚本不跑，也不能删错一个。
  · 删 DLC 前配方已抽出：logs/cleanup_20260917/dlc_recipes_L32_S32.txt；
    配方引用的小文件（校准清单、overrides、建图配置）先拷到 ARCHIVE。
  · 删后核对现网与回滚文件逐个仍在（约束 11 铁律 3 的精神：改了状态就验）。

用法: python scripts/cleanup_20260917.py            # 预演：列清单 + 算 sha256 + 存 manifest
      python scripts/cleanup_20260917.py --apply    # 真删（先存档小文件，删后核对保护集）
"""
import hashlib
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W = os.path.join("D:", os.sep, "ZImage_Work")
P0 = os.path.join(W, "p0_experiments")
P2 = os.path.join(P0, "p2attr")
REPO = os.path.join("D:", os.sep, "LocalDreamZImage")
LOGD = os.path.join(REPO, "logs", "cleanup_20260917")
ARCHIVE = os.path.join(W, "recipes_archive_20260917")

# (路径, 依据) —— 依据全部来自用户 2026-09-17 的勾选 + 删前核实
TARGETS = [
    (os.path.join(W, "deliver_mg"),
     "纯重复：5 个文件 sha256 与 aspect_mg 源、mg 契约三方一致；stage_mg_delivery.py 约 30 s 重建"),
    (os.path.join(P0, "symw"), "§15.16 选择性对称权重，已否定"),
    (os.path.join(P0, "memspeed_probe"), "#162 图切换 E/F 已出局的探针；memspeed_probe_build.py 可重建"),
    (os.path.join("D:", os.sep, "LDZ_FIX3_20260810"), "08-10 旧构建包（用户手工改过的参考代码），用户确认不要"),
    (os.path.join(W, "TIER1_S32"), "L=32 上一代文本编码器，现网为 TIER2_L80；配方已抽出、校准输入已存档"),
] + [
    (os.path.join(P2, "ctx_%s_fp16" % s), "L=32 上一代 transformer context；现网回滚用的是 ctx_%s_fp16_L80（不动）" % s)
    for s in ("part1a", "part1b", "part2a", "part2b")
] + [
    (os.path.join(P2, "%s_fp16_quantized.dlc" % s), "L=32 上一代量化 DLC；配方已抽出，overrides JSON 仍在 p2attr")
    for s in ("part1a", "part1b", "part2a", "part2b")
] + [
    (os.path.join("D:", os.sep, "ZIMAGE", "models", n), "SM8550 context，设备是 SM8750 从未部署；用户批准（#21 需要时重建）")
    for n in ("text_encoder_part1_ctx.SM8550.bin", "text_encoder_part2_ctx.SM8550.bin",
              "text_encoder_part3_ctx.SM8550.bin", "text_encoder_part4_ctx.SM8550.bin",
              "transformer_part1a_ctx.SM8550.bin", "transformer_part1b_ctx.SM8550.bin",
              "transformer_part2_ctx.SM8550.bin", "vae_decoder_ctx.SM8550.bin")
]

# 命中即退出
GUARD = ("_l80", "aspect", "tier2_l80", "zimage_qnn_evidence", "calibration", "device_only_backup",
         "z-image-turbo", "localdreamzimage")

# 删后必须仍在的文件（现网 mg 源 + single 回滚 + 重建 mg 所需 DLC + 源权重）
def protected():
    out = []
    for s in ("part1a", "part1b", "part2a", "part2b"):
        out.append(os.path.join(P0, "aspect_mg", s, "%s_mg.SM8750.bin" % s))
        out.append(os.path.join(P2, "ctx_%s_fp16_L80" % s, "%s_fp16_L80.SM8750.bin" % s))
        out.append(os.path.join(P2, "%s_fp16_L80_quantized.dlc" % s))
        out.append(os.path.join(P2, "%s_fp16_ovr.json" % s))
        for a in ("1184x896", "896x1184", "1280x720", "720x1280"):
            out.append(os.path.join(P0, "aspect", "ctx_%s_%s" % (s, a), "%s_%s.SM8750.bin" % (s, a)))
            out.append(os.path.join(P0, "aspect", "%s_%s_quantized.dlc" % (s, a)))
    out.append(os.path.join(P0, "aspect_mg", "vae", "vae_mg.SM8750.bin"))
    for i in (1, 2, 3, 4):
        d = os.path.join(W, "TIER2_L80", "text_encoder_part%d" % i)
        out.append(os.path.join(d, "text_encoder_part%d_ctx_sm8750.SM8750.bin" % i))
        out.append(os.path.join(d, "text_encoder_part%d_quantized.dlc" % i))
    ev = os.path.join(W, "ZImage_QNN_Evidence")
    out.append(os.path.join(ev, "dlc_pipeline", "vae_decoder", "vae_decoder_ctx_sm8750.SM8750.bin"))
    for n in ("text_encoder_fixed4.onnx.data", "transformer_part1.onnx.data", "transformer_part2.onnx.data"):
        out.append(os.path.join(ev, "onnx", n))
    out.append(os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.single.json"))
    out.append(os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.mg.json"))
    return out


def L(p):
    """Windows 长路径前缀：LDZ_FIX3 里有超过 260 字符的 Gradle 中间文件，不加前缀 stat/删除都会失败。"""
    p = os.path.abspath(p)
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def files_under(p):
    if os.path.isfile(L(p)):
        return [L(p)]
    r = []
    for root, _d, fs in os.walk(L(p)):
        for f in fs:
            r.append(os.path.join(root, f))
    return r


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def archive_small():
    """配方引用的小文件先存档：TIER1_S32 除模型外的全部文件、L32 context 目录的建图配置。"""
    n = 0
    srcs = [(os.path.join(W, "TIER1_S32"), os.path.join(ARCHIVE, "TIER1_S32"))] + [
        (os.path.join(P2, "ctx_%s_fp16" % s), os.path.join(ARCHIVE, "p2attr", "ctx_%s_fp16" % s))
        for s in ("part1a", "part1b", "part2a", "part2b")]
    for src, dst in srcs:
        for f in files_under(src):
            if f.endswith((".bin", ".dlc")):
                continue
            t = os.path.join(dst, os.path.relpath(f, L(src)))
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy2(f, t)
            if os.path.getsize(t) != os.path.getsize(f):
                raise SystemExit("🔴 存档拷贝字节数不符：%s" % f)
            n += 1
    shutil.copy2(os.path.join(LOGD, "dlc_recipes_L32_S32.txt"), os.path.join(ARCHIVE, "dlc_recipes_L32_S32.txt"))
    return n


def main():
    apply = "--apply" in sys.argv
    os.makedirs(LOGD, exist_ok=True)
    if not os.path.isfile(os.path.join(LOGD, "dlc_recipes_L32_S32.txt")):
        raise SystemExit("🔴 配方未抽出（logs/cleanup_20260917/dlc_recipes_L32_S32.txt），不得删 DLC")
    miss = [p for p in protected() if not os.path.isfile(p)]
    if miss:
        raise SystemExit("🔴 删之前保护集就已缺文件，先查清再删：\n  " + "\n  ".join(miss))
    for p, _ in TARGETS:
        low = p.lower()
        if any(g in low for g in GUARD):
            raise SystemExit("🔴 目标命中保护名单：%s" % p)
    manifest, tot = [], 0
    print("模式：%s\n" % ("🔴 真删（--apply）" if apply else "预演（加 --apply 才真删）"))
    for p, why in TARGETS:
        if not os.path.exists(L(p)):
            raise SystemExit("🔴 目标不存在（清单与盘上不符，先查清）：%s" % p)
        fs = files_under(p)
        size = sum(os.path.getsize(f) for f in fs)
        tot += size
        big = [{"path": f, "bytes": os.path.getsize(f), "sha256": sha256(f)}
               for f in fs if os.path.getsize(f) >= 100 << 20]
        manifest.append({"target": p, "why": why, "files": len(fs), "bytes": size, "big_files": big})
        print("  %7.2f GiB  %4d 文件  %s\n              ← %s" % (size / 2**30, len(fs), p, why))
    print("\n合计 %.2f GiB" % (tot / 2**30))
    mf = os.path.join(LOGD, "manifest%s.json" % ("" if apply else "_dryrun"))
    json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "apply": apply, "total_bytes": tot,
               "targets": manifest}, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("manifest -> %s" % mf)
    if not apply:
        return 0
    n = archive_small()
    print("已存档小文件 %d 个 -> %s" % (n, ARCHIVE))
    for p, _ in TARGETS:
        if os.path.isdir(L(p)):
            shutil.rmtree(L(p))
        else:
            os.remove(L(p))
        if os.path.exists(L(p)):
            raise SystemExit("🔴 删除后仍存在：%s" % p)
    miss = [p for p in protected() if not os.path.isfile(p)]
    if miss:
        raise SystemExit("🔴🔴 删除后保护集缺文件：\n  " + "\n  ".join(miss))
    print("✅ 删除完成；保护集 %d 个文件逐个仍在" % len(protected()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
