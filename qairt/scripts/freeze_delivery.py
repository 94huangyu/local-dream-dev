# -*- coding: utf-8 -*-
"""冻结当前交付态：把「宿主产物 + 设备现态」逐项现读成一份可复核的快照。

## 为什么需要它
交付态散在多处：APK 在构建目录、模型在设备私有目录、契约在设备上、marker 是个空文件。
**没有一份清单的话，三个月后没人说得清"现在跑的到底是哪一版"**，
而本项目已经吃过亏（#171：清理后才发现不知道哪份是回滚源）。

本脚本**只读不写设备**，产出 `docs/DELIVERY_FROZEN.md`。
凡是能现读的都现读（sha256、字节数），不抄文档里的旧值（约束 1）。

用法: python scripts/freeze_delivery.py            # 需要插线
      python scripts/freeze_delivery.py --host     # 只记宿主侧（不碰设备）
"""
import hashlib
import io
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D
import p2_s6_session as P

OUT = os.path.join(D.REPO, "docs", "DELIVERY_FROZEN.md")
DEV_FILES = [
    "final_qnn_contract.json",
    "SHARE_SPILLFILL",
    "LOAD_MMAP",
    "models/transformer_part1a_mg_ctx.SM8750.bin",
    "models/transformer_part1b_mg_ctx.SM8750.bin",
    "models/transformer_part2a_mg_ctx.SM8750.bin",
    "models/transformer_part2b_mg_ctx.SM8750.bin",
    "models/vae_mg_ctx.SM8750.bin",
    "models/text_encoder_part1_L80_ctx_sm8750.SM8750.bin",
    "models/text_encoder_part2_L80_ctx_sm8750.SM8750.bin",
    "models/text_encoder_part3_L80_ctx_sm8750.SM8750.bin",
    "models/text_encoder_part4_L80_ctx_sm8750.SM8750.bin",
]


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    lines = ["# 交付态冻结快照", "",
             "> 由 `python scripts/freeze_delivery.py` **现读**生成，不抄任何文档里的旧值（约束 1）。",
             "> 冻结时间：**%s**" % time.strftime("%Y-%m-%d %H:%M"), ""]

    lines += ["## 一、宿主侧产物", "", "| 项 | 值 |", "|---|---|"]
    apk = P.NEW_APK
    if os.path.isfile(apk):
        with zipfile.ZipFile(apk) as z:
            so = z.read("lib/arm64-v8a/libstable_diffusion_core.so")
        lines.append("| APK | `%s` |" % os.path.basename(apk))
        lines.append("| APK 字节数 | %d |" % os.path.getsize(apk))
        lines.append("| APK sha256 | `%s` |" % sha_of(apk))
        lines.append("| `libstable_diffusion_core.so` sha256 | **`%s`** |"
                     % hashlib.sha256(so).hexdigest())
        lines.append("| 构建时间 | %s |" % time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(apk))))
    for name, p in (("noscatA part1a（新件）", P.NEW_PART1A),
                    ("part1a 原件（回滚源）", P.OLD_PART1A)):
        if os.path.isfile(p):
            lines.append("| %s | %d 字节 |" % (name, os.path.getsize(p)))
            lines.append("| ↳ sha256 | `%s` |" % sha_of(p))
    lines.append("")

    if "--host" in sys.argv:
        lines += ["## 二、设备侧现态", "", "（本次以 `--host` 运行，未读设备）", ""]
    else:
        import lab_dev as dev
        dev.require_online("freeze")
        lines += ["## 二、设备侧现态（现读）", "",
                  "| 文件 | 字节数 | sha256 |", "|---|---|---|"]
        for rel in DEV_FILES:
            p = "files/models/ZIMAGE/" + rel
            n = dev.sh("run-as %s stat -c %%s %s 2>/dev/null || echo -"
                       % (S_PKG, p)).strip()
            if n == "-":
                lines.append("| `%s` | **缺失** | — |" % rel)
                continue
            if rel in ("SHARE_SPILLFILL", "LOAD_MMAP"):
                body = dev.sh("run-as %s cat %s 2>/dev/null || true" % (S_PKG, p)).strip()
                lines.append("| `%s` | %s | 内容：`%s` |" % (rel, n, body or "(空文件=开关打开)"))
                continue
            h = dev.sh("run-as %s sha256sum %s" % (S_PKG, p), timeout=3600).strip().split()[0]
            lines.append("| `%s` | %s | `%s` |" % (rel, n, h))
        apk_dev = dev.sh("pm path %s" % S_PKG).strip().replace("package:", "").splitlines()[0]
        lines.append("")
        lines.append("设备上安装的 APK 路径：`%s`" % apk_dev.strip())
        lines.append("")

    lines += ["## 三、验收基线（任何改动后都要能复现）", "",
              "- **出图 sha256（金标准）**：`%s`" % P.GOLDEN_SHA,
              "  - 请求：`scripts/app_generate.sh`，seed 42，默认 prompt，1024×1024",
              "  - 这一条是**逐字节**判据：不相同就说明改动影响了数值，不得交付。", ""]
    io.open(OUT, "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")
    print("已写出 %s（%d 行）" % (OUT, len(lines)))
    return 0


S_PKG = D.PKG

if __name__ == "__main__":
    sys.exit(main())
