# -*- coding: utf-8 -*-
"""把 mg 交付物按契约文件名 staged 到固定目录，并逐个核 sha256。

## 为什么要有它
`deploy_aspect.sh` 要的是「按交付文件名放好的目录」。而 mg 产物在
`p0_experiments/aspect_mg/<seg>/<seg>_mg.SM8750.bin`，名字对不上；
上一个会话把它们 staged 在**会话临时目录**里（随时会被清理）。
⇒ 放到 D 盘固定位置，**并用契约里的 sha256 逐个核对**——
   核对本身就是一道价值很高的门：它同时验证「产物没损坏」和「契约里的哈希是真的」。

## 判据
每个文件：**字节数 == 契约 size_bytes** 且 **sha256 == 契约 sha256**。
任一不符 ⇒ 非零退出，**不得带着坏产物去插线**。

用法: python scripts/stage_mg_delivery.py [--force]
"""
import hashlib
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join("D:", os.sep, "LocalDreamZImage")
CONTRACT = os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.mg.json")
SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg")
DST = os.path.join("D:", os.sep, "ZImage_Work", "deliver_mg")

# 源文件 -> 交付文件名（交付名取自契约，源名取自建图脚本的输出布局）
MAP = {
    "transformer_part1a_mg_ctx.SM8750.bin": os.path.join("part1a", "part1a_mg.SM8750.bin"),
    "transformer_part1b_mg_ctx.SM8750.bin": os.path.join("part1b", "part1b_mg.SM8750.bin"),
    "transformer_part2a_mg_ctx.SM8750.bin": os.path.join("part2a", "part2a_mg.SM8750.bin"),
    "transformer_part2b_mg_ctx.SM8750.bin": os.path.join("part2b", "part2b_mg.SM8750.bin"),
    "vae_mg_ctx.SM8750.bin": os.path.join("vae", "vae_mg.SM8750.bin"),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def want():
    """契约里对每个文件名声明的 (size, sha256)。🔴 同名文件只允许一个 sha256。"""
    c = json.load(open(CONTRACT, encoding="utf-8"))
    out = {}
    def add(e):
        n = e["actual_filename"]
        v = (e.get("size_bytes"), e.get("sha256"))
        if n in out and out[n] != v:
            raise SystemExit("🔴 契约里 %s 声明了两个不同的 sha256/size" % n)
        out[n] = v
    for e in c["models"]:
        add(e)
    for sv in c.get("size_variants", {}).values():
        for e in sv.get("models", []):
            add(e)
    return out


def main():
    force = "--force" in sys.argv
    w = want()
    os.makedirs(DST, exist_ok=True)
    print("契约: %s" % CONTRACT)
    print("staged 到: %s\n" % DST)
    bad = 0
    for name, rel in MAP.items():
        s = os.path.join(SRC, rel)
        d = os.path.join(DST, name)
        if name not in w:
            print("🔴 %s 不在契约里" % name); bad += 1; continue
        exp_size, exp_sha = w[name]
        if not os.path.isfile(s):
            print("🔴 缺源文件 %s" % s); bad += 1; continue
        if not os.path.isfile(d) or os.path.getsize(d) != exp_size or force:
            print("  复制 %s -> %s（%.2f GiB）…" % (rel, name, exp_size / 2**30), flush=True)
            shutil.copyfile(s, d)
        got_size = os.path.getsize(d)
        got_sha = sha256(d)
        ok = (got_size == exp_size) and (got_sha == exp_sha)
        print("  %-46s %12d %s %s"
              % (name, got_size, got_sha[:16], "✅" if ok else "🔴 与契约不符"))
        if not ok:
            print("     期望 size=%s sha=%s" % (exp_size, (exp_sha or "")[:16]))
            bad += 1
    # 契约里还要求 4 个 text_encoder：它们与 single 形态相同，**设备上已有**，不重复推
    te = [n for n in w if n.startswith("text_encoder")]
    print("\n  ⊕ 契约还引用 %d 个 text_encoder 文件（与现网 single 相同，设备上已有，本次不推）：" % len(te))
    for n in sorted(te):
        print("     %s" % n)
    total = sum(os.path.getsize(os.path.join(DST, n)) for n in MAP if
                os.path.isfile(os.path.join(DST, n)))
    print("\n  staged 合计 %.2f GiB" % (total / 2**30))
    if bad:
        print("\n🔴 %d 个文件不符，**不要插线**" % bad)
        return 1
    print("\n✅ 全部与契约逐字节相符，可用于交付")
    return 0


if __name__ == "__main__":
    sys.exit(main())
