# -*- coding: utf-8 -*-
"""部署门：设备上现算的 sha256 必须与契约声明逐一相同。

🔴 这个文件的存在本身是个教训：deploy_aspect.sh 第一版把这一步写成了
   `sha256sum ... ; cat`，**只打印不比对**——判据写在注释里、执行时不生效，
   等于没有这道门（约束 8：判据的操作化必须真的成立）。
"""
import io, json, sys

sys.stdout.reconfigure(encoding="utf-8")
# 🔴 2026-09-17 补 `--names`：deploy_aspect.sh 原来在设备上算 `sha256sum *_ctx.SM8750.bin`，
#    而文本编码器叫 `text_encoder_part1_L80_ctx_sm8750.SM8750.bin`（`_ctx_sm8750.`，不是 `_ctx.`）
#    ⇒ **通配符漏掉了它们**，本门判「缺失」，mg 交付第二次回滚。
#    旧契约的 size_variants 恰好不含文本编码器，所以此前一直没暴露。
#    ⇒ 「要算哪些文件」必须**从契约派生**（约束 11·补：同一份清单不得两处维护），
#       顺带不再白算设备上不相关的 ~30 GiB。
names_mode = sys.argv[1] == "--names"
if names_mode:
    sys.argv.pop(1)
c = json.load(io.open(sys.argv[1], encoding="utf-8"))
want = {}
for v in (c.get("size_variants") or {}).values():
    for m in v["models"]:
        want[m["actual_filename"]] = m["sha256"]
if names_mode:
    print(" ".join(sorted(want)))
    sys.exit(0 if want else 1)
if not want:
    print("    契约里没有 size_variants —— 无可校验对象，判失败（避免空集合假通过）")
    sys.exit(1)
dev = {}
for line in io.open(sys.argv[2], encoding="utf-8"):
    p = line.split()
    if len(p) == 2:
        dev[p[1]] = p[0]
bad = [f for f, h in sorted(want.items()) if dev.get(f) != h]
for f, h in sorted(want.items()):
    print("    %s %s" % ("OK  " if dev.get(f) == h else "FAIL", f))
if bad:
    print("    🔴 sha256 不符或缺失: %s" % bad)
    sys.exit(1)
print("    ✅ %d 个文件设备现算 sha256 与契约逐一相同" % len(want))
