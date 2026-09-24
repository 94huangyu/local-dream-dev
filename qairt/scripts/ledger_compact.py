# -*- coding: utf-8 -*-
"""把 MAINLINE.md §4（未关闭表）里已经 ✅ 关闭的条目搬到 §5（一行索引）+ LEDGER_CLOSED.md。

## 为什么需要它
`CLAUDE.md` 约束 6 明写台账拆成两份的理由：
「49 条已关闭条目占 MAINLINE 全文 42%，**每个新线程都要为它付一次上下文**」。
但 §4 会随时间重新长胖 —— 条目关闭时人往往只把状态改成 ✅，忘了搬走。
2026-09-06 实测：§4 有 35 条、平均 1899 字符/条，其中 **12 条已 ✅ 关闭、占 15930 字符**；
而 §5 平均 136 字符/条。⇒ 新线程白读约 8.7k token。

## 做法（不丢信息，符合约束 2）
- 全文原样追加到 `docs/LEDGER_CLOSED.md`
- §4 删除该行，§5 插入一行索引：`| N | 标题 | 结论一句话 |`

用法: python ledger_compact.py [--apply]      # 不带 --apply 只预演
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

MAIN = "MAINLINE.md"
CLOSED = "docs/LEDGER_CLOSED.md"
apply = "--apply" in sys.argv


def cells(row):
    """按未转义的 | 切分表格行。"""
    out, cur, i = [], "", 0
    while i < len(row):
        if row[i] == "\\" and i + 1 < len(row) and row[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if row[i] == "|":
            out.append(cur)
            cur = ""
            i += 1
            continue
        cur += row[i]
        i += 1
    out.append(cur)
    return [c.strip() for c in out]


lines = io.open(MAIN, encoding="utf-8").read().split("\n")
s4 = next(i for i, l in enumerate(lines) if l.startswith("## 4. 线索台账"))
s5 = next(i for i, l in enumerate(lines) if l.startswith("## 5. 线索台账"))
# §5 表头之后的第一行（分隔线的下一行）
ins = next(i for i in range(s5, len(lines)) if lines[i].startswith("|---")) + 1

move, keep = [], []
for i in range(s4, s5):
    l = lines[i]
    if re.match(r"^\| \d+ \|", l) and "✅" in l[:80]:
        move.append((i, l))

print("§4 里已 ✅ 关闭、应搬走的条目：")
tot = 0
for i, l in move:
    c = cells(l)
    tot += len(l)
    print("  #%-4s %6d 字符  %s" % (c[1], len(l), c[2][:52]))
print("  合计 %d 字符 ≈ %d token" % (tot, int(tot * 0.55)))

if not apply:
    print("\n（预演，未修改。加 --apply 执行）")
    sys.exit(0)

# 1) 全文追加到 LEDGER_CLOSED
add = ["", "", "## 【2026-09-06 从 MAINLINE §4 搬入】以下条目关闭时未及时搬走，全文逐字保留", ""]
add += ["| # | 线索 | 状态 | 要点 / 证据 |", "|---|---|---|---|"]
add += [l for _, l in move]
io.open(CLOSED, "a", encoding="utf-8", newline="").write("\n".join(add) + "\n")

# 2) §5 插入一行索引；§4 删除
onelines = []
for _, l in move:
    c = cells(l)
    num, title, status = c[1], c[2], c[3]
    # 结论一句话：取状态里的核心 + 指向全文
    onelines.append("| %s | %s | %s —— 全文见 `docs/LEDGER_CLOSED.md` |"
                    % (num, title, status))
drop = {i for i, _ in move}
out = []
for i, l in enumerate(lines):
    if i == ins:
        out.extend(onelines)
    if i in drop:
        continue
    out.append(l)
io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(out))
print("\n已搬 %d 条。MAINLINE 由 %d 减到 %d 字符" %
      (len(move), sum(len(x) for x in lines), sum(len(x) for x in out)))
