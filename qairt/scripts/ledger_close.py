# -*- coding: utf-8 -*-
"""把 MAINLINE.md §4 里指定编号的条目标记为已关闭并搬走（全文进 LEDGER_CLOSED，§5 留一行）。

配合 `ledger_compact.py` 用：那个搬的是**已经标了 ✅** 的；这个用于
**工作已完成但状态忘了改**的条目 —— 2026-09-06 实测 `#86` 就是这种：
任务已交付（#161 记录），条目却仍挂「未开工」，**一条就占未关闭表的 58%（30875 字符）**，
每个新线程都要为一个已完成的任务付约 17k token。

用法: python ledger_close.py <编号> "<一行结论>" [<编号> "<一行结论>" ...] [--apply]
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

MAIN = "MAINLINE.md"
CLOSED = "docs/LEDGER_CLOSED.md"

args = [a for a in sys.argv[1:] if a != "--apply"]
apply = "--apply" in sys.argv
if len(args) % 2:
    raise SystemExit(__doc__)
want = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

lines = io.open(MAIN, encoding="utf-8").read().split("\n")
s4 = next(i for i, l in enumerate(lines) if l.startswith("## 4. 线索台账"))
s5 = next(i for i, l in enumerate(lines) if l.startswith("## 5. 线索台账"))
ins = next(i for i in range(s5, len(lines)) if lines[i].startswith("|---")) + 1

hit = []
for i in range(s4, s5):
    m = re.match(r"^\| (\d+) \|", lines[i])
    if m and m.group(1) in want:
        hit.append((i, m.group(1), lines[i]))

for i, n, l in hit:
    print("  #%-4s %6d 字符 -> 搬走，§5 留一行" % (n, len(l)))
print("  合计 %d 字符 ≈ %d token" % (sum(len(l) for _, _, l in hit),
                                    int(sum(len(l) for _, _, l in hit) * 0.55)))
miss = set(want) - {n for _, n, _ in hit}
if miss:
    print("  ⚠️ 在 §4 里没找到: %s" % sorted(miss))

if not apply:
    print("\n（预演，未修改。加 --apply 执行）")
    sys.exit(0)

add = ["", "", "## 【2026-09-06 关闭并从 §4 搬入】工作已完成但状态未及时更新的条目，全文逐字保留", ""]
add += ["| # | 线索 | 状态 | 要点 / 证据 |", "|---|---|---|---|"]
add += [l for _, _, l in hit]
io.open(CLOSED, "a", encoding="utf-8", newline="").write("\n".join(add) + "\n")

drop = {i for i, _, _ in hit}
onelines = ["| %s | %s | 全文见 `docs/LEDGER_CLOSED.md` |" % (n, want[n])
            for _, n, _ in hit]
out = []
for i, l in enumerate(lines):
    if i == ins:
        out.extend(onelines)
    if i in drop:
        continue
    out.append(l)
io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(out))
print("\n已关闭 %d 条。MAINLINE %d -> %d 字符" %
      (len(hit), sum(len(x) for x in lines), sum(len(x) for x in out)))
