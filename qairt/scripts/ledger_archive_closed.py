# -*- coding: utf-8 -*-
"""把 MAINLINE §4（未关闭台账）里**已关闭**的条目搬到 §5 一行索引 + LEDGER_CLOSED 全文。

## 为什么需要它
2026-08-17 拆表的理由是「49 条已关闭条目占 MAINLINE 全文 42%，每个新线程都要为它付一次
上下文」。到 2026-08-29 又长回 19 条 —— 说明**靠自觉搬运不成立，得有工具**。

## 纪律（台账 #118 的教训）
- 只做**整行**操作：整行删除 / 整行插入。**禁止字符偏移插入**。
- 原文**按字节切片**从源行取出，不得凭记忆重打。
- 拆单元格时用**手写扫描**识别未转义的 `|`，不用带反斜杠的正则
  （2026-08-29 实测：heredoc 会把 `(?<!\\)` 吃成 `(?<!\)`，正则当场崩）。
- 改完必须跑 doc_audit --strict（含 D6 表格结构 / D7 行内重复）与 ledger_lint。

用法:
    python ledger_archive_closed.py --dry     # 只报告要搬哪些，不动文件
    python ledger_archive_closed.py           # 执行
"""
import io
import os
import re
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
M = r"D:\LocalDreamZImage\MAINLINE.md"
C = r"D:\LocalDreamZImage\docs\LEDGER_CLOSED.md"


def cells(row):
    """按**未转义**的 `|` 切单元格。手写扫描，不用正则（见文件头）。"""
    out, buf, i = [], [], 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            buf.append(row[i:i + 2]); i += 2; continue
        if ch == "|":
            out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(ch); i += 1
    out.append("".join(buf))
    return out


def main():
    dry = "--dry" in sys.argv
    src = io.open(M, encoding="utf-8").read()
    lines = src.split(chr(10))
    a = next(i for i, l in enumerate(lines) if l.startswith("## 4."))
    b = next(i for i, l in enumerate(lines) if l.startswith("## 5."))

    move = []
    for i in range(a, b):
        c = cells(lines[i])
        if len(c) < 6 or not c[1].strip().isdigit():
            continue
        num, name, st, body = int(c[1].strip()), c[2].strip(), c[3].strip(), c[4].strip()
        if "✅" in st[:8] or "❌" in st[:8]:
            move.append((num, i, name, st, body))
    print("§4 里已关闭的 %d 条：%s" % (len(move), " ".join("#%d" % m[0] for m in move)))
    if not move:
        return 0

    cur = io.open(C, encoding="utf-8").read()
    already = [m[0] for m in move if ("### #%d\u3000" % m[0]) in cur]
    if already:
        print("  归档里已有（只做搬移，不重复写全文）：%s"
              % " ".join("#%d" % n for n in already))
    if dry:
        print("[--dry] 未改动任何文件")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    for p in (M, C):
        shutil.copy2(p, p + ".bak_" + ts)
    print("已备份两份文件（后缀 .bak_%s）" % ts)

    # ---- 1) 全文追加到 LEDGER_CLOSED（原文按字节从源行取） ----
    chunks = []
    for num, i, name, st, body in move:
        if num in already:
            continue
        chunks.append(chr(10) + "---" + chr(10) * 2
                      + "### #%d\u3000%s" % (num, name) + chr(10) * 2
                      + "**状态**：%s" % st + chr(10) * 2 + body + chr(10))
    if chunks:
        io.open(C, "a", encoding="utf-8", newline=chr(10)).write("".join(chunks))
    print("追加 %d 条全文到 %s" % (len(chunks), os.path.basename(C)))

    # ---- 2) 从 §4 整行删除，在 §5 表头后整行插入一行索引 ----
    idx_rows = []
    for num, i, name, st, body in move:
        short = st
        if len(short) > 120:
            short = short[:117] + "…"
        idx_rows.append("| %d | %s | %s 全文见 `docs/LEDGER_CLOSED.md` |" % (num, name, short))

    for _num, i, _n, _s, _b in sorted(move, key=lambda x: -x[1]):
        del lines[i]

    # §5 表头之后（重新定位，行号已变）
    b2 = next(i for i, l in enumerate(lines) if l.startswith("## 5."))
    hdr = None
    for i in range(b2, min(b2 + 40, len(lines))):
        if lines[i].startswith("|") and set(lines[i].replace(" ", "")) <= set("|-:"):
            hdr = i
            break
    assert hdr is not None, "找不到 §5 的分隔行"
    lines[hdr + 1:hdr + 1] = idx_rows

    # ---- 3) 断言：§5 每行列数与表头一致 ----
    ncol = len(cells(lines[hdr - 1])) - 2
    bad = []
    for r in idx_rows:
        if len(cells(r)) - 2 != ncol:
            bad.append(r[:60])
    if bad:
        print("🔴 列数不符（表头 %d 列）：" % ncol)
        for x in bad:
            print("   " + x)
        return 1

    io.open(M, "w", encoding="utf-8", newline=chr(10)).write(chr(10).join(lines))
    print("§4 删除 %d 行，§5 插入 %d 行索引（每行 %d 列，与表头一致）"
          % (len(move), len(idx_rows), ncol))
    return 0


if __name__ == "__main__":
    sys.exit(main())
