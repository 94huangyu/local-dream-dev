# -*- coding: utf-8 -*-
"""把已关闭的 #126 整行从 MAINLINE §4 移入 docs/LEDGER_CLOSED.md，§4 换成 §5 一行索引。

纪律（约束 6 / 台账 #118）：
  · 只做**按内容锚定的整行删除 / 整行插入**，禁止字符偏移插入
  · 原文按**字节切片**拷贝，不得凭记忆重打
  · 改完断言：原行逐字出现在 LEDGER_CLOSED、且不再出现在 MAINLINE §4
"""
import io
import sys

MAIN = "MAINLINE.md"
CLOSED = "docs/LEDGER_CLOSED.md"

ONE_LINER = (
    "| 126 | 🔴 一次生图写 N 条历史 + UI 卡在翻页中间 | ✅ **已关闭·根因定位并现场复现"
    "（2026-08-26）**：`MainActivity` 缺 `launchMode` ⇒ Activity 叠栈 ⇒ 进程内 N 个 "
    "`ModelRunScreen` composition 各自消费同一个**全局** `Complete` ⇒ N 条历史行；"
    "卡翻页是同根因（第二个 consumer 抢先 `markBitmapConsumed()` ⇒ 状态回 Idle ⇒ "
    "`LaunchedEffect` 重启 ⇒ 动画协程被取消在半途）。**修法未交付**，随 #132 打包。"
    "完整证据见 `docs/LEDGER_CLOSED.md` §126，方案 `scripts/EXP_PLAN_DUP_HISTORY.md` |"
)


def main():
    main_txt = io.open(MAIN, encoding="utf-8").read()
    lines = main_txt.split("\n")

    # 1) 找到 §4 里的 #126 整行（唯一）
    hits = [i for i, l in enumerate(lines) if l.startswith("| 126 | ")]
    if len(hits) != 1:
        print("FAIL: §4 中 '| 126 | ' 开头的行有 %d 条，应为 1 条" % len(hits))
        return 1
    idx = hits[0]
    row = lines[idx]

    # 2) §5 一行索引插在 §5 表的第一条数据行之前
    h5 = [i for i, l in enumerate(lines) if l.startswith("## 5. 线索台账")]
    if len(h5) != 1:
        print("FAIL: §5 标题定位失败")
        return 1
    ins = None
    for i in range(h5[0], len(lines)):
        if lines[i].startswith("|---"):
            ins = i + 1
            break
    if ins is None:
        print("FAIL: §5 表头分隔行未找到")
        return 1

    # 3) 先删后插（删的下标 < 插的下标，故插入点要 -1）
    del lines[idx]
    ins -= 1
    lines.insert(ins, ONE_LINER)
    new_main = "\n".join(lines)

    # 4) 原行逐字追加到 LEDGER_CLOSED
    closed = io.open(CLOSED, encoding="utf-8").read()
    if not closed.endswith("\n"):
        closed += "\n"
    closed += (
        "\n## 2026-08-26 新增：#126 全文（从 MAINLINE §4 原样移入）\n\n"
        "> 判据事前锁定于 `scripts/EXP_PLAN_DUP_HISTORY.md` §三，"
        "判读脚本 `scripts/dup126_analyze.py`。现场截图 "
        "`logs/dup126_20260826/stuck_screen.png`；DB / logcat 原始产物同目录。\n\n"
        "| # | 线索 | 状态 | 要点 / 缺口 |\n|---|---|---|---|\n" + row + "\n"
    )

    # 5) 断言后再落盘
    assert row in closed, "原行未逐字进入 LEDGER_CLOSED"
    assert row not in new_main, "原行仍留在 MAINLINE"
    assert new_main.count(ONE_LINER) == 1, "§5 索引行数不为 1"
    assert len(new_main.split("\n")) == len(main_txt.split("\n")), "MAINLINE 行数应不变（删1插1）"

    io.open(MAIN, "w", encoding="utf-8", newline="\n").write(new_main)
    io.open(CLOSED, "w", encoding="utf-8", newline="\n").write(closed)
    print("OK: #126 已移入 LEDGER_CLOSED；MAINLINE §5 留一行索引")
    print("    移动的原行长度 = %d 字符" % len(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
