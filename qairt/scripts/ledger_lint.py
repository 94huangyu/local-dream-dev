"""台账体检器：把「过早收口 / 过度声称 / 小样本外推」变成**机器能拦的东西**。

用法: python ledger_lint.py            # 扫 MAINLINE.md，列出违规条目
      python ledger_lint.py --strict   # 有违规则退出码 1（可挂进流程）

四条规则（每条都对应本项目发生过的真实错误类型，不再靠人自觉）：

  R1 方向性关闭必须有对照实验
     凡条目里出现「方向错」「此路不通」「整族关闭」等措辞，
     必须同时出现「对照」「control」「空白名单」之一 —— 否则可能是把
     「当前尝试失败」误读成「这条路不通」。

  R2 倍数/规律类断言必须带样本量
     出现「N 倍」「低估」「普遍」「一律」等规律性措辞时，
     必须出现「样本」「N 个」「实测 N」之类的量词。

  R3 「已查明 / 已证明」必须标注证据类型
     出现这类强断言时必须同时出现 ①实测 / 实测 / 设备实测 / 官方文档 之一，
     否则应降级为「推理」。

  R4 状态标记合法性
     每条必须含 ✅/🔄/🔶/⛔/⚠️/❌/🟢/🟡 之一（约束 6：状态不许含糊）。
  R5 台账单行不得超过 1000 字符
     实测：123 条的中位是 102 字符，2026-08-24 之前的最长是 968。
     行太长 => 读不完、改坏了看不出来、每个新线程都要为它付上下文。
     详细证据应移进 HANDOVER / LEDGER_CLOSED，台账只留结论 + 指针。

"""
import io
import os
import re
import sys


# 🔴 本工具是强制门（MAINLINE §0.4）。在 GBK 控制台上打印 emoji 会抛
#    UnicodeEncodeError 而整个崩掉——门崩掉和门放行一样危险，都不报问题。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MAINLINE.md")

DIRECTION = ("方向错", "方向不可行", "此路不通", "整族关闭", "整族已关闭", "本族")
CONTROL = ("对照", "control", "空白名单", "控制臂", "对照臂")
RATIO = ("倍", "低估", "高估", "普遍", "一律", "总是")
SAMPLE = ("样本", "个样本", "实测 ", "个张量", "N 个", "四段", "11 个", "个实测")
STRONG = ("已查明", "已证明", "确认无误", "根因已")
EVIDENCE = ("①实测", "实测", "设备实测", "官方文档", "文档实查")
STATUS = ("✅", "🔄", "🔶", "⛔", "⚠️", "❌", "🟢", "🟡", "🔴")


def rows(txt):
    for ln, line in enumerate(txt.split("\n"), 1):
        m = re.match(r"^\|\s*(\d+[a-z]?)\s*\|", line)
        if m:
            yield ln, m.group(1), line


def main():
    txt = io.open(MAIN, encoding="utf-8").read()
    bad = []
    for ln, num, line in rows(txt):
        hits = []
        if any(k in line for k in DIRECTION) and not any(k in line for k in CONTROL):
            hits.append("R1 方向性关闭但未提对照实验")
        if any(k in line for k in RATIO) and not any(k in line for k in SAMPLE):
            hits.append("R2 规律性断言但未标样本量")
        if any(k in line for k in STRONG) and not any(k in line for k in EVIDENCE):
            hits.append("R3 强断言但未标证据类型")
        if not any(k in line for k in STATUS):
            hits.append("R4 缺状态标记")
        # R5：单行过长。阈值贴着「2026-08-24 之前的历史最长值 968」设，
        #     即新写的条目不得比历史上最啰嗦的那条更啰嗦。详见台账 #118。
        if len(line) > 1000:
            hits.append("R5 单行 %d 字符 > 1000：详细证据应移入 HANDOVER / "
                        "LEDGER_CLOSED，台账只留结论 + 指针" % len(line))
        if hits:
            bad.append((ln, num, hits, line[:110]))
    print("台账体检：扫描 %d 条编号条目" % sum(1 for _ in rows(txt)))
    if not bad:
        print("✅ 无违规")
        return 0
    print("\n发现 %d 条需要处理：\n" % len(bad))
    for ln, num, hits, snippet in bad:
        print("  MAINLINE.md:%d  #%s" % (ln, num))
        for h in hits:
            print("      - %s" % h)
        print("      %s..." % snippet)
    return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if "--strict" in sys.argv else 0)
