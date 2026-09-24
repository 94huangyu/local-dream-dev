# -*- coding: utf-8 -*-
"""#86 状态更新：用户已明确需求为「不同比例」，且已完成首轮实测勘察。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

OLD = (u"| 86 | 产品能力：自定义尺寸 / 非 1:1 | \U0001F536 **未查·可行·不在关键路径** |")
NEW = (u"| 86 | \U0001F7E2 **产品能力：不同比例（非 1:1）—— 用户 2026-08-30 明确点名** "
       u"| \U0001F504 **进行中·勘察已完成·待写方案** |")


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    assert s.count(OLD) == 1, "锚点未唯一命中"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_86status.md"))
    io.open(MAIN, "w", encoding="utf-8", newline="").write(s.replace(OLD, NEW, 1))
    print("OK #86 状态已更新")


if __name__ == "__main__":
    main()
