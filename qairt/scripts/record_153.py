# -*- coding: utf-8 -*-
"""登记 #153：用户明确禁止拿精度换速度 ⇒ 步数杠杆关闭。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

ROW = (
    u"| 153 | \U0001F534 **用户定：不得拿精度换速度 "
    u"⇒ 步数杠杆关闭** | ✅ **已关闭·用户决定"
    u"（2026-08-31）** "
    u"| 背景：本轮订正了 #129 的误读（「步数已到下限」"
    u"实为「我们的 app 写死了 8」），"
    u"于是「8→6/4 步」浮现为**幅度最大的未测杠杆**"
    u"（−25%~−50%）。<br>"
    u"\U0001F534 **用户明确：不可以拿精度换速度** "
    u"⇒ 该杠杆**不得使用**，连「先测数据再定」也不做。<br>"
    u"⊕ **与 2026-08-25 那条合并后的完整约束**："
    u"**两个方向都不允许交换** —— "
    u"既不得拿速度换精度（08-25），也不得拿精度换速度（08-31）。"
    u"⇒ **今后任何手段必须在另一维上中性或正向才能考虑。**<br>"
    u"⇒ **在该约束下，速度线确已接近穷尽**："
    u"每步加载已榚干（F，−21.2%）｜DVFS 已最激进（#129 实测）｜"
    u"`O`/`dlbc` 已关闭（#147，PD 红线）｜步数已关闭（本条）。<br>"
    u"\U0001F536 **仅剩一条未测且可能精度中性的**："
    u"VTCM / spill-fill 调参（#34/#101）—— "
    u"⚠️ 但 #34 已实测 `vtcm_mb` 在 `.bin` 路径下**对单算子无效果**"
    u"（2/4/8 产物相同），剩余空间存疑。<br>"
    u"⚠️ **不得写「速度已到物理极限」** —— "
    u"准确表述是「**在不牺牲精度的前提下**，已知手段用尽」。"
    u"步数杠杆**在技术上仍然有效**（−25%~−50%），"
    u"只是被产品约束排除 —— **两者不是一回事，不得混写** |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 152 | ")]
    assert len(i) == 1 and u"| 153 | " not in s
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_153.md"))
    L.insert(i[0], ROW)
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("OK #153 已登记")


if __name__ == "__main__":
    main()
