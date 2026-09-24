# -*- coding: utf-8 -*-
"""更新 §2.1 一句话 + §6 执行表，把 #86 列为当前唯一在办事项。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

OLD_HEAD = u"### 2.1 一句话（2026-08-30 重写）"
NEW_HEAD = (
    u"### 2.1 一句话（2026-08-31 重写）\n\n"
    u"\U0001F389 **已交付**：生图 **204.25 → 160.87 s（−21.2%）**，"
    u"sha256 逐字节相同 ⇒ 零精度代价（台账 F/#145）。"
    u"从 271 s 算起累计 **−40.6%**。\n\n"
    u"\U0001F534 **两条主线的现状**：\n"
    u"1. **速度**：在「不牺牲精度」前提下已知手段用尽"
    u"（#153：用户定不得拿精度换速度，步数杠杆关闭）。"
    u"⚠️ **不得写「速度到物理极限」** —— "
    u"步数在技术上仍有 −25~50%，只是被产品约束排除。\n"
    u"2. **一致性（顶层目标）**：仍是 **22.58 / 14.25 / 13.47 dB**，"
    u"本轮**推进为零**，且 SmoothQuant 与重标定 encoding "
    u"双双判负（#152）⇒ **已无已知可执行的手段**"
    u"（仅剩 #51 AdaRound/AMP 未试，但预期撞约束）。\n\n"
    u"\U0001F7E2 **当前唯一在办事项：#86 不同比例**。"
    u"路线已换成**多图共享权重**（官方正解），"
    u"考察完成但**未开工**；7 个雷已排，"
    u"生死线是「PD 红线按已启用的图还是整个 context 算」。\n\n"
    u"---\n\n"
    u"**上一版一句话（2026-08-30，保留）**："
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    assert s.count(OLD_HEAD) == 1, "定位 §2.1 标题失败"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_s21D.md"))
    s = s.replace(OLD_HEAD, NEW_HEAD, 1)

    old_d = u"| **D** | 一致性补样本到 n≥5 |"
    assert s.count(old_d) == 1
    s = s.replace(
        old_d,
        u"| \U0001F7E2 **#86** | **不同比例（非 1:1）—— 当前唯一在办** | "
        u"最小验证 1.5h 宿主 + 10min 设备 | "
        u"路线 D（多图共享权重）。**考察完成、未开工**。7 个雷见台账 #86；"
        u"生死线是 PD 红线按图算还是按 context 算 |\n"
        u"| **D** | 一致性补样本到 n≥5 |", 1)
    io.open(MAIN, "w", encoding="utf-8", newline="").write(s)
    print("OK §2.1 与 §6 已更新")


if __name__ == "__main__":
    main()
