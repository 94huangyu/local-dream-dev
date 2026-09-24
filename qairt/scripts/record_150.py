# -*- coding: utf-8 -*-
"""登记 #150：重标定 encoding 的收益（留出集验证通过），以及 SmoothQuant 被搁置。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

ROW = (
    u"| 150 | \U0001F7E2\U0001F7E2 **意外收获：把 encoding 用真实激活重标定，"
    u"part1b 单段误差降 ~22%（留出集验证通过）；SmoothQuant 反而被它比下去** "
    u"| \U0001F504 **进行中·宿主已验·待端到端** "
    u"| ①**实测（`sq_holdout.py`，part1b 单段 QDQ 模拟，标定用 step0、留出用 step1/step4）**：<br>"
    u"　**① 重标定 encoding（H → S a=0，s≡1 故 Div 是恒等，唯一变量就是 encoding）**：<br>"
    u"　　step0 全量 **+22.8%** 主体 +15.1%｜step1 **+22.2%** +17.5%｜step4 **+22.6%** +18.7%<br>"
    u"　⇒ \U0001F7E2 **跨步几乎不衰减，两个口径都同意** ⇒ **不是 train-on-test**。<br>"
    u"　**② SmoothQuant 净效应（S a=0 → S a=0.5，两臂共用同一份 encoding）**：<br>"
    u"　　step0 全量 −21.3% 主体 **+20.5%**｜step1 −17.5% **+9.4%**｜step4 −18.1% **+6.4%**<br>"
    u"　⇒ \U0001F534 **主体收益从 20.5% 衰减到 6.4%**（`s` 在 step0 上算的，典型过拟合特征），"
    u"且**全量口径上始终为负**。<br>"
    u"\U0001F534 **我一度以为 ① 是泄漏**（统计与评测都用 testB/s0），"
    u"专门补了留出集才判清 —— **担心是对的，结论是反的**：泄漏在 ② 上，不在 ① 上。<br>"
    u"⇒ **判决：追 ①，搁置 ②。** 三条依据："
    u"(a) ① 跨步稳、② 衰减且两口径打架；"
    u"(b) ① **只改 encoding、不动权重** ⇒ 不涉及 #52 的 **37.6 倍**权重敏感度，"
    u"而 ② 的本质就是往权重挪难度，**宿主模拟会系统性低估它的设备代价**；"
    u"(c) ① 零代码改动，重量化 **0.1 分钟/段**（#134）+ 建 context ~70 分钟。<br>"
    u"⚠️ \U0001F534 **① 尚未过的关键一关（跨 prompt）**：留出的是**同一 prompt 的不同去噪步**，"
    u"不是不同 prompt。我的 min-max 是在一条 prompt 的激活上取的，"
    u"**换 prompt 若超出该量程就会被钳** —— 这正是 #53/#67 的教训"
    u"（percentile 砍量程换分辨率 ⇒ 净损害）。**上机之前必须先做跨 prompt 检验。**<br>"
    u"⚠️ ③**尚未验证**：单段 ≠ 端到端（#121：L1 无法换算 L3）；宿主模拟 ≠ 设备（#123 曾出现比值 1.826 的离群）。<br>"
    u"⚠️ ①**度量口径**：`unified` 集中度实测 **0.9983**（独立复现了 CLAUDE.md 记的 99.83%）"
    u"⇒ 按约束 7 **全量口径不得用于描述该张量整体**，决策看主体；两个口径在 ① 上一致、在 ② 上相反。<br>"
    u"⊕ **本轮我把判据的操作化写错三次**：G0 的几何平均（无依据、alpha=1 必塌缩）、"
    u"行采样（把离群行整个切掉，见指南 §40）、以及本条的全量口径。"
    u"前两次靠「先用已知样本验证」抓出，第三次靠回头再想一遍 —— **没有一次是事前想周全的** |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 149 | ")]
    assert len(i) == 1, "定位 #149 失败"
    assert u"| 150 | " not in s, "#150 已存在"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_150.md"))
    L.insert(i[0], ROW)
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("OK #150 已登记")


if __name__ == "__main__":
    main()
