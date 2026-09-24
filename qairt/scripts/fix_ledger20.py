# -*- coding: utf-8 -*-
"""修 #20 在两份台账里的矛盾。

LEDGER_CLOSED 说「已关闭·无收益…省掉一项大工程」，
MAINLINE §5 早已重分类为「在旧 proxy 下被否，不得写『已证明无效』」。
按回查协议，想重试这条路的人查的是 LEDGER_CLOSED —— 它说谎会直接劝退。

按约束 2：保留原文，标注推翻/重分类理由，不静默删除。
"""
import io
import os
import shutil

P = os.path.join("D:", os.sep, "LocalDreamZImage", "docs", "LEDGER_CLOSED.md")
BAK = os.path.join("D:", os.sep, "LocalDreamZImage", "logs", "LEDGER_CLOSED.bak_fix20.md")

OLD = (u"#20　massive activation / 通道迁移\n\n"
       u"**状态**：✅ **已关闭·无收益**\n\n")

NEW = (u"#20　massive activation / 通道迁移\n\n"
       u"**状态**：⚠️ **已重分类（在旧 proxy 下被否）"
       u"—— 不得当成「已证明无效」**\n\n"
       u"> \U0001F534 **2026-08-30 订正（两份台账曾互相矛盾）**："
       u"本条原先写「✅ 已关闭·无收益…省掉一项大工程」，"
       u"而 `MAINLINE.md` §5 早已重分类为「在旧 proxy 下被否」。\n"
       u"> **两处说法不一致，而回查协议要求想重试某条路时查的正是本文件**\n"
       u"> ⇒ 任何人想做 SmoothQuant/通道迁移，会在这里读到「无收益」然后放弃。\n"
       u"> 同类事故见 #73（同一件事两处各写一遍，必有一处说谎）。\n"
       u">\n"
       u"> **重分类理由**：原判决的依据是**量化表示误差**"
       u"（主体通道 2.90% / CPU 4.56% / HTP 73.62%），\n"
       u"> 而 **#52 实测证明表示误差不是 HTP 误差的预测量**"
       u"（同一改动 CPU Δcos −0.0014、HTP −0.0526，**37.6 倍**）。\n"
       u"> #9（min-max 校准）正是因同一理由被动摇。\n"
       u">\n"
       u"> ⚠️ **不是说它有效** —— 是说「无效」这个结论的**尺子不合格**，"
       u"结论得重新判。\n"
       u"> ⊕ 且 #67 实测支持它的机理：**激活是离群值主导的**"
       u"（钳 0.0261% 丢 **98.75%** 能量），**而权重不是**"
       u"（裁 0.0520% 只丢 1.12%）\n"
       u"> ⇒ 「把难度从激活挪到权重」在机理上是成立的，且 per-channel scale "
       u"可折进权重 ⇒ **速度中性**。\n\n"
       u"—— 以下为原文，逐字保留 ——\n\n"
       u"~~**状态**：✅ **已关闭·无收益**~~\n\n")


def main():
    s = io.open(P, encoding="utf-8").read()
    assert s.count(OLD) == 1, "锚点未唯一命中，停手（禁止不带 count 的 replace）"
    assert u"2026-08-30 订正" not in s, "似乎已订正过"
    shutil.copyfile(P, BAK)
    s2 = s.replace(OLD, NEW, 1)
    assert len(s2) > len(s)
    io.open(P, "w", encoding="utf-8", newline="").write(s2)
    t = io.open(P, encoding="utf-8").read()
    assert u"37.6 倍" in t and u"98.75%" in t and u"逐字保留" in t
    print("OK #20 已订正；备份 %s" % BAK)


if __name__ == "__main__":
    main()
