# -*- coding: utf-8 -*-
"""记录 O3 的关闭（#147）+ 指南 §39.5 订正。

按 MAINLINE §0.4：内容锚定、先备份、改完跑 doc_audit / ledger_lint。
不用 heredoc（CLAUDE.md 编码陷阱）。
"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")
GUIDE = os.path.join(ROOT, "docs", "QNN_CONVERSION_GUIDE.md")
LOGS = os.path.join(ROOT, "logs")

LEDGER_ADD = (
    u"<br>\U0001F534\U0001F534 **已关闭：O3 不可用（2026-08-30 设备实测，阶段 2 的 G5 门失败）**。<br>"
    u"①**part1a 加 `O:3` 后根本装不进 DSP**：`createFromBinary -> 0x3ea`（=1002，"
    u"即 #57 的 unsigned PD 容量上限报错）。**失败时是第一段、什么都还没装、"
    u"MemAvailable 还有 7717 MiB** ⇒ 与内存无关，是**单 context 超了 PD 红线**。<br>"
    u"①**ION 代价无法从文件大小预测**（四段各不相同）："
    u"part1b 文件 +1.43% / ION **+1.8%**｜part2a +1.66% / **+3.6%**｜"
    u"part2b +2.50% / **+60.4%**（1812 → **2906 MiB**，**两次独立测量逐字节相同** `3417324 kB`）。<br>"
    u"①**即使放弃 part1a、只用三段 O3**：6964 vs CTRL 5755 MiB = **+1209 MiB** ⇒ "
    u"app 内 ION 峰值会从 9100 推到约 10300，**直接威胁 F 那 839 MiB 的余量**，"
    u"而换来的只是部分段的个位数收益。<br>"
    u"⇒ **按事前锁定的 G6：保 F，弃 O3**（F 实测 −21.2%/43.4 秒；O3 端到端③未测、粗估 −5~7%）。<br>"
    u"⚠️ **阶段 1 的结论仍然成立**：`O:3` 确实有约 −9% 的**段级计算**收益"
    u"（part1b，三次独立测量 0.939/0.890/0.930）。"
    u"关闭的是**在本配置下交付它**，不是“O3 无效”——"
    u"将来 context 变小（换量化配置/换模型）可重新评估，产物留档于 "
    u"`/d/ZImage_Work/p0_experiments/o3_all/`。<br>"
    u"\U0001F534 **顺带暴露我建图脚本的盲点**：`o3_build_all.py` 检查了建图日志里有无 "
    u"`available PD`，**但 PD 检查发生在设备端 `contextFinalize`，不在建图时**（#60 已记）"
    u"⇒ 建图报 OK、装载才炸。**建完必须上设备试装一次**，不能以建图成功为准。 |"
)

GUIDE_ADD = u"""

#### 39.5.1 🔴 **但 `O:3` 在本项目最终【不可交付】——两个建图时看不见的代价**（2026-08-30 设备实测）

阶段 1 只测了单段的**计算速度**，没测它对**内存与 PD** 的影响。上设备一试，两条全炸：

**① `O:3` 会把 context 顶过 unsigned PD 红线（#57），而建图时不报错**

①实测：part1a 加 `O:3` 后 `createFromBinary -> 0x3ea`（=1002，即 PD 容量报错）。
失败时是**第一段、什么都还没装、MemAvailable 还有 7717 MiB** ⇒ 与内存无关。

🔴 **这是一个通用陷阱**：`qnn-context-binary-generator` **建图成功不代表能装载**——
真正的 PD 检查在设备端 `contextFinalize`（同 §#60）。
⇒ **建完任何新配置的 context，必须上设备试装一次**，别拿建图日志当验收。
一个 `quadctx_probe`（§37.6）跑一遍就够，不必改 app。

**② `O:3` 的 ION 代价无法从文件大小预测，且可能极大**

①实测四段（文件增长 vs ION 增长）：

| 段 | 文件 | ION | 备注 |
|---|---|---|---|
| part1a | +3.20% | **装不进 PD** | — |
| part1b | +1.43% | **+1.8%** | — |
| part2a | +1.66% | **+3.6%** | — |
| part2b | +2.50% | **+60.4%** | 1812 → **2906 MiB**，两次独立测量**逐字节相同** |

⇒ ②**图优化级别会独立地改变运行时内存，与产物体积不成比例。**
体积 +2.34% 的一组产物，实际 ION 涨了 **+21%**（三段合计 5755 → 6964 MiB）。

**⇒ 实践结论**：`O` 不是一个「零风险的性能开关」。改它之后**必须重测 PD 与 ION**，
而不是只测速度。本项目因此放弃 O3，保留已交付的全常驻方案（§39.1）。
"""


def main():
    # --- 台账 #147 ---
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 147 | ")]
    assert len(i) == 1, "定位 #147 失败: %r" % i
    assert u"0x3ea" not in L[i[0]], "#147 似乎已记录过，不重复追加"
    shutil.copyfile(MAIN, os.path.join(LOGS, "MAINLINE.bak_o3closed.md"))
    L[i[0]] = L[i[0]][:-1].rstrip() + LEDGER_ADD
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("#147 已补记 O3 关闭")

    # --- 指南 §39.5.1 ---
    g = io.open(GUIDE, encoding="utf-8").read()
    anchor = u"⚠️ 措辞：以上是 `qnn-net-run` **离线单段**计时"
    assert g.count(anchor) == 1, "指南锚点定位失败"
    assert u"39.5.1" not in g, "§39.5.1 已存在"
    shutil.copyfile(GUIDE, os.path.join(LOGS, "GUIDE.bak_o3closed.md"))
    tail_start = g.index(anchor)
    g = g[:tail_start] + g[tail_start:].rstrip() + GUIDE_ADD
    io.open(GUIDE, "w", encoding="utf-8", newline="").write(g)
    print("指南 §39.5.1 已新增")


if __name__ == "__main__":
    main()
