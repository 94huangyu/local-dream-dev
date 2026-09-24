# -*- coding: utf-8 -*-
"""把 #86 的全部考察结果写进台账：A 路线作废、D 路线（多图共享权重）成为方案。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

NEW = (
    u"| 86 | \U0001F7E2\U0001F7E2 **不同比例（非 1:1）—— 用户点名的产品需求；"
    u"路线已换成【多图共享权重】** | \U0001F536 **考察完成·方案待定·未开工（2026-08-31）** "
    u"| \U0001F534\U0001F534 **我先做后查，白干了一轮（约 5 小时 + 14.7 GB，已回收）**。"
    u"教训写在最后。<br>"
    u"**四条路径，逐条有证据**：<br>"
    u"　**A 每比例一套 context** ❌ **已实测否决**：zstd 以基座为字典压 4:3 的 part1b = "
    u"**1236.3 MB（84.0%）**，而**不带字典也是 1236.3 MB —— 字典零收益**"
    u"（级别 19 也只到 83.7%，耗时 558 s）⇒ 「权重跨比例相同所以增量小」这个假设"
    u"**在我们的产物上是错的**（HTP 编译后的 context 按形状重排权重/调度，二进制处处不同）。"
    u"⇒ 每比例 **6.8 GB 设备存储 + 5.7 GB 下载**，5 个比例 34 GB，**不可行**。"
    u"⚠️ 划界：local-dream 的 `.patch` 机制在 **SD1.5 的 unet.bin** 上有效，"
    u"**不得因我们这里无效就说官方机制没用**。<br>"
    u"　**B 动态形状** ❌ HTP 要求静态形状。<br>"
    u"　**C inpaint padding**（SDXL/Anima 现用的）\U0001F7E1 **机制已读通**："
    u"`Pipeline.hpp:1168-1172` 每步 `latents = orig_noised*(1-mask) + latents*mask`，"
    u"\U0001F7E2 **与 CFG 无关**（我先前担心的「CFG=0 不行」不成立），与 flow-matching 的 "
    u"`add_noise` 兼容；合成底图的 VAE 编码**按目标尺寸缓存到磁盘**"
    u"（`Pipeline.hpp:445-460`）⇒ 理论上可在宿主预算好随包下发。"
    u"\U0001F534 但缺 **VAE encoder**（契约九个图只有 decoder）与 **inpaint 通路**"
    u"（`PipelineZImage.hpp:62` 直接 throw），且 **8 步蒸馏模型被强制黑边是训练分布外**，"
    u"③画质未知。<br>"
    u"　**D 多图共享权重** \U0001F7E2\U0001F7E2 **官方正解，本轮新发现** —— 见下。<br>"
    u"\U0001F7E2 **D 的证据（官方文档 / SDK 头文件 / 本地工具，非推测）**：<br>"
    u"　① 本地 `qnn-context-binary-generator --help` 原文：*“`--dlc_path` … "
    u"To **compose multiple graphs in the context**, use comma-separated list of DLC files”* "
    u"⇒ **我们正在用的 `--dlc_path` 工作流就支持**。<br>"
    u"　② `QnnHtpContext.h:39` `QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED`（bool）。<br>"
    u"　③ AI Hub Linking 文档原文：*“Link jobs allow multiple models to be combined into a "
    u"single deployable asset with multiple graphs that **share weights** … **A common use case is "
    u"to support different instantiations of the same network, such as different input shapes**”* "
    u"⇒ **「同一网络的不同输入形状」是官方点名的典型用例**。"
    u"唯一限制：*“both inputs will need to be **traced separately**”* —— 我们本来就逐形状做手术，满足。<br>"
    u"　④ \U0001F7E2\U0001F7E2 **`QnnContext.h:122-126` `QNN_CONTEXT_CONFIG_ENABLE_GRAPHS`：**"
    u"*“names of the graphs to **deserialize** from a context binary. All graphs are enabled by default”* "
    u"⇒ **可只解开当前比例那一个图 ⇒ 运行时 ION 与今天相同**。这条是 D 可行的关键。<br>"
    u"　⑤ `QnnContext.h:127` `GRAPH_RETENTION_ORDER` 支持多图切换（我们不需要）。<br>"
    u"\U0001F534 **排到的 7 个雷（开工前必须逐条处理）**：<br>"
    u"　**1** `QnnModel.hpp:1114-1123` **无条件遍历 `m_graphsCount` 取回所有图** ⇒ "
    u"只启用一个时会对被禁用的图报错。**必改**。<br>"
    u"　**2** \U0001F534\U0001F534 **PD 红线（3.3 GB，#57）按「已启用的图」算还是按整个 context 算？"
    u"—— 无文档答案，这是 D 的生死线。** 若按整体算，5 个比例必然击穿。<br>"
    u"　**3** revision history：*“DLC: Fixed issues … when **per-channel block quantization** "
    u"is employed on a **multi-graph DLC**”* —— 我们正用 `--use_per_row_quantization`，需验。<br>"
    u"　**4** ①实测：**独立建出的 context 之间零共享** ⇒ 必须**一次 link 建成**，不能事后合并。<br>"
    u"　**5** 契约 schema 现在是**一图一文件**（`actual_filename` ↔ `internal_graph_name` 一一对应），"
    u"多图需允许多条目指向同一文件。<br>"
    u"　**6** 建图耗时：①实测单图 part1a **59.6 分钟**（比 #134 记的 8.9~19.3 分钟/段高得多，"
    u"③原因未查），多图可能远超。<br>"
    u"　**7** 官方推荐组合是 **`udma=on + weight-sharing + multicore without lora`**，与我们配置不同，未验。<br>"
    u"➡️ **建议的下一步（未开工，待用户定）**：只建 **1:1 + 4:3 两个图**的最小验证，"
    u"一次排掉雷 2/3/4 —— 测 ①文件体积（验证权重真共享）②PD 是否通过 "
    u"③`ENABLE_GRAPHS` 只解一个时的 ION。约 **1.5 小时宿主 + 10 分钟设备**。<br>"
    u"\U0001F534\U0001F534 **本轮最大的教训（比结论重要）**：`--dlc_path` 的多图语法就写在"
    u"**工具的 `--help` 里**；`Model.kt` 那条 *“SDXL, Anima, and **Z-Image** render on a fixed "
    u"1024 canvas and reach non-1:1 outputs via aspect-ratio inpaint padding”* 注释"
    u"**我读过那个文件**（引用 `usesFixedCanvas` 那次），只看进了后半句「SD1.5 用 resolution patches」。"
    u"⇒ ②**两条都是「打开就能看到」的，我却先动手做了一轮图手术。**"
    u"⊕ 与 #148（权威模块落后于交付）、约束 1（文档描述是线索不是事实）同一形态：**信息在手却没连起来**。 |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 86 | ")]
    assert len(i) == 1, "定位 #86 失败"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_86D.md"))
    L[i[0]] = NEW
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    t = io.open(MAIN, encoding="utf-8").read()
    for k in (u"ENABLE_GRAPHS", u"1236.3 MB", u"59.6 分钟", u"7 个雷"):
        assert k in t, k
    print("OK #86 已重写为 D 路线方案 + 7 个雷")


if __name__ == "__main__":
    main()
