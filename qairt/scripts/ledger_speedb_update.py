# -*- coding: utf-8 -*-
"""把 EXP_PLAN_SPEEDB 的实测结论写入台账。

纪律（MAINLINE §0.4）：
  - 行级、内容锚定；禁止字符偏移插入；禁止不带 count 的 str.replace
  - 「保留原文」按字节切片拷贝，不凭记忆重打，并断言切片同时存在于新旧文件
  - 先 cp 备份（调用方已做）
"""
import io, os, sys

ROOT = r"D:\LocalDreamZImage"
MAIN = os.path.join(ROOT, "MAINLINE.md")
CLOSED = os.path.join(ROOT, "docs", "LEDGER_CLOSED.md")


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


main = read(MAIN)
lines = main.split("\n")

# ---------- 1. 取出 #133 整行（字节切片，不重打） ----------
idx133 = [i for i, l in enumerate(lines) if l.startswith("| 133 | ")]
assert len(idx133) == 1, "#133 行数异常: %r" % idx133
row133 = lines[idx133[0]]
assert "run-to-run" in row133, "#133 行内容不符预期"
print("取出 #133 原行 %d 字符" % len(row133))

# ---------- 2. 从 §4 删除该行 ----------
del lines[idx133[0]]

# ---------- 3. 在 §4 加入 #145（⛔受阻，活的线索留在本表） ----------
row145 = (
    "| 145 | ⛔ **#132 方案 B（part2 也常驻）被内存门否决 —— 实测，非估算** "
    "| ⛔ **受阻·实测（2026-08-29）·不等于已关闭** "
    "| ①**实测**（`scripts/speedb_measure.sh`，真实 app 全链路，B2/B3 两次逐位相同）："
    "步循环内 ION 是干净的两电平方波 —— **5212 MiB**（part1a+1b 常驻、part2 未加载，79 个采样点）"
    "↔ **7148~7202 MiB**（+ 一个 part2 半段）⇒ **每个 part2 半段占 ION 1936~1990 MiB**。<br>"
    "①**峰值时刻实测 `MemAvailable` 仅 855~980 MiB、`MemFree` 仅 59~64 MiB** "
    "⇒ 方案 B 还需额外约 1936 MiB，**是峰值可用量的两倍**，缺口约 1000~1100 MiB ⇒ 否决。<br>"
    "🔴 **本否决不依赖 #16 的 7300 MB**：实测方案 A 自己就是 **7202 MiB**，"
    "若拿 7300 当 ION 阈值，已交付且稳定运行的方案 A 也会被判死 ⇒ ②**#16 那个数不是 ION 口径，"
    "拿它做阈值是无效操作化**（约束 8）。否决只用两个本轮实测数：额外需求 1936 vs 峰值可用 855。<br>"
    "①**顺带标定**：DSP/ION 占用 ≈ context **文件大小的 1.42×**（part2b 1398 MiB 文件 → 约 1990 MiB ION）"
    "⇒ 与 §3.4 的 PD 估算（1887/1927 MB）吻合，**而与文件大小不符** ⇒ 此前用文件大小做内存预算是错的。<br>"
    "⚠️ **⛔ 不等于已关闭**（约束 6）：若将来段变小（更细切分 / 更小 context），需用同一把尺子重测。"
    "方案与判据：`scripts/EXP_PLAN_SPEEDB.md` |"
)
anchor4 = "| 143 | "
i143 = [i for i, l in enumerate(lines) if l.startswith(anchor4)]
assert len(i143) == 1, "§4 锚点 #143 异常: %r" % i143
lines.insert(i143[0], row145)
print("已在 §4 插入 #145（位于 #143 之前）")

# ---------- 4. 在 §5 加入 #133 / #144 的一行索引 ----------
i_head = [i for i, l in enumerate(lines) if l == "| # | 线索 | 结论 |"]
assert len(i_head) == 1, "§5 表头异常: %r" % i_head
sep = i_head[0] + 1
assert lines[sep].startswith("|---"), "§5 表头下一行不是分隔行: %r" % lines[sep]

idx144 = (
    "| 144 | 🟢🟢 **速度测量协议已标定（G1 0.152%）；且发现内存一直量错了口径** "
    "| ✅ **已关闭·设备实测（2026-08-29）** 全文见 `docs/LEDGER_CLOSED.md` |"
)
idx133 = (
    "| 133 | ~~同一 APK 同一配置的生图耗时 run-to-run 变异约 8%~~ "
    "| ✅ **已关闭·已证否（2026-08-29）**：受控协议下实测 **0.152%**，那 8% 来自 prompt/热/后台三者同时变化 "
    "全文见 `docs/LEDGER_CLOSED.md` |"
)
lines.insert(sep + 1, idx133)
lines.insert(sep + 1, idx144)
print("已在 §5 插入 #144 / #133 一行索引")

write(MAIN, "\n".join(lines))

# ---------- 5. 完整证据写入 LEDGER_CLOSED（含 #133 原行的字节切片） ----------
closed = read(CLOSED)
assert "#133" not in closed.split("\n---\n")[0][:200], "意外：CLOSED 开头已有 #133"

block = (
    "\n---\n\n"
    "### #144\u3000🟢🟢 **速度测量协议已标定；且发现内存一直量错了口径**\n\n"
    "**状态**：✅ **已关闭·设备实测（2026-08-29）**。方案 `scripts/EXP_PLAN_SPEEDB.md`，"
    "测量入口 `scripts/speedb_measure.sh`（B1/B2/B3/M1 强制走同一个脚本，避免口径副本说谎）。\n\n"
    "**协议定义**：`am force-stop` → 静置到 NPU 全部 `nsph*` 分区 ≤ `T_idle+3 °C` → 重启 app 等 8081 监听"
    "（此段不计时）→ 固定 prompt / seed=777 发一次 `/generate`；**主时钟 = `curl time_total`**。\n\n"
    "①**实测（n=3）**：B1 **202.486 s**（试跑，旧装置）／B2 **204.094 s**／B3 **204.404 s**。\n"
    "**G1 判据 `|B2−B3|/mean = 0.152%`（阈值 3%）⇒ 🟢 PASS。** 含试跑三点极差 1.919 s = **0.94%**。\n"
    "三张图 `png sha256` **完全相同**（`5ce17d9a7d1372055666b0b54cf46b8c59b5d7fdb01bda4b1721f38224a3a21c`）；"
    "B2/B3 的 ION 峰值 **逐字节相同**（7,375,036 kB）。\n\n"
    "🔴 **由此证否 #133 的「8% run-to-run 噪声」**：受控协议下是 **0.152%**，低 50 倍。"
    "②那 8% 不是尺子的噪声，而是当时三个样本的 **prompt / 热状态 / 后台负载同时不同**"
    "（#133 自己已标注「不是单变量对照」）⇒ ②**「噪声」这个词用错了：那是未控制的系统性差异。**\n\n"
    "🔴🔴 **量测装置本身先出过错，被步 0 拦下（约束 8 的直接价值）**：\n"
    "- **首版用进程 `VmRSS` 量内存 —— 量的是错的量。** B1 实测：后端进程生成全程 RSS 仅 **~200 MiB**，"
    "峰值 **2356 MiB** 只出现在「把 `.bin` 读进 CPU 缓冲区」那一瞬（≈ part1a 文件 2263 MiB），"
    "交给 DSP 后立即释放 ⇒ **context 常驻在 DSP/ION 侧，进程 RSS 根本看不见它**。\n"
    "- **正确口径是 `/proc/meminfo` 的 `IonTotalUsed`**（本机可读，非 root）。改用后自检门 0c 通过："
    "ION 增量 **6933 MiB** ⇒ 确实看得见 context。\n"
    "- 附带：`VmHWM` 出现过 **小于**瞬时 `VmRSS`（2,412,344 < 2,412,992 kB）。"
    "原因是内核 `update_hiwater_rss()` 惰性更新 ⇒ **`VmHWM` 会滞后，不能当唯一峰值来源**。\n\n"
    "①**顺带实测的热行为（修正 #141 的机制描述）**：**单张图内 NPU 就冲到 103.4 °C**"
    "（B2/B3 峰值逐位相同），起点仅 48~50 °C；步循环内在 **60~103 °C** 间剧烈摆动。\n"
    "⇒ ②#141 的「连续 6 张爬升 +16%」**不是 NPU 慢慢热起来**（单张已打满热区），"
    "累积的应是外壳/电池等慢热体。③**该推论未直接实测**（没测外壳/电池温升曲线）。\n"
    "①**冷机恢复很快**：58.8 °C → 46.1 °C 约 6 分钟（30 秒一采，阻塞式等待）。\n\n"
    "⇒ **对后续的直接含义**：§6 的 B（`O`/`dlbc` 测速）可以直接用这把尺子；"
    "**冷机第一张**这个口径测的不是用户连续使用的稳态体验，引用时必须注明。\n\n"
    "---\n\n"
    "### #133\u3000⚠️ **同一 APK 同一配置的生图耗时 run-to-run 变异约 8%**\n\n"
    "**状态**：✅ **已关闭·已被 #144 证否（2026-08-29）**。以下为**精简前原行逐字保留**（字节切片，未重打）：\n\n"
    + row133 + "\n"
)
write(CLOSED, closed + block)
print("已写入 docs/LEDGER_CLOSED.md，追加 %d 字符" % len(block))

# ---------- 6. 断言：原行在新旧两处都在 ----------
assert row133 in read(CLOSED), "🔴 #133 原行未能在 CLOSED 中找回"
assert row133 not in read(MAIN), "🔴 #133 原行仍留在 MAINLINE §4"
print("断言通过：#133 原行已从 §4 移出、并逐字保存在 LEDGER_CLOSED")
