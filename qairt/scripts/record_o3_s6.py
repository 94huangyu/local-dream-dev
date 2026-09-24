# -*- coding: utf-8 -*-
"""把 §6 的 B 行标记为已关闭（O3 阶段 2 失败）。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

NEW = (
    u"| ~~**B**~~ | ✅ **已关闭（2026-08-30）**：`dlbc` 无效；"
    u"`O:3` 段级计算 **−9%** 但**不可交付** | 已完成 | "
    u"阶段 1（宿主+离线测速）\U0001F7E2：O3 = 0.910× CTRL（三次 0.939/0.890/0.930）｜"
    u"DLBC = 1.008× ❌。<br>阶段 2（设备）❌ **G5 内存门失败**："
    u"**part1a 加 `O:3` 后 `createFromBinary -> 0x3ea`（PD 容量上限，#57）**，"
    u"失败时是第一段、MemAvailable 还有 7717 MiB ⇒ 与内存无关。"
    u"且 ION 代价无法从体积预测（part2b 体积 +2.5% 而 ION **+60.4%**）。"
    u"按事前锁定的 **G6：保 F、弃 O3**。详见 #147 |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| **B** | `O`/`dlbc`")]
    assert len(i) == 1, "定位 §6 B 行失败: %r" % i
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_s6B.md"))
    L[i[0]] = NEW
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("§6 B 行已标记关闭")


if __name__ == "__main__":
    main()
