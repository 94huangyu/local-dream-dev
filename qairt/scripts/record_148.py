# -*- coding: utf-8 -*-
"""登记 #148：权威源模块在 Tier 2 交付后没跟上，会导致第四次取错源。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

ROW = (
    u"| 148 | \U0001F534 **权威源模块 `canonical_sources` 在 Tier 2 交付后没更新 "
    u"—— 差点成为「取错源」第四次** | ✅ **已修（2026-08-30）** "
    u"| ①**事实**：Tier 2（08-29）把部署换成 **L=80**，设备上跑的是 "
    u"`*_clip_L80.onnx` 产出的 context；而 `canonical_sources.ALL` 的表只记到 **L=32**，"
    u"`onnx_for('part1b')` 返回 `transformer_part1b_clip.onnx`（L=32）。"
    u"⇒ **任何人把它读成「当前部署的图」都会取错源。**<br>"
    u"\U0001F534 **形态与 #61 / #136 / #138 完全相同**：不是不知道规矩，是"
    u"**交付改了、权威表没跟上**。而这个模块正是为防前三次而建的 "
    u"⇒ ②**「建了权威模块」不等于「它会自动保持正确」**，它自己也需要一条更新纪律。<br>"
    u"①**为什么这次抓到了**：做 SmoothQuant（#149/E）要选源 ONNX，"
    u"顺手 `ls` 发现盘上同时有 `transformer_part1b_clip.onnx` 与 "
    u"`transformer_part1b_clip_L80.onnx` ⇒ **是靠列目录发现的，不是靠模块自检**。<br>"
    u"①**修法（非破坏性）**：新增 `DEPLOYED_L = 80` 常量 + `onnx_for(seg, variant, L=None)` "
    u"的 `L` 参数。**没有改 `deployed` 的原语义** —— 因为 `dit_seq_surgery.py` 有 "
    u"`stem == ALL[seg][...]` 的断言，改了会把手术工具搞坏（已回归验证四段两 variant 全过）。"
    u"`describe()` 现在会同时报「L=32 原始表」与「当前部署图」。<br>"
    u"⚠️ **仍未做**：③模块没有**自检**能发现「表落后于交付」。"
    u"真正的根治是让 `DEPLOYED_L` 从**契约**（`final_qnn_contract.*.json` 的 `text_seq_len`）"
    u"派生，而不是手写常量 —— 同 #73「让副本不存在」。本次只补了常量，"
    u"**换代时仍要靠人记得改这一行**，这是已知缺口 |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 147 | ")]
    assert len(i) == 1, "定位 #147 失败"
    assert u"| 148 | " not in s, "#148 已存在"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_148.md"))
    L.insert(i[0], ROW)
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("OK #148 已登记")


if __name__ == "__main__":
    main()
