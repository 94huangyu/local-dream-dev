# -*- coding: utf-8 -*-
"""登记 #151：E 的端到端模拟结果 —— 单段结论被完全推翻。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

ROW = (
    u"| 151 | \U0001F534\U0001F534 **E 的端到端模拟：单段的 +22% 在全链上变成 −0.47 dB；"
    u"SmoothQuant 直接毁图（9.99 dB，纯噪声）** "
    u"| \U0001F504 **已实测·塌陷原因诊断中（可能是我的 bug）** "
    u"| ①**实测（`sq_e2e.py` 三臂，各四段建图 + 8 步 + VAE，共 5.9 小时宿主；"
    u"参考原点一字未改，用 §2.5 面板）**：<br>"
    u"　`H`（部署 encoding，基线）　L1 18.18%｜L2 33.51% 余弦 0.9449｜**L3 20.92 dB**<br>"
    u"　`S0`（重标定 encoding）　　 L1 21.39%｜L2 35.39% 余弦 0.9374｜**L3 20.45 dB（−0.47）**<br>"
    u"　`S5`（重标定+SmoothQuant α=0.5）L1 93.62%｜L2 105.46% **余弦 0.2346**｜**L3 9.99 dB（−10.46）**<br>"
    u"①**亲自看图（约束 1）**：`S0` 是**完全正常的图**（橘猫/木桌/挂画/绿植俱全）；"
    u"`S5` 是**纯噪声，无任何可辨识结构**。余弦 0.2346 < 0.3 ⇒ 按约束 7 **已毁，退出定量比较**。<br>"
    u"\U0001F534\U0001F534 **这把我的单段结论完全推翻了**：part1b 单段上"
    u"「重标定 +22%（留出集验证过）」「SmoothQuant 主体 +6~20%」，"
    u"端到端**两个都是负的**。⇒ ②**#121「L1 无法换算成 L3」在这里以最强形式出现**："
    u"不只是量级不同，是**符号反转**。<br>"
    u"\U0001F534 **我的留出集检验为什么没拦住**：`sq_holdout` 每一步都喂**干净的 FP32 输入**，"
    u"而真实链条是**把上一步的量化输出往下传**。我用 step-0 的激活 min-max 重标了量程，"
    u"在干净输入下不越界；在误差累积的真实链条里激活会漂出该量程被钳 "
    u"⇒ ③**这是假设，未验证**，但与 S0 的表现方向一致。⊕ 同类：#116（段间输入的血统）。<br>"
    u"⚠️ \U0001F534 **不得写「SmoothQuant 无效」（约束 R1：方向错 vs 执行错）**："
    u"part1b **单段**跑同一份 mode S 产物只是轻微变化（全量 −17%），**不是灾难**；"
    u"而另外三段的 mode S **从未单独验证过**。塌成噪声更像**我的实现有 bug**"
    u"（例如某段的激活被 Div(s) 了、而对应权重没被 Mul(s) 补偿 ⇒ 逐通道差一个 s 因子）。"
    u"**逐段诊断 `sq_perseg.py` 进行中，结论未出之前不得给 SmoothQuant 定性。**<br>"
    u"\U0001F7E2 **与诊断无关、已经确定的**：`S0`（只重标定 encoding、不动权重）"
    u"实现简单且图正常，端到端仍是 **−0.47 dB** ⇒ **「重标定 encoding」这条已可判负**，"
    u"不必再投入。<br>"
    u"⚠️ ③模拟 ≠ 设备（#123 出现过比值 1.826）；但本条是**模拟内部的三臂对照**，"
    u"该结论不依赖模拟与设备的换算 |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 150 | ")]
    assert len(i) == 1, "定位 #150 失败"
    assert u"| 151 | " not in s, "#151 已存在"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_151.md"))
    L.insert(i[0], ROW)
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("OK #151 已登记")


if __name__ == "__main__":
    main()
