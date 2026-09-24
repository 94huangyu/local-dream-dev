# -*- coding: utf-8 -*-
"""登记 #149：SmoothQuant 的 G0 代价门结果。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

ROW = (
    u"| 149 | \U0001F7E2\U0001F7E2 **SmoothQuant（§6 的 E）：G0 代价门通过，"
    u"且收益是现网【没有】的** | \U0001F504 **进行中·G0 已过·待 §三 全模型模拟** "
    u"| 方案 `scripts/EXP_PLAN_SMOOTHQUANT.md`（判据线事前锁定，未改）。全程宿主，零设备。<br>"
    u"①**实测（part1b，59 个静态权重 MatMul，真实 X 与 W，alpha 扫 0~1）**："
    u"以**部署里仍是 uFxp_16 的 49 个**为准（另 10 个已浮点回退，我的基线高估了它们）："
    u"**sum E² 1358.9 → 54.9（−96.0%）**；改善**中位仅 +4.2%**，"
    u"但误差能量集中的那几个改善 65~93%：`mul_381` **28.775%→1.965%**、"
    u"`mul_406` 15.121%→1.641%、`mul_331` 14.766%→1.575%、`mul_456` 5.300%→1.482%。<br>"
    u"\U0001F7E2 **收益是加成不是重复**：这 5 个在部署里**仍是 uFxp_16**"
    u"（实查 `part1b_fp16_ovr.json`：旧机制下**缺席 = 浮点回退**，它们都在册 ⇒ 被量化）。"
    u"而 #114 实测把 `mul_406/431/456` 加进 FP16 白名单**反而让 L3 掉 0.51 dB**"
    u"（#117：旧机制浮点沿输入锥扩散，把必然溢出的张量一起拖进去）"
    u"⇒ ②**SmoothQuant 够得着 FP16 够不着的那批**——它不改 dtype，"
    u"无溢出风险、无浮点扩散、**不占内存、不吃速度**。<br>"
    u"\U0001F534 **我自己作废了第一版判据（约束 8）**：原用 "
    u"`净 = sqrt(激活step × 权重step)`，两项量纲不同、合成无依据，"
    u"且 alpha=1 时激活 step 必然塌缩（每通道除以自身 max）⇒ 报的 **97.8% 作废**。"
    u"改为直接量 MatMul 输出误差。**判据线（≥20% 过）一字未改**，改的只是「怎么量」。<br>"
    u"\U0001F534 **行采样被验证否决，拦下一个错误结论**："
    u"全部 4176 行 ⇒ 基线 2.80%、改善 +62.9%；仅前 512 行 ⇒ 基线 0.83%、改善 **+0.1%**。"
    u"因为那 14.77% 的误差**住在极少数行里**，前 512 行没抽到 ⇒ "
    u"「没有东西可修」。若图快直接用 512 行，会得出**错误的否决**。已入指南 §40。"
    u"⚠️ 不得引用 #39「用前 512 行没问题」——那件事对行分布不敏感，本件全部信号都在行分布里。<br>"
    u"⚠️ ③**尚未证明的（不得越说）**：这是 **MatMul 层输出误差**，"
    u"**不是端到端**。#121 已证 L1 无法换算成 L3；#97 已证 RmsNorm 有固有放大。"
    u"⇒ 下一步是 §三 全模型 QDQ 模拟（对照 mode H = 部署配置），仍在宿主。<br>"
    u"⚠️ ①**覆盖率划界**：part1b 共 73 个 MatMul，其中 **14 个是注意力的 A@B**"
    u"（两边都是动态激活，没有离线权重可折 scale）⇒ **SmoothQuant 对它们不适用**，"
    u"覆盖率是 59/73，不得说「全覆盖」 |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 148 | ")]
    assert len(i) == 1, "定位 #148 失败"
    assert u"| 149 | " not in s, "#149 已存在"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_149.md"))
    L.insert(i[0], ROW)
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("OK #149 已登记")


if __name__ == "__main__":
    main()
