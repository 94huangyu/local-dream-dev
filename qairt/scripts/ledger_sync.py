# -*- coding: utf-8 -*-
"""同步过期状态。台账自己说谎会把新线程送去重做已完成的事（#132 的教训）。"""
import io
import os
import re
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

# 编号 -> (旧状态片段, 新状态)
FIX = {
    "149": (u"🔄 **进行中·G0 已过·待 §三 全模型模拟**",
            u"✅ **已关闭（2026-08-31）·被 #152 取代**："
            u"§三全模型模拟已跑完，SmoothQuant 判负"),
    "150": (u"🔄 **进行中·宿主已验·待端到端**",
            u"✅ **已关闭（2026-08-31）·被 #152 取代**："
            u"端到端 **−0.47 dB**，且段级 α=0 ≈ 基线 ⇒ 重标定 encoding 判负"),
    "151": (u"🔄 **已实测·塌陷原因诊断中（可能是我的 bug）**",
            u"✅ **已关闭（2026-08-31）**：诊断完成，**确是我的 bug**"
            u"（转置存放权重未被 Mul(s)）；修好后方向仍判负，见 #152"),
    "129": (u"🔶 **未查·待 15 分钟设备测量**",
            u"✅ **已关闭**：该测量早已完成（#132 斜率截距法），"
            u"并据此交付了方案 A 与 B（累计 271→160.87 s）。"
            u"⚠️ 本条的「步数已到下限」已于 2026-08-31 订正为误读，见 #153"),
    "146": (u"🔶 **未查·不在关键路径**",
            u"🔶 **未查·已降级·不在关键路径**（#145 实测四段本就装得下 ⇒ F 不需要它；"
            u"转更大模型时可取用）"),
    "98": (u"🔶 **未查·最高优先级**",
           u"🔶 **未查·⚠️ 但其落地手段已逐条耗尽**："
           u"#53 代价门否决、#48 收益 0.981、SmoothQuant 判负（#152）、"
           u"重标定 encoding 判负（#152）⇒ **结论仍成立，但已无已知可执行的手段**"),
}


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_sync.md"))
    L = s.split("\n")
    n = 0
    for num, (old, new) in FIX.items():
        idx = [k for k, l in enumerate(L) if l.startswith(u"| %s | " % num)]
        if len(idx) != 1:
            print("  ⚠️ #%s 定位到 %d 行，跳过" % (num, len(idx)))
            continue
        if old not in L[idx[0]]:
            print("  ⚠️ #%s 旧状态未命中，跳过（可能已改）" % num)
            continue
        L[idx[0]] = L[idx[0]].replace(old, new, 1)
        n += 1
        print("  ✅ #%s 状态已同步" % num)
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    print("共同步 %d 条" % n)


if __name__ == "__main__":
    main()
