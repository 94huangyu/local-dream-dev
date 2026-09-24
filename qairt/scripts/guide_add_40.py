# -*- coding: utf-8 -*-
"""指南新增 §40：研究 massive activation 时，行采样会把现象整个切掉。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
GUIDE = os.path.join(ROOT, "docs", "QNN_CONVERSION_GUIDE.md")

ADD = u"""

---

## 四十、🔴🔴 **研究「离群值主导」的量时，行采样会把现象整个切掉**（2026-08-30 实测）

### 40.1 事故形态（差点得出相反结论）

测 SmoothQuant 在 MatMul 输出上的收益，样本量大跑不动，于是想学 §#39 的
「取前 512 行」加速。**先用已知样本核对了一下**（约束 8），两组数字如下：

| 口径 | 基线 E 均值 | SmoothQuant 改善 |
|---|---|---|
| **全部 4176 行** | **2.80%** | **+62.9%** |
| 仅前 512 行 | 0.83% | **+0.1%** |

⇒ ②**结论完全相反**。若直接用 512 行，会得出「SmoothQuant 无效，关闭该方向」——
**一个错误的否决**，而且它看起来完全合理（数字干净、alpha 曲线平坦）。

### 40.2 机制

被研究的现象**本身就住在极少数行里**。本项目实测：某个 MatMul 的基线误差
**14.77%**，而同段其他 MatMul 只有 0.6~1.0% —— 那 14.77% 来自少量
massive-activation 行。取前 512 行没抽到它们 ⇒ 基线误差塌到 0.83%，
**于是「没有东西可修」，改善自然是 0。**

### 40.3 规则

🔴 **凡是研究「离群值主导」的量（激活量程、massive activation、钳位、SmoothQuant），
一律不得行采样 / 不得抽样降维**，除非先证明抽样后**基线量本身没有塌**。

判据很简单，三行代码：

```
基线量(全量) vs 基线量(抽样)   —— 两者不同量级 => 抽样非法，立刻停用
```

⚠️ **不要拿别处「用了 512 行也没问题」当依据**：#39 用前 512 行是在比较
两种定点实现的**算术**差异，那件事对行的分布不敏感；本节这件事**全部信号都在行分布里**。
⇒ ②**同一个加速手段在不同问题上合法性不同，必须逐个验证。**

⊕ 同族教训：约束 7（相对 L2 被离群值主导，低估达 48 倍）、
#67（激活钳 0.0261% 丢 98.75% 能量）—— 都是「少数元素携带绝大部分信号」。
**本项目在这一族上已经栽过三次，每次形态不同。**
"""


def main():
    s = io.open(GUIDE, encoding="utf-8").read()
    assert u"## 四十、" not in s, "§40 已存在"
    assert u"## 三十九、" in s, "§39 应当已在"
    shutil.copyfile(GUIDE, os.path.join(ROOT, "logs", "GUIDE.bak_before40.md"))
    io.open(GUIDE, "w", encoding="utf-8", newline="").write(s.rstrip() + ADD)
    t = io.open(GUIDE, encoding="utf-8").read()
    for k in (u"## 四十、", u"+62.9%", u"+0.1%", u"14.77%"):
        assert k in t, k
    print("OK 指南 §40 已新增")


if __name__ == "__main__":
    main()
