# -*- coding: utf-8 -*-
"""G-T 判据 · 逐元素比较官方 transformer 与我们的四段 ONNX 链（step-0 噪声预测）。

## 为什么不能只看 std
`parity_padding.py` 给出我们链的 `noise_std = 1.5368`，官方日志是 `1.5485`。
**std 相同不等于张量相同** —— 两个完全不同的张量可以有同一个 std。
本脚本做逐元素比较，这才是 `EXP_PLAN_OFFICIAL_PARITY.md` §4 的判据量。

## 三点归因（各臂的产出文件）
| 臂 | 文件 |
|---|---|
| 官方 transformer + 官方 caption | `parity/noise_s0_official_offcap.raw` |
| 官方 transformer + **我们的** caption | `parity/noise_s0_official_ourcap.raw` |
| 我们的四段链 + 我们的 caption | `parity/noise_s0_ours.raw` |

- 「官方+我们caption」vs「我们的链」 ⇒ **transformer 实现差异**（caption 已对齐，单变量）
- 「官方+官方caption」vs「官方+我们caption」 ⇒ **文本编码器差异的下游影响**

## 判据（`EXP_PLAN_OFFICIAL_PARITY.md` §4，事前锁定）
  相对 L2：< 1% 🟢 ／ 1%~5% 🟡 ／ >= 5% 🔴
  ⚠️ 全量与主体两个口径都报（约束 7；G-E 已暴露只写「相对 L2」不写口径的缺陷）。

## 标尺来历（§3.1）
step-0 噪声预测的相对 L2：**15.99% ⇒ 成图完好｜44.80% ⇒ 清晰的猫｜47.07% ⇒ 橙色色块**；
量化造成的同口径误差 21~44%。

用法: python scripts/parity_compare.py
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "parity")
ARMS = {
    "官方+官方caption": "noise_s0_official_offcap.raw",
    "官方+我们caption": "noise_s0_official_ourcap.raw",
    "我们的链":          "noise_s0_ours.raw",
}


def load(name):
    p = os.path.join(W, ARMS[name])
    if not os.path.isfile(p):
        return None
    return np.fromfile(p, np.float32).astype(np.float64)


def cmp(a, b, tag):
    """a = 参考，b = 待测。"""
    rel = np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    e = a * a
    k = max(1, int(round(0.01 * e.size)))
    conc = float(np.sort(e)[-k:].sum() / max(e.sum(), 1e-30))
    m = np.abs(a) <= np.quantile(np.abs(a), 0.99)
    relb = np.linalg.norm(b[m] - a[m]) / max(np.linalg.norm(a[m]), 1e-30)
    cosb = float(a[m] @ b[m] / max(np.linalg.norm(a[m]) * np.linalg.norm(b[m]), 1e-30))
    print("  %-34s 全量 %7.4f%% (cos %.6f) ｜ 主体 %7.4f%% (cos %.6f) ｜ 集中度 %.2f%%"
          % (tag, rel * 100, cos, relb * 100, cosb, conc * 100))
    return rel, relb, conc


def verdict(rel):
    return "🟢 等价" if rel < 0.01 else ("🟡 有差异" if rel < 0.05 else "🔴 实质不一致")


def align_sign(data):
    """🔴 两侧的『噪声』符号约定不同，必须先对齐，否则会报出假警报。

    官方（`official_pipeline_run.py`）：`noise_pred = -model_out`，再交给
    `sched.step` 做 `sample + (σ_next − σ)·noise_pred`。
    我们（`t2_fp32_ref.py`）：直接 `nxt = lat − dt·noise`，其中 `dt = σ_next − σ` 是**负数**。
    ⇒ 两者要给出同一个 `nxt`，必然有 `noise_ours = −noise_official`。

    **已知样本佐证（约束 8）**：step-0 之后的 `latents_std`——
    官方日志 **0.9516**，我们 **0.9517** ⇒ 两条链是一致的，
    所以 cos ≈ −0.99 只能是约定差，不是数值分歧。
    第一版脚本没做这一步，报出 198%『实质不一致』—— **那是我的口径错，不是流水线错。**
    """
    ours = data["我们的链"]
    off = data["官方+我们caption"]
    c = float(ours @ off / max(np.linalg.norm(ours) * np.linalg.norm(off), 1e-30))
    if c < 0:
        print("🔴 检出符号约定相反（cos=%.4f）⇒ 把官方两臂取负后再比（理由见 align_sign 文档）\n" % c)
        for k in ("官方+我们caption", "官方+官方caption"):
            data[k] = -data[k]
    return data


def main():
    data = {k: load(k) for k in ARMS}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        raise SystemExit("🔴 缺产物：%s —— 先跑 parity_transformer.py / parity_padding.py"
                         % ", ".join(missing))
    for k, v in data.items():
        print("%-18s n=%d  std=%.4f  |max|=%.4f" % (k, v.size, v.std(), np.abs(v).max()))
    print()
    data = align_sign(data)

    print("\n=== G-T 主判据：transformer 实现（caption 已对齐 ⇒ 单变量）===")
    rel, relb, _ = cmp(data["官方+我们caption"], data["我们的链"],
                       "官方(我们caption) vs 我们的链")
    print("  ⇒ 全量 **%s**（%.4f%%）｜主体 **%s**（%.4f%%）"
          % (verdict(rel), rel * 100, verdict(relb), relb * 100))

    print("\n=== 参照 1：文本编码器差异的下游影响（同一个 transformer）===")
    r2, r2b, _ = cmp(data["官方+官方caption"], data["官方+我们caption"],
                     "官方(官方cap) vs 官方(我们cap)")
    print("  ⇒ TE 的 0.87%%/1.86%% 输入差异，在 step-0 输出上放大成 **%.4f%%（全量）**" % (r2 * 100))

    print("\n=== 参照 2：端到端（两边各用各的 caption，含全部差异）===")
    r3, r3b, _ = cmp(data["官方+官方caption"], data["我们的链"],
                     "官方(官方cap) vs 我们的链")

    print("\n=== 归因 ===")
    print("  transformer 实现贡献 : 全量 %.4f%%" % (rel * 100))
    print("  文本编码器贡献       : 全量 %.4f%%" % (r2 * 100))
    print("  两者合计（端到端）   : 全量 %.4f%%" % (r3 * 100))
    print("  ⊕ 标尺参照（§3.1）：15.99%% ⇒ 成图完好｜44.80%% ⇒ 清晰的猫｜47.07%% ⇒ 色块；"
          "量化误差 21~44%%")
    print("\n🔴 划界：以上只是 **step 0**。8 步累积与最终成图的关系**未测**，"
          "不得由 step-0 的百分比直接推断成图 PSNR（#121 已证 L1 不能预测 L3）。")


if __name__ == "__main__":
    main()
