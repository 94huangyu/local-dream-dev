# -*- coding: utf-8 -*-
"""G-E · 文本编码器一致性：我们的 ONNX TE vs 官方 `Qwen3Model`（`EXP_PLAN_OFFICIAL_PARITY.md`）。

## 为什么必须做
现存唯一的数据是 **20-token 情形相对 L2 0.8628%**（记在 `official_pipeline_run.py` 头部），
而**当前部署跑的是 L=80，从未验证**。TE 的输出是整条链的输入，它偏了下游全偏。

## 官方口径（逐条照抄 `official_pipeline_run.py`，不得自己发明）
  · 类 = `Qwen3Model`（`model_index.json` 指定；不是 `Qwen3ForCausalLM`）
  · dtype = bfloat16，`input_ids` 补到 **512**、PAD = **151643**
  · 取 **`hidden_states[-2]`**（倒数第二层），再用 attention_mask **只取真实 token**
  · 交给 transformer 的是**变长 list**，只含真实 token，**没有 padding 槽**

## 我们的口径
`cap_r4x3.raw` = [1, 80, 2560] float32，我们的 ONNX TE 产出，**定长 80 槽**（为 QNN 固定形状）。
⇒ 「定长 padded + cap_pad_mask」与官方「变长」是否等价，**本身就是一条未验证的假设**，
   本脚本第二节专门量它。

## 判据（`EXP_PLAN_OFFICIAL_PARITY.md` §4 事前锁定，不得改）
  真实 token 部分的相对 L2：< 0.1% 🟢 ／ 0.1%~1% 🟡 ／ >= 1% 🔴

## 口径纪律（约束 7）
必须同时报「前 1% 元素占 ‖a‖² 的比例」；> 50% 时全量口径**不描述主体**，必须看主体口径。

用法: <venv-official>/python.exe scripts/parity_te.py [--recompute]
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
IDS = os.path.join(P0, "prompt_ids.npy")
OURS = os.path.join(P0, "cap_r4x3.raw")
OUT = os.path.join(P0, "parity", "official_caption_L80.raw")
MAXLEN, PAD = 512, 151643
L_SLOTS = 80


def metrics(a, b, tag):
    """a = 参考（官方），b = 待测（我们）。全量 + 集中度 + 余弦；集中时补主体口径。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    rel = np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    e = a * a
    k = max(1, int(round(0.01 * e.size)))
    conc = float(np.sort(e)[-k:].sum() / max(e.sum(), 1e-30))
    print("  %-26s 相对 L2 = %8.4f%%   余弦 = %.6f   前1%%能量占比 = %.2f%%"
          % (tag, rel * 100, cos, conc * 100))
    relb = None
    if conc > 0.5:
        m = np.abs(a) <= np.quantile(np.abs(a), 0.99)
        relb = np.linalg.norm(b[m] - a[m]) / max(np.linalg.norm(a[m]), 1e-30)
        cosb = float(a[m] @ b[m] / max(np.linalg.norm(a[m]) * np.linalg.norm(b[m]), 1e-30))
        print("      ⚠️ 集中度 > 50%% ⇒ 全量口径不描述主体（约束 7）；"
              "主体(|a|<=p99) 相对 L2 = %.4f%%  主体余弦 = %.6f" % (relb * 100, cosb))
    return rel, cos, conc, relb


def official_caption(ids):
    """官方 Qwen3Model 的 caption。算过一次就落盘复用（每次 72 s）。"""
    n_real = len(ids)
    if os.path.isfile(OUT) and "--recompute" not in sys.argv:
        h = np.fromfile(OUT, np.float32).reshape(n_real, 2560)
        print("复用已落盘的官方 caption：%s（--recompute 强制重算）" % OUT)
        return h
    import torch
    from transformers import Qwen3Model
    input_ids = np.full((1, MAXLEN), PAD, dtype=np.int64)
    attn = np.zeros((1, MAXLEN), dtype=np.int64)
    input_ids[0, :n_real] = ids
    attn[0, :n_real] = 1
    te = Qwen3Model.from_pretrained(os.path.join(MODEL, "text_encoder"),
                                    dtype=torch.bfloat16, low_cpu_mem_usage=True)
    te.eval()
    with torch.no_grad():
        o = te(input_ids=torch.from_numpy(input_ids),
               attention_mask=torch.from_numpy(attn).bool(),
               output_hidden_states=True)
    h = o.hidden_states[-2][0][torch.from_numpy(attn)[0].bool()].float().numpy()
    del te, o
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    h.astype(np.float32).tofile(OUT)
    print("已写出官方 caption -> %s" % OUT)
    return h


def main():
    ids = np.load(IDS)
    n_real = len(ids)
    print("prompt_ids: %d 个 token  <- %s" % (n_real, IDS))

    h = official_caption(ids)
    print("官方 caption: %s  std=%.4f  |max|=%.4f" % (h.shape, h.std(), np.abs(h).max()))

    ours_full = np.fromfile(OURS, np.float32)
    assert ours_full.size == L_SLOTS * 2560, ours_full.size
    ours = ours_full.reshape(L_SLOTS, 2560)
    print("我们 caption: (%d, 2560)  前 %d 行 std=%.4f  |max|=%.4f"
          % (L_SLOTS, n_real, ours[:n_real].std(), np.abs(ours[:n_real]).max()))

    print("\n=== G-E 判据（事前锁定：<0.1%% 🟢 ／ 0.1~1%% 🟡 ／ >=1%% 🔴）===")
    rel, cos, conc, relb = metrics(h, ours[:n_real], "官方 vs 我们（真实 token）")
    verdict = "🟢 等价" if rel < 0.001 else ("🟡 有差异" if rel < 0.01 else "🔴 实质不一致")
    print("  ⇒ **%s**（全量口径 %.4f%%）" % (verdict, rel * 100))
    print("  ⊕ 参照：既有记录是 20-token 情形 **0.8628%%**；本次 %d-token / L=%d 槽"
          % (n_real, L_SLOTS))

    per = np.linalg.norm(ours[:n_real] - h, axis=1) / np.maximum(
        np.linalg.norm(h, axis=1), 1e-30)
    order = np.argsort(-per)
    print("  逐 token 相对误差：中位 %.4f%%  最大 5 个 = %s"
          % (np.median(per) * 100,
             ", ".join("tok%d:%.3f%%" % (i, per[i] * 100) for i in order[:5])))

    # ---- 第二节：padding 槽。官方根本没有这一段，我们有 58 个。----
    print("\n=== 🔴 padding 槽（官方无此概念，我们有 %d 个）===" % (L_SLOTS - n_real))
    pad = ours[n_real:]
    print("  我们的 padding 区: std=%.4f  |max|=%.4f  非零元素占 %.2f%%"
          % (pad.std(), np.abs(pad).max(), 100.0 * (pad != 0).mean()))
    e_real = float((ours[:n_real].astype(np.float64) ** 2).sum())
    e_pad = float((pad.astype(np.float64) ** 2).sum())
    print("  能量：真实 token %.4g ｜ padding %.4g ｜ **padding 占全张量 %.2f%%**"
          % (e_real, e_pad, 100.0 * e_pad / max(e_real + e_pad, 1e-30)))
    print("  ⇒ padding 槽**不是零**。它是否有害，取决于 `cap_pad_mask` 在图内是否真的把它屏蔽。")
    print("  ⚠️ ③**本脚本不能回答这一点** —— 需要单变量实验：把 padding 区置零后重跑，")
    print("     比成图/中间张量。这是 G-T 之后的下一条，已登记。")


if __name__ == "__main__":
    main()
