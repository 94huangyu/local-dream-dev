# -*- coding: utf-8 -*-
"""G0 + G-T · 官方 transformer 单步（`EXP_PLAN_OFFICIAL_PARITY.md`）。

## 一次载入跑两趟，做成三点归因
| 装置 | noise_std | 来源 |
|---|---|---|
| 官方 transformer + **官方** caption | 1.5485 | `official_run.log`（已知答案）|
| 官方 transformer + **我们的** caption | 本脚本测 | — |
| 我们的四段 ONNX 链 + 我们的 caption | 1.5368 | `parity_padding.py` 实测 |

⇒ 第二格 ≈ 1.5368 ⇒ 差异在**文本编码器**；≈ 1.5485 ⇒ 差异在 **transformer 本体**。

## G0 装置门（约束 8：先用已知样本验证操作化）
第一趟必须复现 **noise_std = 1.5485**（相对差 <= 0.1%）。
**不过就停止**，说明我搭的官方臂与产出 `official_pytorch_22tok.png` 的那次不是同一装置，
此前不得解读任何 G-T 数字。

## 口径（逐条照抄 `official_pipeline_run.py`，不得自己发明）
  · transformer = `ZImageTransformer2DModel`，**`torch_dtype=torch.bfloat16`**
    （diffusers 0.39 用 `torch_dtype=`，写成 `dtype=` 会 TypeError）
  · timestep = `(1000 - t) / 1000`，t 取官方 sigmas 的第 0 个 = 1000.0
  · 输入 = `latents.unsqueeze(2)` 后 `list(unbind(dim=0))`；caption 是**变长 list**
  · 输出 = `stack(...).squeeze(2)`，**再取负**

⚠️ 官方是 bf16、我们的链是 fp32，**这本身是一个变量**（方案 §7 已事前写明）。
   本脚本第三趟可选做官方 fp32，用来隔离它。

## 成本
每趟约 20~25 分钟（日志实测 1207 s/步），内存峰值约 12 GB。
🔴 **期间不得在本机并行跑重活。**

用法: <venv-official>/python.exe scripts/parity_transformer.py [--fp32]
"""
import gc
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
WORK = os.path.join(P0, "parity")
LAT = os.path.join(P0, "latents_cxx_seed42.raw")
OURS_CAP = os.path.join(P0, "cap_r4x3.raw")
OFF_CAP = os.path.join(WORK, "official_caption_L80.raw")
N_REAL, L_SLOTS, LATENT_CH = 22, 80, 16
KNOWN_NOISE_STD = 1.5485          # official_run.log 的 step 0
G0_TOL = 0.001                    # 相对差 0.1%


def main():
    import torch
    from diffusers import ZImageTransformer2DModel

    use_fp32 = "--fp32" in sys.argv
    dt = torch.float32 if use_fp32 else torch.bfloat16
    os.makedirs(WORK, exist_ok=True)

    latents = torch.from_numpy(
        np.fromfile(LAT, np.float32).reshape(1, LATENT_CH, 128, 128).copy())
    print("latents %s std=%.4f" % (tuple(latents.shape), latents.std()))

    off = np.fromfile(OFF_CAP, np.float32).reshape(N_REAL, 2560)
    ours = np.fromfile(OURS_CAP, np.float32).reshape(L_SLOTS, 2560)[:N_REAL]
    print("官方 caption %s std=%.4f ｜ 我们 caption(前 %d 行) std=%.4f"
          % (off.shape, off.std(), N_REAL, ours.std()))

    # 官方 sigmas 的第 0 个 timestep = 1000.0（调度器一致性已单独验证）
    t0_val = 1000.0
    timestep = torch.tensor([(1000.0 - t0_val) / 1000.0], dtype=dt)
    print("timestep(step0) = %.4f  dtype=%s" % (float(timestep[0]), dt))

    t_load = time.time()
    tr = ZImageTransformer2DModel.from_pretrained(
        os.path.join(MODEL, "transformer"), torch_dtype=dt, low_cpu_mem_usage=True)
    tr.eval()
    print("transformer 载入 %.0fs（dtype=%s）" % (time.time() - t_load, dt), flush=True)

    def run(cap_np, tag):
        pe = [torch.from_numpy(cap_np.copy()).to(dt)]
        lmi = latents.to(dt).unsqueeze(2)
        t = time.time()
        with torch.no_grad():
            out = tr(list(lmi.unbind(dim=0)), timestep, pe, return_dict=False)[0]
        noise = torch.stack([o.float() for o in out], dim=0).squeeze(2)
        noise = -noise                                  # 官方：取负后交给 scheduler
        arr = noise.numpy().astype(np.float32)
        p = os.path.join(WORK, "noise_s0_official_%s.raw" % tag)
        arr.tofile(p)
        print("  [%s] noise_std=%.4f  (%.0fs)  -> %s"
              % (tag, arr.std(), time.time() - t, p), flush=True)
        return arr

    print("\n=== 第 1 趟：官方 transformer + 官方 caption（G0 装置门）===", flush=True)
    a = run(off, "offcap" + ("_fp32" if use_fp32 else ""))
    rel = abs(float(a.std()) - KNOWN_NOISE_STD) / KNOWN_NOISE_STD
    print("  G0: noise_std %.4f vs 已知 %.4f  相对差 %.4f%%  ⇒ %s"
          % (a.std(), KNOWN_NOISE_STD, rel * 100,
             "🟢 装置可信" if rel <= G0_TOL else "🔴 装置不符，停止"))
    if rel > G0_TOL and not use_fp32:
        print("  🔴 G0 未过 ⇒ 按方案 §6，**不得解读任何 G-T 数字**。")
        return 2

    print("\n=== 第 2 趟：官方 transformer + 我们的 caption（G-T 官方臂）===", flush=True)
    b = run(ours, "ourcap" + ("_fp32" if use_fp32 else ""))

    print("\n=== 三点归因 ===")
    print("  官方 transformer + 官方 caption : noise_std = %.4f" % a.std())
    print("  官方 transformer + 我们 caption : noise_std = %.4f" % b.std())
    print("  我们的链    + 我们 caption      : noise_std = 1.5368（parity_padding.py 实测）")
    d_te = abs(float(b.std()) - float(a.std())) / float(a.std())
    print("  ⇒ 换 caption 造成的 std 变化 = %.4f%%" % (d_te * 100))
    print("  ⇒ 下一步：把本脚本产出的 noise_s0_official_ourcap*.raw 与我们链的 step-0"
          " 噪声预测做**逐元素**比较（std 相同不等于张量相同）。")
    del tr
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
