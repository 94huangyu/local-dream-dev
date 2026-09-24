# -*- coding: utf-8 -*-
"""G-A · 位置编码不一致是不是 15.52% 的根因（`EXP_PLAN_OFFICIAL_PARITY.md` §3 G-A / 台账 #154）。

## 已确认的不一致（两边都是**直接读出来的**，不是转述）
| | 图像块的 F 轴位置 |
|---|---|
| 官方 | `cap_end_pos = cap_cu_len(起点 1) + len(caption)` = **1 + 22 = 23**（源码 `transformer_z_image.py:646-681`）|
| 我们 | `stack_1` 轴0 **恒为 33**（`transformer_part1a_L80.onnx` 的 initializer，shape (1,64,64,3) int32）|

⇒ 差 **10**。而 `1 + 32 = 33` ⇒ 我们导出时用的是 **padded 长度 32**，不是真实 token 数 22；
   Tier 2 把 padding 改成 80 后这个常数也没跟着变（按我们自己的错误约定应是 81）。
⊕ caption 自身的位置两边一致：我们 `val_520 = [1..80]`，官方从 1 开始 ⇒ 前 22 个都是 1..22。
   **所以这是一个干净的单变量。**

## 做法
monkeypatch 官方的 `_pad_with_ids`：**只在图像调用上**把 `pos_start` 换成 (33,0,0)。
判别图像调用的依据 = `pos_grid_size` 的后两维 > 1（caption 调用是 `(len,1,1)`）。
其余一切不动，caption 仍用我们的 `cap_r4x3.raw` 前 22 行（与 G-T 同一份）。

## 判据（事前锁定）
基线：G-T 实测「官方(我们caption) vs 我们的链」= 全量 **15.5172%**。
把官方改成 33 之后重测同一对：

| 结果 | 判读 |
|---|---|
| 降到 **< 4%** | 🟢🟢 **位置编码就是主因**（4% ≈ TE 贡献 2.66% 的量级）|
| 降到 4%~10% | 🟡 **是主因之一**，另有其它来源 |
| 仍 >= 10% | 🔴 **不是主因**，位置差异只是并存的另一个缺陷 |

🔴 无论结果如何都不改判据。

## 成本
一次官方 transformer 单步，约 **20~25 分钟**，内存峰值约 12 GB。

用法: <venv-official>/python.exe scripts/parity_pos33.py
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
W = os.path.join(P0, "parity")
LAT = os.path.join(P0, "latents_cxx_seed42.raw")
OURS_CAP = os.path.join(P0, "cap_r4x3.raw")
N_REAL, LATENT_CH = 22, 16
OUR_F_POS = 33          # 从我们的 ONNX 读出来的值
BASELINE_REL = 0.155172  # G-T 实测（官方 23 时）


def main():
    import torch
    from diffusers import ZImageTransformer2DModel
    from diffusers.models.transformers import transformer_z_image as tz

    os.makedirs(W, exist_ok=True)
    latents = torch.from_numpy(
        np.fromfile(LAT, np.float32).reshape(1, LATENT_CH, 128, 128).copy())
    cap = np.fromfile(OURS_CAP, np.float32).reshape(80, 2560)[:N_REAL]
    timestep = torch.tensor([0.0], dtype=torch.bfloat16)   # (1000-1000)/1000

    orig = tz.ZImageTransformer2DModel._pad_with_ids
    stats = {"cap": 0, "img": 0, "patched": []}

    def patched(self, feat, pos_grid_size, pos_start, device, noise_mask_val=None):
        # caption 调用是 (len,1,1)；图像调用是 (F_t,H_t,W_t) 且 H_t,W_t > 1
        is_img = len(pos_grid_size) == 3 and pos_grid_size[1] > 1 and pos_grid_size[2] > 1
        if is_img:
            stats["img"] += 1
            stats["patched"].append((tuple(pos_grid_size), tuple(pos_start),
                                     (OUR_F_POS, 0, 0)))
            pos_start = (OUR_F_POS, 0, 0)
        else:
            stats["cap"] += 1
        return orig(self, feat, pos_grid_size, pos_start, device, noise_mask_val)

    tz.ZImageTransformer2DModel._pad_with_ids = patched
    print("已 monkeypatch `_pad_with_ids`：图像调用的 pos_start -> (%d,0,0)" % OUR_F_POS)

    t = time.time()
    tr = ZImageTransformer2DModel.from_pretrained(
        os.path.join(MODEL, "transformer"), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    tr.eval()
    print("transformer 载入 %.0fs" % (time.time() - t), flush=True)

    pe = [torch.from_numpy(cap.copy()).to(torch.bfloat16)]
    lmi = latents.to(torch.bfloat16).unsqueeze(2)
    t = time.time()
    with torch.no_grad():
        out = tr(list(lmi.unbind(dim=0)), timestep, pe, return_dict=False)[0]
    noise = -torch.stack([o.float() for o in out], dim=0).squeeze(2)
    arr = noise.numpy().astype(np.float32)
    p = os.path.join(W, "noise_s0_official_ourcap_pos%d.raw" % OUR_F_POS)
    arr.tofile(p)
    print("  noise_std=%.4f  (%.0fs)  -> %s" % (arr.std(), time.time() - t, p), flush=True)

    # 🔴 生效门：patch 必须真的被调用过，且改的是图像那一路
    print("\n=== patch 生效证据（不得以「没报错」判定）===")
    print("  caption 调用 %d 次 ｜ 图像调用 %d 次" % (stats["cap"], stats["img"]))
    for g, old, new in stats["patched"]:
        print("  图像: pos_grid_size=%s  pos_start %s -> %s" % (g, old, new))
    if stats["img"] == 0:
        raise SystemExit("🔴 patch 从未命中图像调用 ⇒ 本次结果无效")
    if stats["patched"] and stats["patched"][0][1] == (OUR_F_POS, 0, 0):
        print("  ⚠️ 原值本来就是 %d ⇒ 本实验没有变量，先查为什么" % OUR_F_POS)

    # ---- 判据 ----
    ours = np.fromfile(os.path.join(W, "noise_s0_ours.raw"), np.float32).astype(np.float64)
    a = arr.astype(np.float64).ravel()
    rel = np.linalg.norm(ours - a) / max(np.linalg.norm(a), 1e-30)
    cos = float(a @ ours / max(np.linalg.norm(a) * np.linalg.norm(ours), 1e-30))
    m = np.abs(a) <= np.quantile(np.abs(a), 0.99)
    relb = np.linalg.norm(ours[m] - a[m]) / max(np.linalg.norm(a[m]), 1e-30)
    print("\n=== G-A 判据（事前锁定）===")
    print("  基线（官方用 23）: 全量 %.4f%%" % (BASELINE_REL * 100))
    print("  本次（官方用 %d）: 全量 **%.4f%%**  主体 %.4f%%  余弦 %.6f"
          % (OUR_F_POS, rel * 100, relb * 100, cos))
    if rel < 0.04:
        v = "🟢🟢 **位置编码就是主因**"
    elif rel < 0.10:
        v = "🟡 是主因之一，另有来源"
    else:
        v = "🔴 不是主因"
    print("  ⇒ %s（%.4f%% -> %.4f%%，降了 %.1f%%）"
          % (v, BASELINE_REL * 100, rel * 100,
             100.0 * (BASELINE_REL - rel) / BASELINE_REL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
