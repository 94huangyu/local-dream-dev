"""用【官方 PyTorch 组件】跑 Z-Image Turbo，产出地面真值参考图。

## 目的
确定「蜡质毛发/虎斑发糊」到底是**我们的 ONNX 导出缺陷**，还是**该模型 8 步的真实水平**。
在这个问题回答之前，所有 HTP 量化优化都可能在追一个错的目标。

## 单变量设计
对照臂 = `scratch_runs/fp32_22tok.png`（我们的 ONNX FP32，**同样 22 token caption**）。
两者共享：初始噪声（`latents_cxx_seed42.raw`）、prompt embeddings、8 步 sigma、CFG=0。
**唯一差异 = PyTorch 原版 vs 我们的 ONNX 导出（transformer + VAE）。**
⚠️ 不要拿 20-token 那张做对照——那会把 caption 长度混进来，又是一次不干净的实验。

## 与官方 `pipeline_z_image.py` 的逐行对应（已核对 diffusers 0.39）
- sigmas：`get_default_z_image_sigmas(n) = linspace(1.0, 1/n, n)`
- shift：`FlowMatchEulerDiscreteScheduler`，`use_dynamic_shifting=false` ⇒ `shift*s/(1+(shift-1)*s)`
- timestep：`(1000 - t) / 1000`
- transformer 输入：`latents.unsqueeze(2)` 后 `list(unbind(dim=0))`
- 输出：`stack(...).squeeze(2)`，**再取负** `noise_pred = -noise_pred`
- 更新：`scheduler.step(noise_pred, t, latents)` ⇒ `sample + dt * model_output`

## 内存（先算后跑）
text_encoder bf16 约 **3.8 GB**，**进程内现算 embeddings 后立即释放**；
transformer bf16 约 **11.7 GB** 全程驻留；VAE 0.16 GB 最后加载。峰值约 12 GB。

⚠️ 早期版本复用了预先算好的 embeddings，被用户质疑依据不足——那份只在 20-token
情形下与我们的 ONNX 比对过（相对 L2 0.8628%），22-token 情形无独立校验，
且用的类是 `Qwen3ForCausalLM` 而非 `model_index.json` 指定的 `Qwen3Model`。
**地面真值不能建立在未验证的假设上**，故改为现算；`cap22.raw` 仅用于事后量化两者差异。

用法: python official_pipeline_run.py <latents.raw> <cap22.raw> <out.png>
"""
import gc
import os
import sys
import time

import numpy as np
import torch

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
STEPS, LATENT_CH = 8, 16


def log(*a):
    print(*a, flush=True)


def main():
    lat_path, cap_path, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
    t_all = time.time()

    # --- 初始噪声：与设备/ONNX 侧逐字节相同 ---
    latents = torch.from_numpy(
        np.fromfile(lat_path, np.float32).reshape(1, LATENT_CH, 128, 128).copy())
    log(f"latents {tuple(latents.shape)} std={latents.std():.4f}  <- {lat_path}")

    # --- prompt embeddings：**进程内用官方组件现算**，不复用任何中间产物 ---
    # 之前的版本复用了我预先算好的 embeddings，但那份只在 20-token 情形下
    # 与我们的 ONNX 导出比对过（相对 L2 0.8628%），**22-token 情形没有独立校验**，
    # 且我用的是 Qwen3ForCausalLM 而 model_index.json 写的是 Qwen3Model。
    # 地面真值不能建立在未验证的假设上 ⇒ 这里按 model_index 指定的类现算，算完即释放。
    from transformers import Qwen3Model
    ids = np.load(os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "prompt_ids.npy"))
    MAXLEN, PAD = 512, 151643
    input_ids = np.full((1, MAXLEN), PAD, dtype=np.int64)
    attn = np.zeros((1, MAXLEN), dtype=np.int64)
    input_ids[0, :len(ids)] = ids; attn[0, :len(ids)] = 1
    t0 = time.time()
    te = Qwen3Model.from_pretrained(os.path.join(MODEL, "text_encoder"),
                                    dtype=torch.bfloat16, low_cpu_mem_usage=True)
    te.eval()
    with torch.no_grad():
        o = te(input_ids=torch.from_numpy(input_ids),
               attention_mask=torch.from_numpy(attn).bool(),
               output_hidden_states=True)
    h = o.hidden_states[-2][0][torch.from_numpy(attn)[0].bool()]      # 官方口径
    prompt_embeds = [h.float()]
    n_real = h.shape[0]
    del te, o, h
    gc.collect()
    log(f"prompt_embeds {n_real} 个真实 token（Qwen3Model 现算，{time.time()-t0:.0f}s，已释放）")

    # 与预先算好的那份比对，量化「Qwen3Model vs Qwen3ForCausalLM」的差异（仅记录，不作判据）
    if os.path.isfile(cap_path):
        ref = np.fromfile(cap_path, np.float32).reshape(32, 2560)[:n_real]
        cur = prompt_embeds[0].numpy()
        rel = np.linalg.norm(cur - ref) / max(np.linalg.norm(ref), 1e-9)
        log(f"  vs 预算好的 embeddings: 相对 L2 = {rel*100:.4f}%"
            f"（Qwen3Model 与 Qwen3ForCausalLM 的差异）")

    # --- 调度器：官方默认 sigmas + config 里的 shift ---
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(MODEL, "scheduler"))
    sigmas = torch.linspace(1.0, 1.0 / STEPS, STEPS).tolist()   # 官方 get_default_z_image_sigmas
    sched.set_timesteps(sigmas=sigmas)
    timesteps = sched.timesteps
    log(f"timesteps={[round(float(t), 2) for t in timesteps]}")

    # --- transformer（bf16 全程驻留）---
    from diffusers import ZImageTransformer2DModel
    t0 = time.time()
    # ⚠️ diffusers 0.39 的 from_pretrained 用 `torch_dtype=`；写成 `dtype=` 会被当作
    # 模型 __init__ 的关键字透传 ⇒ TypeError（transformers 5.x 才改用 dtype）。
    tr = ZImageTransformer2DModel.from_pretrained(
        os.path.join(MODEL, "transformer"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tr.eval()
    log(f"transformer 载入 {time.time()-t0:.0f}s")

    pe = [p.to(torch.bfloat16) for p in prompt_embeds]
    for i, t in enumerate(timesteps):
        ts = time.time()
        timestep = t.expand(latents.shape[0])
        timestep = (1000 - timestep) / 1000                     # 官方口径
        lmi = latents.to(torch.bfloat16).unsqueeze(2)
        with torch.no_grad():
            out = tr(list(lmi.unbind(dim=0)), timestep.to(torch.bfloat16), pe,
                     return_dict=False)[0]
        noise_pred = torch.stack([o.float() for o in out], dim=0).squeeze(2)
        noise_pred = -noise_pred                                # 官方：取负后交给 scheduler
        latents = sched.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
        log(f"  step {i}: noise_std={noise_pred.std():.4f} "
            f"latents_std={latents.std():.4f} ({time.time()-ts:.0f}s)")

    del tr
    gc.collect()

    # --- VAE ---
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    vae.eval()
    sf = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    log(f"VAE scaling={sf} shift={shift}")
    with torch.no_grad():
        img = vae.decode((latents / sf + shift).to(torch.float32), return_dict=False)[0]
    arr = ((img[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
    from PIL import Image
    Image.fromarray(arr).save(out_png)
    log(f"Saved: {out_png}   总耗时 {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
