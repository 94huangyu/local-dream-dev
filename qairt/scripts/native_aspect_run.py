# -*- coding: utf-8 -*-
"""EXP_PLAN_NATIVE_ASPECT —— 官方 PyTorch 链路，单进程双臂。

臂 A（自检/对照）：1024x1024，latent 128x128，用与 official_pytorch_22tok.png 同一份噪声
臂 B（原生 4:3）：1152x864，latent 108x144（H,W），tokens 54x72=3888

装置照抄 scripts/official_pipeline_run.py，唯一改动是把写死的 128x128 参数化，
并把 transformer 加载提到两臂之外（只加载一次）。
G-SELF：臂 A 必须复现那只橘猫，否则臂 B 结果整体作废（约束 8）。
"""
import gc
import os
import sys
import time

import numpy as np
import torch

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
WORK = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
STEPS, LATENT_CH = 8, 16

# (标签, latent_H, latent_W, 噪声来源)
ARMS = [("A_1x1",  128, 128, os.path.join(WORK, "latents_cxx_seed42.raw")),
        ("B_4x3",  108, 144, None)]


def log(*a):
    print(*a, flush=True)


def main():
    t_all = time.time()

    # ---- prompt embeddings：官方组件现算，算完释放（与 official_pipeline_run.py 一致） ----
    from transformers import Qwen3Model
    ids = np.load(os.path.join(WORK, "prompt_ids.npy"))
    MAXLEN, PAD = 512, 151643
    input_ids = np.full((1, MAXLEN), PAD, dtype=np.int64)
    attn = np.zeros((1, MAXLEN), dtype=np.int64)
    input_ids[0, :len(ids)] = ids
    attn[0, :len(ids)] = 1
    t0 = time.time()
    te = Qwen3Model.from_pretrained(os.path.join(MODEL, "text_encoder"),
                                    dtype=torch.bfloat16, low_cpu_mem_usage=True)
    te.eval()
    with torch.no_grad():
        o = te(input_ids=torch.from_numpy(input_ids),
               attention_mask=torch.from_numpy(attn).bool(),
               output_hidden_states=True)
    h = o.hidden_states[-2][0][torch.from_numpy(attn)[0].bool()]
    prompt_embeds = [h.float()]
    log("prompt_embeds %d 个真实 token（%.0fs，已释放 TE）" % (h.shape[0], time.time() - t0))
    del te, o, h
    gc.collect()

    # ---- transformer：只加载一次，两臂共用 ----
    from diffusers import ZImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
    t0 = time.time()
    tr = ZImageTransformer2DModel.from_pretrained(
        os.path.join(MODEL, "transformer"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tr.eval()
    log("transformer 载入 %.0fs" % (time.time() - t0))
    pe = [p.to(torch.bfloat16) for p in prompt_embeds]

    finals = {}
    for tag, H, W, npath in ARMS:
        if npath:
            n = np.fromfile(npath, np.float32)
            exp = LATENT_CH * H * W
            assert n.size == exp, "%s 噪声元素数 %d != 期望 %d" % (tag, n.size, exp)
            latents = torch.from_numpy(n.reshape(1, LATENT_CH, H, W).copy())
            src = os.path.basename(npath)
        else:
            rng = np.random.default_rng(42)
            latents = torch.from_numpy(
                rng.standard_normal((1, LATENT_CH, H, W)).astype(np.float32))
            src = "default_rng(42)"
        log("\n=== 臂 %s  latent %dx%d  像素 %dx%d  tokens %d  噪声=%s ===" %
            (tag, H, W, W * 8, H * 8, (H // 2) * (W // 2), src))
        log("    latents std=%.4f" % latents.std())

        # 每臂用全新调度器（step() 是有状态的，复用会串臂）
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(MODEL, "scheduler"))
        sigmas = torch.linspace(1.0, 1.0 / STEPS, STEPS).tolist()
        sched.set_timesteps(sigmas=sigmas)

        for i, t in enumerate(sched.timesteps):
            ts = time.time()
            timestep = (1000 - t.expand(latents.shape[0])) / 1000
            lmi = latents.to(torch.bfloat16).unsqueeze(2)
            with torch.no_grad():
                out = tr(list(lmi.unbind(dim=0)), timestep.to(torch.bfloat16), pe,
                         return_dict=False)[0]
            noise_pred = -torch.stack([o.float() for o in out], dim=0).squeeze(2)
            latents = sched.step(noise_pred.to(torch.float32), t, latents,
                                 return_dict=False)[0]
            log("    step %d: noise_std=%.4f latents_std=%.4f (%.0fs)" %
                (i, noise_pred.std(), latents.std(), time.time() - ts))
        finals[tag] = latents

    del tr, pe, prompt_embeds
    gc.collect()

    # ---- VAE：两臂各解码一次 ----
    from diffusers import AutoencoderKL
    from PIL import Image
    vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    vae.eval()
    sf = vae.config.scaling_factor
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    log("\nVAE scaling=%s shift=%s" % (sf, shift))
    for tag, lat in finals.items():
        with torch.no_grad():
            img = vae.decode((lat / sf + shift).to(torch.float32), return_dict=False)[0]
        arr = ((img[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
        p = os.path.join(OUT, "nativeaspect_%s.png" % tag)
        Image.fromarray(arr).save(p)
        log("  Saved %s  %s" % (p, arr.shape))

    log("\n总耗时 %.0f 分钟" % ((time.time() - t_all) / 60.0))


main()
