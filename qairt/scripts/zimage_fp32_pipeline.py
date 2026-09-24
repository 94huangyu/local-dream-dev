"""
Phase A of the re-planned diagnosis (see plan file binary-imagining-mango.md):
run the COMPLETE Z-Image Turbo pipeline in pure FP32 ONNX Runtime -- text
encoder, 8-step denoising through transformer_part1a/1b/2, VAE decode -- with
NO quantization anywhere. This answers the single highest-leverage question:
is the visual corruption caused by quantization, or does it already exist at
full precision (pointing at the ONNX export/graph-surgery instead)?

Mirrors exactly what PipelineZImage.hpp / TextEncoder.hpp / ZImagePrompt.hpp /
ZImageFlowMatchScheduler.hpp do in the C++ engine, just in Python against the
FP32 ONNX models instead of the quantized QNN context binaries.

Uses the CORRECT Qwen3 tokenizer throughout (D:\models\Z-Image-Turbo\tokenizer\
tokenizer.json, vocab 151643) -- not the mismatched CLIP tokenizer (vocab
49408) that was originally bundled with the deployed model. That mismatch was
found and already independently confirmed as a real (but not sole) bug via an
on-device single-variable test on 2026-08-13: swapping ONLY the tokenizer
(quantized engine unchanged) turned the output from regular vertical stripes
into a mosaic/blob pattern -- different corruption, still not a coherent
image. D:\ZIMAGE\tokenizer\tokenizer.json and the phone's deployed copy have
both already been patched to the correct file (old one backed up alongside
as tokenizer.json.WRONG_CLIP_backup).

STATUS AS OF 2026-08-13 ~21:15: this script's glue code (tokenizer + scheduler
math) was verified standalone. The full model-chaining code below (text
encoder -> transformer loop -> VAE) is WRITTEN but has NOT been run
end-to-end yet -- that is the next thing to do. Tokenizer is now the correct
one throughout, so running this in FP32 isolates quantization as the single
variable (vs. the on-device quantized+correct-tokenizer mosaic-blob result).
Watch for OOM: loading transformer_part1a+part1b+part2 ONNX sessions
simultaneously pulls in ~13.7GB+~11GB of external weight data; ~15GB free RAM
was available as of this writing. If it OOMs, switch to per-step
load/release (slower, same pattern as the device's lowram code) instead of
keeping all three sessions resident for all 8 steps.
"""
import gc
import os
import sys
import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
TOKENIZER_PATH = r"D:\ZIMAGE\tokenizer\tokenizer.json"
OUT_DIR = r"D:\LocalDreamZImage\scratch_runs"

# --- Config.hpp constants ---
ZIMAGE_LATENT_CHANNELS = 16
ZIMAGE_CANVAS_SIZE = 1024
ZIMAGE_TEXT_MAX_LENGTH = 20
ZIMAGE_TURBO_STEPS = 8
ZIMAGE_VAE_SCALING_FACTOR = 0.3611
ZIMAGE_VAE_SHIFT_FACTOR = 0.1159
QWEN_PAD_ID = 151643

CPU = ["CPUExecutionProvider"]


def format_zimage_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def process_zimage_prompt(tokenizer: Tokenizer, prompt: str):
    ids = tokenizer.encode(format_zimage_prompt(prompt)).ids
    input_ids = np.full((1, ZIMAGE_TEXT_MAX_LENGTH), QWEN_PAD_ID, dtype=np.int32)
    attention_mask = np.zeros((1, ZIMAGE_TEXT_MAX_LENGTH), dtype=np.int32)
    token_count = min(len(ids), ZIMAGE_TEXT_MAX_LENGTH)
    input_ids[0, :token_count] = ids[:token_count]
    attention_mask[0, :token_count] = 1
    return input_ids, attention_mask, token_count


def zimage_flow_match_sigmas(num_steps: int, shift: float = 3.0):
    def apply_shift(sigma):
        return shift * sigma / (1.0 + (shift - 1.0) * sigma)

    sigmas = []
    for i in range(num_steps):
        raw = 1.0 if num_steps == 1 else 1.0 - (1.0 - 1.0 / num_steps) * i / (num_steps - 1)
        sigmas.append(apply_shift(raw))
    sigmas.append(0.0)
    return np.array(sigmas, dtype=np.float64)


def load(name):
    t0 = time.time()
    sess = ort.InferenceSession(f"{EVIDENCE}\\{name}.onnx", providers=CPU)
    print(f"  loaded {name} in {time.time()-t0:.1f}s")
    return sess


def run(sess, **inputs):
    names = [o.name for o in sess.get_outputs()]
    results = sess.run(names, inputs)
    return dict(zip(names, results))


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a cute orange cat sitting on a wooden table, masterpiece, best quality"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print("=== Tokenizer ===")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    input_ids, attention_mask, token_count = process_zimage_prompt(tok, prompt)
    print("prompt:", prompt, "| token_count:", token_count)

    # Where(cap_pad_mask, cap_pad_token, real_caption_feats) per transformer_part1a.onnx --
    # True means "replace with the padding embedding", confirmed 2026-08-13 against
    # diffusers' transformer_z_image.py _pad_with_ids/_prepare_sequence. Real token
    # slots must be False; only the actual padding tail is True.
    cap_pad_mask = np.ones((1, 32), dtype=bool)
    cap_pad_mask[0, :min(token_count, 32)] = False

    print("\n=== Text encoder (FP32, 4 parts) ===")
    t_start = time.time()
    # text_encoder.onnx.data 是 14.96 GiB —— 全流程里最大的单个文件，也是内存峰值所在
    # （2026-08-18 实测：即使 transformer 逐段加载，文本编码阶段仍把可用内存打到 194 MB）。
    # 该阶段对同一 prompt 是确定性的，在环脚本已用**同一份 FP32 ONNX**算过并存盘，
    # md5 可核对 ⇒ FP32_CAPTION 指向该文件即可整段跳过。
    _cap = os.environ.get("FP32_CAPTION", "")
    if _cap:
        caption = np.fromfile(_cap, dtype=np.float32).reshape(1, 32, 2560)
        print(f"caption <- {_cap}（复用在环脚本的 FP32 文本编码结果，跳过 14.96 GiB 阶段）")
    else:
        te1 = load("text_encoder_part1")
        out = run(te1, input_ids=input_ids, attention_mask=attention_mask)
        del te1
        te2 = load("text_encoder_part2")
        out = run(te2, **{"add_2452": out["add_2452"], "attention_mask": attention_mask})
        del te2
        te3 = load("text_encoder_part3")
        out = run(te3, **{"add_4828": out["add_4828"], "attention_mask": attention_mask})
        del te3
        te4 = load("text_encoder_part4")
        out = run(te4, **{"add_7204": out["add_7204"], "attention_mask": attention_mask})
        del te4
        caption = out["caption"]
    print("caption shape:", caption.shape, "in", f"{time.time()-t_start:.1f}s")
    if caption.shape[1] != 32:
        raise RuntimeError(f"Expected caption seq_len 32, got {caption.shape[1]} -- "
                            f"check text_encoder chaining/padding assumptions before continuing")

    print("\n=== Latents init ===")
    latent_count = ZIMAGE_LATENT_CHANNELS * 128 * 128
    # 初始噪声。缺省 numpy PCG64；与【设备/app】做 paired 对照时**必须**吃同一份噪声：
    # C++ 侧用 std::mt19937 + std::normal_distribution<float>，与 PCG64 同 seed 也完全不同
    # ⇒ 否则比较的是同一模型的两个不同采样（2026-08-18 已因此产生过两个无效数字）。
    ext = os.environ.get("FP32_LATENTS", "")
    if ext:
        latents = np.fromfile(ext, dtype=np.float32)
        assert latents.size == latent_count, f"latents {latents.size} != {latent_count}"
        print(f"initial latents <- {ext}（外部提供，与设备侧逐字节相同）")
    else:
        latents = np.random.default_rng(seed).standard_normal(latent_count).astype(np.float32)
    print(f"initial latents: min={latents.min():.4f} max={latents.max():.4f} "
          f"mean={latents.mean():.4f} std={latents.std():.4f}")

    sigmas = zimage_flow_match_sigmas(ZIMAGE_TURBO_STEPS)
    timesteps = sigmas[:-1] * 1000.0

    print("\n=== Transformer denoising loop (FP32, part1a+1b+2, 8 steps) ===")
    # 三段常驻会同时吃掉 p1a/p1b 共享的 12.82 GiB + p2 的 10.11 GiB ≈ 22.9 GiB，
    # 在 23.7 GB 的机器上必然打穿（2026-08-18 实测：可用内存跌到 598 MB 被守卫终止）。
    # FP32_LOWRAM=1 改为逐段加载/释放，峰值降到单段最大约 12.8 GiB，代价是每步重新加载。
    LOWRAM = os.environ.get("FP32_LOWRAM", "") == "1"
    if LOWRAM:
        print("LOWRAM: 逐段加载/释放（峰值≈单段最大，代价是每步重载）")
        p1a = p1b = p2 = None
    else:
        print("Loading transformer sessions once (kept resident for all 8 steps)...")
        t0 = time.time()
        p1a = load("transformer_part1a")
        p1b = load("transformer_part1b")
        p2 = load("transformer_part2")
        print(f"all transformer sessions loaded in {time.time()-t0:.1f}s")

    latents_4d = latents.reshape(1, ZIMAGE_LATENT_CHANNELS, 128, 128)
    caption_f32 = caption.astype(np.float32)

    for step in range(ZIMAGE_TURBO_STEPS):
        t_step = time.time()
        timestep = np.array([1.0 - timesteps[step] / 1000.0], dtype=np.float32)

        if LOWRAM:
            # 一次只留一段。实测（2026-08-18）：p1a+p1b 同时驻留 13.21 GiB，
            # 再加 part1a 推理的中间激活就打穿 23.7 GB 的机器（第 4 次尝试死在这里）。
            # 单段实测：p1a 8.04 GiB / p1b 约 5.2 GiB / p2 10.08 GiB；
            # 且 ORT 的 del+gc 确实把内存还给系统（实测 5002 MB -> 18483 MB）。
            _a = load("transformer_part1a")
            o1a = run(_a, latents=latents_4d, timestep=timestep,
                      caption=caption_f32, cap_pad_mask=cap_pad_mask)
            del _a
            gc.collect()
            _b = load("transformer_part1b")
            o1b = run(_b, adaln_input=o1a["adaln_input"], add_131=o1a["add_131"],
                      add_138=o1a["add_138"], select_45=o1a["select_45"],
                      select_46=o1a["select_46"], tanh_19=o1a["tanh_19"])
            del _b
            gc.collect()
            # 用 DCE 版：与 `_fixed`（生产 DLC 的源）计算等价，但去掉 84 个死节点。
            # 那些死节点消费 30 个 [1,30,4128,4128]（单个 1.9 GiB）的注意力矩阵（#62），
            # 是 part2 推理把 23.7 GB 机器打穿的直接原因（2026-08-18 第 5 次尝试实测）。
            # ⚠️ DCE 版没有 latents_shape 输入（已折成常量），不能再传。
            _2 = load("transformer_part2_fixed_dce")
            o2 = run(_2, unified=o1b["unified"], unified_mask=o1a["unified_mask"],
                     unified_freqs=o1a["unified_freqs"], adaln_input=o1a["adaln_input"])
            del _2
            gc.collect()
        else:
            o1a = run(p1a, latents=latents_4d, timestep=timestep,
                      caption=caption_f32, cap_pad_mask=cap_pad_mask)
            o1b = run(p1b, adaln_input=o1a["adaln_input"], add_131=o1a["add_131"],
                      add_138=o1a["add_138"], select_45=o1a["select_45"],
                      select_46=o1a["select_46"], tanh_19=o1a["tanh_19"])
            o2 = run(p2, unified=o1b["unified"], unified_mask=o1a["unified_mask"],
                     unified_freqs=o1a["unified_freqs"], adaln_input=o1a["adaln_input"],
                     latents_shape=o1a["latents_shape"])
        noise = o2["latents"].reshape(-1).astype(np.float64)

        sigma = timesteps[step] / 1000.0
        next_sigma = timesteps[step + 1] / 1000.0 if step + 1 < ZIMAGE_TURBO_STEPS else 0.0
        dt = next_sigma - sigma
        # pipeline_z_image.py negates the transformer output before scheduler.step(),
        # whose formula is sample + dt*model_output -> combined: sample - dt*v_raw.
        latents_flat = latents_4d.reshape(-1).astype(np.float64) - dt * noise
        latents_4d = latents_flat.astype(np.float32).reshape(1, ZIMAGE_LATENT_CHANNELS, 128, 128)

        print(f"  step {step}: dt={dt:.4f} noise[std={noise.std():.4f}] "
              f"latents[std={latents_flat.std():.4f}] ({time.time()-t_step:.1f}s)")

    del p1a, p1b, p2
    gc.collect()

    # 15.21.6 教训：中间产物默认落盘。此前官方那次 final latents 没存，
    # 导致做 VAE 双臂对照时只能改用别的 latents。
    _tl = os.environ.get("FP32_TAG", "")
    _lp = OUT_DIR + os.sep + "fp32_final_latents" + (("_" + _tl) if _tl else "") + ".raw"
    np.ascontiguousarray(latents_4d.astype(np.float32)).tofile(_lp)
    print("[落盘] final latents -> " + _lp, flush=True)

    print("\n=== VAE decode (FP32) ===")
    # 🔴 2026-08-21 修正（HANDOVER 15.22 / 台账 #80）：vae_decoder.onnx 的首节点就是
    #    Div(vae_latents, 0.3611)，反缩放【图内已做】。此处原本又做了一遍 lat/s+shift，
    #    等于除了两遍（对官方 23.51 dB / 高频 1.3049x）。正确喂法是 lat + shift*s
    #    （对官方 79.92 dB / 高频 1.0000x）。改前不要动，先读 §二十。
    vae_latents = (latents_4d.astype(np.float64)
                   + ZIMAGE_VAE_SHIFT_FACTOR * ZIMAGE_VAE_SCALING_FACTOR).astype(np.float32)
    vae = load("vae_decoder")
    pixels = run(vae, vae_latents=vae_latents)["pixels"]  # [1,3,1024,1024]
    del vae

    print("\n=== Saving PNG ===")
    from PIL import Image
    px = pixels[0]  # [3,1024,1024] CHW
    img = np.clip((px + 1.0) * 127.5, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # HWC
    out_path = f"{OUT_DIR}\\zimage_fp32_full_pipeline.png"
    _tag = os.environ.get("FP32_TAG", "")
    if _tag:
        out_path = out_path[:-4] + "_" + _tag + ".png"
    Image.fromarray(img).save(out_path)
    print("Saved:", out_path)
    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
