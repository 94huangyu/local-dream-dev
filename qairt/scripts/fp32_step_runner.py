"""FP32 参考图：**每步一个独立进程**，规避跨步内存累积。

## 为什么要这样（2026-08-18 实测，六次尝试的结论）

单进程跑 8 步会被内存拖死，实测轨迹：
1. 三段常驻 → 22.9 GiB，打穿
2. 逐段但 p1a+p1b 同驻 → 13.21 GiB + 推理激活，打穿
3. 跳过文本编码（14.96 GiB，最大单项）→ 仍打穿
4. 一次只驻一段 → **part2 推理**打穿（84 个死节点消费 30×1.9 GiB，见 #62）
5. 换 `transformer_part2_fixed_dce` → **step 0 成功（246 s）**，step 1 打穿 ⇒ 跨步累积

⇒ 单步能过、多步不能过 ⇒ **每步独立进程**，退出即彻底回收。

## 口径纪律
- part2 用 `transformer_part2_fixed_dce`：与生产 DLC 的源 `_fixed` 计算等价，
  且去掉 84 个死节点（**不是**基础版 `transformer_part2`）
- 初始噪声、caption 由外部提供，保证与设备侧逐字节相同

用法:
  python fp32_step_runner.py step  <step_idx> <latents_in.raw> <latents_out.raw> <caption.raw>
  python fp32_step_runner.py drive <latents_init.raw> <caption.raw> <out.png>
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np

E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT_DIR = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
STEPS, LATENT_CH = 8, 16
VAE_SCALING, VAE_SHIFT = 0.3611, 0.1159


def sigmas(n, shift=3.0):
    f = lambda s: shift * s / (1.0 + (shift - 1.0) * s)
    out = [f(1.0 if n == 1 else 1.0 - (1.0 - 1.0 / n) * i / (n - 1)) for i in range(n)]
    return np.array(out + [0.0], dtype=np.float64)


def load(name):
    import onnxruntime as ort
    t0 = time.time()
    s = ort.InferenceSession(os.path.join(E, name + ".onnx"),
                             providers=["CPUExecutionProvider"])
    print(f"    load {name} {time.time()-t0:.0f}s", flush=True)
    return s


def run(sess, **kw):
    return dict(zip([o.name for o in sess.get_outputs()],
                    sess.run([o.name for o in sess.get_outputs()], kw)))


def one_step(idx, lat_in, lat_out, cap_path):
    latents = np.fromfile(lat_in, np.float32).reshape(1, LATENT_CH, 128, 128)
    caption = np.fromfile(cap_path, np.float32).reshape(1, 32, 2560)
    # cap_pad_mask：缺省 20 个真实 token（与我们导出的 20-token 文本编码器一致）；
    # 做「20 vs 22 token 截断影响」对照时由外部提供（FP32_MASK 指向 float32 raw）。
    mp = os.environ.get("FP32_MASK", "")
    if mp:
        cap_pad_mask = (np.fromfile(mp, np.float32).reshape(1, 32) > 0.5)
        print(f"    cap_pad_mask <- {mp}  真实槽 {int((~cap_pad_mask).sum())}", flush=True)
    else:
        cap_pad_mask = np.ones((1, 32), dtype=bool)
        cap_pad_mask[0, :20] = False
    ts = sigmas(STEPS)[:-1] * 1000.0
    timestep = np.array([1.0 - ts[idx] / 1000.0], dtype=np.float32)

    a = load("transformer_part1a")
    o1a = run(a, latents=latents, timestep=timestep, caption=caption,
              cap_pad_mask=cap_pad_mask)
    del a; gc.collect()
    b = load("transformer_part1b")
    o1b = run(b, adaln_input=o1a["adaln_input"], add_131=o1a["add_131"],
              add_138=o1a["add_138"], select_45=o1a["select_45"],
              select_46=o1a["select_46"], tanh_19=o1a["tanh_19"])
    del b; gc.collect()
    c = load("transformer_part2_fixed_dce")
    o2 = run(c, unified=o1b["unified"], unified_mask=o1a["unified_mask"],
             unified_freqs=o1a["unified_freqs"], adaln_input=o1a["adaln_input"])
    del c; gc.collect()

    noise = o2["latents"].reshape(-1).astype(np.float64)
    s0 = ts[idx] / 1000.0
    s1 = ts[idx + 1] / 1000.0 if idx + 1 < STEPS else 0.0
    dt = s1 - s0
    nxt = latents.reshape(-1).astype(np.float64) - dt * noise
    nxt.astype(np.float32).tofile(lat_out)
    print(f"  step {idx}: dt={dt:.4f} noise_std={noise.std():.4f} "
          f"latents_std={nxt.std():.4f}", flush=True)


def drive(lat_init, cap_path, out_png):
    work = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fp32_steps")
    os.makedirs(work, exist_ok=True)
    cur = os.path.join(work, "lat_0.raw")
    np.fromfile(lat_init, np.float32).tofile(cur)
    t0 = time.time()
    for i in range(STEPS):
        nxt = os.path.join(work, f"lat_{i+1}.raw")
        r = subprocess.run([sys.executable, "-u", __file__, "step", str(i), cur, nxt, cap_path])
        if r.returncode != 0 or not os.path.isfile(nxt):
            raise SystemExit(f"step {i} 失败 rc={r.returncode}")
        cur = nxt
        print(f"[{time.time()-t0:.0f}s] step {i} 完成", flush=True)

    lat = np.fromfile(cur, np.float32).reshape(1, LATENT_CH, 128, 128)
    # 🔴 2026-08-21 修正（HANDOVER 15.22 / 台账 #80）：vae_decoder.onnx 的首节点就是
    #    Div(vae_latents, 0.3611)，反缩放【图内已做】。此处原本又做了一遍 lat/s+shift，
    #    等于除了两遍（对官方 23.51 dB / 高频 1.3049x）。正确喂法是 lat + shift*s
    #    （对官方 79.92 dB / 高频 1.0000x）。改前不要动，先读 §二十。
    vae_in = (lat.astype(np.float64) + VAE_SHIFT * VAE_SCALING).astype(np.float32)
    v = load("vae_decoder")
    px = run(v, vae_latents=vae_in)["pixels"]
    from PIL import Image
    img = np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(img).save(out_png)
    print(f"Saved: {out_png}  总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "step":
        one_step(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        drive(sys.argv[2], sys.argv[3], sys.argv[4])
