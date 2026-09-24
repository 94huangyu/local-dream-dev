"""在 QDQ 模拟模型上跑完整 8 步 + VAE，得到 L1/L2/L3（EXP_PLAN_SIM_CEILING §二）。

结构照抄 `fp32_step_runner.py` 的**每步一个独立进程**（2026-08-18 六次尝试的结论：
单进程跑 8 步必被内存拖死，退出即彻底回收）—— 但改成**四段**，与部署布局一致
（part1a → part1b → part2a → part2b），因为 encoding 是按四段导出的。

🔴 措辞纪律：本脚本的结果一律写「**在 QDQ 模拟上**」，
   **不得**写成「设备实测」（同约束 3 对 CPU 参考的要求）。

用法:
  python sim_runner.py step <mode> <idx> <lat_in.raw> <lat_out.raw> <noise_out.raw>
  python sim_runner.py drive <mode> <tag>        # 8 步 + VAE，产出图与 latents
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
OUTB = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
CAP = os.path.join(P0, "htp_inloop", "const", "caption.raw")
MASK = os.path.join(P0, "htp_inloop", "const", "cap_pad_mask.raw")
LAT0 = os.path.join(P0, "htp_inloop", "s0", "latents.raw")
STEPS, LATENT_CH = 8, 16
SEGS = ["transformer_part1a", "transformer_part1b",
        "transformer_part2a_fixed", "transformer_part2b_fixed"]


def sigmas(n, shift=3.0):
    def f(s):
        return shift * s / (1.0 + (shift - 1.0) * s)
    return np.array([f(1.0 - (1.0 - 1.0 / n) * i / (n - 1)) for i in range(n)] + [0.0],
                    dtype=np.float64)


def load(name, mode):
    import onnxruntime as ort
    p = os.path.join(E, name + ("" if mode == "FP32" else "_sim%s" % mode) + ".onnx")
    if not os.path.exists(p):
        sys.exit("FAIL 模型不存在: %s" % p)
    t0 = time.time()
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print("    load %-34s %.0fs" % (os.path.basename(p), time.time() - t0), flush=True)
    return s


def run(sess, **kw):
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, kw)))


def one_step(mode, idx, lat_in, lat_out, noise_out):
    latents = np.fromfile(lat_in, np.float32).reshape(1, LATENT_CH, 128, 128)
    caption = np.fromfile(CAP, np.float32).reshape(1, 32, 2560)
    cap_pad_mask = np.fromfile(MASK, np.float32).reshape(1, 32) > 0.5
    ts = sigmas(STEPS)[:-1] * 1000.0
    timestep = np.array([1.0 - ts[idx] / 1000.0], dtype=np.float32)

    a = load(SEGS[0], mode)
    o1a = run(a, latents=latents, timestep=timestep, caption=caption,
              cap_pad_mask=cap_pad_mask)
    del a
    gc.collect()
    b = load(SEGS[1], mode)
    o1b = run(b, adaln_input=o1a["adaln_input"], add_131=o1a["add_131"],
              add_138=o1a["add_138"], select_45=o1a["select_45"],
              select_46=o1a["select_46"], tanh_19=o1a["tanh_19"])
    del b
    gc.collect()
    c = load(SEGS[2], mode)
    o2a = run(c, unified=o1b["unified"], unified_mask=o1a["unified_mask"],
              unified_freqs=o1a["unified_freqs"], adaln_input=o1a["adaln_input"])
    del c
    gc.collect()
    d = load(SEGS[3], mode)
    feed = {k: o2a[k] for k in ("add_92", "select", "select_1",
                                "split_7_split_2", "split_7_split_3", "val_105")
            if k in o2a}
    feed["adaln_input"] = o1a["adaln_input"]
    o2b = run(d, **feed)
    del d
    gc.collect()

    noise = o2b["latents"].reshape(-1).astype(np.float64)
    noise.astype(np.float32).tofile(noise_out)
    s0 = ts[idx] / 1000.0
    s1 = ts[idx + 1] / 1000.0 if idx + 1 < STEPS else 0.0
    dt = s1 - s0
    nxt = latents.reshape(-1).astype(np.float64) - dt * noise
    nxt.astype(np.float32).tofile(lat_out)
    print("  step %d: dt=%.4f noise_std=%.4f latents_std=%.4f"
          % (idx, dt, noise.std(), nxt.std()), flush=True)


def drive(mode, tag):
    work = os.path.join(P0, "sim_%s" % tag)
    os.makedirs(work, exist_ok=True)
    cur = os.path.join(work, "lat_0.raw")
    np.fromfile(LAT0, np.float32).tofile(cur)
    t00 = time.time()
    for i in range(STEPS):
        nxt = os.path.join(work, "lat_%d.raw" % (i + 1))
        noi = os.path.join(work, "noise_%d.raw" % i)
        # 🔴 每步独立进程：单进程跑 8 步必被内存拖死（fp32_step_runner 的六次实测结论）
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "step", mode,
                            str(i), cur, nxt, noi],
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            sys.exit("FAIL step %d rc=%d" % (i, r.returncode))
        cur = nxt
        print("  [%s] %d/%d 累计 %.1f min" % (tag, i + 1, STEPS, (time.time() - t00) / 60),
              flush=True)
    print("  VAE 解码（独立进程）...", flush=True)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "vae", cur,
                        os.path.join(OUTB, "sim_%s.png" % tag)],
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if r.returncode != 0:
        print("  ⚠️ VAE 失败 rc=%d —— latents 已保存在 %s，可另行解码" % (r.returncode, work),
              flush=True)
    print("  完成，总计 %.1f min" % ((time.time() - t00) / 60), flush=True)


def vae(lat_path, out_png):
    """VAE 解码。🔴 喂法必须是 `lat + shift*scaling`（台账 #80）：
    `vae_decoder.onnx` 首节点就是 Div(vae_latents, 0.3611)，反缩放**图内已做**；
    调用方再除一遍就是除了两遍（对官方 23.51 dB vs 正确喂法 79.92 dB）。"""
    VAE_SCALING, VAE_SHIFT = 0.3611, 0.1159
    lat = np.fromfile(lat_path, np.float32).reshape(1, LATENT_CH, 128, 128)
    vin = (lat.astype(np.float64) + VAE_SHIFT * VAE_SCALING).astype(np.float32)
    import onnxruntime as ort
    s = ort.InferenceSession(os.path.join(E, "vae_decoder.onnx"),
                             providers=["CPUExecutionProvider"])
    px = s.run(["pixels"], {"vae_latents": vin})[0]
    from PIL import Image
    img = np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(img).save(out_png)
    print("  Saved: %s" % out_png, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "vae":
        vae(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "step":
        one_step(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        drive(sys.argv[2], sys.argv[3])
