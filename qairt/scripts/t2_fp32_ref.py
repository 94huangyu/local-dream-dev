# -*- coding: utf-8 -*-
"""L=80 的 FP32 参考图（Tier 2 的 L3 一致性测量需要它）。

## 为什么不能直接用 fp32_step_runner.py
它走**三段**布局（`transformer_part2_fixed_dce`），L=80 下没有这个手术产物。
本脚本改用**已验证的四段 L=80 链**（`t2_chain_L80.py known` 实测：L=32 下四段链的
part2b 输出与盘上 `vs_fp32/latents_fp32_s0.raw` **逐字节相同**）。

## 调度器与 VAE 喂法**逐字照抄** fp32_step_runner.py
`sigmas()`、Euler 更新 `nxt = lat - dt*noise`、以及 VAE 的正确喂法
`lat + VAE_SHIFT*VAE_SCALING`（#80：图内首节点已做反缩放，外面再做就是除两遍）。
**不得自己重新推导**。

## 已知样本验证（约束 8）
`drive` 支持 `--L 32`：用同一份驱动在 L=32 下跑完 8 步，
与盘上 `scratch_runs/fp32_final_latents_inloopref_vaefix.raw` 比对。
**不过就不许信 L=80 的数** —— 否则那是从一条没验证过的流水线里出来的。

## 每步独立进程
沿用 fp32_step_runner.py 的结论（六次尝试换来的）：单进程跑 8 步会被内存拖死。

用法:
  python t2_fp32_ref.py caption <ids.npy> <L> <out_caption.raw> <out_mask.raw>
  python t2_fp32_ref.py step  <L> <idx> <lat_in> <lat_out> <caption> <mask>
  python t2_fp32_ref.py drive <L> <lat_init> <caption> <mask> <out.png>
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                                     # noqa: E402

E = canonical_sources.ONNX_DIR
P0 = r"D:\ZImage_Work\p0_experiments"
OUT_DIR = r"D:\LocalDreamZImage\scratch_runs"
STEPS, LATENT_CH = 8, 16
# 🔴 2026-09-02 追加（台账 #86 路线 D）：比例手术产物的支持。
# 三个环境变量**未设时，本脚本行为与追加前逐字节相同** —— 不动已验证的 L=80 路径。
ZI_TAG = os.environ.get("ZI_TAG", "")               # 例 r4x3；空 = 原 1:1 行为
ZI_LAT_H = int(os.environ.get("ZI_LAT_H", "128"))
ZI_LAT_W = int(os.environ.get("ZI_LAT_W", "128"))
VAE_SCALING, VAE_SHIFT = 0.3611, 0.1159        # 抄自 fp32_step_runner.py，勿改


def sigmas(n, shift=3.0):
    """逐字抄自 fp32_step_runner.py。"""
    f = lambda s: shift * s / (1.0 + (shift - 1.0) * s)      # noqa: E731
    out = [f(1.0 if n == 1 else 1.0 - (1.0 - 1.0 / n) * i / (n - 1)) for i in range(n)]
    return np.array(out + [0.0], dtype=np.float64)


def seg_path(seg, L):
    """L=32 用 canonical base；L=80 用其手术产物。源名一律由 canonical_sources 给出。"""
    p = canonical_sources.onnx_for(seg, "base")
    if L == 32:
        return p
    stem = os.path.splitext(os.path.basename(p))[0]
    suf = ("_L%d_%s" % (L, ZI_TAG)) if ZI_TAG else ("_L%d" % L)
    q = os.path.join(os.path.dirname(p), "%s%s.onnx" % (stem, suf))
    if not os.path.isfile(q):
        raise FileNotFoundError("缺 %s（先跑 dit_seq_surgery.py %d --variant=base）" % (q, L))
    return q


def _sess(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        return ort.InferenceSession(os.path.basename(path), so,
                                    providers=["CPUExecutionProvider"])
    finally:
        os.chdir(cwd)


def _run(sess, feed):
    import onnx
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feed)))


def caption(ids_npy, L, out_cap, out_mask):
    """用 FP32 ONNX text_encoder 链产 L 槽的 caption（忠于部署链，见 t2_caption_L80.py）。"""
    import onnxruntime as ort
    from onnx import TensorProto                              # noqa: F401
    PAD = 151643
    ids = np.load(ids_npy).tolist()
    n = min(len(ids), L)
    a = np.full((1, L), PAD, dtype=np.int64); a[0, :n] = ids[:n]
    m = np.zeros((1, L), dtype=np.int64); m[0, :n] = 1
    t = {"input_ids": a, "attention_mask": m}
    # 🔴 text_encoder 的槽数是**烘焙在图里**的，源文件名与 L 的对应关系
    #    必须查表，不能按“默认就是 32”这种朴素规则猜（#61/#138 同类）：
    #      L=20 -> 原始导出（`text_encoder_partN.onnx`，图内写死 {1,20,-1,128}）
    #      L=32 -> Tier 1 手术产物 `_s32`
    #      L=80 -> Tier 2 手术产物 `_L80`
    #    2026-08-29 实测：按旧规则拿 L=32 去跑原始图，Reshape 当场报
    #    `Input shape:{1,32,1024}, requested shape:{1,20,-1,128}`。
    SUF = {20: "", 32: "_s32", 80: "_L80"}
    if L not in SUF:
        raise ValueError("L=%d 没有对应的 text_encoder 产物" % L)
    suf = SUF[L]
    cwd = os.getcwd(); os.chdir(E)
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        for p in (1, 2, 3, 4):
            f = "text_encoder_part%d%s.onnx" % (p, suf)
            se = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
            feed = {}
            for inp in se.get_inputs():
                v = t[inp.name]
                feed[inp.name] = (v.astype(np.int32) if "int32" in inp.type
                                  else v.astype(np.float32) if "float" in inp.type else v)
            for o, val in zip(se.get_outputs(), se.run(None, feed)):
                t[o.name] = val
            del se
            print("   te part%d ok" % p, flush=True)
    finally:
        os.chdir(cwd)
    cap = np.ascontiguousarray(np.asarray(t["caption"], np.float32).reshape(-1, 2560))
    assert cap.shape[0] == L, "caption 行数 %d != %d" % (cap.shape[0], L)
    mask = np.ones((1, L), np.float32); mask[0, :n] = 0.0     # 0=真实 1=pad（实测极性）
    cap.tofile(out_cap); mask.tofile(out_mask)
    print("caption %s -> %s ；mask 真实槽 %d / pad %d -> %s"
          % (cap.shape, out_cap, n, L - n, out_mask))
    return 0


def one_step(L, idx, lat_in, lat_out, cap_path, mask_path):
    from onnx import TensorProto
    import onnx
    latents = np.fromfile(lat_in, np.float32).reshape(1, LATENT_CH, ZI_LAT_H, ZI_LAT_W)
    cap = np.fromfile(cap_path, np.float32).reshape(1, L, 2560)
    mask = np.fromfile(mask_path, np.float32).reshape(1, L)
    ts = sigmas(STEPS)[:-1] * 1000.0
    timestep = np.array([1.0 - ts[idx] / 1000.0], dtype=np.float32)

    t = {"latents": latents, "timestep": timestep, "caption": cap, "cap_pad_mask": mask}
    for seg in ("part1a", "part1b", "part2a", "part2b"):
        path = seg_path(seg, L)
        m = onnx.load(path, load_external_data=False)
        need = [(i.name, i.type.tensor_type.elem_type) for i in m.graph.input]
        miss = [n for n, _ in need if n not in t]
        if miss:
            raise KeyError("%s 缺输入 %s" % (seg, miss))
        se = _sess(path)
        feed = {n: (t[n].astype(bool) if e == TensorProto.BOOL else t[n]) for n, e in need}
        for o, v in zip(se.get_outputs(), se.run(None, feed)):
            t[o.name] = (np.ascontiguousarray(v, np.float32) if v.dtype != np.bool_
                         else v.astype(np.float32))
        del se, m
        gc.collect()

    noise = t["latents"].reshape(-1).astype(np.float64)
    s0 = ts[idx] / 1000.0
    s1 = ts[idx + 1] / 1000.0 if idx + 1 < STEPS else 0.0
    dt = s1 - s0
    nxt = np.fromfile(lat_in, np.float32).reshape(-1).astype(np.float64) - dt * noise
    nxt.astype(np.float32).tofile(lat_out)
    print("  step %d: dt=%.4f noise_std=%.4f latents_std=%.4f"
          % (idx, dt, noise.std(), nxt.std()), flush=True)
    return 0


def drive(L, lat_init, cap_path, mask_path, out_png):
    work = os.path.join(P0, "fp32_steps_L%d%s" % (L, ("_" + ZI_TAG) if ZI_TAG else ""))
    os.makedirs(work, exist_ok=True)
    cur = os.path.join(work, "lat_0.raw")
    np.fromfile(lat_init, np.float32).tofile(cur)
    t0 = time.time()
    for i in range(STEPS):
        nxt = os.path.join(work, "lat_%d.raw" % (i + 1))
        r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "step",
                            str(L), str(i), cur, nxt, cap_path, mask_path])
        if r.returncode != 0 or not os.path.isfile(nxt):
            raise SystemExit("step %d 失败 rc=%d" % (i, r.returncode))
        cur = nxt
        print("[%.0fs] step %d 完成" % (time.time() - t0, i), flush=True)

    # 🔴🔴 2026-09-03：这里原本是【固定文件名】，被每次 drive 覆盖。
    # 后果（真实发生）：我为 VAE 校准与对照臂各跑了多次 drive，于是
    #   fp32_final_latents_L80.raw        实际存着 seed44
    #   fp32_final_latents_L80_1184x896.raw 实际存着 seed46
    # 而我仍拿它当「seed42 的参考」去比 PSNR ⇒ 两臂的 seed42 数据点全是
    # 「seed42 量化输出 vs 另一个 seed 的 FP32」，落在 12 dB（无关图的基线），
    # 让我一度误判「换比例造成 9.61 dB 回退」。
    # ⚠️ `htp_inloop_pipeline.py` 文件头早就警告过「唯一副本会被下一次运行覆盖」——
    #    我读过那个文件却没把纪律用到这里（信息在手却没连起来）。
    # 修法：文件名带上【初始噪声的 basename】，天然按输入隔离。
    _seed_tag = os.path.splitext(os.path.basename(lat_init))[0]
    fin = os.path.join(OUT_DIR, "fp32_final_latents_L%d%s__%s.raw"
                       % (L, ("_" + ZI_TAG) if ZI_TAG else "", _seed_tag))
    np.fromfile(cur, np.float32).tofile(fin)
    lat = np.fromfile(cur, np.float32).reshape(1, LATENT_CH, ZI_LAT_H, ZI_LAT_W)
    # #80：vae_decoder.onnx 首节点已做 Div(vae_latents, 0.3611)，外面只能加 shift*scale
    vae_in = (lat.astype(np.float64) + VAE_SHIFT * VAE_SCALING).astype(np.float32)
    v = _sess(os.path.join(E, "vae_decoder%s.onnx" % (("_" + ZI_TAG) if ZI_TAG else "")))
    px = _run(v, {"vae_latents": vae_in})["pixels"]
    del v
    from PIL import Image
    img = np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(img).save(out_png)
    print("Saved: %s  终点 latents: %s  总耗时 %.0fs" % (out_png, fin, time.time() - t0),
          flush=True)

    # 已知样本验证：L=32 必须复现盘上的 FP32 终点 latents
    if L == 32 and not ZI_TAG:
        ref = os.path.join(OUT_DIR, "fp32_final_latents_inloopref_vaefix.raw")
        if os.path.isfile(ref):
            a = np.fromfile(fin, np.float32); b = np.fromfile(ref, np.float32)
            same = np.array_equal(a, b)
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            print("\nG0-known（L=32 复现盘上 FP32 终点 latents）：")
            print("  逐字节相同=%s  相对 L2=%.6f%%" % (same, rel * 100))
            print("  判定: %s" % ("PASS 驱动可信，可用于 L=80" if same or rel < 1e-4
                                  else "FAIL 驱动与盘上参考不符，L=80 的数不许用"))
            return 0 if (same or rel < 1e-4) else 1
        print("\n⚠️ 找不到 %s，无法做 G0-known" % ref)
    return 0


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "caption":
        sys.exit(caption(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]))
    if c == "step":
        sys.exit(one_step(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5],
                          sys.argv[6], sys.argv[7]))
    if c == "drive":
        sys.exit(drive(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    sys.exit("未知模式 %s" % c)
