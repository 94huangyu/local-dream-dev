"""EXP_PLAN_GAMMA_BW 第 2 步：跑三段 FP32 ONNX 链，取 step-0 噪声预测。

用法: python gamma_bw_run.py {A|B}
  A = 原始 ONNX（装置门 / 已知答案样本，必须复现 latents_fp32_s0.raw）
  B = gamma 8-bit Q/DQ 版

⚠️ 内存：part1 外部权重 13.7 GB、part2 10.9 GB，本机总 25.49 GB。
   **两臂必须分进程串行**，不得并发（2026-08-15 曾吃穿 23.7 GB 压死宿主）。

输入与 scripts/fp32_noise_ref.py 逐字节相同（同 seed / caption / timestep），
代码路径也照抄该脚本，**不另起炉灶**——V0 门就是用来抓"我的装置和产生原点那次不同"的。
"""
import gc, os, sys, time
import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")
EVID = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
WORK = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "htp_inloop")
OUT  = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "gamma_bw")

P1A_OUTS = ["adaln_input", "add_131", "add_138", "latents_shape", "select_45",
            "select_46", "tanh_19", "unified_freqs", "unified_mask"]
P1B_INS = ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]
P2_INS  = ["unified", "unified_mask", "unified_freqs", "adaln_input", "latents_shape"]
SHAPES = {"latents": (1, 16, 128, 128), "caption": (1, 32, 2560),
          "timestep": (1,), "cap_pad_mask": (1, 32)}


def run(model, feeds, want):
    p = os.path.join(EVID, model + ".onnx")
    t0 = time.time()
    print("  跑 %s ..." % model, flush=True)
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    r = s.run(want, feeds)
    del s
    gc.collect()
    print("     完成 %.1f s" % (time.time() - t0), flush=True)
    return dict(zip(want, r))


def main():
    arm = sys.argv[1].upper()
    assert arm in ("A", "B")
    sfx = "" if arm == "A" else "_gq8"
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "latents_%s_s0.raw" % arm)
    t0 = time.time()

    feeds = {
        "latents": np.fromfile(os.path.join(WORK, "s0", "latents.raw"), np.float32
                               ).reshape(SHAPES["latents"]),
        "caption": np.fromfile(os.path.join(WORK, "const", "caption.raw"), np.float32
                               ).reshape(SHAPES["caption"]),
        "timestep": np.fromfile(os.path.join(WORK, "s0", "timestep.raw"), np.float32
                                ).reshape(SHAPES["timestep"]),
        # ONNX 侧 cap_pad_mask 是 bool（设备/QNN 侧才一律 float32）。极性见 fp32_noise_ref.py 注释。
        "cap_pad_mask": np.fromfile(os.path.join(WORK, "const", "cap_pad_mask.raw"), np.float32
                                    ).reshape(SHAPES["cap_pad_mask"]).astype(bool),
    }
    print("臂 %s   模型后缀 '%s'" % (arm, sfx), flush=True)
    for k, v in feeds.items():
        print("  输入 %-14s %s  std=%.4f" % (k, v.shape, v.std()), flush=True)

    o1a = run("transformer_part1a" + sfx, feeds, P1A_OUTS)
    del feeds; gc.collect()
    o1b = run("transformer_part1b" + sfx, {k: o1a[k] for k in P1B_INS}, ["unified"])
    f2 = {"unified": o1b["unified"]}
    f2.update({k: o1a[k] for k in P2_INS if k != "unified"})
    del o1a, o1b; gc.collect()
    o2 = run("transformer_part2" + sfx, f2, ["latents"])

    noise = np.ascontiguousarray(o2["latents"].reshape(-1).astype(np.float32))
    noise.tofile(dst)
    print("\n写出 %s (%d 元素, std=%.4f)  总耗时 %.1f s"
          % (dst, noise.size, noise.std(), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
