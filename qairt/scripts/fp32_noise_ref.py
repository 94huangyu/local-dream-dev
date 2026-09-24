"""把"噪声预测"这把标定尺的【原点】补上，并永久保存。

为什么必须做（约束 7 / 约束 9 第 1 条，2026-08-16 自审抓到）：
本项目唯一有刻度的量是 step-0 噪声预测相对 L2 ——
  **15.988% ⇒ 成图完好；47.07% ⇒ 橙色色块**
但这两个刻度的参考是 **FP32 transformer**，而该 FP32 噪声预测**从未被保存**。
我一度拿"距 Test B CPU 参考的 45.72%"去和这两个刻度比 —— **参考系不同，无效比较**。

本脚本用 FP32 ONNX 跑一遍 part1a → part1b → part2，得到 step-0 噪声预测，
存为 `vs_fp32/latents_fp32_s0.raw`，此后所有配置一律以它为原点。

输入取 `htp_inloop` 那次运行**原封不动的同一批**（同 seed、同 caption、同 timestep），
保证与设备侧逐字节相同。

⚠️ 内存：part1 外部权重 13.7 GB、part2 10.9 GB，onnxruntime 会实际载入。
   **必须独占 PC 运行**，不得与量化/建图并发（2026-08-15 已因并发吃穿 23.7 GB 内存）。

用法: python fp32_noise_ref.py
"""
import gc
import os

import numpy as np
import onnxruntime as ort

EVID = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
WORK = r"D:\ZImage_Work\p0_experiments\htp_inloop"
OUT = r"D:\ZImage_Work\p0_experiments\vs_fp32"

# 输入/输出清单**逐字取自 ONNX 本身**，不照抄在环脚本：
#   in-loop 脚本喂 part2 只列了 4 个（`.bin` 路径下 latents_shape 已固化进图），
#   但 **FP32 ONNX 要求 5 个输入**，少一个直接报 Required inputs missing（①实测）。
P1A_OUTS = ["adaln_input", "add_131", "add_138", "latents_shape", "select_45",
            "select_46", "tanh_19", "unified_freqs", "unified_mask"]
P1B_INS = ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]
P2_INS = ["unified", "unified_mask", "unified_freqs", "adaln_input", "latents_shape"]

SHAPES = {"latents": (1, 16, 128, 128), "caption": (1, 32, 2560),
          "timestep": (1,), "cap_pad_mask": (1, 32)}


def run(model, feeds, want):
    p = f"{EVID}\\{model}.onnx"
    print(f"  跑 {model} ...", flush=True)
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    r = s.run(want, feeds)
    del s
    gc.collect()
    return dict(zip(want, r))


def main():
    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}\\latents_fp32_s0.raw"
    if os.path.exists(dst):
        print(f"已存在，直接复用: {dst}")
        return

    # 与设备侧 step0 逐字节相同的输入
    feeds = {
        "latents": np.fromfile(f"{WORK}\\s0\\latents.raw", np.float32
                               ).reshape(SHAPES["latents"]),
        "caption": np.fromfile(f"{WORK}\\const\\caption.raw", np.float32
                               ).reshape(SHAPES["caption"]),
        "timestep": np.fromfile(f"{WORK}\\s0\\timestep.raw", np.float32
                                ).reshape(SHAPES["timestep"]),
        # ⚠️ ONNX 侧 cap_pad_mask 是 **bool**（设备/QNN 侧才是一律 float32，见 EXECUTION_MODEL 规则2）。
        # 极性与参考实现 testB_hybrid_pipeline.py:180 一致：float32 的 0=真实 token、1=padding，
        # astype(bool) 后 False=真实 token、True=padding。本项目该张量的极性出过 bug，不得想当然。
        "cap_pad_mask": np.fromfile(f"{WORK}\\const\\cap_pad_mask.raw", np.float32
                                    ).reshape(SHAPES["cap_pad_mask"]).astype(bool),
    }
    for k, v in feeds.items():
        print(f"  输入 {k:<14} {v.shape}  std={v.std():.4f}")

    o1a = run("transformer_part1a", feeds, P1A_OUTS)
    del feeds
    gc.collect()

    o1b = run("transformer_part1b", {k: o1a[k] for k in P1B_INS}, ["unified"])
    f2 = {"unified": o1b["unified"]}
    f2.update({k: o1a[k] for k in P2_INS if k != "unified"})
    del o1a, o1b
    gc.collect()

    o2 = run("transformer_part2", f2, ["latents"])
    noise = np.ascontiguousarray(o2["latents"].reshape(-1).astype(np.float32))
    noise.tofile(dst)
    print(f"\n写出 {dst}  ({noise.size} 元素, std={noise.std():.4f})")
    print("此后所有噪声预测比较一律以它为原点。")


if __name__ == "__main__":
    main()
