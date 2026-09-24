"""
Test B（方案见 scripts/EXP_PLAN_V2.md）：唯一能分离"量化"与"C++实现"的实验。

同一个 Python 脚本、同一 prompt、同一 seed(42)，只把去噪循环里的 part1a/1b/2
从 onnxruntime 换成 snpe-net-run 跑量化 DLC，其余全部保持 FP32：

  文本编码(FP32) -> [8步: part1a(量化) -> part1b(量化) -> part2(量化) -> 调度器(FP32)] -> VAE(FP32) -> PNG

产出的图与 scratch_runs/zimage_fp32_full_pipeline.png 直接可比（同 prompt 同 seed）。

验收标准（执行前定稿于 EXP_PLAN_V2.md，不得事后修改）：
  出图≈清晰的猫   -> C1 被否定，transformer 量化不是根因，转查 C++ 实现
  出图≈橙色色块   -> C1 被确认，量化足以造成该失败
  介于两者之间     -> C1 有贡献但不足以解释全部
"""
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
DLC_DIR = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline"
TOKENIZER_PATH = r"D:\ZIMAGE\tokenizer\tokenizer.json"
OUT_DIR = r"D:\LocalDreamZImage\scratch_runs"
WORK = r"D:\ZImage_Work\p0_experiments\testB"
SNPE = r"D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\snpe-net-run.exe"
QNN_LIB = r"D:\qairt\2.48.0.260626\lib\x86_64-windows-msvc"
QNN_BIN = r"D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc"

TEXT_MAX_LEN, QWEN_PAD_ID = 20, 151643
LATENT_CH, STEPS = 16, 8
VAE_SCALING, VAE_SHIFT = 0.3611, 0.1159
PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"
SEED = 42

# part1a 的输出里，下游真正要用到的
P1A_OUTS = ["adaln_input", "add_131", "add_138", "select_45", "select_46",
            "tanh_19", "unified_mask", "unified_freqs"]
P1B_INS = ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]
# 注意：part2 的 DLC 里【没有】latents_shape 这个输入（转换时被常量折叠掉了），
# 用 snpe-dlc-info 核对过。多喂会报 "invalid input size provided"。
P2_INS = ["unified", "unified_mask", "unified_freqs", "adaln_input"]

# 输入数据类型：见 ../docs/EXECUTION_MODEL.md 规则 1 ——
# snpe-net-run 在未加 --use_native_input_files 时，【所有】输入一律写 float32，
# 与 DLC 内部声明的 Bool_8 / uFxp_16 / Int_32 无关。
#
# 实测反例（务必不要重犯）：把 cap_pad_mask 按 Bool_8 写成 32 字节 uint8 后，snpe 不报错，
# 但用「输入字节数 ÷ 期望字节数」= 32/128 = 1/4 推断批次，导致【所有输出张量变成 1/4 尺寸】。
DLC_IN_DTYPE = {}   # 全部 float32


def log(*a):
    print(*a, flush=True)


def sigmas_of(num_steps, shift=3.0):
    def sh(s):
        return shift * s / (1.0 + (shift - 1.0) * s)
    out = []
    for i in range(num_steps):
        raw = 1.0 if num_steps == 1 else 1.0 - (1.0 - 1.0 / num_steps) * i / (num_steps - 1)
        out.append(sh(raw))
    out.append(0.0)
    return np.array(out, dtype=np.float64)


def ort_run(name, feeds):
    s = ort.InferenceSession(f"{EVIDENCE}\\{name}.onnx", providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()]
    r = dict(zip(outs, s.run(outs, feeds)))
    del s
    return r


def write_raw(name, arr, path):
    """按 DLC 声明的数据类型写输入。
    V1 检查（加强版）：字节数必须等于 元素数 × 该 dtype 的字节宽度。
    旧版只校验 "元素数×4"，无法发现 Bool_8 被当 float32 写这类错误。"""
    dt = DLC_IN_DTYPE.get(name, np.float32)
    a = np.ascontiguousarray(arr.astype(dt))
    a.tofile(path)
    got, expect = os.path.getsize(path), a.size * np.dtype(dt).itemsize
    assert got == expect, f"V1 FAIL {path}: {got} != {expect} (dtype={dt.__name__})"
    return path


def snpe_run(part, feeds, want, shapes, tag):
    """跑一段量化 DLC。feeds: {name: ndarray}；want: 要导出的张量名列表"""
    d = f"{WORK}\\{tag}_{part}"
    os.makedirs(d, exist_ok=True)
    raws = {k: write_raw(k, v, f"{d}\\{k}.raw") for k, v in feeds.items()}
    lst = f"{d}\\list.txt"
    with open(lst, "w") as f:
        f.write("%" + " ".join(want) + "\n")
        f.write(" ".join(f"{k}:={v}" for k, v in raws.items()) + "\n")
    env = dict(os.environ, PATH=f"{QNN_LIB};{QNN_BIN};" + os.environ["PATH"])
    t0 = time.time()
    r = subprocess.run([SNPE, "--container", f"{DLC_DIR}\\{part}\\{part}_quantized.dlc",
                        "--input_list", lst, "--output_dir", f"{d}\\out"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout[-3000:]); log(r.stderr[-3000:])
        raise RuntimeError(f"snpe-net-run failed: {part} ({tag})")
    res = {}
    for w in want:
        # ../docs/EXECUTION_MODEL.md 规则 2：输出一律 float32，即使张量声明为 Bool_8 / Int_32
        a = np.fromfile(f"{d}\\out\\Result_0\\{w}.raw", dtype=np.float32)
        exp = int(np.prod(shapes[w]))
        # 这条断言是喂错输入的唯一防线：snpe 输入尺寸不对时不报错，只让输出按比例缩水
        assert a.size == exp, (f"V1 FAIL {w}: 输出 {a.size} != 期望 {exp} "
                               f"（比值 {a.size/exp:.3f}）——通常意味着某个输入的 .raw 尺寸不对，"
                               f"snpe 按比例缩了批次。见 ../docs/EXECUTION_MODEL.md 规则 1")
        res[w] = a.reshape(shapes[w])
    log(f"    {part} ({tag}) done in {time.time()-t0:.1f}s")
    return res


def main():
    os.makedirs(WORK, exist_ok=True)

    log("=== 分词 + FP32 文本编码（与 FP32 基准完全一致）===")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    ids = tok.encode(f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n").ids
    input_ids = np.full((1, TEXT_MAX_LEN), QWEN_PAD_ID, dtype=np.int32)
    attention_mask = np.zeros((1, TEXT_MAX_LEN), dtype=np.int32)
    n = min(len(ids), TEXT_MAX_LEN)
    input_ids[0, :n] = ids[:n]
    attention_mask[0, :n] = 1
    log(f"token_count={n}")

    t = {"input_ids": input_ids, "attention_mask": attention_mask}
    for name, ins in [("text_encoder_part1", ["input_ids", "attention_mask"]),
                      ("text_encoder_part2", ["add_2452", "attention_mask"]),
                      ("text_encoder_part3", ["add_4828", "attention_mask"]),
                      ("text_encoder_part4", ["add_7204", "attention_mask"])]:
        t.update(ort_run(name, {k: t[k] for k in ins}))
    caption = t["caption"].astype(np.float32)
    log(f"caption shape={caption.shape}")

    # cap_pad_mask: True = 该位置是 padding，应被替换为 pad token
    # （2026-08-13 对照 diffusers transformer_z_image.py 确认的极性）
    cap_pad_mask = np.ones((1, 32), dtype=np.float32)
    cap_pad_mask[0, :min(n, 32)] = 0.0

    rng = np.random.default_rng(SEED)
    latents = rng.standard_normal(LATENT_CH * 128 * 128).astype(np.float32)
    latents_4d = latents.reshape(1, LATENT_CH, 128, 128)
    log(f"initial latents std={latents.std():.4f}")

    sig = sigmas_of(STEPS)
    timesteps = sig[:-1] * 1000.0

    shapes_1a = {"adaln_input": (1, 256), "add_131": (1, 1, 3840), "add_138": (1, 4128, 3840),
                 "select_45": (1, 4128, 1, 64), "select_46": (1, 4128, 1, 64),
                 "tanh_19": (1, 1, 3840), "unified_mask": (1, 4128),
                 "unified_freqs": (1, 4128, 64, 2)}
    shapes_1b = {"unified": (1, 4128, 3840)}
    shapes_2 = {"latents": (1, LATENT_CH, 128, 128)}

    log("\n=== 去噪循环（transformer 三段全部走量化 DLC）===")
    for step in range(STEPS):
        ts = np.array([1.0 - timesteps[step] / 1000.0], dtype=np.float32)
        log(f"  step {step}: timestep={ts[0]:.4f}")

        o1a = snpe_run("transformer_part1a",
                       {"latents": latents_4d, "timestep": ts,
                        "caption": caption, "cap_pad_mask": cap_pad_mask},
                       P1A_OUTS, shapes_1a, f"s{step}")

        # 步0：与 FP32 对照，作为有效性检查 + 早期信号
        if step == 0:
            log("    [step0 对照] 同输入跑 FP32 part1a...")
            f1a = ort_run("transformer_part1a",
                          {"latents": latents_4d, "timestep": ts, "caption": caption,
                           "cap_pad_mask": cap_pad_mask.astype(bool)})
            for k in ["add_138", "adaln_input"]:
                e = np.abs(o1a[k] - f1a[k])
                log(f"    [step0] {k}: 误差max={e.max():.4f} "
                    f"相对={e.max()/max(np.abs(f1a[k]).max(),1e-9)*100:.2f}%")

        o1b = snpe_run("transformer_part1b", {k: o1a[k] for k in P1B_INS},
                       ["unified"], shapes_1b, f"s{step}")
        feeds2 = {"unified": o1b["unified"]}
        feeds2.update({k: o1a[k] for k in P2_INS if k != "unified"})
        o2 = snpe_run("transformer_part2", feeds2, ["latents"], shapes_2, f"s{step}")

        noise = o2["latents"].reshape(-1).astype(np.float64)

        if step == 0:
            log("    [step0 对照] 同输入跑 FP32 part1b+part2...")
            f1b = ort_run("transformer_part1b", {k: f1a[k] for k in P1B_INS})
            # FP32 的 part2.onnx 仍然声明了 latents_shape 输入（只有量化 DLC 里被折叠掉了）
            ff2 = {"unified": f1b["unified"]}
            ff2.update({k: f1a[k] for k in P2_INS if k != "unified"})
            ff2["latents_shape"] = f1a["latents_shape"]
            f2 = ort_run("transformer_part2", ff2)
            nf = f2["latents"].reshape(-1).astype(np.float64)
            rel = np.linalg.norm(noise - nf) / np.linalg.norm(nf) * 100
            log("    " + "=" * 60)
            log(f"    [step0 关键指标] 噪声预测相对误差 = {rel:.3f}%")
            log(f"    FP32 noise std={nf.std():.4f}  量化 noise std={noise.std():.4f}")
            log("    " + "=" * 60)

        s0 = timesteps[step] / 1000.0
        s1 = timesteps[step + 1] / 1000.0 if step + 1 < STEPS else 0.0
        dt = s1 - s0
        flat = latents_4d.reshape(-1).astype(np.float64) - dt * noise
        latents_4d = flat.astype(np.float32).reshape(1, LATENT_CH, 128, 128)
        log(f"    dt={dt:.4f} noise_std={noise.std():.4f} latents_std={flat.std():.4f}")

    log("\n=== VAE 解码（FP32）===")
    # 🔴 2026-08-21 修正（HANDOVER 15.22 / 台账 #80）：vae_decoder.onnx 的首节点就是
    #    Div(vae_latents, 0.3611)，反缩放【图内已做】。此处原本又做了一遍 lat/s+shift，
    #    等于除了两遍（对官方 23.51 dB / 高频 1.3049x）。正确喂法是 lat + shift*s
    #    （对官方 79.92 dB / 高频 1.0000x）。改前不要动，先读 §二十。
    vae_in = (latents_4d.astype(np.float64) + VAE_SHIFT * VAE_SCALING).astype(np.float32)
    px = ort_run("vae_decoder", {"vae_latents": vae_in})["pixels"]

    from PIL import Image
    img = np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    out = f"{OUT_DIR}\\testB_hybrid_quantized_transformer.png"
    Image.fromarray(img).save(out)
    log("Saved:", out)


if __name__ == "__main__":
    main()
