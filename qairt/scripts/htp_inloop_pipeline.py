"""EXP_PLAN_HTP_INLOOP：在设备 HTP 上重做 Test B。

与 testB_hybrid_pipeline.py 【唯一】的差别：去噪循环里的 transformer 三段
从 `snpe-net-run 跑量化 .dlc（CPU 参考）` 换成 `qnn-net-run 跑 .bin（设备 HTP）`。
分词、FP32 文本编码、调度器、VAE、prompt、seed 全部保持不变。

Test B 的同一条流水线出的是【清晰的猫】(PSNR 23.89dB)。若本脚本出【橙色色块】，
则假设 C（HTP 执行 ≠ CPU 参考）在同一实验框架内被坐实为根因。

判据见 scripts/EXP_PLAN_HTP_INLOOP.md 第 3 节，不得事后修改。
"""
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
TOKENIZER_PATH = r"D:\ZIMAGE\tokenizer\tokenizer.json"
OUT_DIR = r"D:\LocalDreamZImage\scratch_runs"
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
T = "/data/local/tmp/htpcmp"          # 设备工作目录（已有 qnn-net-run + 各段 .bin + 全套库）
DEV = "3B1F65EA9BBUMSHZ"

# 配置标识。WORK 也随它变：s0/latents_dev.raw 是"step-0 噪声预测"的唯一副本，
# 不隔离就会被下一次运行覆盖掉（43.41% 那份就在 htp_inloop/s0 里）。
# PNG 在 15.17.6 已经差点被覆盖过一次，这里是同一类风险的另一半。
TAG = os.environ.get("INLOOP_TAG", "")
# 段 .bin 的名字后缀：设 INLOOP_BIN_SUFFIX=_fp16 则读 part1a_fp16.bin 等。
# 目的：**不覆盖设备上部署的那四个 .bin**（约束 11 铁律 2：覆盖前先备份 / 能不覆盖就不覆盖）。
BIN_SUFFIX = os.environ.get("INLOOP_BIN_SUFFIX", "")
WORK = r"D:\ZImage_Work\p0_experiments\htp_inloop" + (f"_{TAG}" if TAG else "")

# part2 的执行形态：`split` = part2a_fixed -> part2b_fixed（四段），其余 = 原三段
P2_MODE = os.environ.get("INLOOP_P2", "")

TEXT_MAX_LEN, QWEN_PAD_ID = 20, 151643
LATENT_CH, STEPS = 16, 8
VAE_SCALING, VAE_SHIFT = 0.3611, 0.1159
# 默认是强约束 prompt（明确指定「橘色的猫」「木头桌子」）。
# INLOOP_PROMPT 可换成别的 —— EXP_PLAN_PROMPT_SENS 要用手机上那个弱 prompt
# 来区分「prompt 约束强度」与「量化文本编码器」这两个假设。
PROMPT = os.environ.get(
    "INLOOP_PROMPT",
    "a cute orange cat sitting on a wooden table, masterpiece, best quality")
SEED = 42

P1A_OUTS = ["adaln_input", "add_131", "add_138", "select_45", "select_46",
            "tanh_19", "unified_mask", "unified_freqs"]
P1B_INS = ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]
# part2a 的输入集合与 baseline part2 【完全相同】（本轮从两个 .bin 的元数据 dump 逐个核对，
# 不由 ONNX 或文件名推断）⇒ 喂料代码原样复用。
P2_INS = ["unified", "unified_mask", "unified_freqs", "adaln_input"]
# 切口：part2a 的 6 个输出 = part2b 的输入（+ adaln_input，取自 part1a）
CUT = ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3", "val_105"]

# 🔴 2026-08-29（Tier 2）：SHAPES 原本把 4128 写死。Tier 2 把 caption 槽扩到 80 后
#    unified 变成 4176，而判据 V4（约束 3：防 qnn-net-run 静默缩批）拿 L=32 的期望值
#    去比 L=80 的真实产出，当场报 `add_92: 64143360 != 63406080`。
#    **设备算的是对的，是这张表停在旧长度上** —— 与 #73/#138 同一个模式
#    （同一个常量在多处各写一遍）。
#    ⇒ 改成从 caption 槽数派生。槽数由 INLOOP_CAPTION 的实际字节数反推，
#      没传就退回 32（保持既有行为逐字不变）。
#    🔴 判据本身不许放松：V4 是抓静默缩批的唯一防线。
def _cap_slots():
    p = os.environ.get("INLOOP_CAPTION", "")
    if p and os.path.isfile(p):
        n = os.path.getsize(p) // 4 // 2560
        if n * 4 * 2560 != os.path.getsize(p):
            raise SystemExit("INLOOP_CAPTION 字节数 %d 不是 2560*4 的整数倍"
                             % os.path.getsize(p))
        return n
    return 32


CAP_SLOTS = _cap_slots()
UNI = 4096 + CAP_SLOTS                 # unified 长度：图像 4096 固定 + caption 槽

SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840), "add_138": (1, UNI, 3840),
          "select_45": (1, UNI, 1, 64), "select_46": (1, UNI, 1, 64),
          "tanh_19": (1, 1, 3840), "unified_mask": (1, UNI),
          "unified_freqs": (1, UNI, 64, 2), "unified": (1, UNI, 3840),
          "add_92": (1, UNI, 3840), "select": (1, UNI, 1, 64),
          "select_1": (1, UNI, 1, 64), "split_7_split_2": (1, 1, 3840),
          "split_7_split_3": (1, 1, 3840), "val_105": (1, 1, 1, UNI),
          "latents": (1, LATENT_CH, 128, 128)}


def log(*a):
    print(*a, flush=True)


def sh(cmd, timeout=1800):
    """adb shell。**adb 自身失败必须立刻炸**，不得把空字符串当成"命令跑了但没输出"。

    2026-08-17 实测教训：设备中途掉线时本函数返回空串，
    调用方 `htp_run` 于是报 "qnn-net-run failed"，**把一次 USB 掉线伪装成模型执行失败**。
    诊断信息（`adb: device not found`）全在被丢弃的 stderr 里。
    """
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"adb shell 失败 (rc={r.returncode}): {cmd[:120]}\n"
                           f"stderr: {r.stderr.strip()}\nstdout: {r.stdout.strip()}")
    return r.stdout


def require_device(where):
    """每段执行前确认设备仍在。掉线要在"跑模型"之前被认出来，而不是之后被误判成模型问题。"""
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    if not any(DEV in l and "device" in l.split() for l in out.splitlines()[1:]):
        raise RuntimeError(f"设备 {DEV} 不在线（{where}）——USB 掉线或设备重启。\n"
                           f"adb devices:\n{out}")


def push(local, remote):
    r = subprocess.run([ADB, "-s", DEV, "push", local, remote],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"adb push failed: {local}\n{r.stderr}")


def pull(remote, local):
    r = subprocess.run([ADB, "-s", DEV, "pull", remote, local],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"adb pull failed: {remote}\n{r.stderr}")


def sigmas_of(num_steps, shift=3.0):
    def f(s):
        return shift * s / (1.0 + (shift - 1.0) * s)
    out = []
    for i in range(num_steps):
        raw = 1.0 if num_steps == 1 else 1.0 - (1.0 - 1.0 / num_steps) * i / (num_steps - 1)
        out.append(f(raw))
    out.append(0.0)
    return np.array(out, dtype=np.float64)


def ort_run(name, feeds):
    s = ort.InferenceSession(f"{EVIDENCE}\\{name}.onnx", providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs()]
    r = dict(zip(outs, s.run(outs, feeds)))
    del s
    return r


def put_raw(name, arr, step):
    """写 float32 到本地再推到设备。EXECUTION_MODEL 规则 1：一律 float32。"""
    os.makedirs(f"{WORK}\\s{step}", exist_ok=True)
    p = f"{WORK}\\s{step}\\{name}.raw"
    a = np.ascontiguousarray(arr.astype(np.float32))
    a.tofile(p)
    assert os.path.getsize(p) == a.size * 4, f"V1 FAIL {p}"
    push(p, f"{T}/loop/in/{name}.raw")
    return f"{T}/loop/in/{name}.raw"


def htp_run(part, feeds_dev, outdir, tag):
    """在设备 HTP 上跑一段 .bin。feeds_dev: {张量名: 设备上的 .raw 路径}"""
    require_device(f"{part}{BIN_SUFFIX}/{tag} 执行前")
    line = " ".join(f"{k}:={v}" for k, v in feeds_dev.items())
    sh(f"mkdir -p {T}/{outdir} && rm -rf {T}/{outdir}/Result_0")
    sh(f"printf '%s\\n' '{line}' > {T}/{outdir}/list.txt")
    t0 = time.time()
    out = sh(f"cd {T} && export LD_LIBRARY_PATH={T} && export ADSP_LIBRARY_PATH={T} && "
             f"./qnn-net-run --retrieve_context {part}{BIN_SUFFIX}.bin --backend libQnnHtp.so "
             f"--input_list {outdir}/list.txt --output_dir {T}/{outdir} --log_level error "
             f"2>&1 | tail -3")
    if "Finished Executing Graphs" not in out:
        raise RuntimeError(f"qnn-net-run failed [{part}/{tag}]:\n{out}")
    log(f"    {part} ({tag}) {time.time()-t0:.1f}s")
    return f"{T}/{outdir}/Result_0"


def check_dev_sizes(dev_dir, names, tag):
    """判据 V4（约束 3）：切口张量在【设备上】的字节数必须等于 float32 期望值。

    输出字节数是抓 `qnn/snpe-net-run` 静默缩批的唯一防线，而切口是本实验新引入、
    从未在环上跑过的一组张量 —— 恰恰是最该查的地方。
    """
    out = sh("stat -c '%s %n' " + " ".join(f"{dev_dir}/{n}.raw" for n in names))
    got = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 2 and p[0].isdigit():
            got[p[1].rsplit("/", 1)[-1][:-4]] = int(p[0])
    for n in names:
        exp = int(np.prod(SHAPES[n])) * 4
        assert got.get(n) == exp, (f"判据V4 FAIL [{tag}] {n}: {got.get(n)} != {exp} "
                                   f"(比值 {(got.get(n) or 0)/exp:.3f})，疑似静默缩批")
    log(f"    V4 切口 {len(names)} 个张量字节数全部正确 "
        f"（合计 {sum(int(np.prod(SHAPES[n]))*4 for n in names)/1048576:.1f} MB）")


def fetch(dev_dir, name, step):
    """把设备上的一个输出张量拉回来（只对真正需要回宿主的张量用）。"""
    loc = f"{WORK}\\s{step}\\{name}_dev.raw"
    pull(f"{dev_dir}/{name}.raw", loc)
    a = np.fromfile(loc, dtype=np.float32)
    exp = int(np.prod(SHAPES[name]))
    assert a.size == exp, (f"判据0 FAIL {name}: {a.size} != {exp} "
                           f"(比值 {a.size/exp:.3f})，疑似输入尺寸不对导致缩批")
    return a.reshape(SHAPES[name])


def main():
    os.makedirs(WORK, exist_ok=True)
    log(f"配置 INLOOP_TAG={TAG or '(无)'}  INLOOP_P2={P2_MODE or '(三段)'}")
    log(f"WORK = {WORK}")
    sh(f"mkdir -p {T}/loop/in")

    log("=== 分词 + FP32 文本编码（与 Test B 逐字节相同）===")
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

    cap_pad_mask = np.ones((1, 32), dtype=np.float32)
    cap_pad_mask[0, :min(n, 32)] = 0.0

    # 🔴 2026-08-29（Tier 2 / L=80）：允许**直接吃外部算好的 caption 与 mask**。
    #    动机：Tier 2 把 caption 槽扩到 80，而本脚本上面那段是按 TEXT_MAX_LEN=20 的
    #    原始 text_encoder 写死的（`(1, 32)` 也是写死的副本）。与其在这里再复制一份
    #    长度常量（#73/#138 反复栽的模式），不如让调用方把**与 FP32 参考臂逐字节相同**
    #    的 caption/mask 直接传进来 —— 这也正是 #131 确立的正确做法：
    #    **两臂都用 FP32 TE 算出的同一份 caption，把 TE 量化这个变量排除掉。**
    cap_ext = os.environ.get("INLOOP_CAPTION", "")
    msk_ext = os.environ.get("INLOOP_MASK", "")
    if cap_ext or msk_ext:
        if not (cap_ext and msk_ext):
            raise SystemExit("INLOOP_CAPTION 与 INLOOP_MASK 必须同时给")
        cap2 = np.fromfile(cap_ext, dtype=np.float32)
        msk2 = np.fromfile(msk_ext, dtype=np.float32)
        if cap2.size % 2560 != 0:
            raise SystemExit("caption 元素数 %d 不是 2560 的整数倍" % cap2.size)
        Lext = cap2.size // 2560
        if msk2.size != Lext:
            raise SystemExit("mask 长度 %d != caption 槽数 %d" % (msk2.size, Lext))
        caption = cap2.reshape(1, Lext, 2560)
        cap_pad_mask = msk2.reshape(1, Lext)
        log(f"[外部 caption] {cap_ext}  槽数={Lext}  真实槽={int((cap_pad_mask==0).sum())}")

    # 全程不变的两个输入只推一次
    os.makedirs(f"{WORK}\\const", exist_ok=True)
    for nm, arr in [("caption", caption), ("cap_pad_mask", cap_pad_mask)]:
        p = f"{WORK}\\const\\{nm}.raw"
        np.ascontiguousarray(arr.astype(np.float32)).tofile(p)
        push(p, f"{T}/loop/in/{nm}.raw")

    # 初始噪声。缺省用 numpy PCG64；但要与 app 做对照时**必须**改吃 app 那侧的噪声：
    # C++ 用 `std::mt19937 + std::normal_distribution<float>`，与 numpy 的 PCG64
    # **同 seed 也完全不同** ⇒ 两边生成的是同一模型的不同采样，像素比较无意义
    # （2026-08-17 我据此做过一次无效对照）。
    # 用 INLOOP_LATENTS 指向设备端 `gen_latents` 产出的 raw，即可逐字节对齐。
    ext = os.environ.get("INLOOP_LATENTS", "")
    if ext:
        latents = np.fromfile(ext, dtype=np.float32)
        exp = LATENT_CH * 128 * 128
        assert latents.size == exp, f"latents 元素数 {latents.size} != {exp}: {ext}"
        log(f"initial latents <- {ext}（外部提供，与 app 逐字节相同）")
    else:
        latents = np.random.default_rng(SEED).standard_normal(
            LATENT_CH * 128 * 128).astype(np.float32)
    latents_4d = latents.reshape(1, LATENT_CH, 128, 128)
    log(f"initial latents std={latents.std():.4f}")

    sig = sigmas_of(STEPS)
    timesteps = sig[:-1] * 1000.0

    n_seg = 4 if P2_MODE == "split" else 3
    log(f"\n=== 去噪循环（transformer {n_seg} 段全部走【设备 HTP】）===")
    for step in range(STEPS):
        ts = np.array([1.0 - timesteps[step] / 1000.0], dtype=np.float32)
        log(f"  step {step}: timestep={ts[0]:.4f}")

        put_raw("latents", latents_4d, step)
        put_raw("timestep", ts, step)

        d1a = htp_run("part1a", {
            "latents": f"{T}/loop/in/latents.raw",
            "timestep": f"{T}/loop/in/timestep.raw",
            "caption": f"{T}/loop/in/caption.raw",
            "cap_pad_mask": f"{T}/loop/in/cap_pad_mask.raw"}, "loop/o1a", f"s{step}")

        # 段间中间张量全程留在设备上，不做无谓往返
        d1b = htp_run("part1b", {k: f"{d1a}/{k}.raw" for k in P1B_INS},
                      "loop/o1b", f"s{step}")

        feeds2 = {"unified": f"{d1b}/unified.raw"}
        feeds2.update({k: f"{d1a}/{k}.raw" for k in P2_INS if k != "unified"})

        if P2_MODE == "split":
            d2a = htp_run("part2a_fixed", feeds2, "loop/o2a", f"s{step}")
            if step == 0:
                check_dev_sizes(d2a, CUT, f"s{step}")
            feeds2b = {k: f"{d2a}/{k}.raw" for k in CUT}
            feeds2b["adaln_input"] = f"{d1a}/adaln_input.raw"
            d2 = htp_run("part2b_fixed", feeds2b, "loop/o2b", f"s{step}")
        else:
            d2 = htp_run("part2", feeds2, "loop/o2", f"s{step}")

        noise = fetch(d2, "latents", step).reshape(-1).astype(np.float64)

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
    # 输出名必须带配置标识：本脚本此前固定写 htp_inloop_transformer.png，
    # 2026-08-15 跑 per-row 时**差点把基线那张唯一副本覆盖掉**（HANDOVER 15.17.6）。
    # 用环境变量 INLOOP_TAG 区分配置；不设则退回原名（向后兼容）。
    suffix = f"_{TAG}" if TAG else ""
    out = f"{OUT_DIR}\\htp_inloop_transformer{suffix}.png"
    Image.fromarray(img).save(out)
    log("Saved:", out)


if __name__ == "__main__":
    main()
