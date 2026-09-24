"""三层指标面板（MAINLINE §2.5）：一条命令给出与 FP32 参考的一致性。

用法: python panel3.py [tag ...]      # tag 即 INLOOP_TAG，不给则列全部已知配置

顶层目标是**与原始模型的一致性**，不是"图好不好"。三层同时报：
  L1  step-0 噪声预测相对 L2（E_all）      —— 快，可用于筛选
  L2  终点 latents 相对 L2 + 主体余弦      —— 轨迹终点
  L3  成图 PSNR / 像素相对 L2              —— **报告结论时必须给这一层**

⚠️ 同条件同 seed 是前提：本脚本自动核对各配置的 s0/latents.raw 与 caption.raw 的 md5，
   不一致直接拒绝比较（#74：同 seed ≠ 同噪声，跨实现比较必须先对齐随机源）。
"""
import hashlib
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
S = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
REF_S0 = os.path.join(B, "vs_fp32", "latents_fp32_s0.raw")
REF_FIN = os.path.join(S, "fp32_final_latents_inloopref_vaefix.raw")
REF_IMG = os.path.join(S, "recheck_new_fp32ref.png")
DT7 = -0.3000  # 最后一步的 dt（时间表固定）

KNOWN = [
    ("htp_inloop", "htp_inloop_transformer_perrow4seg.png", "per-row 基线"),
    ("htp_inloop_fp16seg", "htp_inloop_transformer_fp16seg.png", "FP16 26 张量"),
    ("htp_inloop_fp16seg2", "htp_inloop_transformer_fp16seg2.png", "FP16 32 张量"),
    ("htp_inloop_fp16seg3", "htp_inloop_transformer_fp16seg3.png", "FP16 29 张量"),
    ("htp_inloop_fp16best", "htp_inloop_transformer_fp16best.png", "FP16 26 复现"),
    ("htp_inloop_dtseg", "htp_inloop_transformer_dtseg.png", "dtype 26 张量"),
    # dtype 机制 + part1b 白名单 2->5（加 mul_406/431/456）。与上一行**唯一变量 = 那 3 个张量**：
    # 其余三段沿用同一批 dtype 产物；dtype 零扩散 => 不会像旧机制那样把 linear_129/137/145
    # （实测 |a|max 94k~121k）拖进 FP16 变 inf（#117）。这是去掉混淆后的 #112 重做。
    ("htp_inloop_dtseg5", "htp_inloop_transformer_dtseg5.png", "dtype 29 张量"),
    # 2026-08-25 #111：旧机制 + 在 ONNX 里显式 Clip(±65504)，白名单 = old-26 的 26 个
    # 加上 18 个被钳的 massive activation。QDQ 模拟预测 L1 约 22%（当前最优 30.68%）。
    ("htp_inloop_clip", "htp_inloop_transformer_clip.png", "Clip+FP16"),
]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rel(a, b):
    return 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)


def cos_bulk(a, b):
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def main():
    want = sys.argv[1:]
    rows, seeds = [], {}
    a0 = np.fromfile(REF_S0, np.float32).astype(np.float64)
    af = np.fromfile(REF_FIN, np.float32).astype(np.float64)
    from PIL import Image
    ai = np.asarray(Image.open(REF_IMG).convert("RGB"), dtype=np.float64)

    for d, img, tag in KNOWN:
        if want and d not in want and tag not in want:
            continue
        dd = os.path.join(B, d)
        p0 = os.path.join(dd, "s0", "latents_dev.raw")
        n0 = os.path.join(dd, "s0", "latents.raw")
        c0 = os.path.join(dd, "const", "caption.raw")
        p7 = os.path.join(dd, "s7", "latents.raw")
        n7 = os.path.join(dd, "s7", "latents_dev.raw")
        ip = os.path.join(S, img)
        if not all(os.path.exists(x) for x in (p0, n0, c0, p7, n7)):
            continue
        seeds[tag] = (md5(n0)[:16], md5(c0)[:16])
        l1 = rel(a0, np.fromfile(p0, np.float32).astype(np.float64))
        fin = (np.fromfile(p7, np.float32).astype(np.float64)
               - DT7 * np.fromfile(n7, np.float32).astype(np.float64))
        l2, l2c = rel(af, fin), cos_bulk(af, fin)
        if os.path.exists(ip):
            bi = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float64)
            dif = bi - ai
            rmse = float(np.sqrt((dif * dif).mean()))
            l3p, l3r = 20 * np.log10(255.0 / max(rmse, 1e-9)), rel(ai, bi)
        else:
            l3p = l3r = float("nan")
        rows.append((tag, l1, l2, l2c, l3p, l3r))

    print("=== 前提核对：同噪声同 caption（#74）===")
    vals = set(seeds.values())
    for t, v in seeds.items():
        print("  %-16s noise=%s caption=%s" % (t, v[0], v[1]))
    if len(vals) > 1:
        print("  🔴 输入不一致 —— 拒绝比较")
        return 1
    print("  ✅ 全部一致")

    print("\n=== 三层指标（与 FP32 参考的一致性；目标是一致，不是好看）===")
    print("%-16s %11s %11s %10s %10s %11s"
          % ("配置", "L1 E_all", "L2 终点L2", "L2 余弦", "L3 PSNR", "L3 像素L2"))
    print("-" * 74)
    for t, l1, l2, l2c, l3p, l3r in rows:
        print("%-16s %10.2f%% %10.2f%% %10.4f %9.2f dB %10.2f%%" % (t, l1, l2, l2c, l3p, l3r))
    print("\n参照：我们的 FP32 vs 官方 = **41.61 dB**（§15.22.10）")
    print("⚠️ 误差分布不均匀：主体（受 caption 强约束）收敛，背景/场景（弱约束）不收敛")
    print("   ⇒ 判断一致性时**背景比主体灵敏**，看图时优先看背景。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
