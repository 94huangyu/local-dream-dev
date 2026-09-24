"""EXP_PLAN_HTP_INLOOP 判据 2：把 HTP 在环出图放到已实测的尺子上。

已有阶梯（HANDOVER 14.2 / MAINLINE Test B 结论）：
  FP32 vs CPU 跑量化 transformer : 平均|像素差| 9.32   PSNR 23.89 dB  -> 清晰的猫
  FP32 vs 设备 app 真机输出      : 平均|像素差| 46.40  PSNR 11.56 dB  -> 橙色色块

⚠️ 2026-08-16 修正：本脚本原先**完全忽略命令行参数**，只比对写死的
`htp_inloop_transformer.png`。我曾按 `compare_images.py <基准> <目标>` 的形式调用它，
参数被静默丢弃——与 `snpe-net-run` 静默缩批、overrides 名字错了静默失效属同一类。
（当时那个 43.56 之所以是对的，纯属巧合：写死的那个文件刚好被该次运行覆盖成了目标图。）
现在支持 `python compare_images.py [目标.png ...]`，不给参数则用默认清单。
"""
import sys

import numpy as np
from PIL import Image

R = r"D:\LocalDreamZImage\scratch_runs"
BASE = f"{R}\\zimage_fp32_full_pipeline.png"

TARGETS = [
    ("CPU 跑量化 transformer (Test B)", f"{R}\\testB_hybrid_quantized_transformer.png"),
    ("设备 app 真机输出",              f"{R}\\sm8750_test2.png"),
    ("HTP 在环 (本实验)",              f"{R}\\htp_inloop_transformer.png"),
]
if len(sys.argv) > 1:
    TARGETS = [(p.replace("\\", "/").split("/")[-1], p) for p in sys.argv[1:]]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def metrics(a, b):
    d = np.abs(a - b)
    mse = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    return d.mean(), psnr


base = load(BASE)
print(f"基准: zimage_fp32_full_pipeline.png  {base.shape}")
print()
print(f"{'对照':<36}{'平均|像素差|':>14}{'PSNR(dB)':>12}")
print("=" * 62)
res = {}
for tag, p in TARGETS:
    try:
        img = load(p)
    except FileNotFoundError:
        print(f"{tag:<36}{'(缺文件)':>14}")
        continue
    if img.shape != base.shape:
        print(f"{tag:<36}  形状不符 {img.shape}")
        continue
    m, ps = metrics(base, img)
    res[tag] = (m, ps)
    print(f"{tag:<36}{m:>14.2f}{ps:>12.2f}")

print()
print("=" * 62)
print("判据 2（EXP_PLAN_HTP_INLOOP 第 3 节，事前定稿）")
print("=" * 62)
k = "HTP 在环 (本实验)"
if k in res:
    m, ps = res[k]
    print(f"  HTP 在环: 平均|像素差| = {m:.2f}   PSNR = {ps:.2f} dB")
    if m <= 15 and ps >= 20:
        print("  => 与【CPU 量化】同类 ⇒ 假设 C 被否定为根因")
    elif m >= 35 and ps <= 14:
        print("  => 与【设备真机】同类 ⇒ 假设 C 被坐实为根因")
    else:
        print("  => 落在中间 ⇒ C 有贡献，但不足以解释全部")
    print()
    print("  提醒：判据 3 —— 若本指标与肉眼所见冲突，以【肉眼】为准。")
    print("        必须自己打开 PNG 看，不得只报指标（CLAUDE.md 约束 1）。")
