# -*- coding: utf-8 -*-
"""为新比例生成 FP32 参考 latents —— 一份产物同时供两处使用：

  1. **VAE 量化的标定集**（`vae_quant_aspect.py` 要 `vae_calib_<tag>/latent_s{42..46}.raw`）
  2. **G-QUALITY 的 FP32 参考臂**（D2 形式，seed 42/43/44）

🔴 **绝不使用固定文件名。** 2026-09-03 就是 `t2_fp32_ref.py drive` 写到固定名、
   被下一个 seed 覆盖，导致「本 seed 的量化输出 vs 另一 seed 的 FP32 参考」，
   PSNR 落在 12 dB 的无关图基线上，害我报出「换比例回退 9.61 dB」的错误结论。
   现在 `t2_fp32_ref.py` 按输入名派生输出名，本脚本据此取回，并加 md5 互异门。
   旧的 `vae_calib_1184x896/run.sh` 仍是固定名写法，**已作废，不要照抄**。

用法: python aspect_fp32_refs.py <tag> <宽> <高> [--seeds 42,43,44,45,46]
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
SCRATCH = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
PY = os.path.join("D:", os.sep, "ZImage_Work", "venv-official", "Scripts", "python.exe")
CAP = os.path.join(P0, "cap_r4x3.raw")     # caption 与几何无关（[1,80,2560]）
MSK = os.path.join(P0, "mask_r4x3.raw")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seeds = [42, 43, 44, 45, 46]
    if "--seeds" in sys.argv:
        seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    lh, lw = H // 8, W // 8
    want = 16 * lh * lw
    out = os.path.join(P0, "vae_calib_%s" % tag)
    os.makedirs(out, exist_ok=True)

    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "ZI_TAG": tag,
                "ZI_LAT_H": str(lh), "ZI_LAT_W": str(lw)})
    print("=== %s  latent %dx%d  seeds %s ===" % (tag, lh, lw, seeds), flush=True)

    made = {}
    for s in seeds:
        noise = os.path.join(P0, "lat_init_%s_rng%d.raw" % (tag, s))
        if not os.path.isfile(noise):
            print("  🔴 缺噪声 %s" % noise)
            return 1
        n = os.path.getsize(noise) // 4
        if n != want:
            print("  🔴 噪声元素数 %d != %d" % (n, want))
            return 1
        dst = os.path.join(out, "latent_s%d.raw" % s)
        if os.path.isfile(dst) and os.path.getsize(dst) == want * 4:
            print("  seed %d 已有产物，跳过" % s, flush=True)
            made[s] = dst
            continue
        t0 = time.time()
        r = subprocess.run(
            [PY, os.path.join(HERE, "t2_fp32_ref.py"), "drive", "80", noise, CAP, MSK,
             os.path.join(SCRATCH, "calib_%s_s%d.png" % (tag, s))],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("  🔴 seed %d FP32 失败 rc=%d" % (s, r.returncode))
            print((r.stdout or "")[-1500:])
            return 1
        # 按输入名派生的产物名（t2_fp32_ref.py 的修复后行为），**不接受固定名**
        stem = os.path.splitext(os.path.basename(noise))[0]
        src = os.path.join(SCRATCH, "fp32_final_latents_L80_%s__%s.raw" % (tag, stem))
        if not os.path.isfile(src):
            print("  🔴 没找到按 seed 隔离命名的产物 %s" % src)
            print("     （若 t2_fp32_ref.py 又退回固定名，必须先修它，不得将就）")
            return 1
        if os.path.getsize(src) != want * 4:
            print("  🔴 产物字节数 %d != %d" % (os.path.getsize(src), want * 4))
            return 1
        shutil.copyfile(src, dst)
        made[s] = dst
        print("  seed %d OK  %.1f 分钟" % (s, (time.time() - t0) / 60.0), flush=True)

    print("\n=== 门：五个参考必须两两不同（防覆盖事故重演）===", flush=True)
    h = {s: md5(p) for s, p in sorted(made.items())}
    for s, v in h.items():
        print("  seed %-3d md5=%s" % (s, v[:16]))
    if len(set(h.values())) != len(h):
        print("  🔴 有重复 md5 —— 说明发生了覆盖，产物不可用")
        return 1
    print("  ✅ %d 个参考两两不同" % len(h))
    return 0


sys.exit(main())
