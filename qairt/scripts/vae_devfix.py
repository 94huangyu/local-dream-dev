"""#81 设备门：一行喂法修改在【量化 QNN VAE】上是否恢复画质？

方案：scripts/EXP_PLAN_VAE_DEVFIX.md（判据事前锁定）
单变量：同一台设备、同一份 vae.bin、同一份 latents，唯一变量 = 喂法。
"""
import os, sys, time, subprocess
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
LAT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fp32_steps", "lat_8.raw")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SCALING, SHIFT = 0.3611, 0.1159
PIX_BYTES = 1 * 3 * 1024 * 1024 * 4
V3_HF, V3_HF_TOL = 4.021, 0.05
V3_D, V3_D_TOL = 13.28, 0.50


def sh(cmd, timeout=1800):
    """adb shell：adb 自身失败必须立刻炸（#65：掉线曾被伪装成模型执行失败）"""
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("adb shell rc=%d: %s\nstderr: %s\nstdout: %s"
                           % (r.returncode, cmd[:120], r.stderr.strip(), r.stdout.strip()))
    return r.stdout


def require_device(where):
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    if not any(DEV in l and "device" in l.split() for l in out.splitlines()[1:]):
        raise RuntimeError("设备 %s 不在线（%s）\nadb devices:\n%s" % (DEV, where, out))


def push(local, remote):
    r = subprocess.run([ADB, "-s", DEV, "push", local, remote],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("adb push failed: %s\n%s" % (local, r.stderr))


def pull(remote, local):
    r = subprocess.run([ADB, "-s", DEV, "pull", remote, local],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("adb pull failed: %s\n%s" % (remote, r.stderr))


def to_img(px):
    return np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def grad_energy(a):
    g = np.gradient(a.astype(np.float64).mean(axis=2))
    return float(np.sqrt(g[0] ** 2 + g[1] ** 2).mean())


def run_arm(tag, arr):
    require_device("%s 执行前" % tag)
    lp = os.path.join(OUT, "devfix_in_%s.raw" % tag)
    a = np.ascontiguousarray(arr.astype(np.float32))
    a.tofile(lp)
    assert os.path.getsize(lp) == a.size * 4, "输入字节数不对"
    push(lp, "%s/devfix_%s.raw" % (T, tag))
    sh("mkdir -p %s/devfix_%s && rm -rf %s/devfix_%s/Result_0" % (T, tag, T, tag))
    sh("printf '%%s\\n' 'vae_latents:=%s/devfix_%s.raw' > %s/devfix_%s/list.txt" % (T, tag, T, tag))
    t0 = time.time()
    out = sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
             "./qnn-net-run --retrieve_context vae.bin --backend libQnnHtp.so "
             "--input_list devfix_%s/list.txt --output_dir %s/devfix_%s --log_level error "
             "2>&1 | tail -3" % (T, T, T, tag, T, tag))
    if "Finished Executing Graphs" not in out:
        raise RuntimeError("V1 FAIL qnn-net-run [%s]:\n%s" % (tag, out))
    print("  [V1] %s  qnn-net-run OK  %.1fs" % (tag, time.time() - t0))
    rp = "%s/devfix_%s/Result_0/pixels.raw" % (T, tag)
    nb = int(sh("stat -c %%s %s" % rp).strip())
    if nb != PIX_BYTES:
        raise RuntimeError("V2 FAIL %s: pixels.raw %d bytes, expect %d" % (tag, nb, PIX_BYTES))
    print("  [V2] %s  pixels.raw = %d bytes  OK" % (tag, nb))
    dst = os.path.join(OUT, "devfix_pixels_%s.raw" % tag)
    pull(rp, dst)
    return np.fromfile(dst, np.float32).reshape(1, 3, 1024, 1024).astype(np.float64)


def main():
    from PIL import Image
    lat = np.fromfile(LAT, np.float32).reshape(1, 16, 128, 128)
    assert abs(lat.std() - 1.0996) < 0.01, "身份门不过"
    pxA = np.load(os.path.join(OUT, "vae_gain_pxA.npy")).astype(np.float64)
    A = to_img(pxA).astype(np.float64)
    gA = grad_energy(A)
    print("[ref] 官方 PyTorch: grad_energy=%.4f" % gA)

    arms = {
        "current": lat.astype(np.float64) / SCALING + SHIFT,
        "fixed": lat.astype(np.float64) + SHIFT * SCALING,
    }
    res = {}
    for tag, inp in arms.items():
        print("\n=== 臂 %s   输入 std=%.4f  范围 [%.3f, %.3f] ==="
              % (tag.upper(), inp.std(), inp.min(), inp.max()))
        px = run_arm(tag, inp)
        B = to_img(px).astype(np.float64)
        d = float(np.abs(A - B).mean())
        psnr = 10 * np.log10(255.0 ** 2 / max(((A - B) ** 2).mean(), 1e-9))
        gB = grad_energy(B)
        res[tag] = (d, gB, gB / gA)
        Image.fromarray(B.astype(np.uint8)).save(os.path.join(OUT, "devfix_%s.png" % tag))
        print("  对官方: 平均|像素差|=%.3f  PSNR=%.2f dB  高频=%.4f (%.4fx)"
              % (d, psnr, gB, gB / gA))
        print("  saved scratch_runs/devfix_%s.png" % tag)

    dc, gc_, rc = res["current"]
    df, gf, rf = res["fixed"]

    print("\n=== V3 基线复现门（对照 §15.21.7）===")
    ok_hf = abs(gc_ - V3_HF) <= V3_HF * V3_HF_TOL
    ok_d = abs(dc - V3_D) <= V3_D_TOL
    print("  CURRENT 高频 %.4f  期望 %.3f±5%%   %s" % (gc_, V3_HF, "OK" if ok_hf else "FAIL"))
    print("  CURRENT 像素差 %.3f  期望 %.2f±%.2f   %s" % (dc, V3_D, V3_D_TOL, "OK" if ok_d else "FAIL"))
    if not (ok_hf and ok_d):
        print("  ⚠️ V3 不过 ⇒ 与 §15.21.7 并非同一配置，**不得跨实验比较**；本轮只做自比。")

    R = 1.0 - df / max(dc, 1e-9)
    print("\n=== 主判据 ===")
    print("  d_CURRENT=%.3f   d_FIXED=%.3f   改善率 R=%.4f" % (dc, df, R))
    print("  高频比: CURRENT %.4fx -> FIXED %.4fx  (官方=1.0000x)" % (rc, rf))
    if R >= 0.70:
        v = "🔴 修复在设备上有效 ⇒ 可进入 APK 交付（约束 11 四条铁律）"
    elif R <= 0.30:
        v = "❌ 设备上基本无效 ⇒ 不得交付，回查设备侧损害来源"
    else:
        v = "🔶 部分有效 ⇒ 交付仍做，但须登记剩余损害并另开线索"
    print("  判定: %s" % v)
    print("\n  [附带·非判据] FIXED 臂的残余 %.3f 即【纯量化损害】" % df)
    print("     依据：fp32 + 正确喂法对官方已是 0.001 / 79.92 dB")
    print("     ⇒ 残余不可能来自导出或喂法")


if __name__ == "__main__":
    main()
