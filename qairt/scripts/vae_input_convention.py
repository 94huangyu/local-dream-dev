"""根因验证：vae_decoder.onnx 的第一个节点是 Div(vae_latents, 0.3611)
=> 图内部自己做反缩放，宿主/app 在外面又做了一次 => 除了两遍。

官方语义：  decoder( lat/0.3611 + 0.1159 )
我们的 ONNX：decoder( input/0.3611 )
=> 正确输入应为  input = lat + 0.1159*0.3611 = lat + 0.041856

判据（事前锁定）：
  「+shift*scaling」这一臂 vs 官方 PyTorch：
     平均|像素差| < 1.0  且  高频能量比 in [0.98, 1.02]  => 根因确认，导出无缺陷
对照臂：现行喂法 lat/0.3611+0.1159 ／ 裸 lat（无 shift）
"""
import os, sys, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
LAT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fp32_steps", "lat_8.raw")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SCALING, SHIFT = 0.3611, 0.1159


def to_img(px):
    return np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def grad_energy(a):
    g = np.gradient(a.astype(np.float64).mean(axis=2))
    return float(np.sqrt(g[0] ** 2 + g[1] ** 2).mean())


def main():
    lat = np.fromfile(LAT, np.float32).reshape(1, 16, 128, 128)
    assert abs(lat.std() - 1.0996) < 0.01, "identity gate failed"

    pxA = np.load(os.path.join(OUT, "vae_gain_pxA.npy")).astype(np.float64)  # official PyTorch
    A = to_img(pxA).astype(np.float64)
    gA = grad_energy(A)
    print("[ref] official PyTorch arm reused: grad_energy=%.4f" % gA)

    arms = {
        "CURRENT  lat/s + shift   (现行喂法)": (lat.astype(np.float64) / SCALING + SHIFT),
        "RAW      lat             (无 shift)": lat.astype(np.float64),
        "FIXED    lat + shift*s   (推定正确)": (lat.astype(np.float64) + SHIFT * SCALING),
    }

    import onnxruntime as ort
    s = ort.InferenceSession(os.path.join(ONNX, "vae_decoder.onnx"),
                             providers=["CPUExecutionProvider"])
    print("")
    print("%-38s%12s%10s%12s%12s" % ("arm", "meanAbsDiff", "PSNR", "gradEnergy", "vs official"))
    print("-" * 84)
    res = {}
    for name, inp in arms.items():
        t0 = time.time()
        out = s.run(["pixels"], {"vae_latents": inp.astype(np.float32)})[0].astype(np.float64)
        B = to_img(out).astype(np.float64)
        d = float(np.abs(A - B).mean())
        psnr = 10 * np.log10(255.0 ** 2 / max(((A - B) ** 2).mean(), 1e-9))
        gB = grad_energy(B)
        res[name] = (d, gB / gA)
        print("%-38s%12.3f%10.2f%12.4f%11.4fx" % (name, d, psnr, gB, gB / gA))
        tag = name.split()[0].lower()
        from PIL import Image
        Image.fromarray(B.astype(np.uint8)).save(os.path.join(OUT, "vae_conv_%s.png" % tag))
        print("    saved scratch_runs/vae_conv_%s.png   (%.0fs)" % (tag, time.time() - t0), flush=True)

    d, gr = res["FIXED    lat + shift*s   (推定正确)"]
    ok = d < 1.0 and 0.98 <= gr <= 1.02
    print("")
    print("=== 判据 ===")
    print("  FIXED arm: meanAbsDiff=%.3f (<1.0)   gradRatio=%.4f (0.98~1.02)" % (d, gr))
    print("  VERDICT: %s" % ("🔴 根因确认 —— 双重反缩放，VAE 导出本身无缺陷"
                             if ok else "❌ 未达判据，双重反缩放不足以解释全部差异"))


if __name__ == "__main__":
    main()
