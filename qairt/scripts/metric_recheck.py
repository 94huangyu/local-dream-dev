"""#84：三把尺子的"反相关"是不是坏 VAE 的钳位假象？

方案：scripts/EXP_PLAN_METRIC_RECHECK.md（判据事前锁定，不得事后修改）

范围已在方案 §〇 收窄：E_all 算在 latents 上、完全不经过 VAE ⇒ 不受影响、无需复查。
本脚本只查经过 VAE 的两把尺子：平均|像素差| 与 PSNR。
"""
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SCALING, SHIFT = 0.3611, 0.1159
SIGMA7 = 0.3000                      # 末步 dt = 0 - sigma7；FP32 日志 step 7 dt=-0.3000 已确认
V2_EXPECT = {"3seg": 41.37, "4seg": 44.52}
V2_TOL = 0.50


def to_img(px):
    return np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def rebuild_final(tag):
    """末步 latents 重建：final = lat_s7 - dt*noise = lat_s7 + sigma7*noise_s7"""
    d = os.path.join(P0, tag, "s7")
    lat = np.fromfile(os.path.join(d, "latents.raw"), np.float32).reshape(1, 16, 128, 128)
    noise = np.fromfile(os.path.join(d, "latents_dev.raw"), np.float32).reshape(1, 16, 128, 128)
    return (lat.astype(np.float64) + SIGMA7 * noise.astype(np.float64)).astype(np.float32)


def main():
    from PIL import Image
    import onnxruntime as ort

    arms = {
        "FP32ref": np.fromfile(os.path.join(OUT, "fp32_final_latents_inloopref_vaefix.raw"),
                               np.float32).reshape(1, 16, 128, 128),
        "3seg": rebuild_final("htp_inloop_ctl3seg"),
        "4seg": rebuild_final("htp_inloop_perrow4seg"),
    }
    print("=== 末步 latents ===")
    for k, v in arms.items():
        print("  %-8s std=%.4f  range=[%.3f, %.3f]" % (k, v.std(), v.min(), v.max()))

    feeds = {
        "OLD": lambda x: (x.astype(np.float64) / SCALING + SHIFT).astype(np.float32),
        "NEW": lambda x: (x.astype(np.float64) + SHIFT * SCALING).astype(np.float32),
    }
    s = ort.InferenceSession(os.path.join(ONNX, "vae_decoder.onnx"),
                             providers=["CPUExecutionProvider"])
    imgs = {}
    for fname, f in feeds.items():
        for aname, lat in arms.items():
            px = s.run(["pixels"], {"vae_latents": f(lat)})[0].astype(np.float64)
            im = to_img(px).astype(np.float64)
            imgs[(fname, aname)] = im
            Image.fromarray(im.astype(np.uint8)).save(
                os.path.join(OUT, "recheck_%s_%s.png" % (fname.lower(), aname.lower())))
            print("  decoded %s/%s" % (fname, aname), flush=True)
    del s

    def stats(fname, aname):
        A, B = imgs[(fname, "FP32ref")], imgs[(fname, aname)]
        d = float(np.abs(A - B).mean())
        psnr = 10 * np.log10(255.0 ** 2 / max(((A - B) ** 2).mean(), 1e-9))
        clip = float(((B <= 0) | (B >= 255)).mean() * 100)
        return d, psnr, clip

    print("")
    print("%-6s %-8s %12s %10s %12s" % ("喂法", "臂", "平均|像素差|", "PSNR", "钳位像素%"))
    print("-" * 56)
    res = {}
    for fname in ("OLD", "NEW"):
        rc = float(((imgs[(fname, "FP32ref")] <= 0) | (imgs[(fname, "FP32ref")] >= 255)).mean() * 100)
        print("%-6s %-8s %12s %10s %11.3f%%" % (fname, "FP32ref", "-", "-", rc))
        for aname in ("3seg", "4seg"):
            d, p, c = stats(fname, aname)
            res[(fname, aname)] = (d, p)
            print("%-6s %-8s %12.3f %10.2f %11.3f%%" % (fname, aname, d, p, c))

    # ---- V2 门 ----
    print("")
    print("=== V2 门：旧喂法必须复现历史记录（验证末步 latents 重建是否正确）===")
    ok2 = True
    for aname in ("3seg", "4seg"):
        d = res[("OLD", aname)][0]
        exp = V2_EXPECT[aname]
        good = abs(d - exp) <= V2_TOL
        ok2 &= good
        print("  %s 旧喂法像素差 %.3f   历史 %.2f±%.2f   %s"
              % (aname, d, exp, V2_TOL, "✅" if good else "❌"))
    if not ok2:
        print("  ⚠️ V2 未过 ⇒ 按方案 §三：**只比较两臂的相对次序**，")
        print("     且不与历史绝对值比较（可能是 FP32 参考不同源造成的系统性偏移）。")

    # ---- 主判据 ----
    d3o, d4o = res[("OLD", "3seg")][0], res[("OLD", "4seg")][0]
    d3n, d4n = res[("NEW", "3seg")][0], res[("NEW", "4seg")][0]
    gap_o, gap_n = d4o - d3o, d4n - d3n
    print("")
    print("=== 主判据 ===")
    print("  旧喂法: d3=%.3f  d4=%.3f   差距 d4-d3 = %+.3f  (%s)"
          % (d3o, d4o, gap_o, "四段更差" if gap_o > 0 else "四段更好"))
    print("  新喂法: d3=%.3f  d4=%.3f   差距 d4-d3 = %+.3f  (%s)"
          % (d3n, d4n, gap_n, "四段更差" if gap_n > 0 else "四段更好"))
    if d4n < d3n:
        v = "🔴 反相关是钳位假象 ⇒ #69/#70 须重新表述；解除 §2.3 标量决策暂停；P1 三层指标面板必要性下调"
    elif gap_n >= gap_o / 2.0:
        v = "✅ 反相关成立 ⇒ #69/#70 维持，#84 关闭；三层指标面板仍需要"
    else:
        v = "🔶 部分是钳位假象 ⇒ 两条结论都要加限定，不得原样引用"
    print("  判定: %s" % v)
    print("")
    print("  PSNR 次序（是否与像素差同步）：")
    for fname in ("OLD", "NEW"):
        p3, p4 = res[(fname, "3seg")][1], res[(fname, "4seg")][1]
        print("    %s: 3seg %.2f dB  vs  4seg %.2f dB  ⇒ %s"
              % (fname, p3, p4, "四段更差" if p4 < p3 else "四段更好"))


if __name__ == "__main__":
    main()
