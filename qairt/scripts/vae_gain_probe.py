"""🔴 2026-08-21 作废声明（HANDOVER 15.22 / 台账 #80）
本脚本的 ONNX 臂喂的是 lat/0.3611+0.1159，而 vae_decoder.onnx 图内首节点已有
Div(vae_latents,0.3611) ⇒ 除了两遍。**本脚本产出的一切数字与图像一律作废，不得引用。**
正确喂法与三臂对照见 scripts/vae_input_convention.py。
按约束 2 保留原文，不删除。
"""
"""VAE 导出缺陷的性质判别：全局增益 还是 结构性锐化？

方案定稿：scripts/EXP_PLAN_VAE_GAIN.md（判据事前锁定，不得事后修改）

主判据：把 ONNX 臂（**钳位前 float**）整体对齐到官方臂的 mean/std，再过同一个 to_img，
        算平均 |像素差|
        < 3.0 ⇒ 纯全局增益 ｜ > 8.0 ⇒ 增益不是主因 ｜ 3.0~8.0 ⇒ 两者兼有
V 门  ：未校正时必须复现 13.326 ± 0.20，否则实验作废
G-check：C1 纯增益构造须 < 3.0；C2 模糊构造须 > 3.0（约束 8）

用法: python vae_gain_probe.py
"""
import os, sys, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LAT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fp32_steps", "lat_8.raw")
MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SCALING, SHIFT = 0.3611, 0.1159
LAT_STD_EXPECT = 1.0996          # vae_ab.py 记录值，身份门
V_GATE_EXPECT, V_GATE_TOL = 13.326, 0.20


def to_img(px):                   # 与 vae_ab.py 逐字相同
    return np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def verdict(pxA, pxB):
    """判据流程：pxB 对齐到 pxA 的 mean/std -> to_img -> 平均|像素差|"""
    corr = (pxB - pxB.mean()) / pxB.std() * pxA.std() + pxA.mean()
    A = to_img(pxA).astype(np.float64)
    B = to_img(corr).astype(np.float64)
    return float(np.abs(A - B).mean())


def raw_diff(pxA, pxB):
    A = to_img(pxA).astype(np.float64)
    B = to_img(pxB).astype(np.float64)
    return float(np.abs(A - B).mean())


def box3(x):
    p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    s = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            s += p[:, :, dy:dy + x.shape[2], dx:dx + x.shape[3]]
    return s / 9.0


def top1_share(a):
    """约束 7：前 1% 元素占 ||a||^2 的比例"""
    e = np.sort((a.ravel().astype(np.float64)) ** 2)[::-1]
    k = max(1, int(round(len(e) * 0.01)))
    return float(e[:k].sum() / max(e.sum(), 1e-30))


def main():
    os.makedirs(OUT, exist_ok=True)
    lat = np.fromfile(LAT, np.float32).reshape(1, 16, 128, 128)
    print(f"[身份门] {LAT}\n          std={lat.std():.4f} (期望 {LAT_STD_EXPECT})", flush=True)
    if abs(lat.std() - LAT_STD_EXPECT) > 0.01:
        sys.exit("❌ 身份门不过：latents 不是 vae_ab.py 用的那一份，实验作废")
    vae_in = (lat.astype(np.float64) / SCALING + SHIFT).astype(np.float32)

    pA = os.path.join(OUT, "vae_gain_pxA.npy")
    pB = os.path.join(OUT, "vae_gain_pxB.npy")
    if os.path.exists(pA) and os.path.exists(pB):
        pxA = np.load(pA).astype(np.float64)
        pxB = np.load(pB).astype(np.float64)
        print(f"[复用] 已落盘的两臂 float 输出  shape={pxA.shape}", flush=True)
        return analyze(pxA, pxB)

    # ---------- 臂 A：官方 PyTorch ----------
    import torch
    from diffusers import AutoencoderKL
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    vae.eval()
    with torch.no_grad():
        pxA = vae.decode(torch.from_numpy(vae_in), return_dict=False)[0].numpy().astype(np.float64)
    print(f"[臂A] 官方 PyTorch VAE {time.time()-t0:.0f}s  shape={pxA.shape}", flush=True)
    del vae

    # ---------- 臂 B：我们的 ONNX ----------
    import onnxruntime as ort
    t0 = time.time()
    s = ort.InferenceSession(os.path.join(ONNX, "vae_decoder.onnx"),
                             providers=["CPUExecutionProvider"])
    pxB = s.run(["pixels"], {"vae_latents": vae_in})[0].astype(np.float64)
    print(f"[臂B] 我们的 ONNX VAE  {time.time()-t0:.0f}s  shape={pxB.shape}", flush=True)
    del s

    # ---------- 中间产物落盘（15.21.6 教训）----------
    np.save(os.path.join(OUT, "vae_gain_pxA.npy"), pxA.astype(np.float32))
    np.save(os.path.join(OUT, "vae_gain_pxB.npy"), pxB.astype(np.float32))
    print("[落盘] " + os.path.join(OUT, "vae_gain_px[AB].npy"), flush=True)

    return analyze(pxA, pxB)


def analyze(pxA, pxB):

    # ---------- V 门 ----------
    d_raw = raw_diff(pxA, pxB)
    ok_v = abs(d_raw - V_GATE_EXPECT) <= V_GATE_TOL
    print(f"\n=== V 门（有效性）===")
    print(f"  未校正平均|像素差| = {d_raw:.3f}   期望 {V_GATE_EXPECT}±{V_GATE_TOL}"
          f"   {'✅ 通过' if ok_v else '❌ 不通过 ⇒ 实验作废'}")
    if not ok_v:
        sys.exit("❌ V 门不过：执行方法错误（输入或环境），不得下任何结论")

    # ---------- G-check（约束 8）：判据二次锁定版，见方案 §九 ----------
    def ef_rs(pxRef, pxCmp, d_c2=None):
        dr, dc = raw_diff(pxRef, pxCmp), verdict(pxRef, pxCmp)
        ef = 1.0 - dc / max(dr, 1e-12)
        return dr, dc, ef, (dc / d_c2 if d_c2 else float("nan"))

    blur = box3(pxA)
    ctrl = {
        "C1 纯增益 1.2815x+0.05": (1.2815 * pxA + 0.05, "EF>=0.90", lambda e: e >= 0.90),
        "C2 3x3 模糊(结构性)   ": (blur,                "EF<=0.50", lambda e: e <= 0.50),
        "C3 锐化(竞争假设本身) ": (pxA + 0.5 * (pxA - blur), "EF<=0.50", lambda e: e <= 0.50),
    }
    print()
    print(f"=== G-check（判据操作化的已知样本验证，三组全过才继续）===")
    print(f"  {'控制组':<24}{'d_raw':>8}{'d_corr':>8}{'EF':>8}   要求")
    d_c2, ok_all = None, True
    for name, (arr, req, fn) in ctrl.items():
        dr, dc, ef, _ = ef_rs(pxA, arr)
        if name.startswith("C2"):
            d_c2 = dc
        ok = fn(ef)
        ok_all &= ok
        print(f"  {name:<24}{dr:>8.3f}{dc:>8.3f}{ef:>8.3f}   {req}  {'✅' if ok else '❌'}")
    if not ok_all:
        sys.exit("❌ G-check 不过：判据操作化本身有问题，结论作废（不得改阈值迁就结果）")
    print(f"  [标尺] C2 的 d_corr = {d_c2:.3f}  ⇒ RS 以它为 1.0 倍")

    # ---------- 主判据 ----------
    d_corr = verdict(pxA, pxB)
    EF = 1.0 - d_corr / d_raw
    RS = d_corr / d_c2
    if EF >= 0.90 and RS <= 1.0:
        v = "🔴 增益主导  ⇒ #71 探针改为「逐层输出 std 之比」"
    elif EF <= 0.50:
        v = "✅ 增益非主因 ⇒ 我的假设被证否，回原方案（高频发散定位）"
    else:
        v = "🔶 两者兼有  ⇒ #71 两个量都记"
    print()
    print(f"=== 主判据 ===")
    print(f"  d_raw = {d_raw:.3f}   d_corr = {d_corr:.3f}")
    print(f"  EF（增益解释率） = {EF:.4f}   要求 >=0.90（增益主导）/ <=0.50（非主因）")
    print(f"  RS（残差 / C2模糊）= {RS:.3f}   要求 <=1.0")
    print(f"  判定: {v}")

    # ---------- 附带记录（不是判据）----------
    print(f"\n=== 附带记录（仅供 #71 设计参考，非判据）===")
    print(f"  float 输出   官方 mean={pxA.mean():+.5f} std={pxA.std():.5f}")
    print(f"               ONNX  mean={pxB.mean():+.5f} std={pxB.std():.5f}")
    print(f"  全局增益比 k = std_B/std_A = {pxB.std()/pxA.std():.4f}")
    for c in range(3):
        print(f"    ch{c}: std_A={pxA[0,c].std():.5f} std_B={pxB[0,c].std():.5f} "
              f"k={pxB[0,c].std()/pxA[0,c].std():.4f}")
    A1, B1 = pxA.ravel(), pxB.ravel()
    a = float(np.cov(A1, B1)[0, 1] / np.var(A1))
    b = float(B1.mean() - a * A1.mean())
    res = B1 - (a * A1 + b)
    Bc = B1 - B1.mean()
    rel = float(np.linalg.norm(res) / np.linalg.norm(Bc))
    print(f"  最小二乘仿射 px_B ≈ {a:.4f}·px_A + {b:+.5f}")
    print(f"    残差相对L2 = {rel*100:.4f}%   R² = {1-rel**2:.6f}"
          f"   [约束7] 分母前1%元素占 ||a||² = {top1_share(Bc)*100:.2f}%")
    for n, x in (("官方", pxA), ("ONNX", pxB)):
        lo, hi = float((x < -1).mean()), float((x > 1).mean())
        print(f"  {n} 超出[-1,1]: 低端 {lo*100:.3f}%  高端 {hi*100:.3f}%  合计 {(lo+hi)*100:.3f}%")


if __name__ == "__main__":
    main()
