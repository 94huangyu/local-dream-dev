"""🔴 2026-08-21 作废声明（HANDOVER 15.22 / 台账 #80）
本脚本的 ONNX 臂喂的是 lat/0.3611+0.1159，而 vae_decoder.onnx 图内首节点已有
Div(vae_latents,0.3611) ⇒ 除了两遍。**本脚本产出的一切数字与图像一律作废，不得引用。**
正确喂法与三臂对照见 scripts/vae_input_convention.py。
按约束 2 保留原文，不删除。
"""
"""#77：定位 VAE 导出的 x1.3324 全局增益在哪一层累积。

方案：scripts/EXP_PLAN_VAE_LAYER.md（判据事前锁定）
锚点：两侧的 Conv（35 个，4 维，不被 Reshape 分解）—— 取代上次失败的 group_norm 按序匹配。
主量：逐层输出 std 之比（#79：高频能量主要在反映增益，不得再用）。
"""
import os, sys, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
LAT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fp32_steps", "lat_8.raw")
MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SCALING, SHIFT = 0.3611, 0.1159
SUB = 100_000
END_GAIN, END_TOL = 1.3324, 0.02


def sub_idx(i, n):
    """两侧完全一致的固定子样下标：只由 (层号, 元素数) 决定"""
    rng = np.random.default_rng(1234 + i * 7919 + n % 1000003)
    return rng.choice(n, min(SUB, n), replace=False)


def stat(i, a):
    a = a.astype(np.float64)
    return dict(shape=tuple(a.shape), mean=float(a.mean()), std=float(a.std()),
                sub=a.ravel()[sub_idx(i, a.size)].copy())


def top1_share(a):
    e = np.sort(a.astype(np.float64) ** 2)[::-1]
    k = max(1, int(round(len(e) * 0.01)))
    return float(e[:k].sum() / max(e.sum(), 1e-30))


def main():
    lat = np.fromfile(LAT, np.float32).reshape(1, 16, 128, 128)
    assert abs(lat.std() - 1.0996) < 0.01, "身份门不过"
    vae_in = (lat.astype(np.float64) / SCALING + SHIFT).astype(np.float32)

    # ---------------- PyTorch 臂 ----------------
    import torch
    from diffusers import AutoencoderKL
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    vae.eval()
    convs = [m for m in vae.decoder.modules() if isinstance(m, torch.nn.Conv2d)]
    pqc = getattr(vae, "post_quant_conv", None)
    print(f"[结构] decoder 内 Conv2d = {len(convs)}   "
          f"post_quant_conv = {type(pqc).__name__ if pqc is not None else 'None'}")
    pt, hs = [], []

    def mk():
        def h(mod, inp, out):
            pt.append(stat(len(pt), out.detach().numpy()))
        return h
    for m in convs:
        hs.append(m.register_forward_hook(mk()))
    with torch.no_grad():
        vae.decode(torch.from_numpy(vae_in), return_dict=False)
    for h in hs:
        h.remove()
    del vae
    print(f"[臂A] PyTorch 抓到 {len(pt)} 个 Conv 输出  {time.time()-t0:.0f}s", flush=True)

    # ---------------- ONNX 臂 ----------------
    import onnx
    from onnx import helper
    import onnxruntime as ort
    m = onnx.load(os.path.join(ONNX, "vae_decoder.onnx"), load_external_data=True)
    names = [n.output[0] for n in m.graph.node if n.op_type == "Conv"]
    print(f"[结构] ONNX Conv 节点 = {len(names)}")

    # ---- V1 数量门 ----
    if len(names) != len(pt):
        sys.exit(f"❌ V1 不过：Conv 数量 ONNX {len(names)} vs PyTorch {len(pt)} ⇒ 锚点选择错误")
    print(f"[V1] Conv 数量一致 = {len(names)}  ✅")

    have = {o.name for o in m.graph.output}
    for nm in names:
        if nm not in have:
            m.graph.output.append(helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
    tmp = os.path.join(os.environ.get("TEMP", "."), "vae_conv_probe.onnx")
    onnx.save(m, tmp, save_as_external_data=True, all_tensors_to_one_file=True,
              location="vae_conv_probe.data", size_threshold=1024)
    del m
    s = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])

    ox = [None] * len(names)
    BATCH = 4
    for b0 in range(0, len(names), BATCH):
        req = names[b0:b0 + BATCH]
        t1 = time.time()
        outs = s.run(req, {"vae_latents": vae_in})
        for j, a in enumerate(outs):
            ox[b0 + j] = stat(b0 + j, a)
        del outs
        print(f"[臂B] Conv {b0}..{b0+len(req)-1}  {time.time()-t1:.0f}s", flush=True)
    del s

    # ---- V2 形状门 ----
    for i, (a, b) in enumerate(zip(pt, ox)):
        if a["shape"] != b["shape"]:
            sys.exit(f"❌ V2 不过：Conv#{i} 形状 PyTorch {a['shape']} vs ONNX {b['shape']} "
                     f"⇒ 匹配错位，立即停止（上次正是这条断言救的场）")
    print(f"[V2] 全部 {len(pt)} 个 Conv 形状逐个相等  ✅")

    # ---- V3 端点门 ----
    r_end = ox[-1]["std"] / pt[-1]["std"]
    ok3 = abs(r_end - END_GAIN) <= END_TOL
    print(f"[V3] 末层 std 之比 = {r_end:.4f}   期望 {END_GAIN}±{END_TOL}  "
          f"{'✅' if ok3 else '❌ 测量改变了被测对象（融合被抑制？）'}")
    if not ok3:
        sys.exit("❌ V3 不过：结论作废")

    # ---------------- 主判据 ----------------
    print()
    print("=== 逐层结果 ===")
    print(f"{'#':<4}{'shape':<22}{'std_pt':>10}{'std_onnx':>10}{'r=比值':>10}"
          f"{'q=增量':>9}{'残差s':>9}{'top1%':>8}")
    print("-" * 82)
    rs, qs, ss = [], [], []
    prev = 1.0
    for i, (a, b) in enumerate(zip(pt, ox)):
        r = b["std"] / max(a["std"], 1e-12)
        q = r / prev
        prev = r
        x, y = a["sub"], b["sub"]
        k = float(np.cov(x, y)[0, 1] / max(np.var(x), 1e-30))
        c = float(y.mean() - k * x.mean())
        res = y - (k * x + c)
        yc = y - y.mean()
        sres = float(np.linalg.norm(res) / max(np.linalg.norm(yc), 1e-30))
        rs.append(r); qs.append(q); ss.append(sres)
        flag = " 🔴" if q >= 1.10 else (" 🔶" if q >= 1.05 else "")
        print(f"{i:<4}{str(a['shape']):<22}{a['std']:>10.5f}{b['std']:>10.5f}"
              f"{r:>10.4f}{q:>9.4f}{sres*100:>8.2f}%{top1_share(yc)*100:>7.1f}%{flag}")

    qmax = max(qs)
    print()
    print("=== 主判据 A：增益累积形态 ===")
    if qmax >= 1.10:
        top = sorted(range(len(qs)), key=lambda i: -qs[i])[:3]
        verdict = "🔴 单点跳变 ⇒ 定位到该层，查该层之前的算子"
        detail = "  最大跳变层: " + "  ".join(f"Conv#{i}(q={qs[i]:.3f})" for i in top)
    elif qmax < 1.05:
        verdict = "🔴 均匀累积 ⇒ 指向归一化层的普遍性数值问题"
        detail = (f"  最大单层增量 q={qmax:.4f} < 1.05；"
                  f"若 {len(qs)} 层均匀分摊，每层需 {rs[-1]**(1/len(qs)):.4f}")
    else:
        top = sorted(range(len(qs)), key=lambda i: -qs[i])[:3]
        verdict = "🔶 混合形态 ⇒ 报 top-3 跳变层，逐个查"
        detail = "  top-3: " + "  ".join(f"Conv#{i}(q={qs[i]:.3f})" for i in top)
    print(f"  最大 q = {qmax:.4f}   末层 r = {rs[-1]:.4f}")
    print(f"  判定: {verdict}")
    print(detail)

    first = next((i for i, v in enumerate(ss) if v >= 0.03), None)
    print()
    print("=== 主判据 B：结构残差引入点（s_i >= 3%）===")
    print(f"  {'第一个越线层: Conv#' + str(first) + f'  s={ss[first]*100:.2f}%' if first is not None else '全程未越线（s_i < 3%）'}")
    print(f"  末层 s = {ss[-1]*100:.2f}%   （成图上该残差为 9.86%）")

    np.save(os.path.join(OUT, "vae_layer_r.npy"), np.array(rs))
    np.save(os.path.join(OUT, "vae_layer_s.npy"), np.array(ss))
    print()
    print("[落盘] scratch_runs/vae_layer_{r,s}.npy")


if __name__ == "__main__":
    main()
