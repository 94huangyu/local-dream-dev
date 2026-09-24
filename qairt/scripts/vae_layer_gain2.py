"""🔴 2026-08-21 作废声明（HANDOVER 15.22 / 台账 #80）
本脚本的 ONNX 臂喂的是 lat/0.3611+0.1159，而 vae_decoder.onnx 图内首节点已有
Div(vae_latents,0.3611) ⇒ 除了两遍。**本脚本产出的一切数字与图像一律作废，不得引用。**
正确喂法与三臂对照见 scripts/vae_input_convention.py。
按约束 2 保留原文，不删除。
"""
"""#77 v2：按【权重字节】匹配的 VAE 逐层增益定位。

v1 作废原因见 EXP_PLAN_VAE_LAYER.md 第五节：Conv 0~10 输出形状全相同，
「形状相等」这条门允许错位全过。v2 改用权重哈希做无歧义双射。
"""
import os, sys, time, hashlib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
LAT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fp32_steps", "lat_8.raw")
MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
SCALING, SHIFT = 0.3611, 0.1159
SUB = 100_000


def wh(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def sub_idx(n):
    """只依赖元素数 n，与配对顺序无关（v1 的种子依赖层号，是缺陷）"""
    return np.random.default_rng(n % 2147483647).choice(n, min(SUB, n), replace=False)


def stat(a):
    a = a.astype(np.float64)
    return dict(shape=tuple(a.shape), mean=float(a.mean()), std=float(a.std()),
                sub=a.ravel()[sub_idx(a.size)].copy())


def top1(a):
    e = np.sort(a.astype(np.float64) ** 2)[::-1]
    return float(e[:max(1, len(e) // 100)].sum() / max(e.sum(), 1e-30))


def main():
    lat = np.fromfile(LAT, np.float32).reshape(1, 16, 128, 128)
    assert abs(lat.std() - 1.0996) < 0.01, "identity gate failed"
    vae_in = (lat.astype(np.float64) / SCALING + SHIFT).astype(np.float32)

    # ---------------- PyTorch arm ----------------
    import torch
    from diffusers import AutoencoderKL
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    vae.eval()
    name_of = {id(m): n for n, m in vae.decoder.named_modules()}
    pt, hs = [], []

    def hook(mod, inp, out):
        pt.append(dict(name=name_of.get(id(mod), "?"), h=wh(mod.weight.detach().numpy()),
                       wshape=tuple(mod.weight.shape), **stat(out.detach().numpy())))

    for m in vae.decoder.modules():
        if isinstance(m, torch.nn.Conv2d):
            hs.append(m.register_forward_hook(hook))
    with torch.no_grad():
        vae.decode(torch.from_numpy(vae_in), return_dict=False)
    for x in hs:
        x.remove()
    del vae
    print("[armA] PyTorch %d Conv (true call order)  %.0fs" % (len(pt), time.time() - t0), flush=True)

    # ---------------- ONNX weight hashes ----------------
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    m = onnx.load(os.path.join(ONNX, "vae_decoder.onnx"), load_external_data=True)
    init = {t.name: t for t in m.graph.initializer}
    const = {n.output[0]: n for n in m.graph.node if n.op_type == "Constant"}
    ox_meta = []
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        w = n.input[1]
        if w in init:
            arr = numpy_helper.to_array(init[w])
        elif w in const:
            arr = numpy_helper.to_array(const[w].attribute[0].t)
        else:
            sys.exit("ERR: Conv %s weight %s is neither initializer nor Constant" % (n.name, w))
        ox_meta.append(dict(out=n.output[0], h=wh(arr), wshape=tuple(arr.shape)))
    print("[armB] ONNX %d Conv nodes" % len(ox_meta))

    # ---------------- V1 / V2a / V2b ----------------
    if len(ox_meta) != len(pt):
        sys.exit("V1 FAIL: Conv count %d vs %d" % (len(ox_meta), len(pt)))
    pt_by_h = {}
    for i, r in enumerate(pt):
        pt_by_h.setdefault(r["h"], []).append(i)
    dup = {k: v for k, v in pt_by_h.items() if len(v) > 1}
    if dup:
        sys.exit("V2a FAIL: duplicate weight hashes on PyTorch side: %s" % dup)
    pair, miss = {}, []
    for j, o in enumerate(ox_meta):
        idx = pt_by_h.get(o["h"])
        if idx is None:
            miss.append((j, o["wshape"]))
        elif idx[0] in pair:
            sys.exit("V2b FAIL: PyTorch#%d hit by two ONNX Convs" % idx[0])
        else:
            pair[idx[0]] = j
    if miss:
        sys.exit("V2a FAIL: %d ONNX Convs unmatched: %s" % (len(miss), miss[:5]))
    if len(pair) != len(pt):
        sys.exit("V2b FAIL: only %d/%d paired" % (len(pair), len(pt)))
    perm = [pair[i] for i in range(len(pt))]
    print("[V2a/b] weight-hash bijection OK 35<->35")
    print("[perm] onnx_node_index for pytorch_exec_order = %s" % perm)
    print("       %s" % ("IDENTICAL -> v1 ordering was actually correct"
                         if perm == list(range(len(pt)))
                         else "DIFFERENT -> v1 sequential matching was indeed misaligned"))

    # ---------------- capture ONNX activations ----------------
    have = {o.name for o in m.graph.output}
    names = [ox_meta[pair[i]]["out"] for i in range(len(pt))]
    for nm in names:
        if nm not in have:
            m.graph.output.append(helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
    tmp = os.path.join(os.environ.get("TEMP", "."), "vae_conv_probe2.onnx")
    onnx.save(m, tmp, save_as_external_data=True, all_tensors_to_one_file=True,
              location="vae_conv_probe2.data", size_threshold=1024)
    del m
    s = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    ox = [None] * len(names)
    for b in range(0, len(names), 4):
        req = names[b:b + 4]
        t1 = time.time()
        outs = s.run(req, {"vae_latents": vae_in})
        for j, a in enumerate(outs):
            ox[b + j] = stat(a)
        del outs
        print("[armB] %d..%d  %.0fs" % (b, b + len(req) - 1, time.time() - t1), flush=True)
    del s

    # ---------------- V2c / V2d ----------------
    for i, (a, bb) in enumerate(zip(pt, ox)):
        if a["shape"] != bb["shape"]:
            sys.exit("V2c FAIL: #%d %s shape %s vs %s" % (i, a["name"], a["shape"], bb["shape"]))
    print("[V2c] paired shapes all equal  OK")
    r0 = ox[0]["std"] / pt[0]["std"]
    print("[V2d] first layer %s std ratio = %.6f  (expect 1.000 +- 0.001)  %s"
          % (pt[0]["name"], r0, "OK" if abs(r0 - 1) <= 1e-3 else "FAIL"))
    if abs(r0 - 1) > 1e-3:
        sys.exit("V2d FAIL: known-answer sample measured wrong, results void")

    # ---------------- results ----------------
    print("")
    print("%-4s%-34s%-22s%9s%8s%8s%7s" % ("#", "module", "shape", "r", "q", "s", "top1%"))
    print("-" * 92)
    rs, qs, ss = [], [], []
    prev = 1.0
    for i, (a, bb) in enumerate(zip(pt, ox)):
        r = bb["std"] / max(a["std"], 1e-12)
        q = r / prev
        prev = r
        x, y = a["sub"], bb["sub"]
        k = float(np.cov(x, y)[0, 1] / max(np.var(x), 1e-30))
        res = y - (k * x + (y.mean() - k * x.mean()))
        yc = y - y.mean()
        sr = float(np.linalg.norm(res) / max(np.linalg.norm(yc), 1e-30))
        rs.append(r)
        qs.append(q)
        ss.append(sr)
        fl = " <<<" if q >= 1.10 else (" <<" if q >= 1.05 else "")
        print("%-4d%-34s%-22s%9.4f%8.4f%7.2f%%%6.1f%%%s"
              % (i, a["name"][:33], str(a["shape"]), r, q, sr * 100, top1(yc) * 100, fl))

    qmax = max(qs)
    top = sorted(range(len(qs)), key=lambda i: -qs[i])[:3]
    print("")
    print("=== criterion A: how the gain accumulates ===")
    print("  final r = %.4f   max q = %.4f" % (rs[-1], qmax))
    if qmax >= 1.10:
        print("  VERDICT: SINGLE JUMP -> " + "  ".join(
            "#%d:%s(q=%.3f)" % (i, pt[i]["name"], qs[i]) for i in top))
    elif qmax < 1.05:
        print("  VERDICT: UNIFORM ACCUMULATION (per-layer %.4f if evenly split over %d)"
              % (rs[-1] ** (1.0 / len(qs)), len(qs)))
    else:
        print("  VERDICT: MIXED -> " + "  ".join(
            "#%d:%s(q=%.3f)" % (i, pt[i]["name"], qs[i]) for i in top))
    f3 = next((i for i, v in enumerate(ss) if v >= 0.03), None)
    print("=== criterion B: structural residual onset (s >= 3%) ===")
    print("  first: " + ("#%d %s  s=%.2f%%" % (f3, pt[f3]["name"], ss[f3] * 100)
                         if f3 is not None else "none"))
    print("  final s = %.2f%%" % (ss[-1] * 100))
    np.savez(os.path.join(OUT, "vae_layer_v2.npz"), r=rs, q=qs, s=ss,
             names=[a["name"] for a in pt], shapes=[str(a["shape"]) for a in pt])
    print("[saved] scratch_runs/vae_layer_v2.npz")


if __name__ == "__main__":
    main()
