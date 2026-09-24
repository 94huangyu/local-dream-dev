"""🔴 2026-08-21 作废声明（HANDOVER 15.22 / 台账 #80）
本脚本的 ONNX 臂喂的是 lat/0.3611+0.1159，而 vae_decoder.onnx 图内首节点已有
Div(vae_latents,0.3611) ⇒ 除了两遍。**本脚本产出的一切数字与图像一律作废，不得引用。**
正确喂法与三臂对照见 scripts/vae_input_convention.py。
按约束 2 保留原文，不删除。
"""
"""逐层定位：ONNX VAE 与官方 PyTorch VAE 的数值发散从哪一层开始。

## 背景（2026-08-19 实测）
同一份 final latents，三臂 VAE 的高频能量：
  官方 PyTorch fp32  2.892 （基准）
  我们的 ONNX fp32   3.773 （+30.5%）  ← 导出引入
  设备 QNN 量化      4.021 （+39.0%）  ← 量化再叠加 8.5pp
且 VAE 的**图结构与权重与官方完全等价**（35 Conv/30 GroupNorm/29 SiLU/3 nearest Resize/
1 attention，eps 均 1e-6，73 个权重逐位吻合、4 个注意力权重转置后 0.000000）
⇒ 缺陷是**运行时数值行为**，需定位到具体层。

## 方法
两侧都取 **GroupNorm 的输出**（ONNX 里是 `node_group_norm*` 的 Add 输出，
与官方 decoder 的 30 个 GroupNorm 模块 1:1 对应，按执行序匹配）。

## 判据（事前定稿）
· 某个 GroupNorm 输出**突然**出现明显相对误差 ⇒ ③「GroupNorm→InstanceNorm 分解」假设成立
· 误差从第一层就有且**逐层缓慢累积** ⇒ 是普遍性数值差异，非单点缺陷
· GroupNorm 处都很小、却在别处放大 ⇒ 转查 Resize / Conv

⚠️ 只抓前 N 个（默认 10，均在 128×128、每个 33.5 MB）以控内存；不足以定位再往后延。

用法: python vae_layerwise.py <vae_in.raw> [N]
"""
import os, sys
import numpy as np

E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")


def main():
    vae_in = np.fromfile(sys.argv[1], np.float32).reshape(1, 16, 128, 128)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # ---- ONNX：把前 N 个 group_norm 输出加为 graph output ----
    import onnx
    from onnx import helper
    import onnxruntime as ort
    m = onnx.load(os.path.join(E, "vae_decoder.onnx"), load_external_data=True)
    names = [n.output[0] for n in m.graph.node if n.name.startswith("node_group_norm")][:N]
    for nm in names:
        m.graph.output.append(helper.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, None))
    tmp = os.path.join(os.environ.get("TEMP", "."), "vae_probe.onnx")
    onnx.save(m, tmp, save_as_external_data=True, all_tensors_to_one_file=True,
              location="vae_probe.data", size_threshold=1024)
    s = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    outs = s.run(names, {"vae_latents": vae_in})
    onnx_acts = dict(zip(names, outs))
    print(f"ONNX 抓到 {len(onnx_acts)} 个 group_norm 输出", flush=True)
    del s

    # ---- PyTorch：hook 前 N 个 GroupNorm ----
    import torch
    from diffusers import AutoencoderKL
    v = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"),
                                      torch_dtype=torch.float32, low_cpu_mem_usage=True)
    v.eval()
    caught, hs = [], []
    def mk(i):
        def h(mod, inp, out):
            if len(caught) < N: caught.append(out.detach().numpy())
        return h
    gns = [mo for mo in v.decoder.modules() if isinstance(mo, torch.nn.GroupNorm)]
    for i, mo in enumerate(gns[:N + 6]):
        hs.append(mo.register_forward_hook(mk(i)))
    with torch.no_grad():
        v.decode(torch.from_numpy(vae_in), return_dict=False)
    for h in hs: h.remove()
    print(f"PyTorch 抓到 {len(caught)} 个 GroupNorm 输出", flush=True)

    # ---- 对照 ----
    def hf(a):
        x = a.reshape(a.shape[1], -1) if a.ndim == 4 else a
        return float(np.abs(np.diff(a[0], axis=-1)).mean())
    print(f"\n{'#':<4}{'shape':<22}{'相对L2':>10}{'余弦':>10}{'HF(ONNX/PT)':>14}")
    print("-" * 62)
    for i, nm in enumerate(names):
        if i >= len(caught): break
        a, b = caught[i].astype(np.float64), onnx_acts[nm].astype(np.float64)
        if a.shape != b.shape:
            print(f"{i:<4}形状不符 {a.shape} vs {b.shape} —— 匹配错位，停止"); break
        rel = np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12)
        cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
        ha, hb = hf(a), hf(b)
        print(f"{i:<4}{str(a.shape):<22}{rel*100:>9.4f}%{cos:>10.6f}{hb/max(ha,1e-12):>13.3f}x")


if __name__ == "__main__":
    main()
