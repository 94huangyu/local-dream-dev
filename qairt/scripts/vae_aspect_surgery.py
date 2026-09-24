# -*- coding: utf-8 -*-
"""把 vae_decoder.onnx 从 1024x1024 改到目标长宽比。

实证必需：`vae_decoder.onnx` 输入写死 `[1,16,128,128]`，4:3 的 latents 喂进去直接报
`Got: 108 Expected: 128`（2026-09-02 实测）。

🔴 **歧义陷阱**：`128/256/512` 既是**通道数**也是**空间尺寸**
   （`val_482 = [1,512,512,512]` 里三个 512 含义不同；`val_144 = [1,512,128,128]`）
   ⇒ **只能按位置处理**（NCHW 的第 2/3 位是空间），**禁止按值全局替换**（约束 8）。

空间层级（VAE 8x 上采样，三级各 x2）：
   latent 128 -> 256 -> 512 -> 1024(像素)
   4:3 时     108x144 -> 216x288 -> 432x576 -> 864x1152
注意力块把空间展平：`16384 = 128x128` -> `15552 = 108x144`。

验收（另见 vae_aspect_check.py）：手术后 ONNX 解码同一份 latents，
与 **PyTorch AutoencoderKL** 的解码结果比 PSNR。参照 §2.2 的 VAE 导出保真度 79.92 dB。

用法: python vae_aspect_surgery.py <tag> <宽> <高>
"""
import os
import sys

import numpy as np
import onnx
from onnx import external_data_helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import canonical_sources                      # noqa: E402

ONNX = canonical_sources.ONNX_DIR
SRC = "vae_decoder.onnx"
OLD_LAT = 128


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    assert W % 16 == 0 and H % 16 == 0
    lh, lw = H // 8, W // 8
    # 空间层级映射：旧的正方尺寸 -> 新的 (h, w)
    smap = {}
    for k in range(4):
        old = OLD_LAT * (2 ** k)
        smap[old] = (lh * (2 ** k), lw * (2 ** k))
    fmap = {o * o: n[0] * n[1] for o, n in smap.items()}     # 展平后的 H*W
    print("空间映射 %s" % {k: v for k, v in sorted(smap.items())}, flush=True)
    print("展平映射 %s" % {k: v for k, v in sorted(fmap.items())}, flush=True)

    cwd = os.getcwd()
    os.chdir(ONNX)
    try:
        m = onnx.load(SRC, load_external_data=False)
        g = m.graph
        n_c = n_f = n_io = 0

        for t in g.initializer:
            if t.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                continue
            if t.data_location == 1:
                continue
            a = numpy_helper.to_array(t)
            if a.size == 0 or a.size > 16:
                continue
            v = [int(x) for x in a.reshape(-1)]
            nv = list(v)
            touched = False
            # 规则 A：长度 4 的 NCHW，且第 2/3 位相等且是已知空间尺寸 -> 按位置改
            if len(v) == 4 and v[2] == v[3] and v[2] in smap:
                nv[2], nv[3] = smap[v[2]]
                touched = True
                n_c += 1
            # 规则 B：任何位置出现展平后的 H*W -> 换成新的（16384 等值不与通道数撞）
            for i, x in enumerate(nv):
                if x in fmap:
                    nv[i] = fmap[x]
                    touched = True
                    if not (len(v) == 4 and v[2] == v[3] and v[2] in smap):
                        n_f += 1
            if not touched:
                continue
            arr = np.array(nv, dtype=a.dtype).reshape(a.shape)
            t.CopyFrom(numpy_helper.from_array(arr, name=t.name))

        # IO + value_info：同样按位置
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            d = vi.type.tensor_type.shape.dim
            dv = [x.dim_value if x.HasField("dim_value") else None for x in d]
            if len(dv) == 4 and dv[2] is not None and dv[2] == dv[3] and dv[2] in smap:
                d[2].dim_value, d[3].dim_value = smap[dv[2]]
                n_io += 2
            for i, x in enumerate(dv):
                if x in fmap:
                    d[i].dim_value = fmap[x]
                    n_io += 1

        # 门 V1：输入/输出形状必须正是目标
        gi = {i.name: [x.dim_value for x in i.type.tensor_type.shape.dim] for i in g.input}
        go = {o.name: [x.dim_value for x in o.type.tensor_type.shape.dim] for o in g.output}
        assert gi["vae_latents"] == [1, 16, lh, lw], "输入 %s" % gi["vae_latents"]
        assert go["pixels"] == [1, 3, H, W], "输出 %s" % go["pixels"]

        onnx.checker.check_model(m, full_check=False)
        dst = "vae_decoder_%s.onnx" % tag
        onnx.save(m, dst)
        print("形状常量 %d | 展平常量 %d | IO+value_info 维 %d -> %s"
              % (n_c, n_f, n_io, dst), flush=True)
        print("输入 %s  输出 %s" % (gi["vae_latents"], go["pixels"]), flush=True)
    finally:
        os.chdir(cwd)
    return 0


sys.exit(main())
