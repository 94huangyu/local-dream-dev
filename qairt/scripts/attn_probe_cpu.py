"""EXP_PLAN_ATTENTION 的宿主半边：导出注意力段入口的 5 个张量。

依赖链（trace_dag.py 已验证）：
  transpose_72(Q) -> val_2587(缩放) ┐
  val_2590(K 缩放) -> val_2590_converted_unsigned_symmetric ┤-> val_2591 = MatMul(QK^T)
                                                             -> val_2592 = Softmax
  transpose_74(V) -> transpose_74_converted_unsigned_symmetric ┘-> scaled_dot_product_attention_18

本轮只取入口 5 个小张量（315 MB）。若入口余弦已掉 -> 已定位，无需再取 2GB 的注意力矩阵。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snpe_runner

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
WORK = r"D:\ZImage_Work\p0_experiments\attn_probe\cpu"

IN_SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840),
             "add_138": (1, 4128, 3840), "select_45": (1, 4128, 1, 64),
             "select_46": (1, 4128, 1, 64), "tanh_19": (1, 1, 3840)}

PROBES = [
    ("val_2587", (1, 30, 4128, 128)),                                  # Q x scale
    ("val_2590", (1, 30, 128, 4128)),                                  # K^T x scale
    ("val_2590_converted_unsigned_symmetric", (1, 30, 128, 4128)),     # Convert(K)
    ("transpose_74", (1, 30, 4128, 128)),                              # V
    ("transpose_74_converted_unsigned_symmetric", (1, 30, 4128, 128)),  # Convert(V)
]


def main():
    os.makedirs(WORK, exist_ok=True)
    feeds = {n: np.fromfile(f"{TB}\\{n}.raw", np.float32).reshape(s)
             for n, s in IN_SHAPES.items()}
    want = [p[0] for p in PROBES]
    extra = dict(PROBES)
    mb = sum(int(np.prod(s)) for _, s in PROBES) * 4 / 1e6
    print(f"请求 {len(want)} 个张量，合计约 {mb:.0f} MB", flush=True)
    res = snpe_runner.run("transformer_part1b", feeds, want, WORK, extra_shapes=extra)
    print("\n导出成功：")
    for n, _ in PROBES:
        a = res[n]
        print(f"  {n:<46} {str(a.shape):<22} std={a.std():.6f}")


if __name__ == "__main__":
    main()
