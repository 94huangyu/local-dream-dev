"""第二轮探针的宿主半边：SNPE CPU 参考导出最后一个 block 链路上的 17 个张量。

注意力矩阵 val_2591/val_2592 各 2 GB，本轮不取（若定位指向注意力内部再单独取）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snpe_runner
from probes2_lastblock import PROBES2

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
WORK = r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu"

IN_SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840),
             "add_138": (1, 4128, 3840), "select_45": (1, 4128, 1, 64),
             "select_46": (1, 4128, 1, 64), "tanh_19": (1, 1, 3840)}


def main():
    os.makedirs(WORK, exist_ok=True)
    feeds = {n: np.fromfile(f"{TB}\\{n}.raw", np.float32).reshape(s)
             for n, s in IN_SHAPES.items()}

    want = [p[0] for p in PROBES2]
    extra = dict(PROBES2)
    mb = sum(int(np.prod(s)) for _, s in PROBES2) * 4 / 1e6
    print(f"请求 {len(want)} 个张量，合计约 {mb:.0f} MB", flush=True)
    print("跑 SNPE CPU 参考（snpe_runner，四项契约检查）...", flush=True)

    res = snpe_runner.run("transformer_part1b", feeds, want, WORK, extra_shapes=extra)

    print("\n导出成功：")
    for n, _ in PROBES2:
        a = res[n]
        print(f"  {n:<34} {str(a.shape):<24} std={a.std():.6f}")


if __name__ == "__main__":
    main()
