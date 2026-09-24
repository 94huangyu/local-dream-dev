"""判据 4.5（对照）：新的 selective-symmetric-FC 量化 DLC 在 **SNPE CPU 参考**上跑一遍。

目的：确认对称权重本身没有把精度改坏。若 CPU 侧也显著变差，
即使 HTP 侧改善也不能直接采纳（见 EXP_PLAN_SYM_FC_OVR 4.5）。

约束 3：必须走 snpe_runner.py，禁止自己拼 input_list / 写 .raw。
输入取 testB s0 的 part1b 输入（与实验 A / HTP 侧逐字节相同）。

用法: python symfc_cpu_ref.py <quantized.dlc> <workdir> <out.raw>
"""
import os
import sys

import numpy as np

import snpe_runner

SRC = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
NAMES = ("add_138", "add_131", "tanh_19", "adaln_input", "select_45", "select_46")


def main():
    dlc, workdir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    feeds = {n: np.fromfile(os.path.join(SRC, n + ".raw"), dtype=np.float32) for n in NAMES}
    for n, a in feeds.items():
        print(f"  输入 {n:12} {a.size} 元素 / {a.nbytes} 字节  <- {SRC}")
    res = snpe_runner.run("transformer_part1b", feeds, ["unified"], workdir, dlc_path=dlc)
    u = res["unified"]
    print(f"unified: shape={u.shape} 元素={u.size} std={u.std():.4f}")
    u.astype(np.float32).tofile(out_path)
    print(f"写出 {out_path} ({os.path.getsize(out_path)} 字节)")


if __name__ == "__main__":
    main()
