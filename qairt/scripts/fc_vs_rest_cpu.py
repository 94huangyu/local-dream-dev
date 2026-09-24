"""EXP_PLAN_FC_VS_REST 宿主半边：CPU 参考导出 part1b 前两个 block 的 21 个张量。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snpe_runner

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
WORK = r"D:\ZImage_Work\p0_experiments\fc_probe\cpu"

IN_SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840),
             "add_138": (1, 4128, 3840), "select_45": (1, 4128, 1, 64),
             "select_46": (1, 4128, 1, 64), "tanh_19": (1, 1, 3840)}

# (名, shape, 算子类型, Id) —— 依赖链顺序
PROBES = [
    ("mul_304", (1, 4128, 3840), "RmsNorm", 0),
    ("linear_97_fc", (4128, 3840), "FullyConnected", 12),
    ("mul_308", (1, 4128, 3840), "RmsNorm", 14),
    ("mul_311", (1, 4128, 3840), "RmsNorm", 24),
    ("linear_99_fc", (4128, 3840), "FullyConnected", 29),
    ("linear_100_fc", (4128, 3840), "FullyConnected", 31),
    ("mul_314", (1, 4128, 30, 128), "RmsNorm", 35),
    ("linear_102_fc", (4128, 3840), "FullyConnected", 82),
    ("mul_326", (1, 4128, 3840), "RmsNorm", 84),
    ("linear_105_fc", (4128, 3840), "FullyConnected", 99),
    ("mul_333", (1, 4128, 3840), "RmsNorm", 101),
    ("add_153", (1, 4128, 3840), "Eltwise_Binary", 103),
]


def main():
    os.makedirs(WORK, exist_ok=True)
    feeds = {n: np.fromfile(f"{TB}\\{n}.raw", np.float32).reshape(s)
             for n, s in IN_SHAPES.items()}
    want = [p[0] for p in PROBES]
    extra = {p[0]: p[1] for p in PROBES}
    mb = sum(int(np.prod(p[1])) for p in PROBES) * 4 / 1e6
    print(f"请求 {len(want)} 个张量，合计约 {mb:.0f} MB", flush=True)
    res = snpe_runner.run("transformer_part1b", feeds, want, WORK, extra_shapes=extra)
    print("\n导出成功：")
    for n, sh, ty, oid in PROBES:
        a = res[n]
        print(f"  Id={oid:<5}{ty:<18}{n:<34}{str(a.shape):<22} std={a.std():.6f}")


if __name__ == "__main__":
    main()