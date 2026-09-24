"""导出 FC 的输入激活（CPU 参考），用于离线精确模拟 A16W8 真定点执行。

背景：`--enable_cpu_fxp` 被 SNPE 拒绝
（error_code=703: "CPU fixed point execution is selected with DLC having INT16 activation"）
⇒ 本 SDK 无法提供 A16W8 的真定点 CPU 参考。
⇒ 改为自己用 numpy 精确模拟，这样可以同时对比 FP32 与 HTP，
   回答"真定点本身是否就会崩"这个 CPU-fxp 回答不了的问题。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snpe_runner

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
WORK = r"D:\ZImage_Work\p0_experiments\fc_inputs\cpu"

IN_SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840),
             "add_138": (1, 4128, 3840), "select_45": (1, 4128, 1, 64),
             "select_46": (1, 4128, 1, 64), "tanh_19": (1, 1, 3840)}

# FC 的输入（Reshape 输出）——与我已有 HTP 输出的那批 FC 对应
PROBES = [
    ("node_linear_97_pre_reshape", (4128, 10240)),   # -> linear_97_fc
    ("node_linear_99_pre_reshape", (4128, 3840)),    # -> linear_99_fc
    ("node_linear_100_pre_reshape", (4128, 3840)),   # -> linear_100_fc
    ("node_linear_102_pre_reshape", (4128, 3840)),   # -> linear_102_fc
    ("node_linear_105_pre_reshape", (4128, 10240)),  # -> linear_105_fc
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
    for n, sh in PROBES:
        a = res[n]
        print(f"  {n:<34}{str(a.shape):<18} std={a.std():.6f}  "
              f"min={a.min():.4f} max={a.max():.4f}")


if __name__ == "__main__":
    main()
