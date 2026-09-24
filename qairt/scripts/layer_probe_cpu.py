"""EXP_PLAN_LAYER_LOCALIZE 的宿主半边：用 SNPE CPU 参考导出 part1b 的逐层探针张量。

走 snpe_runner.py（约束 3），输入与"实验A 隔离 part1b"逐字节相同。
设备半边用 `qnn-net-run --dlc_path --debug` 导出同名张量后比对。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snpe_runner

TB = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
WORK = r"D:\ZImage_Work\p0_experiments\layer_probe\cpu"

IN_SHAPES = {"adaln_input": (1, 256), "add_131": (1, 1, 3840),
             "add_138": (1, 4128, 3840), "select_45": (1, 4128, 1, 64),
             "select_46": (1, 4128, 1, 64), "tanh_19": (1, 1, 3840)}

# 沿图均匀分布的探针（Id 见注释）。val_2474 (1,30,4128,4128)=2.04GB 暂不取。
PROBES = [
    ("mul_304", (1, 4128, 3840)),                  # Id=0   RmsNorm
    ("stack_28", (1, 4128, 30, 64, 2)),            # Id=50  Concat
    ("add_153", (1, 4128, 3840)),                  # Id=103 Eltwise_Binary
    ("view_125", (1, 4128, 30, 128)),              # Id=153 Reshape
    ("linear_115_fc", (4128, 3840)),               # Id=203 FullyConnected
    ("linear_118_fc", (4128, 3840)),               # Id=256 FullyConnected
    ("mul_394", (1, 4128, 30, 64)),                # Id=306 Eltwise_Binary
    ("node_linear_129_pre_reshape", (4128, 10240)),# Id=359 Reshape
    ("mul_424", (1, 4128, 30, 64)),                # Id=409 Eltwise_Binary
    ("mul_436", (1, 4128, 3840)),                  # Id=459 RmsNorm
    ("select_160_pre_reshape", (1, 4128, 30, 64, 1)),  # Id=562 Gather
    ("mul_484", (1, 4128, 3840)),                  # Id=624 Eltwise_Binary
    ("unified", (1, 4128, 3840)),                  # 图输出，用于判据 0 自洽检查
]


def main():
    os.makedirs(WORK, exist_ok=True)
    feeds = {n: np.fromfile(f"{TB}\\{n}.raw", np.float32).reshape(s)
             for n, s in IN_SHAPES.items()}
    for n, a in feeds.items():
        print(f"  输入 {n:<14} {a.shape}")

    want = [p[0] for p in PROBES]
    extra = {n: s for n, s in PROBES}
    total_mb = sum(int(np.prod(s)) for _, s in PROBES) * 4 / 1e6
    print(f"\n请求 {len(want)} 个张量，合计约 {total_mb:.0f} MB")
    print("跑 SNPE CPU 参考（snpe_runner，四项契约检查）...", flush=True)

    res = snpe_runner.run("transformer_part1b", feeds, want, WORK, extra_shapes=extra)

    print("\n导出成功：")
    for n, _ in PROBES:
        a = res[n]
        print(f"  {n:<30} {str(a.shape):<24} std={a.std():.6f}")

    # 自洽检查：unified 必须与既有 CPU 参考逐位相同
    ref = np.fromfile(f"{TB}\\out\\Result_0\\unified.raw", np.float32)
    got = res["unified"].reshape(-1)
    same = np.array_equal(ref, got)
    print(f"\n自洽检查 unified 与既有 CPU 参考逐位相同: {'[OK]' if same else '[FAIL]'}")
    if not same:
        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        print(f"  最大差 = {d.max():.6g}（应为 0）")


if __name__ == "__main__":
    main()
