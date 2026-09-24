"""EXP_PLAN_P2_ATTR 宿主半边：CPU 参考导出 part2a 的 12 个探针张量。

表述纪律（MAINLINE §7 / 约束 3）：本脚本产出的是「在 SNPE CPU 参考实现上」的结果，
**不得写成"设备实测"**。且按 #35，CPU 是**浮点执行**量化 DLC（无 --enable_cpu_fxp）。

输入与设备侧**逐字节相同**（testB/s0_transformer_part2 那一份），保证两侧可配对。
"""
import os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import snpe_runner

SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "testB", "s0_transformer_part2")
WORK = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr", "cpu")
DLC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2split", "part2a_fixed_perrow.dlc")
IN_SHAPES = {"unified": (1, 4128, 3840), "unified_mask": (1, 4128),
             "unified_freqs": (1, 4128, 64, 2), "adaln_input": (1, 256)}
PROBES = [("mul_1", (1, 4128, 3840), "RmsNorm"), ("linear_1_fc", (4128, 3840), "FullyConnected"),
          ("mul_4", (1, 4128, 30, 128), "RmsNorm"), ("mul_6", (1, 4128, 30, 128), "RmsNorm"),
          ("scaled_dot_product_attention", (1, 30, 4128, 128), "MatMul"),
          ("linear_4_fc", (4128, 3840), "FullyConnected"), ("mul_16", (1, 4128, 3840), "RmsNorm"),
          ("add_8", (1, 4128, 3840), "Eltwise"), ("mul_19", (1, 4128, 3840), "RmsNorm"),
          ("linear_5_fc", (4128, 10240), "FullyConnected"), ("mul_23", (1, 4128, 3840), "RmsNorm"),
          ("linear_9_fc", (4128, 3840), "FullyConnected")]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(WORK, exist_ok=True)
    feeds = {}
    for n, s in IN_SHAPES.items():
        p = os.path.join(SRC, n + ".raw")
        feeds[n] = np.fromfile(p, np.float32).reshape(s)
        print("  输入 %-14s %s  md5=%s" % (n, s, md5(p)[:16]), flush=True)
    want = [p[0] for p in PROBES]
    extra = {p[0]: p[1] for p in PROBES}
    mb = sum(int(np.prod(p[1])) for p in PROBES) * 4 / 1e6
    print("请求 %d 个张量，合计约 %.0f MB" % (len(want), mb), flush=True)
    res = snpe_runner.run("transformer_part2a", feeds, want, WORK,
                          extra_shapes=extra, dlc_path=DLC)
    print("\n导出成功（**SNPE CPU 参考实现**，浮点执行，非设备实测）：")
    for n, sh, ty in PROBES:
        a = res[n]
        print("  %-30s %-16s %-22s std=%.6f" % (n, ty, str(a.shape), a.std()))


if __name__ == "__main__":
    main()
