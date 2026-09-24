"""#30 单 MatMul 探针：生成 3 个只含一个 MatMul 的 ONNX + 校准/测试数据。

方案：`scripts/EXP_PLAN_MATMUL_PROBE.md`（上一线程已定稿，判据不得改）。
问题：**HTP 内部有没有与 K 无关的硬性精度地板？**
  · H-floor：内部把 A16 隐式截断到 ~12 bit ⇒ 误差与 K 基本无关
  · H-acc  ：累加/分块策略随 K 恶化      ⇒ 误差随 K 显著增长
区分标志就是**误差对 K 的依赖曲线**，所以必须是同一设计、只变 K。

## 两个方案没写死、由本脚本定下的实现细节（显式声明，不静默选择）

1. **量化配置取"主模型基线"四项**（`--act_bitwidth 16 --weights_bitwidth 8
   --bias_bitwidth 32 --*_quantizer_calibration min-max`），**不加 per-row**。
   理由：判据是 `E_htp / E_fxp` 这个**比值**，而 `E_fxp` 用的是与 HTP 完全相同的
   encoding ⇒ encoding 好坏在比值里对消，per-row 与否不影响判别力。
   （#44 已证 `--use_per_channel_quantization` 对零 Conv 模型空转，故不加。）

2. **输入的"条件数"取良态**：`x ~ N(0, σ)`，σ 使量程与 `add_138`
   （实测 encoding `min -131.585, max 1574.002, scale 0.0260`，跨度 1705.6）**同数量级**，
   但**不注入 massive activation 离群点**。
   ⇒ 这是**对 HTP 有利**的一侧：若在良态输入下 HTP 仍丢多位，
   那就是**与条件数无关的硬地板**，结论更强。
   ⚠️ 反过来不成立：若这里正常，**不能**推出真实模型里也正常
   （真实模型是恶态：`unified` 的 99.83% 能量集中在前 1%）。这条写进结果，不许外推。

## 形状

`y = x @ W`，`x[1,K]`、`W[K,64]`，`K ∈ {64, 1024, 3840}`（3840 = 注意力投影真实 K）。
权重最大 3840x64 = 245K 参数 ⇒ 三个 context binary 都极小，**设备内存占用可忽略**。
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = r"D:\ZImage_Work\p0_experiments\matmul_probe"
KS = [64, 1024, 3840]
N = 64
N_CALIB = 16          # 校准样本数（min-max 校准，样本多少只影响量程覆盖）
# 测试样本数。N=64 个输出 ⇒ 单个样本只有 64 个误差值，std 的相对不确定度约 9%，
# 而本实验的核心信号是"误差随 K 怎么变"这条曲线 —— 噪声会直接糊掉它。
# 32 个样本 => 2048 个误差值，判别力足够。
# ⚠️ 这不改判据（判据仍是 E_htp/E_fxp 与等效位宽），只是把测量噪声压下去。
N_TEST = 32
SEED_W, SEED_X = 20260817, 42

# add_138 的实测 encoding（part1a_dlcinfo.txt，①实测，非估计）
ADD138_MIN, ADD138_MAX = -131.585418701172, 1574.002197265625
X_SIGMA = (ADD138_MAX - ADD138_MIN) / 8.0     # ±4σ 覆盖同量级跨度 => σ ≈ 213


def build(k):
    d = os.path.join(OUT, f"k{k}")
    os.makedirs(os.path.join(d, "calib"), exist_ok=True)

    w = np.random.default_rng(SEED_W + k).standard_normal((k, N)).astype(np.float32)
    w /= np.sqrt(k)                       # 保持 y 的量级不随 K 爆炸，便于跨 K 比较

    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"], name="probe_matmul")],
        f"probe_k{k}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N])],
        [numpy_helper.from_array(w, "W")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    p = os.path.join(d, f"probe_k{k}.onnx")
    onnx.save(m, p)

    rng = np.random.default_rng(SEED_X + k)
    lines = []
    for i in range(N_CALIB):
        x = (rng.standard_normal((1, k)) * X_SIGMA).astype(np.float32)
        f = os.path.join(d, "calib", f"x_{i:04d}.raw")
        np.ascontiguousarray(x).tofile(f)
        lines.append(f"x:={f}")
    with open(os.path.join(d, "input_list.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # 测试输入：与校准**同分布但不同 seed**，避免"在校准集上测"这种自欺
    os.makedirs(os.path.join(d, "test"), exist_ok=True)
    rt = np.random.default_rng(9000 + k)
    tl = []
    for i in range(N_TEST):
        xt = (rt.standard_normal((1, k)) * X_SIGMA).astype(np.float32)
        f = os.path.join(d, "test", f"x_{i:04d}.raw")
        np.ascontiguousarray(xt).tofile(f)
        tl.append(f)
    with open(os.path.join(d, "input_list_test.txt"), "w") as f:
        f.write("\n".join(f"x:={t}" for t in tl) + "\n")
    np.save(os.path.join(d, "W.npy"), w)

    xt_all = np.stack([np.fromfile(t, np.float32) for t in tl])
    y = xt_all.astype(np.float64) @ w.astype(np.float64)
    print(f"K={k:>5}  W{w.shape}  x 量程 [{xt_all.min():>9.2f}, {xt_all.max():>8.2f}]  "
          f"y std={y.std():.4f}  测试样本 {N_TEST}  -> {p}")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"x 的 sigma = {X_SIGMA:.2f}（add_138 跨度 {ADD138_MAX - ADD138_MIN:.1f} 的 1/8）")
    print(f"校准 {N_CALIB} 样本 / 测试 {N_TEST} 样本（不同 seed）\n")
    for k in KS:
        build(k)
    print(f"\n产物根目录: {OUT}")


if __name__ == "__main__":
    main()
