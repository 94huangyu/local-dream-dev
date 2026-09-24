"""第二轮探针：沿【真实 DAG】覆盖 part1b 最后一个 block（layers.14）的完整链路。

第一轮（13 个沿 Id 均匀分布的探针）的教训：
  Id 顺序 ≠ 依赖顺序，图上有并行分支与残差连接，
  相对误差沿 Id 既不单调也不可比 —— EXP_PLAN_LAYER_LOCALIZE 的判据 1/3 因此作废。

本轮改为沿依赖链取点，可回答：mul_484 处 k=0.4077（HTP 只算出 40.8% 幅值）
是【在该算子产生】还是【从上游继承】。

两个 2 GB 的注意力矩阵（val_2591 打分、val_2592 softmax 输出）本轮不取；
若定位指向注意力块内部，再单独跑一轮取它们。
"""

# (张量名, shape)  —— 按依赖顺序，不是 Id 顺序
PROBES2 = [
    # --- 注意力路径 ---
    ("transpose_72",                    (1, 30, 4128, 128)),   # Id=589 Q
    ("scaled_dot_product_attention_18", (1, 30, 4128, 128)),   # Id=601 注意力输出
    ("linear_150_fc",                   (4128, 3840)),         # Id=604 输出投影
    ("mul_476",                         (1, 4128, 3840)),      # Id=606 RmsNorm
    ("mul_477",                         (1, 4128, 3840)),      # Id=607 门控
    ("add_222",                         (1, 4128, 3840)),      # Id=608 残差（unified 的另一路）
    # --- FFN 路径（mul_484 的上游）---
    ("mul_479",                         (1, 4128, 3840)),      # Id=609 RmsNorm
    ("mul_480",                         (1, 4128, 3840)),      # Id=610 门控
    ("linear_151_fc",                   (4128, 10240)),        # Id=613
    ("linear_152_fc",                   (4128, 10240)),        # Id=615
    ("val_2609",                        (1, 4128, 10240)),     # Id=617 ElementWiseNeuron(SIGMOID)
    ("silu_19",                         (1, 4128, 10240)),     # Id=618 SiLU = x*sigmoid(x)
    ("mul_481",                         (1, 4128, 10240)),     # Id=619
    ("linear_153_fc",                   (4128, 3840)),         # Id=621
    ("mul_483",                         (1, 4128, 3840)),      # Id=623 RmsNorm ← mul_484 的输入之一
    ("tanh_33",                         (1, 1, 3840)),         # Id=543 门控 ← mul_484 的另一输入
    ("mul_484",                         (1, 4128, 3840)),      # Id=624 k=0.4077 的那个
]

# 依赖关系（由 trace_dag.py 读出，用于判据）
DEPS = {
    "unified":  ["add_222", "mul_484"],          # 残差加
    "mul_484":  ["tanh_33", "mul_483"],          # 门控乘
    "mul_483":  ["linear_153"],                  # RmsNorm
    "add_222":  ["add_213", "mul_477"],
    "mul_477":  ["tanh_32", "mul_476"],
    "silu_19":  ["linear_152_fc", "val_2609"],
    "mul_481":  ["silu_19", "linear_151_fc"],
}

if __name__ == "__main__":
    tot = sum(4 * __import__("numpy").prod(s) for _, s in PROBES2) / 1e6
    print(f"探针 {len(PROBES2)} 个，合计 {tot:.0f} MB")
    print(",".join(n for n, _ in PROBES2))
