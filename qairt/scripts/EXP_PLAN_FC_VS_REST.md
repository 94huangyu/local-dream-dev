# EXP_PLAN_FC_VS_REST —— 误差主要来自 FullyConnected，还是其余算子？

定稿时间：2026-08-15（执行前定稿，判据不得事后修改）
主线归属：A → "如何解决 HTP 的数值问题" → **决定选择性 FP16 是否可行**
> 🔴🔴 **2026-08-22 追加：本方案 §0 的否决前提已被官方文档推翻，但方案主体仍然有效。**
>
> §0 写着 *"浮点化 FullyConnected 则必然出局"*（+6154 MB 撞 3.3 GB PD 红线），
> 前提是「算子转 FP16 = 权重也转 FP16」。**HTP OpDef supplement 实查：不是。**
> FullyConnected 支持 `FP16 激活 | SFIXED_POINT_8 权重 | FP16 bias | FP16 输出`
> （= `wFxp_actFP`，正是 `--keep_weights_quantized` 的用途）⇒ **权重字节数不变，体积红线不成立**。
>
> ⇒ **「H-fc 成立 ⇒ 选择性 FP16 不可行」这条推论作废**：即使误差主要由 FC 产生，
> 也可以在**不增加权重体积**的前提下把 FC 的激活放到 FP16。
> 原文保留见下，依据见 `docs/QNN_CONVERSION_GUIDE.md` §二十七。
>
> ⚠️ 仍未验证：量化器实际会不会产出该配置、PD 估算是否仍在红线内、HTP 上 FP16 的速度代价。



## 0. 为什么这个问题是决定性的

已确立（`scripts/repr_vs_compute.py`）：**问题在 HTP 的定点计算，不在量化表示**
（主体通道：表示极限 2.90% / CPU 4.56% / **HTP 73.62%**）。
⇒ 唯一剩下的、不换硬件的方向是**把出问题的算子放到浮点**。

而体积可行性已算清（三段合计 8-bit 权重 6154 M 参数）：

| 算子类型 | 参数量 | 占比 | 浮点化体积增量 |
|---|---|---|---|
| **FullyConnected** | 6153.9 M | **99.99%** | **+6154 MB** ❌ 撞红线 |
| RmsNorm | 0.5 M | 0.01% | +0.5 MB ✅ |
| MatMul | **0**（两输入皆激活） | 0 | **0** ✅ |
| 其余 | ~0 | ~0 | ~0 ✅ |

⇒ **把 FullyConnected 以外全部浮点化，体积只增 ~1 MB、速度基本不变。**
⇒ **但浮点化 FullyConnected 则必然出局。**

**所以本实验只回答一件事：误差主要由 FullyConnected 产生，还是由其余算子产生？**

## 1. 假设

**H-rest**：误差主要由 FullyConnected **以外**的算子产生
（RmsNorm / Eltwise / MatMul / Softmax 等）。
⇒ 若成立，存在近乎零成本的解。

零假设 H-fc：误差主要由 FullyConnected 产生 ⇒ 选择性 FP16 此路不通。

## 2. 实现路径

沿**真实依赖链**（`trace_dag.py`，非 Id 顺序）在 **part1b 的 Id 0~111** 密集取点，
覆盖前两个完整 transformer block。该区间主体余弦 0.9996 → 0.9574，**误差仍可比**
（约束 7：在余弦 < 0.3 的已毁区域做定位无意义）。

21 个探针（约 1754 MB），排除两个 2 GB 的注意力矩阵：

```
Id=0   RmsNorm  mul_304        Id=4   FC  linear_95_fc     Id=6   FC  linear_96_fc
Id=12  FC  linear_97_fc        Id=14  RmsNorm mul_308      Id=24  RmsNorm mul_311
Id=29  FC  linear_99_fc        Id=31  FC  linear_100_fc    Id=33  FC  linear_101_fc
Id=35  RmsNorm mul_314         Id=36  RmsNorm mul_316      Id=50  Concat stack_28
Id=79  MatMul  scaled_dot_product_attention_12
Id=82  FC  linear_102_fc       Id=84  RmsNorm mul_326      Id=87  RmsNorm mul_329
Id=91  FC  linear_103_fc       Id=93  FC  linear_104_fc    Id=99  FC  linear_105_fc
Id=101 RmsNorm mul_333         Id=103 Eltwise add_153
```

设备侧 `qnn-net-run --dlc_path --set_output_tensors`；宿主侧 `snpe_runner.py` 取同名张量。

## 3. 判据（事后不得修改）

**度量按约束 7**：主体相对 L2（`|a| ≤ p99`）+ 主体余弦 + 前 1% 能量占比。
余弦 < 0.3 的张量退出定量比较。

**判据 0（有效性）**：两侧元素数一致；`unified` 与既有运行逐位相同。

**判据 1（主判据，按算子类型归因）**：
对链上每个算子 X，计算**主体余弦的下降量** `Δcos(X) = max(cos(输入)) − cos(输出)`
（输入取该算子在链上的非 STATIC 输入；Reshape 视为透明，取其上游）。
按算子类型聚合 `Δcos` 的**中位数**与**总和**：

- 若 `FullyConnected 的 Δcos 总和 ≥ 2 ×（其余类型合计）` ⇒ **H-fc 成立**，
  选择性 FP16 **不可行**（因为 FC 占 99.99% 权重，浮点化必撞体积红线）。
- 若 `其余类型合计 ≥ 2 × FullyConnected` ⇒ **H-rest 成立**，
  选择性 FP16 **可行且近乎零成本**，进入方案设计。
- 介于两者之间 ⇒ 两类都有实质贡献，报告比例；
  此时需评估"只浮点化非 FC + 部分 FC"的折中，**不得直接宣称可行**。

**判据 2（操作化自检，按约束 8）**：
"沿依赖链比较输入/输出误差"这一操作化已在第二轮验证（`trace_dag.py` 读出真实 DAG）。
执行前须确认所选 21 个张量确实构成连通的依赖链，且 Reshape 的透明处理正确。

**判据 3（防误判）**：若某算子的**所有**输入都不在探针集内，
该算子不参与归因统计（避免第二轮那种"漏统计输入导致误报放大点"的错误）。

## 4. 成功 / 失败判断

- **成功**：给出按算子类型的 Δcos 归因，并据判据 1 给出明确分支。
- **失败·执行方法错误**：设备内存不足 / 张量名不存在 / 元素数不符 ⇒ 修方法重跑，判据不变。
- **失败·方向错误**：链上所有 Δcos 都 ≈ 0，而端到端误差却很大
  ⇒ 说明误差不由单个算子产生，或探针太稀疏，需加密。

## 5. 预算与放弃条件

- 一轮设备（约 40 分钟）+ 一轮宿主导出。
- 若判据 1 判定 H-fc 成立 ⇒ **选择性 FP16 出局**，
  届时不换硬件的路径全部用尽，应转向"向高通提 case + 交付主线 B"。
- **执行后必须回台账更新，无论结论如何。**
