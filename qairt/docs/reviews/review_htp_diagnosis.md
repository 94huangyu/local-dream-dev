# 审查报告：Z-Image Turbo HTP 色块问题诊断

> 审查对象：[DIAGNOSIS_HTP_ZIMAGE.md](file:///Users/huangyu/Documents/Codex/local-dream-ZIT/DIAGNOSIS_HTP_ZIMAGE.md)
> 审查日期：2026-08-15

---

## 总体评价

这是一份**质量很高**的诊断报告。实验设计严谨（单变量对照）、度量选择周到（主体口径 vs 全量口径的区分是一个亮点）、排除过程有据可依。以下审查按三个维度展开：**判断方向是否正确**、**逻辑是否完备**、**还有哪些方向可以排查**。

---

## 一、判断方向审查：根因定位是否正确

### ✅ §3 根因实验：方向正确，设计严密

单变量对照实验（仅切换 transformer 后端 CPU→HTP，其余逐字节相同）是定位类问题的金标准。PSNR 从 23.89 dB 掉到 12.06 dB，且与真机 11.56 dB 几乎吻合——这确实把问题锁死在 HTP 执行路径上。

**无异议。**

### ✅ §4.2 "表示 vs 计算" 判别：方向正确，但有一个微妙假设需要注意

> [!IMPORTANT]
> **核心论证链回顾：** 量化表示极限 2.90% → CPU 实测 4.56% → HTP 实测 73.62%，因此"不是量化表示的问题，是 HTP 计算过程的问题"。

这个推理在逻辑上是成立的：如果同一份量化模型在 CPU 参考实现上能跑到接近表示极限（4.56% vs 2.90%），说明量化本身不是瓶颈。HTP 73.62% 是表示极限的 25 倍，显然是计算引入的。

**但有一个微妙前提值得明确化**：这里隐含假设了 CPU 参考实现和 HTP 使用的是**完全相同的量化参数**（scale/offset）。如果 `qnn-context-binary-generator` 在为 HTP 编译时对 encoding 做了某种变换（比如为适配 HTP 硬件而改变了 scale/zero-point 的精度、或者做了 per-tensor 到 per-channel 的合并），那么"表示极限"可能不同于你在 CPU 上算的那个。

> [!TIP]
> **建议验证**：dump HTP context binary 中实际使用的 scale/offset，与 CPU 参考使用的做逐个比对。如果一致，则 §4.2 论证完全封闭；如果不一致，需要用 HTP 实际的 encoding 重新算表示极限。

### ✅ §4.3 误差分解：方向正确

HTP 全局斜率 0.854（系统性增益偏低 14.6%）+ 随机噪声 12.534% 的分解是合理的。特别是 `cos∠(CPU误差, HTP误差) = -0.33` 说明 HTP 的误差模式和 CPU 不同方向，这排除了"HTP 只是把 CPU 误差放大"的假设——它产生了**结构上不同的误差**。

**但 0.854 的系统性增益偏差本身值得深挖**——在正确的定点实现中不应出现全局增益偏差。这指向两个可能：
1. **定点累加器溢出/截断**导致的系统性下偏
2. **反量化/requant 步骤的 scale 计算**在 HTP 上有精度损失

### ⚠️ §5 排除方向中的一个潜在问题

第 129 行排除"只修某一段"的实验：

> 把 part2 修到完美（用 CPU 的 part2 跑 HTP 的 unified），最终误差 47.07% → 52.87%，收益为负

这说明在 part1b 已经严重偏离真值的情况下，用正确的 part2 反而会让结果更差——因为**正确的 part2 会"忠实地"处理已经错误的输入**，而 HTP 的 part2 恰好对错误输入有某种"将错就错"的效果。

> [!NOTE]
> 这不是一个排除理由，而是一个**级联效应的特征**。它说明误差已经在 part1b 阶段严重到了非线性传播的程度。但"只修某一段不够"不等于"修某一段没用"——如果 part1b 被修正，part2 是否还需要修？这个实验没有回答。

---

## 二、逻辑完备性审查

### 2.1 关于 massive activation 的论证链有缺口

§4.5 指出 ch85 占 99.58% 能量，§4.2 按 massive activation 通道分离了误差。但关键推理缺了一环：

**massive activation 如何具体影响 HTP 定点计算？** 报告只是说"可能与问题相关"和"transformer 特有形态"，但没有给出机制假说。实际上，massive activation 可能通过以下机制造成 HTP 定点精度灾难：

1. **定点累加器动态范围不足**：当一个通道的值是其他通道的 $\sqrt{99.58\% / 0.42\%} \approx 15.4$ 倍时，per-channel 量化会给每个通道分配不同的 scale。但在 MatMul 中，多个通道的定点积需要**累加到同一个累加器**——如果 HTP 的累加器宽度不足（比如 32-bit 累加 16-bit×8-bit 乘积），massive activation 通道的大值乘积可能**挤压**其他通道的有效位数。

2. **RmsNorm 的非线性放大**：RmsNorm 计算 $x / \sqrt{\text{mean}(x^2)}$，当 ch85 贡献了 99.58% 的 $\sum x^2$ 时，分母几乎完全由 ch85 决定。对主体通道来说，它们被一个"与自身几乎无关的分母"做了缩放——定点下的 reciprocal sqrt 如果精度不足，误差会被 massive activation 的这种不均匀结构放大。

> [!IMPORTANT]
> **建议**：对 part1b 中的 RmsNorm 算子做定向探针——比较 HTP 和 CPU 在 RmsNorm 前后的误差变化。如果 RmsNorm 是误差放大器，那么主体通道在 RmsNorm 后的误差应该比 RmsNorm 前显著增大。

### 2.2 §4.3 增益 0.854 的物理解释缺失

全局增益偏差 0.854 是一个非常重要的线索，但报告没有给出物理解释。0.854 ≈ 1 - 1/6.85，不是任何显然的位宽截断特征值。但考虑到：

- HTP 等效精度为 11.34 bit（丢 4.66 bit）
- 主体通道被系统性低估

一个可能的解释是：**HTP 在 W8A16 MatMul 中使用了比预期更窄的中间精度**。比如：
- 如果 HTP 内部把 A16 截断到 A12 后再与 W8 相乘（12+8=20 bit 乘积，32-bit 累加），会丢约 4 bit，与观测的 4.66 bit 损失吻合
- 或者累加器在 requant 之前做了**右移截断**而非**四舍五入**，导致系统性偏低

> [!TIP]
> **建议**：构造一个**全 1 输入**或**线性梯度输入**的测试 case，用极简模型（1 个 MatMul）在 HTP 上跑，分析输出的系统性偏差模式。这可以直接暴露 HTP 的内部位宽和舍入策略。

### 2.3 §6 横向参照的推理不够严密

> "SD1.5/SDXL 的 W8A16 在同设备上跑通 ⇒ HTP 的 W8A16 路径本身能正确跑扩散模型"

这个推论成立，但后半段的分析不够到位：

- SD1.5/SDXL 用的是 **UNet**（卷积为主），Z-Image 用的是 **DiT**（MatMul 为主）
- UNet 中**没有 massive activation**（卷积层通常不会出现单通道占 99.6% 能量的情况）
- 关键差异可能不在于"DiT vs UNet"这个架构差异本身，而在于**具体算子类型和数值分布的组合**

> [!WARNING]
> SD1.5/SDXL 使用的是 QNN SDK 2.28/2.39，而本项目用 2.48。**SDK 版本差异也是一个未排除的变量**。特别是如果 2.48 的 HTP compiler 对某些算子图模式做了新的优化（如算子融合、精度降级），可能引入了新的精度问题。§9.6 已提到但未验证——建议提升优先级。

### 2.4 "选择性 FP16 fallback"方向的逻辑审查

§8 的体积分析是清晰的：

- FullyConnected 占 99.99% 参数量，浮点化必然出局
- 其余（RmsNorm + LayerNorm + MatMul）仅 ~1 MB 增量

**但存在一个逻辑漏洞**：报告问的是"误差主要由 FullyConnected 产生，还是由其余算子产生？"——这是一个好问题，但即使误差由 RmsNorm/MatMul 产生（最佳情况），**单纯浮点化这些算子可能不够**：

1. FullyConnected 的**输出**仍然是定点的，如果 FullyConnected 本身引入了 0.854 的增益偏差，那么即使 RmsNorm 用 FP16 完美执行，它处理的输入已经偏了
2. 更精确的问题应该是：**误差的放大主要由哪些算子完成**？一个算子可以自身精度没问题，但因为上游输入的微小误差在其非线性变换下被放大

> [!IMPORTANT]
> 建议把问题重新表述为：**在 CPU 输入 + HTP 单算子执行的条件下，哪些算子的输出误差最大？** 这才是定位"误差引入点"的正确实验——而非在已经累积了误差的流水线中观察。

---

## 三、尚可排查的方向

### 3.1 🔍 HTP Graph Compiler 的算子融合行为

> [!IMPORTANT]
> **高优先级**

QNN HTP compiler 会做大量的算子融合（fusion）——比如把 MatMul + RmsNorm 融合为一个超级算子，或者把 Elementwise Add 吸收到前序算子的 requant 中。融合后的超级算子使用的内部精度策略**可能与单独算子不同**。

**排查方式**：
- 用 `qnn-net-run --profiling_level detailed` 或 `--log_level verbose` 获取 HTP 实际执行的算子图（而非输入的 ONNX 图），对比看有哪些融合
- 检查 `qnn-context-binary-generator` 的日志中是否有 "fusing" / "merging" 相关输出
- 尝试 `--vtcm_mb 0` 或其他禁用某些优化的选项

### 3.2 🔍 HTP 的 W8A16 MatMul 内部精度路径

> [!IMPORTANT]
> **高优先级**

Hexagon v79 的 HVX 向量单元的乘法和累加精度是问题的核心。具体来说：

- HVX 的 `vmpy` 指令在处理 8×16 bit 乘法时，内部乘积宽度是多少？(应为 24 bit)
- 累加器宽度？(HVX 提供 32-bit 累加器，但在某些配置下可能用 16-bit)
- 在长累加链（FullyConnected 的 reduce 维度可能很大）中，是否存在**中间 requant**（在累加完成前就做一次缩放截断）？

**如果 HTP 在内部对 A16 做了隐式截断到更窄的位宽**（比如为了利用 8×8 的更快乘法路径），就能完美解释所有观测：
- 系统性增益偏低（截断导致系统性下偏）
- 丢 4.66 bit（从 16 bit 截到 ~11 bit）
- 逐位确定性（确定性截断）
- CPU 不受影响（CPU 用浮点算定点，没有硬件截断）

> [!TIP]
> **建议验证**：用 `--act_bitwidth 8` 看 HTP 是否反而更好。如果 HTP 内部本来就在把 A16 截到某个中间位宽，那么让模型直接量化到 A8 或许能让 scale 参数更"匹配" HTP 的实际硬件路径（虽然 §5 说 A8 表示误差 80%——那是全张量口径，massive activation 通道的 A8 可能还行）。

### 3.3 🔍 per-channel scale 的定点表示精度

W8 per-channel 量化意味着每个输出通道有一个 scale（和 offset）。HTP 上这些 scale 是如何存储的？

- 如果 scale 用 FP16 存储，那么极端的 massive activation 通道（值域远大于主体通道）可能导致 **FP16 scale 的精度不足以同时准确表示极端通道和主体通道的 scale**
- 特别是 FullyConnected 的权重 per-channel scale 如果被压缩到了不足的精度，会导致系统性增益偏差

**排查方式**：检查 HTP context binary 中 encoding 参数的格式和精度。

### 3.4 🔍 SDK 版本对比实验

> [!WARNING]
> §9.6 提到但未执行

SD1.5/SDXL 用 SDK 2.28，本项目用 2.48。版本跨越了 20 个版本，HTP compiler 的优化策略可能发生了重大变化。

**建议**：用 SDK 2.28（或至少 2.39）重新编译 Z-Image 的 context binary，看问题是否消失。这是**低成本高信息量**的实验。

### 3.5 🔍 `--keep_weights_quantized false` 或 `--float_fallback` 选项

QNN 2.48 可能提供了一些控制 HTP 精度的新选项。具体检查：

- `QNN_GRAPH_CONFIG_OPTION_CUSTOM` 中是否有关于 MatMul/FullyConnected 精度的选项
- `converter` 阶段的 `--float_fallback` 选项是否能让特定算子 fallback 到浮点
- HTP backend 的 `vtcm_mb` 配置是否影响精度（VTCM 是片上缓存，其大小可能影响分块策略从而影响累加精度）

### 3.6 🔍 单算子 HTP 误差定位实验（§9.1 的正确做法）

§9.1 已承认两轮定位实验失败。正确做法应该是：

1. **选择 part1b 前半部分**（Id 0~306，主体余弦 0.9996→0.9574）
2. **逐层而非逐算子**：以 transformer block 为单位，把每个 block 的入口/出口用 CPU 结果"钳住"（即只让一个 block 用 HTP 跑，其余用 CPU 的正确中间结果），观察单 block 引入的误差
3. 这样可以定位到"哪个 block 最先出现显著误差"
4. 再在该 block 内部做更细粒度的定位

### 3.7 🔍 DiT 的 Timestep Embedding 和 AdaLN

DiT 架构的一个特殊之处是使用 **Adaptive Layer Normalization (AdaLN)**——normalization 的 scale 和 shift 参数来自 timestep embedding，是动态的。如果 HTP 在处理这种"动态 scale/shift"时有精度问题（与静态 weight 不同的代码路径），可能是一个误差来源。

**排查方式**：检查 ONNX 图中 AdaLN 相关的算子（可能表现为 Mul + Add 跟在 LayerNorm/RmsNorm 后面），看这些算子在 HTP 上的误差是否异常大。

### 3.8 🔍 quantization encoding 的 offset 处理

W8A16 的 asymmetric 量化意味着权重有 zero-point offset。在 HTP 定点 MatMul 中：

$$y = \sum_i (w_i - zp) \cdot x_i = \sum_i w_i \cdot x_i - zp \cdot \sum_i x_i$$

如果 HTP 在处理 offset 项时有精度问题（比如 $zp \cdot \sum_i x_i$ 这个大值项的累加溢出），可能导致系统性偏差。**对 massive activation 通道来说，$\sum_i x_i$ 可能极大**，使得 offset 项的精度问题被放大。

> [!TIP]
> **建议验证**：尝试 `symmetric` 权重量化（zero-point = 0）。§5 提到这条路因为 RmsNorm 的 OpDef 不支持 SFxp8 而被阻断——但可以尝试只对 FullyConnected 用 symmetric、RmsNorm 保持 asymmetric。或者检查是否有其他 data type 可用。

### 3.9 🔍 VTCM 分块策略与部分和精度

HTP 在执行大矩阵乘法时需要在 VTCM（Vector Tightly-Coupled Memory, 通常 8MB）中做分块。分块策略影响：
- 每个分块的累加长度
- 部分和（partial sum）在写回主存前的精度
- 如果部分和在 requant 后再累加（而非直接在高精度累加器中累加），精度会显著下降

`--vtcm_mb` 参数可以控制 VTCM 使用量，间接影响分块策略。值得实验不同的 VTCM 配置。

---

## 四、对 §10 "外部评审问题" 的回应

### Q1: 除"选择性 FP16"外，是否还有绕开 HTP 定点精度不足的手段？

1. **Mixed-precision quantization**：对 massive activation 相关的 FullyConnected 层用 W8A8（而非 W8A16），如果 HTP 的 8×8 路径比 8×16 路径精度更高（因为不存在隐式截断），这反而可能更好
2. **权重 clipping**：在转换时对权重做温和的 clipping（比如 percentile 99.99%），减少极端权重值对定点累加的压力
3. **Activation smoothing**（不是 SmoothQuant 改表示，而是改模型权重）：在模型层面吸收 massive activation 的尺度差异到权重中，使得推理时激活值分布更均匀
4. **尝试 per-tensor 激活量化**（如果当前是 per-channel）：改变量化粒度可能影响 HTP 选择的计算路径

### Q2: DiT 在 v79 上的 W8A16 成功案例？

据我所知，截至 2026 年中，DiT 在 Hexagon v79 上的公开成功案例非常稀少。Qualcomm AI Hub 上的扩散模型示例主要是 SD1.5/SDXL（UNet 架构）。DiT 的部署案例多见于 v81+（SM8 Gen 4 及以后）。

### Q3: §4.2 的判别是否成立？

成立。在同一组 encoding 参数下，如果表示误差只有 2.90% 而执行误差 73.62%，计算过程是唯一的解释。但如前所述，需要确认 HTP 实际使用的 encoding 参数与你计算表示极限时使用的一致。

### Q4: massive activation 通过何种机制影响 HTP 定点计算？

见 §2.1 中的分析：最可能的机制是**定点累加器动态范围不足**和 **RmsNorm 的 reciprocal sqrt 在定点下的精度损失**。验证方式已在对应章节给出。

### Q5: QNN/HTP 侧的诊断手段？

- `qnn-net-run --profiling_level detailed` 可获取每个算子的执行统计
- `QNN_LOG_LEVEL_DEBUG` 可能输出内部精度选择的日志
- Qualcomm 内部有 **HTP Simulator**（非公开工具），可以逐指令观察累加器状态——建议通过 Qualcomm 技术支持渠道获取
- `snpe-diagview` / `qnn-profile-viewer` 可以分析 profiling 数据

---

## 五、总结与优先级建议

| 优先级 | 排查方向 | 预期信息量 | 成本 |
|---|---|---|---|
| 🔴 P0 | 单 block HTP 误差定位（§3.6） | 极高——定位到具体 block/算子 | 中 |
| 🔴 P0 | 确认 HTP 实际使用的 encoding 参数（§一·验证建议） | 高——封闭 §4.2 的前提 | 低 |
| 🟠 P1 | HTP graph compiler 融合行为分析（§3.1） | 高——可能发现隐式精度降级 | 低 |
| 🟠 P1 | SDK 2.28 vs 2.48 对比（§3.4） | 高——可直接定位到 SDK 回归 | 中 |
| 🟡 P2 | 简单模型探针（单 MatMul）测 HTP 内部位宽（§3.2/§2.2） | 高——直接暴露硬件行为 | 中 |
| 🟡 P2 | `high_precision_sigmoid` / `advanced_activation_fusion` 正确重试（§5.1） | 中 | 低 |
| 🟡 P2 | VTCM 分块配置实验（§3.9） | 中 | 低 |
| ⚪ P3 | per-channel scale 精度检查（§3.3） | 中 | 低 |
| ⚪ P3 | offset 项精度（symmetric 量化尝试）（§3.8） | 中 | 中 |
| ⚪ P3 | AdaLN 动态 scale/shift 定向检查（§3.7） | 中 | 中 |

> [!CAUTION]
> **最关键的未完成工作**是 §9.1（误差定位到具体算子）。当前已知 part1b 是主要问题段，但不知道是哪个（些）算子引入了误差。没有这个信息，"选择性 FP16 fallback"的方案就无法判断可行性。建议用 §3.6 的方法（逐 block 钳住实验）尽快完成定位。
