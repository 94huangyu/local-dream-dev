# EXP_PLAN_PERROW —— FC 权重逐行量化（scale 细化 3.9 倍），能否救回 HTP？

定稿时间：2026-08-15 21:0x（**执行前定稿，判据不得事后修改**）
台账归属：**#47**，动机来自 **#52**（HTP 对权重量化步长的敏感度 = CPU 的 37.6 倍）
前置实验：`EXP_PLAN_SYM_FC_OVR`（#7 被否定；但它同时**产出了本实验最关键的对照组**）

## 0. 假设

**H-row**：HTP 的误差对**权重量化步长**高度敏感（#52）；把 FC 权重从 per-tensor 换成
per-row（每行一个 scale），量化步长中位细化 **3.4~3.9 倍**，设备侧主体余弦应显著回升。

**证据基础**：
- ①实测（#52）：scale 变**粗** 1.073 倍 ⇒ 设备 cos_bulk **0.6033 → 0.5507**（CPU 只掉 0.0014，放大 **37.6 倍**）
- ①实测（`scripts/perrow_feasibility.py`，59 个 FC 权重全测）：
  | 口径 | 主体表示误差中位 | 最大 | scale 中位倍数 |
  |---|---|---|---|
  | per-tensor（现状） | **3.7279%** | 10.7941% | 1.000 |
  | per-row 轴0 | **1.0244%** | 1.4300% | **×0.2583** |
  | per-row 轴1 | 1.0893% | 1.3395% | ×0.2920 |
  ⇒ 表示误差降到 **0.28 倍**，量化步长细化约 **3.9 倍**
- ①事实（官方帮助）：`--use_per_row_quantization` = *"enable rowwise quantization of
  **Matmul and FullyConnected** ops"* ⇒ 正好命中本模型（99.99% 权重在 FC）。
  对照 #44：`--use_per_channel_quantization` 限定 *"convolution-based op weights"*，
  本模型零 Conv ⇒ **一直是空转**。

## 0.1 【先决条件 · 已完成】官方 OpDef 的硬约束（约束 5）

`HtpOpDefSupplement.html` → FullyConnected → INT16 → in[1] 原文：

> UFIXED_POINT_8 weights must be 8bit, 4bit or 2bit and **must have rank 2**.
> Given a 2D weight of dimensions **[m n]**, `QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET`
> and `QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET` is supported **with only axis 'm'**
> and the values are expected to be **signed and symmetrically quantized**.
> The Tensor must be static when using `BW_AXIS_SCALE_OFFSET`.

Quantization Parameters Config 表同样写死：`Encoding: AXIS_SCALE_OFFSET / Axis: 0 /
**Symmetry: Symmetric**`（UFxp8、SFxp8、UFxp16 三种权重类型都是如此）。

> 🔴 **由此得到一条本实验设计上的关键事实**：
> **逐行量化在 HTP 上强制要求对称权重** ⇒ 它**必然捆绑** #7 刚被否定的那个改动。
> 因此 per-row 与 per-tensor-asymmetric 的差异**不是单变量**（同时变了「对称化」与「细化」）。
> **但 `EXP_PLAN_SYM_FC_OVR` 已经把「对称化 + per-tensor」单独测过了（cos 0.5507）**
> ⇒ 用它当对照组，就能把**细化**的效应干净地分离出来。
> 这是 #7 那次"失败"实验最有价值的产出。

## 1. 实现路径（比 #7 更简单：**不需要 overrides、不需要重新转换**）

在**基线的量化命令**上只加一个开关：

```
qairt-quantizer --input_dlc  transformer_part1b_fp32.dlc      <- 基线那份，未改动
                --output_dlc part1b_perrow_quantized.dlc
                --input_list <基线同一份 input_list_raw.txt>
                --weights_bitwidth 8 --act_bitwidth 16 --bias_bitwidth 32
                --use_per_channel_quantization                 <- 照抄基线（对本模型空转）
                --act_quantizer_calibration min-max --param_quantizer_calibration min-max
                --use_per_row_quantization                     <- 唯一新增
```

⇒ **连 `--float_bitwidth 32` 的坑都绕开了**（#46 只在用 overrides 时触发）。
⇒ 不加 `--enable_per_row_quantized_bias`（保持单变量；若建图因 bias 失败，再作为修方法处理）。

后续与 #7 完全相同：`qnn-context-binary-generator --htp_socs sm8750` → 设备 HTP 跑
（输入取 `testB/s0_transformer_part1b`，与实验 A 逐字节相同）→ `metrics_unified.py`。

## 2. 先决门（Gate）

**G1 —— 逐行量化真的生效**
量化后 DLC 里，59 个 FC 权重的 encoding 必须是 **per-axis（多个 scale）**，
而不是单一 scale。判据：至少有一个 FC 权重张量在 `snpe-dlc-info` 里给出 **> 1 个** encoding。
- 若仍是单一 scale ⇒ 该开关对本模型空转（与 #44 同样的结局），**立即停止**，
  按"执行方法错误"排查（是否需要 overrides / 是否要求 rank-2）。

**G2 —— 只动了权重**
- 44 个 RmsNorm gamma 与基线**逐字段相同**
- 6 个图输入 encoding 与基线**逐字段相同**
（沿用 `scripts/check_sym_fc.py` 的 G3-2 / G4；G3-1 需按 per-axis 改判读。）

**G3 —— 建得出 `.bin`**
`qnn-context-binary-generator` EXIT=0 且无 `validateOpConfig` 报错。
- 若报错 ⇒ 记录原文，按 OpDef 逐条比对，**不得**改判据。

**G4 —— 设备侧有效性自检**
`.bin` 宿主/设备 md5 一致；输出 63,406,080 字节；同输入两次运行 md5 相同。

## 3. 验收标准（判据，事后不得修改）

真值 = `vs_fp32/unified_fp32.raw`；口径 = 主体（`|a| ≤ p99`）；工具 = `metrics_unified.py`。
已实测基线（全部来自同一脚本同一口径）：

| 对照组 | cos_bulk | 用途 |
|---|---|---|
| **A. 设备 HTP · 基线（per-tensor 非对称）** | **0.6033** | **主判据的比较对象**（"到底有没有用"） |
| **B. 设备 HTP · per-tensor 对称（#7）** | **0.5507** | **次判据的比较对象**（分离"细化"的净效应） |
| C. CPU 参考 · 基线 | 0.9975 | 可达上界 |
| 无效变更本底 | ±0.001 | O=3 实测标定 |

### 判据 1（主判据，实用性）：`Δ_A = cos_row − 0.6033`

| 结果 | 判定 |
|---|---|
| `Δ_A ≥ +0.20`（弥合 ≥ 50% 的 0.3942 缺口） | **H-row 成立且是主因** ⇒ 三段全量重量化 + 在环出图 |
| `+0.05 ≤ Δ_A < +0.20` | **部分成立**，报告弥合份额 `Δ_A/0.3942`，**不得**称"找到根因" |
| `\|Δ_A\| < 0.005` | **无效**（本底 5 倍以内）⇒ H-row 否定 |
| `Δ_A ≤ −0.005` | **更差** ⇒ H-row 否定，且说明细化也救不回来 |

### 判据 2（机制，分离"细化"的净效应）：`Δ_B = cos_row − 0.5507`

per-row 与 B 组**同为对称权重**，唯一差别是 scale 细化约 3.9 倍。
- `Δ_B ≥ +0.05` ⇒ **#52 的③假设成立**：HTP 误差确实随权重量化步长单调变化，
  这是一条可外推的机制结论（即使判据 1 只落在"部分成立"）。
- `\|Δ_B\| < 0.005` ⇒ **#52 的③假设被否定**：步长不是 HTP 误差的驱动量，
  #7 的恶化另有原因（例如 HTP 对 `sFxp8` 路径本身处理更差）。**这条比判据 1 更有信息量。**

### 判据 3（对照，防止改坏）
新 DLC 在 SNPE CPU 参考上跑：`cos_bulk_cpu` 必须 ≥ 0.99（基线 0.9975，#7 版 0.9961）。
按表示误差测算（3.73% → 1.02%），CPU 侧**应当变好**；若反而变差，说明理解有误，需查清。

## 4. 成功 / 失败判断

- **成功**：拿到 `cos_row` 并按判据 1 / 2 给出明确分支。
- **失败 · 执行方法错误**：G1 空转 / 建图失败 / OOM ⇒ 修方法重跑，**判据不变**。
- **失败 · 方向错误**：门全过但 `Δ_A` 与 `Δ_B` 均落在无效带 ⇒ H-row 被否定，
  且连带否定 #52 的③假设，**主线必须转向"非步长"类解释**（#30 单 MatMul 探针、#32 SDK 版本）。

## 5. 预算与放弃条件

| 步骤 | 预算 |
|---|---|
| 量化（无需重新转换） | 45~60 min（#7 实测 54 min） |
| context binary | ~12 min（#7 实测） |
| 设备实测 + CPU 对照 | ~5 min + ~4 min |

- **只做 part1b。** 拿到结论前不得动 part1a/part2。
- G1 若空转，**立刻停**，不要继续往建图烧 12 分钟。

---

## 6. 追加实验 PERROW-INLOOP：**只换 part1b 的 `.bin`，在环出图**（判据在执行前定稿）

**为什么加这一步**：① 顶层问题问的是"让设备生成正确的图"，张量指标只是代理量；
② 顺带关闭台账 **#31**（"只修 part1b"从未测过——此前"单段修复无用"只测过 part2，属过度声称）。

**唯一变量**：设备 `/data/local/tmp/htpcmp/part1b.bin` 换成 per-row 版
（原件 md5 `83f16066e57a5ed6baa040b21b65b9e9`，实验后必须还原）。
part1a / part2 的 `.bin`、prompt、seed 42、分词、FP32 文本编码、调度器、VAE 全部不变，
脚本沿用 `scripts/htp_inloop_pipeline.py`（与 `EXP_PLAN_HTP_INLOOP` 逐字节同一条流水线）。

**已实测的对照点（同一流水线、同一 prompt/seed）**：

| 配置 | 平均\|像素差\| | PSNR | 图像（本人亲自打开看过） |
|---|---|---|---|
| Test B：CPU 参考跑量化 transformer | 9.32 | 23.89 dB | 清晰的橘猫 |
| HTP 在环（三段全基线 `.bin`） | 46.73 | 12.06 dB | 橙色色块，无猫无桌 |

**判据（事后不得修改）**，记新图与 FP32 基准的平均 |像素差| 为 `d`：

| 结果 | 判定 |
|---|---|
| `d ≤ 15` **且**图中能辨认出猫 | **只修 part1b 就足以救回成图** ⇒ 立即推进三段全量 per-row |
| `15 < d < 40` **或**出现可辨识结构（哪怕不完整） | **部分改善** ⇒ #31 被证伪（单段修复**有**效果），继续修 part1a/part2 |
| `d ≥ 40` **且**仍是无定形色块 | 单段修复不足以出图；张量层面的改善**未能**传导到成图 |

⚠️ **约束 1**：必须**自己打开 PNG 看**，不得用形容词转述；
"有没有猫"这一项由我直接看图判定，不由 `d` 单独决定。

---

# 执行结果（2026-08-15，判据未做任何事后修改）

## 7. 门禁

| 门 | 结果 |
|---|---|
| G1 逐行量化生效 | ✅ **66 个 axis-quant 张量**（59 FC 权重 + 7 个 adaLN bias），**全部 `axis: 0`**，num_elements 分布 `{3840:36, 10240:16, 15360:14}`；59 个 FC 权重全为 `sFxp_8` 且 **offset 全 0**（符合 OpDef 强制的 signed symmetric）。<br>转储原文：`axis-quant: axis: 0, num_elements: 3840 (above encoding is only for the first (channel_0) of 3840 channels)`。<br>实例 `val_1800` 第 0 行 scale **0.015012 → 0.005844**（细 2.57 倍）。DLC 体积 1385.5 → **1400.0 MB**（+14.5 MB = 多出的 scale 数组） |
| G2 只动了权重 | ✅ 44/44 RmsNorm gamma 与基线逐字段相同；6/6 图输入 encoding 与基线逐字段相同 |
| G3 建得出 `.bin` | ✅ EXIT=0，1472.9 MB，约 14 分钟，无 `validateOpConfig` 报错 |
| G4 设备有效性 | ✅ 宿主/设备 md5 一致；输出 63,406,080 字节；**同输入两次运行 md5 相同**（`bc7249a9…`） |
| 观测项 | 下游激活 scale 漂移 >1% 的仅 **11 个**（最大 3.22%），比 #7 那版（35 个 / 3.70%）更小 |

> ⚠️ 工具口径提示：per-axis 的转储打的是 `encoding for channel_0:` 而**不是** `encoding :`，
> `check_sym_fc.py` 的旧正则匹配不到，会把 FC 权重显示成"0 个"并误报 ❌。
> 已给该脚本加上 per-axis 识别与显式提示。**这是工具口径不适用，不是门禁失败，判据未改。**

## 8. 主结果

| 配置 | E_all | E_bulk | **cos_bulk** | **slope_bulk** | 主体 std |
|---|---|---|---|---|---|
| CPU 参考 · 基线 | 1.1757% | 7.0863% | 0.9975 | 1.0057 | 0.6286 |
| 设备 HTP · 基线（per-tensor 非对称） | 19.2406% | 114.798% | **0.6033** | 0.8622 | 0.8910 |
| 设备 HTP · per-tensor 对称（#7） | 21.0129% | 125.825% | **0.5507** | 0.8217 | 0.9303 |
| **设备 HTP · per-row（本实验）** | **7.8665%** | **78.771%** | **0.7986** | **1.0435** | 0.8146 |

### 判据 1（主判据）：`Δ_A = 0.7986 − 0.6033 = +0.1953`

**落在 `+0.05 ≤ Δ_A < +0.20` ⇒ 「部分成立」。**
弥合份额 `0.1953 / 0.3942 = 49.5%`——**差 0.0047 没够到我事前写的 50% 线**，
按约束 4「判据事前定稿、事后不得修改」，**不得**宣称"找到根因"。

同时记录（不用于判定）：E_all **19.24% → 7.87%**（降 2.4 倍）；
E_bulk **首次跌破 100%**（114.80% → 78.77%）⇒ 该张量不再属于约束 7 所说的"已毁"区间。

### 判据 2（机制，事前即声明"比判据 1 更有信息量"）：`Δ_B = 0.7986 − 0.5507 = +0.2479`

**≥ +0.05 ⇒ #52 的③假设成立。**
per-row 与 #7 版**同为对称权重**（OpDef 强制），唯一差别是 scale 细化约 3.9 倍。
⇒ ②严格推论：**HTP 的误差随权重量化步长单调变化**——粗 1.073 倍 ⇒ 余弦 −0.0526；
细约 3.9 倍 ⇒ 余弦 +0.2479。这是可外推到新模型的机制结论。

### 判据 3（CPU 对照）：⛔ **无法执行，受阻，不计为通过**

```
error_code=202; Invalid fixed point parameter.
Dequantization of axis-quant tensor is not supported for FullyConnected; error_component=Dl System
```
SNPE CPU 参考**不支持 axis-quant 的 FullyConnected**（与 #36 的 `--enable_cpu_fxp`
拒绝 INT16 属同类工具限制）。
- 替代证据（**不等价，仅供参考**）：宿主侧权重表示误差 3.7279% → **1.0244%**，严格更好。
- ⇒ "per-row 会不会把模型本身改坏"这一问**在本 SDK 上无法用 CPU 参考回答**。

## 9. 顺带解释掉的旧现象：15.13.1 的「系统性增益缩小」

`slope_bulk` 三次实测完全同向：

| 配置 | 权重 scale 相对基线 | slope |
|---|---|---|
| #7 对称 per-tensor | ×1.073（**粗**） | **0.8217** |
| 基线 | ×1.000 | 0.8622 |
| #47 per-row | ≈×0.26（**细**） | **1.0435** |

⇒ ②推论：15.13.1 记录的「HTP 系统性把输出缩小 14.6%」**由权重量化步长驱动**，
   **不是** offset 项（#40）驱动的——这也与 #7 的失败一致。

## 10. 追加实验 PERROW-INLOOP 的结果（判据未事后修改）

产物：`scratch_runs/htp_inloop_perrow_0815.png`
对照：`scratch_runs/htp_inloop_transformer_BASELINE_0815.png`
（⚠️ `htp_inloop_pipeline.py` 会**覆盖** `htp_inloop_transformer.png`，
已在覆盖前把基线那张备份出来，否则对照组的图会永久丢失。）

| 配置 | 平均\|像素差\| | PSNR |
|---|---|---|
| Test B：CPU 参考跑量化 transformer | 9.32 | 23.89 dB |
| HTP 在环 · 三段全基线 | 46.73 | 12.06 dB |
| **HTP 在环 · 只把 part1b 换成 per-row** | **43.56** | **12.60 dB** |

### 【①实测·本人亲自打开两张 PNG 逐一查看】图像层面的变化

- **基线那张**：纯渐变。左上米白 → 下方橙色，**没有任何边界、纹理或物体**，
  除了颜色过渡以外什么都没有。（与我此前的记录一致。）
- **per-row 这张**：**出现了全局构图结构**——
  ① 下方约 1/3 处有一条**横贯画面的水平分界**（位置与"桌面边缘"相符）；
  ② 画面中央有一团**带纹理的主体块**，纵向约占画面中部，内部有深浅斑纹；
  ③ 右中部有一个**带硬边的深色矩形物体**（约 x 700~790、y 555~635）；
  ④ 全图有明显的块状/条带状伪影（量化/分块痕迹）。
- **但是：没有猫。** 中央主体块无法辨认为猫或任何具体物体，
  没有耳朵、眼睛、毛发或可辨识的轮廓。

### 判据判定

判据三档中，**第二档命中**（`15 < d < 40` **或** 出现可辨识结构（哪怕不完整））：
`d = 43.56` 不满足前半句，但**后半句成立**（这是个"或"）。
第三档要求 `d ≥ 40` **且**"仍是无定形色块"，后半句**不成立** ⇒ 第三档不命中。

⇒ **判定：部分改善。台账 #31 就此关闭——"只修一段"确实有效果，
但不足以救回成图。**

> ⚠️ **必须同时说清的两件事（约束 2）**：
> ① 像素指标几乎没动（46.73 → 43.56，仅 −3.2），仍稳稳落在"废图"区间；
> ② 但图像从**零结构**变成了**有全局构图**。
> **单看指标会得出"没变化"的错误结论**——这正是约束 1 存在的理由。
> 反过来也**不得**说"接近成功"：没有猫，就是没有猫。
