# EXP_PLAN_SYM_WEIGHTS —— 权重改对称量化，能否消掉 HTP 的 19.24% 偏差？

定稿时间：2026-08-15（执行前定稿，判据不得事后修改）
前置：`EXP_PLAN_VS_FP32`（HTP 距 FP32 19.24%，CPU 参考仅 1.18%，差 16.4 倍）
主线归属：A → C 的机制层 → **第一个可修性验证**

## 0. 假设与它的证据基础

**H-sym**：A16W8 配置下**权重非对称量化**在 HTP（v79）上导致系统性精度损失，
表现为「增益缩小 14.6% ⊕ 噪声 12.5% = 19.24%」，而 SNPE CPU 参考不受影响。

**已有证据（均已核实，见 HANDOVER 15.13）**：
- 官方文档**通用最佳实践**：「It is recommended to always use symmetrical quantization of
  weights ... to obtain best accuracy on HTP based targets. Activation data is recommended
  to be asymmetric.」（适用于本项目）
- 官方旁证：`SHORT_DEPTH_CONV_ON_HMX_OFF` 的说明承认「weights are not symmetric」的图
  需要额外 flag「to guarantee accurate results」（该条针对 short-depth Conv，
  对 MatMul 是否适用**未验证**）
- ①实测：本项目 242 个权重张量中 **241 个（99.6%）非对称**（offset ≠ -128）
- ①实测：HTP 比正确的 16-bit 实现多丢 **4.02 bit**

> ⚠️ 以上**全部只说明"违反了建议"**，不能证明"它就是那 19.24% 的原因"。
> 本实验就是补这个因果证据。**在本实验出结果前，H-sym 只是假设。**

## 1. 实现路径（单变量）

复现原始量化命令，**只改 `--param_quantizer_schema` 一项**：
`asymmetric` → `symmetric`。其余全部照抄（从 `transformer_part1b_dlcinfo.txt` 的
Quantizer command 字段逐项核对过）：

```
act_bitwidth=16            act_quantizer=tf              act_quantizer_calibration=min-max
act_quantizer_schema=asymmetric   <-- 激活保持非对称（官方建议如此，不动）
weights_bitwidth=8         bias_bitwidth=32              use_per_channel_quantization=True
use_dynamic_16_bit_weights=True   param_quantizer_calibration=min-max
backend=HTP                float_fallback=False          restrict_quantization_steps=[]
input_list=<原校准列表，逐字节相同>
```

输入：`transformer_part1b_fp32.dlc`（5.54 GB）+ 原校准 `input_list_raw.txt`（**同一份**）。

流程：量化 → `qnn-context-binary-generator` 生成 sm8750 `.bin` → 推设备 →
用与实验 A **逐字节相同**的输入跑 HTP → 与 FP32 真值比。

## 2. 验收标准（判据，事后不得修改）

基准（均为已实测值，part1b 的 `unified`，同一输入）：
`E_cpu = 1.1757%`、`E_htp(非对称) = 19.2406%`、HTP 增益斜率 `k = 0.8540`。

记新测得的对称权重版为 `E_htp_sym`、`k_sym`。

**判据 0（有效性）**：新 `.dlc` 的权重张量必须**确实变成对称**
（用 `scripts/check_weight_symmetry.py` 复核，offset 应为 -128）。
若仍是非对称 ⇒ 命令没生效，本次无效，不得解读。
输出 `unified` 必须是 15,851,520 个 float32。

**判据 1（主判据）**：
- **`E_htp_sym < 3%`** ⇒ **H-sym 成立**，权重非对称是主因，**找到可修的根因**。
  下一步：三段全部重新量化 + 在环出图复核。
- **`E_htp_sym > 15%`**（即基本没降）⇒ **H-sym 被否定**，
  权重对称性不是主因，转查其它 HTP 配置项。
- 3% ~ 15% ⇒ 部分改善，按 `(19.24 - E_htp_sym)/19.24` 报告消掉的份额，
  **不得**宣称"找到根因"，需继续找剩余部分的来源。

**判据 2（增益项）**：单独看斜率 `k_sym`。
若 `|1-k_sym| < 0.03`（即增益误差从 14.6% 降到 3% 以内）
⇒ 说明**系统性增益误差**确实由权重非对称造成，即使总误差没完全消掉，
这也是一条明确的机制结论。

**判据 3（对照，防止"改坏了别处"）**：新 `.dlc` 也要在 **SNPE CPU 参考**上跑一次。
`E_cpu_sym` 必须仍 ≤ 3%（原为 1.1757%）。
若 CPU 侧反而变差很多 ⇒ 说明对称量化本身损失了精度，
此时即使 HTP 侧改善也**不能**直接采纳，需权衡。

## 3. 成功 / 失败判断

- **成功**：拿到 `E_htp_sym`、`k_sym`、`E_cpu_sym`，按判据 1 给出明确分支。
- **失败·执行方法错误**：量化 OOM / 参数名不对 / 建图失败。修方法重跑，判据不变。
- **失败·方向错误**：新旧 `.dlc` 的 md5 相同 ⇒ 参数没生效，必须查清。

## 4. 预算与放弃条件

- part1b 量化预估 **45~90 分钟**（原始日志：23:44 → 00:27，约 43 分钟）；
  context binary 生成预估 10~25 分钟；设备重测 12 秒。
- 只做 part1b。**不要**在拿到 part1b 结论前就去重量化 part1a/part2（那是 3 段 × 数小时）。
- 若判据 1 落在"被否定"分支 ⇒ 停止本方向，回文档查其它 HTP 配置
  （按约束 5：先查文档，不要再猜）。

---

# 执行结果（2026-08-15）：❌ **方案在 HTP 上根本无法构建，H-sym 无法按此路径验证**

## 量化本身成功

`--param_quantizer_schema symmetric` 生效，产出 `transformer_part1b_symw_quantized.dlc`（1.39 GB）。

**判据 0 的操作化写错了，特此说明**：我写的判据是「对称 ⇔ offset == -2^(bw-1) == -128」，
那是**无符号**表示下的写法。量化器实际把权重换成了**有符号 int8**：

| | dtype | offset | range |
|---|---|---|---|
| 原始 | `uFxp_8`（无符号） | -2 / -12 / -121 / -23 … | `[-0.0138, 1.7469]` 等，不对称 |
| 新 | `sFxp_8`（**有符号**） | **全部 0** | `[-1.763780, 1.750000]` |

`-1.763780 / 1.750000 = 1.007874 = 128/127`，正是有符号 int8 `[-128s, 127s]` 的标准对称形式。
⇒ 判据的**指标定错了**，但它要问的问题（权重是否关于零对称）答案明确是**是**。
（非事后放宽判据——原判据在有符号表示下不适用。）

## 🔴 但 context binary 生成失败

```
<E> None of the combinations match the provided case
<E> QnnBackend_validateOpConfig failed 3110
<E> Failed to validate op rms_norm_node_ with error 0xc26
Validate OpConfig failed: QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
Graph composer failed, no graphs were created
```

## 官方 OpDef 给出的硬约束（`docs/QAIRT-Docs/QNN/OpDef/HtpOpDefSupplement.html`，RmsNorm 段）

`in[0]` = UFxp16（本项目的激活类型）时，`in[1]`（权重）**只允许三种**：

| in[0] 激活 | in[1] 权重 | in[2] 偏置 | out[0] |
|---|---|---|---|
| UFxp16 | **UFxp16** | UFxp16 / SFxp16 / UFxp8 / SFxp32 | UFxp16 |
| UFxp16 | **SFxp16** | UFxp16 / SFxp16 / UFxp8 / SFxp32 | UFxp16 |
| UFxp16 | **UFxp8** | UFxp16 / SFxp16 / UFxp8 / SFxp32 | UFxp16 |

- 本项目**原始**配置 `UFxp16 / UFxp8 / SFxp32` —— **在表中，合法**。
- **对称量化后** `UFxp16 / SFxp8 / …` —— **SFxp8 根本不在允许列表里**。

⇒ ①事实：**HTP 的 RmsNorm 不支持有符号 8-bit 权重。**
⇒ ②严格推论：官方"权重用对称量化"的**通用建议**，
   **不能通过 `--param_quantizer_schema symmetric` 在本模型上整体套用**——
   它会把 8-bit 权重变成 sFxp_8，而本模型 44 个 RmsNorm 全部因此无法通过算子校验。

## H-sym 的状态：**未被验证，也未被否定**

本实验**没有**产出 `E_htp_sym`，因此 H-sym 既没成立也没被推翻。
不得把"构建失败"解读为"对称权重无效"。**台账里 #7 保持【未查】，并注明受阻原因。**

## 这次失败反而指出了一条有依据的路

支持表里 `in[1] = **SFxp16**`（有符号 16-bit 权重，即**对称的 16-bit 权重**）
与 UFxp16 激活的组合**是被支持的**——这正是官方所说的 **A16W16**，
且文档明确写「A16W16 models are expected to achieve better accuracy than A16W8 models
with Post-Training quantization」。

⇒ 它同时满足：① 权重对称（符合通用建议）② 在 RmsNorm 的支持表内 ③ 官方声明精度更好。
⇒ 台账 #11 从"未查"升级为**首选候选**，但仍须先完成定位（#13）再决定是否投入。

**代价提醒**：A16W16 需要 `--restrict_quantization_steps "-0x8000 0x7F7F"`
（Hexagon 硬件限制，INT16 权重范围是 0x8000~0x7F7F 而非满 16-bit），
且官方注明"This feature is enabled only on selected SoCs"——**v79 是否支持尚未确认**。

## 5. 已知的备选方向（本实验失败时按序尝试，均需先查文档）

1. `--act_quantizer_calibration percentile`（收紧量程，见 15.12）
2. A16W16（官方称 PTQ 下精度优于 A16W8）——但需确认 v79 是否支持、
   且需 `--restrict_quantization_steps "-0x8000 0x7F7F"`
3. `--use_per_channel_quantization` 的开关对照
4. HTP 图配置项（`O` 级别、`SHORT_DEPTH_CONV_ON_HMX_OFF` 等）
