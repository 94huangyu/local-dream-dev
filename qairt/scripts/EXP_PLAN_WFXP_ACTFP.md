# EXP_PLAN_WFXP_ACTFP —— 选择性 FP16（#48）的最小确认：单算子探针

定稿时间：**2026-08-22**（执行前定稿，**判据不得事后修改**）
线索：**#48 / #28**（体积否决已作废，见指南 §二十七）

---

## 一、为什么做这个而不是直接上全模型

全模型走一遍是 **3h 量化 + 3h 建图 + 上机**。而其中**风险最高、最便宜**的一问是：

> **QAIRT 2.48 的工具链，实际上会不会产出 `FP16 激活 + SFxp8 权重` 这个配置？**

这一问**纯宿主、单算子、分钟级**。不过就直接关闭，设备一次都不用碰。

### 依据（文档实查，非推测）

- HTP OpDef supplement：`FullyConnected` 有 `FP16 : FLOAT_16 | SFIXED_POINT_8 | FLOAT_16 | FLOAT_16`
  ⇒ **后端认这个组合，且权重字节数不变**
- `tools.html`：`--keep_weights_quantized` … *"Required to enable wFxp_actFP configurations"*
  ⚠️ 紧跟 *"These modes are not supported by all runtimes. Please check … Backend OpDef supplement"*
- `quantization.html`：`--enable_float_fallback` 在 qairt-quantizer **已是 no-op**；
  浮点回退发生在 **converter**，机制是「**缺 encoding 的张量落成浮点**」

⇒ 配方：**overrides 里只给权重 encoding，故意不给激活 encoding**，
再配 `--keep_weights_quantized`。**该配方本身未经实测，正是本实验要验的。**

---

## 二、装置（🔴 照抄订正后的构建命令，不自己拼）

复用 `scripts/p0b_build.py` 的单算子 FC 台（`node_linear_99`，M/K/N = 4128/3840/3840），
输入取同一份 `node_linear_99_pre_reshape.raw`（与 baseline/perrow 两臂逐字节相同）。

🔴 **`p0b_build.py` 里的 context 构建命令是坏的**——它只传 `--htp_socs sm8750`，
**没有 `--config_file`**，正是 #95 那个把整套 P0-B 结论作废的装置错误。
本实验一律照抄 `scripts/o_dlbc_probe.py` 里订正后的写法：

```
detail = {"graphs":[{"graph_names":["fc99_fp32"]}],
          "devices":[{"soc_model":69,"dsp_arch":"v79"}]}
ext    = {"backend_extensions":{"shared_library_path":"...QnnHtpNetRunExtensions.dll",
                                "config_file_path": detail}}
qnn-context-binary-generator ... --htp_socs sm8750 --config_file <ext>
```

---

## 三、阶段划分（阶段 1 不过就不进阶段 2）

### 阶段 1 —— 纯宿主，不占设备

| 门 | 内容 | 判定 |
|---|---|---|
| **V1 产出门（主判据①）** | 从量化 DLC 读回：`x` 与 `linear_99_fc` 的 datatype 必须是 **FLOAT_16**；`val_1800` 必须是 **SFIXED_POINT_8** | 任一不符 ⇒ **工具链不产出该配置** |
| **V2 生效性门** | 按 §7.3 纪律：**两个不同取值 ⇒ 两个不同产物**。带 / 不带 `--keep_weights_quantized` 两次产物 md5 必须**不同** | 相同 ⇒ 选项被静默忽略，V1 的"通过"不可信 |
| **V0 装置门** | context 用 `check_ctx_identity.py --expect-arch 79 --expect-vtcm 8` 验；并 `--diff` 与 perrow 参照臂比对画像 | 不过 ⇒ 装置不等同，作废 |

### 阶段 2 —— 设备（仅在阶段 1 三门全过时进行）

跑该单算子 context，取输出。

**真值** = 用**原始 fp32 权重**做的精确 float64 matmul（三臂共用同一真值）。
**度量**（约束 7）：全量相对 L2 + 主体相对 L2（`|a| ≤ p99`）+ 主体余弦 + **前 1% 能量占比**。

---

## 四、🔒 判据（事前锁定）

设 `E_wfxp` / `E_perrow` = 各臂相对精确 FP32 真值的**主体**相对 L2。
`E_perrow` 取本实验**同装置重跑**的值，**不引用历史数字**（#95 教训：历史值可能来自坏装置）。

| 结果 | 判定 | 后续 |
|---|---|---|
| `E_wfxp < 0.5 × E_perrow` | 🟢 **显著更好** | 值得进全模型验证（3h+3h），并**先测 PD 估算是否仍在 3.3 GB 红线内** |
| `0.5 × E_perrow ≤ E_wfxp < E_perrow` | 🟡 **有改善但不显著** | 登记，暂不投入全模型 |
| `E_wfxp ≥ E_perrow` | ❌ **不更好** | 关闭 #48 的这条形态，写明理由 |

**并行的硬约束（速度）**：报告两臂 `qnn-net-run` 单次耗时。
若 `t_wfxp > 3 × t_perrow` ⇒ 🔴 **标注「产品级不可行」**，即使精度更好也不得直接进全模型
（依据 #16：A16W16 因生图时间翻倍出局）。

---

## 五、失败归因

- **V1 不过** ⇒ 区分两种：① 配方错（overrides 写法/开关组合）② 工具链不支持。
  只允许换**一次**配方，仍不过 ⇒ 判为 ②，关闭并登记。

> 🔴 **2026-08-22 执行中的偏离声明（明写，不静默扩权）**：
> 定稿时把"允许的改动"限定为「`--float_bitwidth` 显式取值 / 是否保留激活 encoding」两项。
> 第一轮实测 V1 FAIL（激活仍 `uFxp_16`），根因是 **quantizer 拿 `--input_list` 把所有激活都量化了**，
> 与上述两项都无关。实际需要的改动是**第三项**：把 `--input_list` 换成 `--enable_float_fallback`。
> 依据是本方案 §一 已引用的原文：*"`--enable_float_fallback` and `--input_list` are **mutually exclusive**
> options. **One of them is mandatory** argument for quantizer."* ——不给校准集正是"不要量化激活"的官方表达。
> **写方案时未想透该机制，故范围列错。** 仍只用掉**一次**换配方预算；判据 §四与止损 §七**一字未改**。
- **V2 不过** ⇒ 选项落到空处（同 #93 的 `graph_names` 类错误），先修配置再谈。
- **V0 不过** ⇒ 装置问题，修一次。

---

## 六、🔴 划界（事前写死，防越界引用）

- 本实验是**单算子**、且是 **part1b 的 `linear_99`**。
  §3.2 实测 **part2 才是支配项**；**精度结论不得外推到 part2，也不得外推到端到端**。
- #30 已划界：**探针输入是良态的**，单算子结论不得外推到真实模型。
- 本实验**不回答**「全模型 PD 估算是否超红线」——那只能靠真实建图。
- ⚠️ 该配置要求权重 **SFxp8（有符号对称 per-axis）**。**不得引用 #7「对称更差」**——
  #7 是在**全定点**设定下测的；本设定激活是 FP16，
  按约束 4·补 规则 2，机制迁移必须先验证迁移成立，不得直接搬证据。
- ⚠️ #10/#27 的划界：图里出现**真 fp16 段落**时它们**需重测**（本实验正是第一次制造 fp16 段落）。

---

## 七、止损

| 触发 | 动作 |
|---|---|
| 阶段 1 宿主耗时 > **60 分钟** | 停，登记【未查·成本超预算】 |
| 阶段 2 单次设备占用 > **20 分钟** | 停 |
| 任一 V 门不过且已修过一次 | 关闭并登记，**不再尝试** |
| 落 🟡 或 ❌ | **不做任何后续单算子实验**，回 P1 其余两条 |

**不得扩张**：不做多层、不做全模型、不调校准方法、不试 BF16、不改 M/K/N。
