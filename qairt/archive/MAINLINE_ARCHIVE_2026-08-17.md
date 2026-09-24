# 主线追踪（每次开始新实验前必须读，每个分支点必须更新）

> 存在意义：防止"遇到问题解决问题"式的漂移。本项目已经发生过一次严重漂移，
> 在一条**前提从未验证**的支线上做了 7 层深挖（详见文末复盘）。

## 顶层问题（唯一的主线，任何时候都不能忘）

> **【2026-08-15 主线推进】原顶层问题已关闭，主线随之前移，不是变成支线。**

### 当前顶层问题：**如何解决 HTP 的数值问题，让设备生成正确的图？**

围绕它展开的机制分析与配置验证**都是主线工作**；其余（v79 支持成熟度等）是支线。
除非判断不完备——即除 HTP 外还有其他主因——否则 HTP 就是主线。

### 已关闭的原顶层问题（保留，供追溯）

**为什么设备生成的是橙色色块，而同 prompt 同流程的 FP32 生成的是完美的猫？**
**答案**：HTP 执行量化 transformer 的 `.bin`，与 CPU 参考执行同一份模型不等价，
这一件事单独就足以毁图（`EXP_PLAN_HTP_INLOOP` 单变量对照，见下）。

证据（本人亲自看图确认，非转述）：
- `scratch_runs/zimage_fp32_full_pipeline.png`：照片级橘猫坐在木桌上，完全符合 prompt
- `scratch_runs/sm8750_test2.png`：橙色无定形色块，无猫无桌，有横向条带伪影

## 假设树（标注每个节点的验证状态）

```
Q: 设备 vs FP32 的差异从何而来？
├─ A. 量化
│   ├─ A1. transformer 量化           【已否定 2026-08-14 Test B】全量化 transformer + 其余FP32
│   │                                  仍生成清晰的猫（PSNR 23.89dB vs FP32基准）
│   ├─ A2. text encoder 量化          【部分验证】caption 张量的量化表示无损（余弦≥0.99995），
│   │                                  但【量化后的 text encoder 计算】未验证（CPU后端不支持Gather）
│   └─ A3. VAE 量化                   【已否定 2026-08-14 实验A3】量化VAE解码真实latents，
│                                      PSNR 51.11dB、平均像素差0.32，几乎无损
└─ B. C++ 实现 ≠ Python 实现          【完全未排查】← 现在的首要嫌疑
    └─ 已知该实现出过 2 个 bug（cap_pad_mask 极性、Euler 符号），
       无证据排除还有第 3 个
```

**关键认识**：FP32 那张好图是 **Python 脚本**跑的，设备那张坏图是 **C++ 实现**跑的。
Test B 已经把"量化"这个因素单独摘出来测过了 —— **它不是根因**。剩下的差异集中在
A2 / A3 / **B**。

## 已标定的参考点（很有用，别忘了）

**噪声预测的 L2 相对误差达到 15.988% 时，成图仍然完好。**
（Test B step0 实测：量化 transformer vs FP32 transformer 的噪声预测差异）
⇒ 以后再看到"某中间张量误差 xx%"时，先想想这个标定点：扩散过程对这个量级的误差是鲁棒的。
⇒ 反过来说，能把图毁成橙色色块的原因，一定是比"量化噪声"**性质不同**的东西
   （结构性错误、条件信息丢失、算子语义错误之类），而不是精度不够。

## 当前进度

| 实验 | 目标 | 状态 |
|---|---|---|
| Test A | 检验 A2（text encoder 量化是否破坏 caption 条件） | **部分完成**：caption 张量的量化表示无损（余弦 ≥0.99995、rms 相对误差 0.16%）。但量化 text encoder 的**计算**未验证——`snpe-net-run` CPU 后端不支持 embedding 的 Gather 算子（error_code=1002）。这是模拟器限制，不代表 DLC 有问题。 |
| Test B | 检验 A1 → **唯一能分离 A 和 B 的实验** | **已完成，A1 被否定**。见下 |
| OPID_MAPPING | 把 part2 编译期报错的 OpID 映射到具体算子 | **已完成，身份确定；但该线索被自己证伪**。见下 |

### OPID_MAPPING 结论（2026-08-14）

方案 `scripts/EXP_PLAN_OPID_MAPPING.md`，工具 `scripts/map_opid.py`，全程离线。

**`OpID: 0x6994000000a2` = `node_Where_105`**（`Eltwise_Ternary`，DLC 图内 Id=66），
即 part2 的注意力加性掩码。定位靠两点精确复算，逐位命中日志：
原始 `5.192376e33/1.5259e-9 = 3.4e42` → float32 溢出 = `inf`；
maskfix `(100/65535)/(1e-4/65535)` = **整 1e6** = 日志的 `1000000.000000`。

真正的病理是 **`Where` 的输出 `val_105` 的 encoding 是 `[0, 1e-4]`、offset=0 的无符号 u16
——无法表示任何负数**。上次 maskfix 只改了输入常量 `val_104`，没改输出 `val_105`，
所以报错照旧。（这推翻了 14.18/14.21「该 inf 不能由单个张量的 encoding 解释」的归因，
详见 HANDOVER 14.22.4。）

> 🔴 **但这条线索被同一次实验证伪了**：实测 `unified_mask.raw` 全 4128 个值**全为 1**，
> 没有任何被掩码的位置 ⇒ `Where` 永远选 `val_103`(≈0) 分支，
> `val_104` 分支**一次都没被选中** ⇒ **病理常量在死路径上**。
> 而且这份 mask 正是 Test B **出好猫**那次用的输入。
> ⇒ 该编译期报错**很可能是红鲱鱼**（且它从不阻断构建，`EXIT=0`，`.bin` 正常产出）。
>
> **应用规则 1 的直接收益**：拿到身份后**先验证"它能否造成观察到的现象"**，
> 于是在写任何修复方案之前就把这条线掐掉了。若跳过这一步，
> 下一轮必然是"改 `val_105` encoding → 重新量化 → 重建 → 装机 → 仍是色块"，
> 又一次 34 分钟起步的无效循环。

### Test B 结论（2026-08-14）

配置：FP32 文本编码 + **量化 transformer（part1a/1b/2 全部走 `_quantized.dlc`）** +
FP32 调度器 + FP32 VAE，同 prompt 同 seed(42)。

| 对比 | 平均\|像素差\| | PSNR |
|---|---|---|
| FP32 基准 vs **混合(量化transformer)** | **9.32** | **23.89 dB** |
| FP32 基准 vs 设备真机输出 | 46.40 | 11.56 dB |

产物：`scratch_runs/testB_hybrid_quantized_transformer.png` —— 清晰的橘猫，虎斑/胡须/瞳孔/
木纹/背景虚化全部正常。

**按执行前定稿的判据（EXP_PLAN_V2 第4节）：A1 被否定。**

### 实验 A3 结论（2026-08-14）

配置：同一份**真实** latents（VAE 校准 sample_0000）分别用 FP32 VAE(onnx) 与
量化 VAE(`vae_decoder_quantized.dlc`) 解码。量化侧走 `snpe_runner.py` 强制契约检查。

- 平均像素差 **0.32**，PSNR **51.11 dB** —— 几乎无损
- V2 有效性检查通过：FP32/量化两侧解码出的都是清晰自然图像（虎斑猫在木桌上）
- 产物：`scratch_runs/testA3_vae_fp32.png`、`testA3_vae_quant.png`

**按执行前定稿的判据（EXP_PLAN_A3 第6节）：A3 被否定。**

### 由 A1+A3 两次否定得到的更强判断

三个量化环节里已有两个被实测排除，且都**接近无损**而非勉强通过。结合"噪声预测 15.988%
相对误差仍出好图"这个标定点：

> **量化这条线整体上不足以解释"图像变成橙色无定形色块"这种性质的失败。**
> 量化误差表现为细节劣化，不会抹除语义结构。真正的原因应当是**结构性/语义性**的
> ——条件信息丢失、算子语义错误、数据布局错位、系数写错之类。

这类错误的已知发生地是 **C++ 实现**（已抓到过 2 个：`cap_pad_mask` 极性、Euler 符号）。

## 上述分叉点已在同一天关闭（2026-08-14）

问的是「设备 C++ 喂给 part2 的 `unified_mask` 是什么」。**答案：C++ 根本不喂它。**

`unified_mask` 是 **transformer_part1a 的输出**，由 `runGraph` 自动传给 part2；
C++ 一方源码里 `unified_mask` 一次都没出现。⇒ 设备与 Python 参考**同源**。
⇒ `node_Where_105` 的证伪对设备同样成立 ⇒ **这条线索结案，不是根因。**
（详见 HANDOVER 14.23。唯一残留的口子是 part1a 在 HTP 上是否算出不同的
`unified_mask`——但那已经是假设 C 的一般情形，不是这条线索特有的。）

**顺带差点误报一个根因**：C++ 里 `cap_pad_mask` 是 32 字节 uint8，而契约 JSON 写 128，
和 CLAUDE.md 约束 3 里"32/128 → 输出缩到 1/4"的实测反例数字完全一致，看起来就是那个 bug。
**但三处独立守卫否定了它**（`ZImageQnnContract.hpp:164`、`PipelineZImage.hpp:390`、
`QnnModel.hpp:89-94`，最后一处直接比对 `.bin` 里 QNN 张量的 `client_buf.dataSize`）。
真相是：**"宿主侧一律 float32"是 SNPE `.dlc` 路径的规则，不是设备 QNN `.bin` 路径的规则**，
两条路径各自都对。已写入 `../docs/EXECUTION_MODEL.md` 规则 2·补。**不要去"修"那个 32 字节。**

## 🔴 顶层问题已有答案（2026-08-15）

> **设备生成橙色色块的根因是：HTP 后端执行量化 transformer 的 `.bin`，
> 与 SNPE CPU 参考执行同一份量化模型不等价，且这个不等价单独就足以毁图。**

**决定性证据（`EXP_PLAN_HTP_INLOOP`，单变量对照）**：
把 Test B 的流水线原封不动地跑一遍，**只把 transformer 三段的执行后端
从「CPU 参考跑 `.dlc`」换成「设备 HTP 跑 `.bin`」**，其余（分词、FP32 文本编码、
prompt、seed 42、调度器、VAE）逐字节相同：

| 配置 | 平均\|像素差\| | PSNR | 图像（本人亲自打开看） |
|---|---|---|---|
| Test B：CPU 参考跑量化 transformer | 9.32 | 23.89 dB | 清晰的橘猫 |
| **本实验：设备 HTP 跑同一模型** | **46.73** | **12.06 dB** | **橙色色块，无猫无桌** |
| 设备 app 真机输出 | 46.40 | 11.56 dB | 橙色色块 |

**HTP 在环几乎逐点复现了真机失败（46.73 vs 46.40）。**

## 假设树最终状态

```
Q: 设备 vs FP32 的差异从何而来？
├─ A. 量化                        ❌ 全部否定
│   ├─ A1 transformer 量化         ❌ Test B（PSNR 23.89dB，出猫）
│   ├─ A2 text encoder 量化        ❌ 本轮排除：HTP 在环实验用 FP32 文本编码，照样出色块
│   └─ A3 VAE 量化                 ❌ 实验A3（51.11dB）+ 本轮再次确认（用 FP32 VAE 照样色块）
├─ B. C++ 实现 ≠ Python 实现       ❌ 本轮排除：实验里根本没用到 C++，照样出色块
└─ C. HTP 执行 .bin ≠ CPU 参考     🔴 **已坐实为根因**
```

**同一份量化模型，在 CPU 上出猫、在 HTP 上出色块** —— 这一句话就把量化、C++、
文本编码、VAE、调度器全部排除了。

## 已测定的量化边界

- 噪声预测相对 L2 **15.988%** ⇒ 成图**完好**（Test B 实测）
- 噪声预测相对 L2 **47.07%** ⇒ 成图**变色块**（本轮实测）
- ⇒ 真正的阈值在 16%~47% 之间，**未测定**（也不必急着测）

## 逐段归因

> ⚠️ **【2026-08-15 度量订正】下表的中间张量数字（part1a / part1b 两行）已作废。**
> 它们用的是全量相对 L2，而这两个张量的 `||a||²` 有 **98.2% / 99.8%** 来自最大的 1% 元素，
> 该口径实质只在度量那 1%。主体口径重算：part1a `add_138` **6.61%**（原 2.35%）、
> part1b `unified` **114.05%**（原 19.47%）。
> **part2 `latents` 那一行不受影响**（能量集中度仅 8.15%，两口径倍差 1.03）。
> 详见约束 7 与 `scripts/metric_proof.py`。

隔离条件下（两侧喂相同的 CPU 参考上游输出）各段自身的 HTP vs CPU 相对 L2（**旧口径，见上方订正**）：

| 段 | 相对 L2 | 备注 |
|---|---|---|
| part1a | **2.35%** | `add_138`；差异呈 42σ 集中离群 |
| part1b | **19.47%** | `unified` |
| part2 | **43.04%** | `latents`；已弥散化，无 >20σ 离群 |

**差异逐段自生**（级联 47.07% vs 隔离 43.04%，仅放大 1.09 倍）⇒
不是"某个上游坏张量污染全链"，而是 HTP 在每一段都持续偏离。
且 HTP **逐位确定性**（同输入两次运行 md5 相同），是系统性偏差而非运行时噪声。
（这两条基于 `latents`，**不受度量订正影响**。）

## 【2026-08-15】定位工作的真实状态：**两轮都无效，需重做**

- **第一轮**（13 个探针沿 Id 均匀分布）：判据假设"误差沿 Id 单调"，
  而 Id 顺序 ≠ 依赖顺序 ⇒ 判据 1/3 作废。
- **第二轮**（17 个探针覆盖最后一个 block）：改用主体口径后发现，
  该 block 内张量的**主体余弦已掉到 0.11~0.48，与真值基本去相关**
  ⇒ 在已毁区域比较误差大小无意义，"RmsNorm 是放大点"**同样作废**。

**下一轮该探测的区间**（由第一轮 bulk 数据读出，误差仍可比、余弦 > 0.95）：

| Id | 主体误差 | 主体余弦 |
|---|---|---|
| 0 | 2.77% | 0.9996 |
| 103 | 7.02% | 0.9976 |
| 203 | 8.64% | 0.9963 |
| 306 | **29.29%** | **0.9574** |
| 459 | 99.69% | 0.7139（已劣化） |

⇒ **主体误差是从图前段逐步长起来的，应在 Id 0~306 区间沿依赖链细分探测**，
   而不是在最后一个 block。

## "为什么 HTP 会偏"——已排除一条，剩一条（2026-08-15）

### ❌ 已排除：宿主离线 prepare 编错了

`EXP_PLAN_ONDEVICE_PREPARE` 实测：把 graph prepare 从 PC 换到设备本机
（`qnn-net-run --dlc_path`，设备在线建图 16m26s），输出与宿主编的 `.bin`
**逐位相同**（相对 L2 = 0.0000%，最大差 = 0），两者对 CPU 参考的偏离都是 **19.4727%**。

⇒ **prepare 在哪跑都一样，不是元凶。**
⇒ 连带否定了"`Unsupported HTP Arch 1 79` 说明按错误架构编译"这个猜想——**确认为虚惊**
   （宿主 x86 侧本就没有任何 HTP stub 与 cdsprpc，该告警与 `libcdsprpc.dll` 同类）。

> 查证纪律：交叉项为 0 曾触发"是不是偷用了缓存 `.bin`"的警报，
> 按事前判据先查证再采信——日志有 `Failed in cacheSelection`（未命中缓存）、
> 有 `Composing Graphs`/`Finalizing Graphs`（真在线建图）、
> 目录里没生成任何缓存、耗时差 80 倍。警报解除后才下的结论。

### 🔴 剩下的唯一方向：HTP 后端的执行语义 / 量化精度配置

偏离来自 HTP 本身的算子实现、累加精度或重量化策略，与 SNPE CPU 参考不等价。
候选配置项（来自量化命令）：`act_bitwidth=16`、`weights_bitwidth=8`、
`use_dynamic_16_bit_weights=True`、`use_per_channel_quantization=True`、`bias_bitwidth=32`。

### ❌ 再排除一条：ch85 的极端点不是毁图原因

`EXP_PLAN_CH85_CAUSAL` 实测（part2 四次全部走 CPU 参考，唯一变量是 unified 里换哪些元素）：

| 变体 | part2 输出相对基线的相对 L2 | 占 R_all |
|---|---|---|
| 全 HTP | 52.87% | 1.000 |
| 仅极端点（16 个 ch85 点） | 12.24% | **0.232** |
| 仅弥散本底 | 52.53% | **0.994** |

**弥散本底造成 99% 的下游损害，极端点只有 23%。**
但那 16 个点却占**输入**误差能量的 64.28%——
**"误差能量占比"与"下游致害程度"严重不一致**，这正是必须做因果分离的理由。
（也顺带推翻了我先前"极端点占 ~81% 能量"的估算，实测 64.28%。）

⇒ **不要再查 ch85 / 量程上界。**

### ✅ 已确证：**HTP 是"算错了"，不是"另一种合法舍入"**

`EXP_PLAN_VS_FP32` 实测（FP32 onnxruntime 当真值，输入与隔离实验逐字节相同）：

| | 与 FP32 的相对 L2 | std |
|---|---|---|
| FP32 真值 | — | 14.9664 |
| **SNPE CPU 参考跑量化模型** | **1.1757%** | 15.1097 |
| **设备 HTP 跑同一模型** | **19.2406%** | **12.9183** |

**E_htp / E_cpu = 16.4 倍。**

⇒ **量化模型本身没问题**——CPU 参考把它跑到了距 FP32 仅 1.18%（正常量化噪声）。
⇒ **HTP 的执行是实打实的错**，不是"同样合法的另一种舍入"。
⇒ 顺带把"CPU 参考可当真值"这个一直是默认、从未验证的前提**补上了实测**（1.1757%）。

### 🔴 当前唯一方向：HTP 系统性把幅值算小

三条互相印证的观察（前两条①实测，第三条②推论）：
1. HTP 输出 std **12.92** vs FP32 **14.97** —— 只有 **86.3%**
2. ch85 极端值处，HTP 给出 CPU 值的 **55.8%** —— 幅值越大压得越狠
3. 误差方向 cos∠(CPU误差, HTP误差) = **-0.33**，部分反相关，不是同向放大

⇒ 形态像**重量化 scale 偏小 / 饱和截断**，不像随机舍入噪声。

关键量级（①实测）：`unified` 量化 scale = 0.0978，HTP vs CPU 的弥散差异
p50 ≈ **3.8 个量化步**、p90 ≈ **12.4 个量化步**（纯舍入应 ≤ 0.5 步）。

### 一条可能的缓解手段（**未验证**，成本数小时）

`unified` 的 encoding 是 min-max 校准的 `[-145.277, 6265.993]`，
但张量主体 std 仅 15.11 ⇒ **16 bit 量程里主体只用到约 10 bit**
（±3σ 仅占 1.414% 的量化级），其余全被 0.026% 的离群值占掉。

CPU 参考在同一 encoding 下只差 1.18%，说明**条件数差本身不致命**；
但它可能吃掉了 HTP 的数值余量。若成立，
`--act_quantizer_calibration percentile`（p99.99 可收紧 scale 3.6 倍）是候选缓解。
**需重新量化 + 重建，成本数小时，动手前应先做更廉价的验证。**

## 线索台账（**每发现一条可疑点立刻登记，每轮开工前先读这张表**）

> **存在意义**：长会话必然丢记忆。本项目已多次实际发生：
> ① 14.20 写了"先确定 OpID 对应哪个算子"，做完后顺着新发现走到配置层，把定位整个丢了；
> ② 2026-08-15 把台账建起来后**仍然没有按它执行**——
> `scaled_dot_product_attention_18` 的误差是全部 29 个张量里**最高的（127%/140%）**，
> 却因为一个后来被证伪的"放大点"分析被带去 RmsNorm，此后再没回去。
> **建表不等于执行。每轮开工、每次转向前，必须先读这张表并回答"有没有该查而没查的"。**

**状态只允许四种**：✅已关闭 ／ 🔄进行中 ／ 🔶未查 ／ ⛔受阻（当前做不了，**不等于已关闭**）

### A. HTP 根因确立【前】的线索

| # | 线索 | 状态 | 依据 |
|---|---|---|---|
| 1 | `node_Where_105` requant inf | ✅ | `unified_mask` 全为 1，掩码分支是死路径 |
| 2 | `cap_pad_mask` 32B vs 契约 128B | ✅ | 非 bug，两条路径宿主契约本就不同，设备元数据确认 |
| 3 | ch85 极端离群点（对下游） | ✅ | 因果实验：只造成 23% 下游损害，本底造成 99% |

### B. "HTP 为什么算错"——根因确立【后】的全部怀疑

| # | 线索 | 状态 | 依据 / 缺口 |
|---|---|---|---|
| 4 | `fp16_relaxed_precision` | ✅ | 查文档：2.35 起废弃，且只作用于 fp 模型 |
| 5 | 宿主离线 prepare 编错 | ✅ | 设备在线建图输出**逐位相同**（相对 L2 = 0） |
| 6 | `Unsupported HTP Arch 1 79` | ✅ | 宿主无 HTP stub，属同类告警；#5 佐证 |
| 7 | **权重非对称量化** | ✅ **已关闭·被否定，且方向相反**（2026-08-15 `EXP_PLAN_SYM_FC_OVR`） | 用 overrides 只对 **59 个 FullyConnected** 权重指定对称 `sFxp_8`（RmsNorm 的 gamma 44/44 与基线逐字段相同）。**编得过图**（填上上一轮的空缺）。设备实测：主体余弦 **0.6033 → 0.5507（Δ = −0.0526）**，落入事前判据的「反向恶化」分支；slope **0.8622 → 0.8217**（幅值算小更严重）。CPU 对照 0.9975 → 0.9961 仍健康 ⇒ 不是模型被改坏。<br>⚠️ **与官方通用建议相反**（"recommended to always use symmetrical quantization of weights"）——该建议在本配置（A16W8 + 零 Conv 的 DiT）下**不成立**，勿重试。<br>⚠️ **划界**：否定的是"靠对称化 FC 修 HTP"这条**路径**，**不是** #40 的 offset 项机制本身——对称化同时粗化了 scale（中位 1.073×），两者无法分离 |
| 8 | `PRECISION_COMPENSATION` | ✅（不可用） | 官方要求 min_arch >= 81，本机 v79。**但它的存在佐证 16-bit 激活精度是已知弱点** |
| 9 | min-max 校准（激活） | ✅ | percentile 钳掉 48~99.97% 能量；MSE 最优解**就是** min-max；16-bit 表示误差仅 0.32% |
| 10 | `HIGH_PRECISION_SIGMOID` | ⛔ **受阻·未验证** | 经 `qnn-context-binary-generator --config_file` 设置后，产物 md5 与原始**逐位相同**（对照：O=3 会改 md5）⇒ **被静默忽略**。官方文档：该 flag 通过 **`snpe-dlc-graph-prepare`** 或在线准备设置。**待换工具重试，不计为已排除** |
| 11 | Softmax / 注意力段 | ✅ **已关闭·H-attn 被否定** | 实测注意力**入口**主体余弦已劣化：Q 0.8359 / K 0.8404 / V 0.6054，输出 0.4597。命中判据第一分支：**引入点在注意力段上游，注意力只是传递误差**。两个 2GB 的注意力矩阵因此**无需补测** |
| 12 | RmsNorm gamma 量化质量 | ✅ | 反相关：质量最好的放大最多 |
| 13 | RmsNorm `weight_bias` scale=0 | ✅ | ONNX 里 44 个 norm、**0 个 bias**，是转换器合成的全零占位，无害 |
| 14 | RmsNorm 输出 encoding 余量 | ✅ | 全样本 rho=-0.235（弱）；且 MSE 分析证明表示本身无损 |
| 15 | ~~"RmsNorm 是误差放大点"~~ | ❌ **已撤回** | 度量订正后那些张量主体余弦仅 0.11~0.48，**在废墟里比大小无意义** |
| 16 | A16W16 | ✅ | 内存：part1a+1b 7279M vs 实测被杀线 7300M；运行：加载主导（~110MB/s），字节翻倍→生图时间翻倍 |
| 17 | A8W8 | ✅ | 计算：8-bit 表示误差约 **80%**（16-bit 为 0.32%，x256） |
| 18 | 拆分模型为更多段 | ✅ | 加载主导，总字节数不变，拆分不解决；且用户明确否决（生图已 4-5 分钟） |
| 19 | 图优化级别 `O` | ✅ **已关闭·无影响** | `--optimization_level_override 3` 重建 part1b，元数据确认 `optimizationLevel: 0 → 3`（体积 1471.5 → 1493.8 MB），输出与 O=0 **不逐位相同**（图确实变了）。但精度无改善：主体余弦 **0.6033 → 0.6028（-0.0005）**、主体误差 114.80% → 114.89%，均在噪声量级。⇒ **`O` 是性能选项，不是精度选项** |
| 20 | massive activation / 通道迁移 | ✅ **已关闭·无收益** | `repr_vs_compute.py` 实测：主体通道**量化表示极限仅 2.90%**，CPU 4.56%，而 **HTP 73.62%** ⇒ 表示不是瓶颈，改善表示无意义。**省掉一项大工程** |
| 25 | `Convert` 算子（对称化 MatMul 的 B 输入） | ✅ **已关闭·无损** | 实测 `val_2590` 经 Convert 后主体余弦 **0.8404 → 0.8404（变化 -0.0000）**，scale 变粗 1.004~1.375 倍未造成实质影响。⚠️ `transpose_74_converted` 曾显示余弦 -0.0019、40% 元素为 q=0，**经验证是 `--set_output_tensors` 强制物化被融合算子的探测假象**（判据：注意力输出 mean -0.00003，若 V 真含 40% 的 -212.65 应被拉到 -80 量级） |
| 21 | v79 支持成熟度 | ⛔ **无法验证** | 有 SM8550(v73) 的 `.bin`，但**缺 v73 设备**。**不得因做不了就当不存在** |
| 26 | **误差在进入注意力【之前】就已积累** | 🔶 **未查·由 #11 导出** | #11 实测：注意力入口 Q/K/V 的主体余弦已是 **0.84 / 0.84 / 0.61**。⇒ 引入点在 part1b 前段乃至 part1a。已知 part1b 内主体余弦沿链：Id 0 **0.9996** → Id 103 0.9976 → Id 306 **0.9574** → Id 459 0.7139。**应在余弦仍 >0.95 的 Id 0~306 区间沿依赖链细分探测** |
| 27 | `advanced_activation_fusion`（默认 true） | ⛔ **受阻·未验证** | 同 #10：`--config_file` 设 false 后 md5 逐位相同，被静默忽略。同样需换 `snpe-dlc-graph-prepare` |
| 28 | **误差来自 FullyConnected 还是其余算子** | 🔄 **进行中·决定性** | 体积已算清：三段 8-bit 权重 6154 M 参数中 **FullyConnected 占 99.99%**（6153.9 M），RmsNorm 仅 0.5 M，MatMul **0**（两输入皆激活）。⇒ 浮点化非 FC 算子体积仅增 **~1 MB**；浮点化 FC 则 +6154 MB 撞红线。**本实验决定选择性 FP16 可不可行**（`EXP_PLAN_FC_VS_REST.md`） |
| 29 | **offset 项精度（外部审查提出）** | 🔶 **未查·高优先级** | 非对称权重的定点 MatMul 展开为 `y = Σw·x − zp·Σx`。**对 massive activation 通道 `Σx` 极大 ⇒ offset 修正项极大 ⇒ 精度损失被放大**。这条把"权重非对称"与"通道 85"两个现象**连起来了**，也解释了官方为何建议权重对称量化。验证方式见 #7 的绕法 |
| 30 | **单 MatMul 探针测 HTP 内部位宽（外部审查提出）** | 🔶 **未查·成本最低** | 建极简模型（1 个 MatMul）+ 受控输入，在 HTP 上跑，直接暴露内部位宽与舍入策略。审查者假说：**HTP 内部把 A16 隐式截断到 ~A12**，可同时解释系统性增益偏低、丢 4.66 bit、逐位确定、CPU 不受影响 |
| 31 | **只修 part1b（我先前过度声称）** | ✅ **已关闭·单段修复有效果但不足以出图**（2026-08-15） | 设备上**只把 `part1b.bin` 换成 per-row 版**（part1a/part2/prompt/seed/调度器/VAE 全不变），在环出图：平均\|像素差\| 46.73 → **43.56**（PSNR 12.06 → 12.60 dB）。<br>**①实测·本人亲自打开两张 PNG**：基线是**纯渐变、零结构**；per-row 版**出现了全局构图**——下方 1/3 有横贯的水平分界（似桌面边缘）、中央有带纹理的主体块、右中部有一个带硬边的深色物体。**但没有猫**，主体无法辨认为任何具体物体。<br>⚠️ **单看像素指标（−3.2）会误判为"没变化"**——这正是约束 1 的价值；**反过来也不得说"接近成功"**。<br>⇒ 推翻我先前"任何只修一段的方案都无效"的过度声称 |
| 32 | **SDK 2.28 vs 2.48 数值差异** | 🔶 **未查** | local-dream 的 SD1.5/SDXL 用 **2.28** 转换且跑通；本项目用 **2.48**。跨 20 个版本，HTP compiler 策略可能变化。外部审查列为 P1 |
| 33 | **HTP 算子融合行为** | 🔶 **未查** | HTP compiler 会做算子融合，融合后的超级算子内部精度策略可能与单算子不同。可用 `--profiling_level detailed` / `--log_level verbose` 观察实际执行图 |
| 34 | **VTCM 分块策略对累加精度的影响** | 🔶 **未查** | 大矩阵乘需在 VTCM（8MB）分块；若部分和在 requant 后再累加（而非高精度累加器内累加），精度会显著下降。`--vtcm_mb` 可控 |

### C·补. 外部审查提出并已验证的前提（**下次诊断必须附证据**）

| 质疑 | 状态 | 证据 |
|---|---|---|
| **P0：HTP 实际使用的 encoding 可能与 CPU 不同** | ✅ **已验证一致** | 从 context binary 元数据 dump 出 HTP 实际 scale/offset，与 `.dlc` 声明逐个比对：part1a 全部 **10 个张量完全一致，0 个不一致**。已写入 `../docs/reviews/DIAGNOSIS_HTP_ZIMAGE.md` §4.2.1，**预先挡住该质疑** |

> **教训**：诊断文档要能自己挡住可预见的质疑。凡是"隐含前提"，
> 都应在文档内**先自证**再下结论，而不是等评审提出再补。
| 35 | 🔴 **CPU 基线一直是【浮点】执行，不是定点** | ✅ **已确认·重大** | `snpe-net-run` 有 `--enable_cpu_fxp`（"Enable the fixed point execution on CPU runtime"），而 `snpe_runner.py` **从未使用**。⇒ 此前所有"CPU 参考"都是反量化后 float 计算。**"HTP 比正确的 16-bit 实现丢 4 bit"失去前提**；Test B 的"清晰的猫"是**伪量化**产生的 |
| 36 | **本 SDK 无法提供 A16W8 真定点 CPU 参考** | ✅ **已确认·受限** | 加 `--enable_cpu_fxp` 后工具直接拒绝：`error_code=703; CPU fixed point execution is selected with DLC having INT16 activation`。⇒ SNPE 的 CPU 定点运行时**只支持 INT8**。该判别器不可用 |
| 37 | 自建 numpy 定点模拟器 | ✅ **已完成** | 前提 P1~P4 全过（语义 `x @ W`，方阵方向实测确认，用反量化权重复算 vs CPU 参考 **0.0001~0.65%**）。结果见 #39/#40/#41 |
| 38 | **FC 权重实为 per-tensor 而非 per-channel** | 🔶 **未查** | 量化命令写 `use_per_channel_quantization=True`，但 DLC 里 `val_1800`[3840,3840] 等 FC 权重**只有单个 scale/offset**。per-channel 可能只作用于 Conv。若确认，则 FC 权重量化粒度比预期粗 |

| 39 | 🔴 **注意力投影层：HTP 做的不是"最优定点"** | ✅ **已实测** | 自建模拟器对比（前 512 行）：`linear_99/100/102_fc`（K=3840）**正确定点误差仅 0.012~0.19%，HTP 却是 2.72~2.87%**，差 20~230 倍 ⇒ **H-B：HTP 引入了超出正确定点的误差**。<br>但 FFN 层（K=10240）不同：正确定点自身就有 **1.9~9.7%** ⇒ 那部分是**模型对定点不友好**，两类成因需分开 |
| 40 | 🔴 **offset 项与主项几乎等大（灾难性抵消）** | ✅ **已实测·支持审查者 §3.8** | `acc = Σq_x·q_w − z_w·Σq_x − …`，实测 **offset 项/主项 = 0.891~0.979**。两个大数相减、结果远小于各自量级 ⇒ 中间精度稍有不足即吃掉有效位。**对称量化（z_w=0）会让该项整体消失** |
| 41 | 累加器溢出假说 | ✅ **已否定** | 实测 \|acc\| max 仅占 int32 上限的 **0.0000~0.0309**，余量 30 倍以上，不存在溢出 |
| 42 | `--restrict_quantization_steps` | ✅ **已关闭·不适用** | 用 `-0x8000 0x7F7F` 重量化 part1b：**852 个 encoding 全部未变，0 个改变** ⇒ 参数对本配置无效。**原因**：HTP 文档限定 *"INT16 **weight** ... when using **A16W16**"*，而本项目是 **A16W8**（8-bit 权重）。量化器帮助里那句 "required for 16-bit Matmul" 被我读窄了——**适用范围我早已读到却没连起来**（约束 5 的重复错误，代价 45 分钟）。DLC md5 变化仅因内嵌命令行字符串变长 8 字节 |
| 43 | **`--enable_float_fallback` 单独使用** | ✅ **已关闭·必须配 overrides** | 实测报错：*"fallbackToFloat: Use float fallback flag when user specify encodings through `--quantization_overrides` or input network contains framework trained QAT encodings"* ⇒ 浮点回退是**填补用户未指定 encoding** 的机制，**不能单独用**。⇒ 选择性 FP16 必须走 overrides（见 #45） |
| 44 | **`use_per_channel_quantization` 对本模型无效** | ✅ **已确认** | 帮助原文：*"enable per-channel quantization for **convolution-based op** weights"*。本模型**零 Convolution** ⇒ 该开关一直是空转，FC 权重实为 per-tensor（与 #38 一致） |

| 45 | 🟢 **`--quantization_overrides` 闸门已打通（本轮最关键进展）** | ✅ **已实测可用** | 历史记录（HANDOVER 13.x）称该路径坏掉：`version="2.0.0"` 触发 SDK 崩溃、`"0.6.1"` 撞 `node_MatMul_333` 校验失败。**2026-08-15 实测两个坑都可绕过**：用 **`version="0.6.1"`** + **ONNX 张量名**（`linear_99`，**不是** DLC 的 `linear_99_fc`）→ `Processed 2 quantization encodings`、`INFO_CONVERSION_SUCCESS`，EXIT=0，耗时约 60 秒。<br>⚠️ **第一次失败的原因是名字**：用 DLC 名时日志只打 `Processed 0 quantization encodings` 而**不报错**——**静默失效**，必须每次都检查这一行的计数。<br>官方文档（`converters.html` → *Quantization overrides Usecases* → *Float mixed precision conversion*）：只需**点名张量**并给 `{"bitwidth":16,"dtype":"float"}`，不必提供完整 encoding。<br>**这一条同时解锁 #7（选择性对称权重）与选择性 FP16——两者此前都卡在这个闸门上** |


### B·续. 2026-08-15 晚新登记（读 `qairt-quantizer --help` 全量选项 + 官方文档时发现）

| # | 线索 | 状态 | 依据 / 缺口 |
|---|---|---|---|
| 46 | 🔴 **`--quantization_overrides` 会把浮点位宽默认降到 FP16** | ✅ **已确认·陷阱** | ①实测：`ovr/part1b_ovr.dlc` 张量**全部 Float_16**（1607 个 Float_16 / 0 个 Float_32），而基线 fp32 DLC 是 Float_32；DLC 内嵌命令行仍记录 `float_bitwidth=32`（**记录值 ≠ 实际行为**）。官方 `converters.html`：*"Float bitwidth 16 is the default bitwidth for source model with quantization **encodings or overrides**"*。⇒ 用 overrides 时**必须显式 `--float_bitwidth 32`**，否则单变量被破坏 |
| 52 | 🔴 **HTP 对权重量化步长的敏感度是 CPU 的 37.6 倍** | ✅ **已实测（#7 的副产品）·当前最强的机制线索** | 同一次权重 encoding 改动（59 个 FC 由非对称 uFxp8 改对称 sFxp8，scale 中位变粗 **1.073×**、权重表示误差中位 3.73%→3.97%）：**CPU 参考 Δcos = −0.0014，设备 HTP Δcos = −0.0526，倍数 37.6**。<br>⇒ ②推论：HTP 的误差**对权重量化质量高度敏感**，与 #39 同向。<br>⇒ ③假设（**未验证**）：scale 变粗 ⇒ 更差；那么 **scale 变细应当有收益** ⇒ 直接给 #47 提供了定量动机 |
| 56 | 🔴 **用标定过的尺子重测：per-row 在噪声预测上几乎没有推进** | ✅ **已实测·本轮最重要的订正** | 标定点（已有）：噪声预测相对 L2 **15.988% ⇒ 成图完好**、**47.07% ⇒ 橙色色块**。<br>①实测（参考 = Test B 的 CPU 参考 step-0 噪声预测，即**出清晰猫**的那一版；该张量能量集中度仅 8.15%，两口径一致）：<br>· **2/3 per-row 级联：E_all 45.72%、主体余弦 0.8912** ⇒ 相比 47.07% 只走了 **1.35 pp**，缺口 31 pp<br>· 参照：**基线 part2 单段隔离误差 43.04%**（与台账记录完全复现）<br>🔴 **关键推论**：把上游两段都改好后，级联误差 **45.72% ≈ part2 自己的 43.04%** ⇒ **总误差由 part2 支配，而 part2 正是唯一未被处理的那一段**。<br>⚠️ 同时说明：per-row 把 part1b 的 `unified` 余弦从 0.6033 拉到 0.7986，**这个大幅改善没有传导到噪声预测** —— 中间张量指标与顶层目标之间**不是线性关系** |
| 62 | 🔴 **切分 ONNX 前必须先做死代码消除（DCE）** | ✅ **已实测·本轮最重要的方法论结论** | `transformer_part2_fixed.onnx` 有 **84 个死节点**（IsNaN 15 / Where 15 / Gather 10 / Reshape 14 …），它们消费 **30 个 `[1,30,4128,4128]` 张量（57.1 GiB）**。<br>**不 DCE 直接切：切割集 14 个张量 16.4 GB**；**先 DCE 再切：6 个张量 65.6 MB**。<br>机制：转换器对**完整图**会自己 DCE（生产 part2 DLC 里 `IsNaN` = **0**），**一切分，死节点就被"复活"成子图的必需部分**，量化时每个 1.9044 GiB 的张量都要参与校准。<br>①实测佐证：量化内存增长步长 **1.90~1.91 GB = 单个该张量 fp32 大小**；DCE 后 1 样本量化**峰值 32.36 GB 封顶、5.5 分钟完成**（DCE 前线性涨到 134 GB 后 OOM）。<br>⚠️ 已查：part1a/part1b **死节点为 0**，此问题目前仅 part2 有 |
| 61 | 🔴 **我用错了源 ONNX（base 而非 `_fixed`），整轮返工** | ✅ **已关闭·方法论教训** | 生产 part2 DLC 的 `--input_network` 是 `transformer_part2_fixed.onnx`（从 `01_qairt_converter.log` 核实），我却按文件名朴素程度选了 `transformer_part2.onnx`，两者**差 108 个节点**。<br>**验证材料当时就在手上**——我同一小时读过那个日志，只取了 `input_dim`，没看同一行的 `--input_network`。<br>⇒ **任何输入产物必须从流水线日志核实来源，不得由文件名推断**。<br>⚠️ 附带教训：遇到 `latents_shape` 报错时我**自己重新发明了 `_fixed` 早已做过的修复**——**当一个问题被"重新发现"时，应立刻怀疑是不是用了旧版本** |
| 60 | **external weights / spill-fill buffer** | ✅ **已实测关闭·不减少 PD 估算** | 自建最小 probe（`scripts/extbuf_probe/`，NDK 交叉编译，仅加载不推理）四档实测：T0 普通 ❌0x3ea；T1 仅 spill-fill（260.0 MB 注册成功）❌；T2 仅 weights（2730.5 MB 注册成功）❌；T3 两者 ❌。<br>**FARF 的 `context size estimate 3652345600` 外置前后完全相同** ⇒ 机制虽通（DEFER 拿到 handle、memRegister 全部成功），但**不减少 PD 容量估算**。<br>①首次测得：`WEIGHTS_BUFFER_SIZE` = **2730.5 MB**、`MAX_SPILLFILL_BUFFER_SIZE` = **260.0 MB**。<br>②推论：真正的 PD 检查在 `contextFinalize`，`DEFER_GRAPH_INIT` 只推迟不豁免 |
| 59 | ⚠️ **`qnn-net-run` 2.48 无法使用 `spill_fill_buffer` / `weights_buffer` 配置** | ✅ **已确认·工具缺陷** | JSON schema 要 integer、下游解析器要 string，整数 −1 / 整数正值 / 字符串三种全部失败。**设备端五个关键库与 SDK 2.48.0.260626 md5 逐一核对完全一致 ⇒ 排除版本错配**。<br>`qairt-net-run` 的 schema 反过来要 string（配置能解析），但其 HTP 后端在更早的平台信息解析失败（对照组同样失败），该运行器路径未打通 |
| 58 | ⚠️ **`rpcmem_alloc` 的 size 是 `int`，>2 GB 会 32 位溢出** | ✅ **已实测** | 分配 2730.5 MB 的 weights buffer 直接失败。设备 `libcdsprpc.so` 导出 64 位的 **`rpcmem_alloc2`**，换用后成功。转 Flux.2 Klein 等大模型时必踩 |
| 57 | 🔴 **根因已定位：unsigned PD 容量上限，part2 已无余量** | ✅ **已实测·精确到字节** | 开 FARF（`qnn-net-run.farf` 写 `0x1f`）后拿到 DSP 侧原文：<br>`QnnDsp <E> **Failed to find available PD** for contextId 1 ... with **context size estimate 3652345600**`<br>`context create from binary failed, err = 1002`<br>**不是内存不足** —— 同一时刻 lowmemorykiller 记录 `device has enough memory 8047228Kib`。<br>是 **DSP 保护域（PD）放不下 3.652 GB 的 context**。baseline part2 按元数据推算约 **3.62 GB**、能加载 ⇒ **上限就在这 30 MB 之间，part2 已经贴着天花板在跑**。<br>**已排除的绕法**：`vtcm_override` —— 三份 bin 实测 `vtcmSize` 全 = **8 MB**，已是 v79 上限；`optimization_level` 默认已是 0（#19 实测 O=3 只让体积**变大**）；DLBC 文档原文是压缩 *inputs* 降带宽、非缩 context；Sparse Weights Compression 只对稀疏权重有效（本模型权重稠密）。<br>signed PD 能有更大空间，但官方要求"push a signed dsp image to target"，**我们没有** |
| 55 | 🔴 **宿主离线编的 per-row part2 `.bin` 在设备上【加载失败】** | ✅ **已关闭·根因见 #57**（曾被我降级为🔶，该降级已被 #56 推翻）（曾于 2026-08-16 11:10 被我降级为🔶，**该降级已被 #56 推翻**：降级依据是成图像素差 46.73→41.37，而像素差在废图区间无分辨力；换标定尺后 part2 反而是支配项。**保留原判断与推翻理由，见约束 9 清单第 3 条**） | ①实测：`qnn-net-run --retrieve_context part2_perrow_ctx.SM8750.bin` ⇒ **`Could not create context from binary`**。<br>**已排除的解释**：① 不是内存——同一时刻**原版 part2 `.bin`（2928.7 MB）在 MemAvailable 更低（4.6 GB vs 6.4 GB）时正常加载执行；② 不是文件损坏——宿主/设备 md5 一致，且 `qnn-context-binary-utility` 解析正常（dspArch 79 / socModel 69 / 单图 / blob 2934277376）；③ 不是 spill/VTCM——建图日志的 `spill_bytes=4218912768` 与基线**逐字节相同**；④ 不是那条 `scale too large for requant ... inf` 报错——**基线建图日志里同样有一条**。<br>⑤ 不是 per-row 本身——**part1a（2370.6 MB）与 part1b（1472.9 MB）的 per-row `.bin` 都能正常加载执行**。<br>🔴 **决定性观察**：同一份 per-row part2 DLC 用 `qnn-net-run --dlc_path` 在**设备本机在线建图**，成功越过 `Composing Graphs` 进入 `Finalizing Graphs` ⇒ **模型本身没问题，问题在"加载宿主离线编的 `.bin`"这一步**。<br>⚠️ 这不推翻 #5（那次证明的是 part1b 基线的host/device 输出逐位相同），但**明确划出了 #5 结论的适用边界**：host-prepare 与 device-prepare **并非总是可互换**。<br>③假设（**未验证**）：part2 是三段里唯一 >2.9 GB 的 context，per-row 让它再涨 5.5 MB，可能顶到某个固定上限。<br>**绕法**：SDK 自带 `aarch64-android/qnn-context-binary-generator`，改为**在设备上生成 part2 的 `.bin`** |
| 54 | **三段全部 per-row，在环出图** | ⛔ **受阻于 #55**（`EXP_PLAN_PERROW3`） | part1a/part2 已重量化并确认 per-axis 生效（axis-quant 张量 **109 / 139**，全 axis 0）。<br>⚠️ **补记：这两次重量化我是在没写方案的情况下启动的，违反约束 4**，方案已于结果产生前补齐，判据与决策树事前锁定。<br>⚠️ 已知悲观理由：part2 的注意力 MatMul **两输入皆激活、无静态权重**，per-row 对其**完全不作用** ⇒ part2 收益很可能小于 part1b |
| 53 | 🔴 **#9（激活 min-max 校准）的关闭理由已被 #52 动摇，应重开** | 🔶 **未查·由 #52 导出·已锁定为 #54 分支 B 的下一步** | #9 当初关闭的依据是**表示误差/MSE 论证**（"percentile 钳掉 48~99.97% 能量；MSE 最优解就是 min-max；16-bit 表示误差仅 0.32%"）。但 #52 实测证明：**表示误差不是 HTP 误差的正确预测量**——权重表示误差只从 3.73% 变到 3.97%（+6.5%），设备余弦却掉 0.0526；per-row 把表示误差降到 0.28 倍，余弦涨 0.2479。⇒ **"MSE 最优"不等于"HTP 最优"，#9 的关闭论证在 HTP 上不成立**。15.12 实测 `unified` 的 16-bit 量程里主体只用到约 10 bit，`--act_quantizer_calibration percentile`（p99.99）可收紧 scale **3.6 倍**——与 per-row 对权重做的事同构。<br>成本：重量化 part1b（~54 min）+ 建图（~14 min）+ 设备（~2 min） |
| 47 | ✅ **已实测·`--use_per_row_quantization` 大幅改善**（2026-08-15 `EXP_PLAN_PERROW`） | 66 个 axis-quant 张量全部 axis 0；59 个 FC 权重 `sFxp_8`/offset 0。设备实测 cos_bulk **0.6033 → 0.7986（Δ=+0.1953，弥合 49.5%）**，E_all **19.24% → 7.87%**，E_bulk 首次跌破 100%（→78.77%），**slope 0.8622 → 1.0435**。<br>判据 1 落在「部分成立」（差 0.0047 没够到 +0.20 线，**不宣称找到根因**）；**判据 2 成立**（vs #7 同为对称权重、仅 scale 细 3.9 倍 ⇒ Δ=+0.2479）。<br>⛔ 判据 3（CPU 对照）**无法执行**：`error_code=202; Dequantization of axis-quant tensor is not supported for FullyConnected`（同 #36 类工具限制）。<br>在环出图（只换 part1b）：46.73 → 43.56，**图像从零结构变为有全局构图但仍无猫**（见 #31） | 帮助原文：*"enable **rowwise quantization of Matmul and FullyConnected ops**"*。#44 已确认 `use_per_channel_quantization` 只作用于 **convolution-based op**、对本模型空转 ⇒ **per-row 才是本模型（99.99% 权重在 FC）对应的细粒度量化开关**，此前完全没注意到。配套还有 `--enable_per_row_quantized_bias`。只需重量化，不改图 |
| 48 | `--keep_weights_quantized`（wFxp_actFP） | 🔶 **未查** | 帮助原文：*"keep the weights quantized even when the output of the op is in floating point ... Required to enable **wFxp_actFP** configurations"*。⇒ 这正是审查者建议的「FP16 激活 + 8-bit 权重、权重不翻倍」的官方开关名。做选择性 FP16（执行顺序第 2 条）时必须一起用 |
| 49 | `--param_quantizer_schema unsignedsymmetric` | 🔶 **未查** | schema 取值共四种：asymmetric / symmetric / **unsignedsymmetric** / signedasymmetric。unsignedsymmetric 可能给出「量程关于 0 对称但仍是 uFxp_8」⇒ **RmsNorm 的 OpDef 限制可能不再触发**，全局一把梭无需 overrides。⚠️ 但它**不消除** #40 的 offset 项（z_w=128≠0），机制上与 #7 不等价，只是编译可行性更好 |
| 50 | `--apply_algorithms cle`（Cross Layer Equalization） | ✅ **已关闭·不适用** | 文档明确 CLE 只匹配 `Conv->Batchnorm->activation->Conv...` 模式且**只支持 Relu**；本模型**零 Convolution、零 BatchNorm** ⇒ 无可匹配模式。（**查文档 2 分钟，省掉一次量化**） |
| 51 | `--use_aimet_quantizer` + AdaRound / AMP | 🔶 **未查·需装 aimet-torch** | AMP = 自动混合精度搜索，正是"选择性 FP16 该点名谁"的官方解法；AdaRound 是权重舍入优化。需 `pip install aimet-torch` + 写 dataloader callback，工程量中等。**在 #7 / #47 之后再考虑** |

### D. 2026-08-17 新登记

| # | 线索 | 状态 | 依据 / 缺口 |
|---|---|---|---|
| 63 | 🔴 **A16W8 下"正确定点"自身的等效位宽只有 ~7.4 bit** | ✅ **已实测（#30 探针的宿主侧预检副产品）** | 单 MatMul 探针，32 测试样本，用与 HTP 完全相同的 encoding 做正确定点：`E_fxp` = 1.03~1.34%，**等效位宽 7.04~7.53 bit**（K=64/1024/3840）。<br>机制：权重仅 8 bit，经 K 次累加后**权重量化误差主导**，与激活的 16 bit 无关。<br>🔴 **直接后果**：15.13.2 的"HTP 等效 11.34 bit"与本处的位宽**不是同一个量**（那个是相对 `unified` 自身量程、不含 W8 误差）⇒ **两者不得直接比较**，`EXP_PLAN_MATMUL_PROBE` 的"11~13 bit ⇒ 强支持 H-floor"子判据在该设计下**无分辨力**（见该方案附录 A.2）。<br>⚠️ 划界：本探针权重是 `N(0,1)/√K` 随机阵，与真实模型权重分布不同，**不得外推为"真实模型的定点上限就是 7.4 bit"** |
| 64 | ⚠️ **`qnn-context-binary-generator` 漏 `--htp_socs` 会静默产出【非 SoC 定向】的 context binary** | ✅ **已实测·工具陷阱** | 不加 `--htp_socs sm8750` 时产物是 `probe_kK_ctx.bin`（无 SoC 后缀），日志里**也没有** `Cache binary for SM8750 copied to ...` 那一行，**但 rc=0、无任何警告**。加上后才得到 `*.SM8750.bin`。<br>帮助原文：*"Specify SoC(s) to generate **HTP Offline Cache** for"*。<br>⚠️ 项目里 part1a/part1b/part2/part2a/part2b 的 `.bin` **全部**是 `--htp_socs` 产物 ⇒ 任何新建的对照/探针必须同样加，否则**测的不是同一条执行路径**。**判别方法：看文件名有没有 SoC 后缀。** |
| 65 | ⚠️ **`htp_inloop_pipeline.py` 的 `sh()` 吞掉 adb 错误，把 USB 掉线伪装成模型执行失败** | ✅ **已修复** | 2026-08-17 实测：设备中途掉线时 `sh()` 返回空串，调用方于是抛 `qnn-net-run failed [part1a/s0]` —— 报错指向模型，真实原因 `adb: device not found` 全在被丢弃的 stderr 里。<br>已改：`sh()` 在 rc≠0 时抛异常并带 stderr；每段执行前加 `require_device()`。<br>**教训**：诊断信息被吞会让排查指向错误对象，这类"报错说谎"比报错本身更贵 |

### C. 方法论缺陷（已固化为约束）

| # | 缺陷 | 状态 | 去处 |
|---|---|---|---|
| 22 | 相对 L2 被离群主导（低估达 48 倍） | ✅ | 约束 7 |
| 23 | 误差 >100% 后张量已去相关（余弦 0.11） | ✅ | 约束 7 |
| 24 | 判据操作化未验证（一天内错 3 次） | ✅ | 约束 8 |

### 执行顺序（按「证据强度 x 成本」排，**不按想到的顺序**）—— 2026-08-15 晚重排

**已完成**：#11 Softmax/注意力（否定）、#19 优化级别（无影响）、#25 Convert（无损）、
#37 定点模拟器、#39 HTP 超出正确定点、#40 offset 项、#41 累加器溢出（否定）、
#42 restrict_quantization_steps（不适用）、#43 float_fallback 单独用（不可）、
#44 per_channel（对本模型空转）、**#45 overrides 闸门已通**、
**#7 选择性对称权重（否定，且方向相反）**、#46 float_bitwidth 陷阱、#50 CLE（不适用）、
**#52 HTP 对权重量化步长敏感度 = CPU 的 37.6 倍（新机制线索）**

> **【2026-08-15 晚重排】#7 已执行并被否定（且方向相反）**，顺序据此更新。
> 本轮新增的 **#52（HTP 对权重量化步长的敏感度 = CPU 的 37.6 倍）** 是当前最强的机制线索，
> 它把 #47 从"顺手试试"抬成了"有定量动机的首选"。

> ## 🟠【2026-08-17 08:00 当前状态 —— 四段在环因设备掉线未跑成，已转宿主侧工作】
>
> **四段在环（#54）：⛔ 受阻·设备离线，判据不变，等设备回来原样重跑。**
> 方案 `scripts/EXP_PLAN_INLOOP4.md`（已定稿）；`htp_inloop_pipeline.py` 已扩到四段并通过语法/preflight。
> 07:41 启动后，宿主侧文本编码正常、`adb push` 正常，推进到 part1a 时设备从 adb 消失。
> **原因已确认（用户直接告知）：用户上班带走手机，是他拔的线。** 不是掉线、不是设备重启、与模型无关。
> 设备侧全部停摆至其下班。
>
> ⚠️ **方法论教训（当场记录）**：我当时写的是"USB 掉线**或设备重启**"这个推测，
> 还挂了监听准备回头读 `uptime` 取证——**为一个不存在的问题准备了取证方案**，
> 而且是在用户已经说过"要带手机上班"之后写的。**问一句 5 秒就有答案。**
> ⇒ 见 CLAUDE.md 约束 10。
> 按方案第 4 节：**V1 有效性门未过 ⇒ 执行方法错误，不得据此对 per-row 或四段级联下任何结论。**
>
> **已拿到的有效结果**：
> · **V3 门通过** —— 新 run 的 step-0 输入（`latents`/`timestep`/`caption`/`cap_pad_mask`）
>   与 43.41% 那次 **md5 逐一相同** ⇒ 四段 vs 三段的对照**单变量成立**，这是判据可比性的前提。
> · 契约核实：`part2a` 的输入集合与 baseline `part2.bin` **完全相同**（从两个 `.bin` 元数据 dump 逐个比对）。
> · 设备上 5 个 `.bin` 的身份按**字节数**与宿主产物逐一对上（`part1a.bin` 是 per-row，`part1a_ORIG.bin` 才是基线）。
> · 度量操作化用已知样本验证：复算出 **43.4094% / 28.5156%**，与记录一致（约束 8）。
>
> **#30 单 MatMul 探针：宿主侧已全部就绪，只差设备那 <1 分钟。**
> 三个 `.bin`（`probe_k{64,1024,3840}_ctx.SM8750.bin`，合计 < 0.5 MB）已建好；
> 脚本 `matmul_probe_gen.py` / `matmul_probe_run.py` / `matmul_probe_analyze.py` 齐备。
> **有效性自检已在宿主提前跑掉**：`E_fxp` = 1.03~1.34% ≪ 5% 无效线 ⇒ 主判据有判别力。
> 同时发现该方案的**位宽子判据在本设计下无分辨力**（见 #63 与方案附录 A.2），
> 决策由 `E_htp/E_fxp` 比值判据承担。
>
> ---
>
> ## 🟢【2026-08-17 03:40 上一线程交接状态】
>
> **卡点已突破**：part2 的 per-row context 装不进 unsigned PD（#57），
> 经"DCE → 切分 → 逐位等价 → 两段量化 → 两段建图"全流程打通：
> 每半段 **1887 / 1927 MB**（红线 3506），设备**加载成功**。
>
> **per-row 对 part2 的收益首次实测**（单段隔离，对 FP32 原点）：
> **E_all 41.24% → 28.52%**、主体余弦 0.9089 → 0.9557，**弥合到 CPU 参考缺口的 50.4%**。
> （含拆分本身的混淆，但方向保守：切口边界只会更差 ⇒ per-row 真实收益 ≥ 此值）
>
> **下一步（唯一未答的顶层问题）**：把 `htp_inloop_pipeline.py` 从三段扩到**四段**
> （part1a → part1b → **part2a → part2b**），跑在环出图，测 step-0 噪声预测
> （门槛 **15.99%**；三段全基线 47.07%，part1a+1b per-row 时 43.41%），
> **并自己打开 PNG 看**（约束 1）。
> 设备上 `part2a_fixed.bin` / `part2b_fixed.bin` 已就位并验证可加载；
> 切口传 6 个张量共 65.6 MB。
>
> ---
>
> **【2026-08-16 早 重排】#47 已完成（部分成立，首个有效手段）。当前在 #54。**
> **后续路线已在 `EXP_PLAN_PERROW3` 第 4 节按 A/B/C 三分支事前锁定，不等结果再想。**
>
> | #54 结果 | 已锁定的下一步 | 依据 |
> |---|---|---|
> | A 出猫 | 固化配方 → 真机级别 3 验证 → 测生图时间 | — |
> | **B 有物体无猫** | **#53 激活 percentile（先只做 part1b）** | #52 步长单调性 + 15.12 主体只用 10 bit + CH85_CAUSAL 离群点只造成 23% 损害 |
> | C 仍无结构 | **#30 单 MatMul 探针**（测 HTP 内部有效位宽），据其结果转 #32 或 #48 | 张量改善未传导 ⇒ 继续试量化配置期望收益低 |

**剩余，按序**：

0. 🔄 **#54 三段全量 per-row 在环出图** —— **当前进行中**
1. ✅ **#47 `--use_per_row_quantization`** —— **已完成：首个有效手段，弥合 49.5%**
   - 官方帮助原文：*"enable rowwise quantization of **Matmul and FullyConnected** ops"*
     ⇒ 正好命中本模型（99.99% 权重在 FC；#44 已证 per-channel 只作用于 Conv、一直空转）
   - #52 实测：scale 变**粗** 1.073× ⇒ 设备主体余弦掉 0.0526；逐行量化让每行用自己的 scale
     ⇒ 中位数上显著变**细**。③假设：应有收益（**未验证**）
   - 成本：不改图、不重转换，只需重量化（~54 min）+ 建图（~12 min）+ 设备实测（~2 min）
   - ⚠️ 事前必须先做：把逐行 scale 与当前 per-tensor scale 的**中位比值**算出来
     （若细化不明显，就没有做的必要）；并查 OpDef 确认 per-row 的数据类型组合被支持
2. **选择性 FP16（#43 的正确形态）** —— 闸门已通（#45）。
   #7 失败后它上升为"若 #47 也失败则必做"，应优先点名 #39 里 `E_sim` 与 `E_htp` 差距最大的那几个 FC。
   ⚠️ 必须配 `--keep_weights_quantized`（#48，wFxp_actFP，权重不翻倍）
3. **#26 向上游定位** —— 沿 Id 0~306（余弦 >0.95 的可比区间）细分探测
4. **#20 通道迁移 / SmoothQuant** —— 已按审查者 §3.7 **重开**；需 ONNX 图手术，工程量最大
5. **#31 只修 part1b** —— 从未测过（此前"单段修复无用"是**过度声称**，只测过 part2）
6. **#10 / #27 `high_precision_sigmoid` / `advanced_activation_fusion`** ——
   此前用错工具（`qnn-context-binary-generator` 会**静默忽略**），文档指向 `snpe-dlc-graph-prepare`
7. **#32 SDK 2.28 vs 2.48 A/B** —— local-dream 的转换脚本锁定 2.28 且在同款设备上跑通 SD1.5/SDXL
8. **#21 v79 成熟度** —— 🚫 **受阻**（缺 v73 设备），**保留不删**

> **零成本开关已全部用尽**：#9 校准（min-max 已是 MSE 最优）、#19 优化级别、#25 Convert、
> #42 restrict_quantization_steps、#44 per_channel —— 五条全部证否或不适用。
> #16/#17/#18（A16W16 / A8W8 / 拆分）因体积或速度出局。
> **剩下的每一条都要动 encoding 或动模型结构，而这些现在都要经过 #45 这个闸门。**
