# 扩散模型 → QNN/HTP 转换量化流程（可复用配方）

> **用途**：把 Z-Image Turbo 及其微调版、Flux.2 Klein 等扩散模型转换并量化到骁龙 HTP。
> 本文件与排查文档分开维护：`HANDOVER_*.md` 记录"这次为什么坏"，本文件记录"怎么正确地做"。
>
> **写作纪律同 CLAUDE.md 约束 2**：每条都标注证据来源；未验证的进第九节，不得当事实用。
>
> 🔴 **要上手一个新模型，从 [§四十七 路线图](#四十七路线图把一个新模型从零做到可交付并提速) 开始读，不要从这里顺读。**
> 顺序：**§47.10 五种思维模式 → §47.1 阶段表 → §47.1.1 六张执行卡片（照着跑）→ §47.2 坑清单（已按卡片索引）**。
> 要做到**手机上真的能跑**，还要读 **§四十八（接进 app）** —— 卡片走到卡 E 只是"设备上数值正确"，不是交付。
> 其余各章是**证据库**，被卡片指到时才翻。
>
> ⚠️ **【2026-09-20 一致性订正】本文档是逐日累积的，早期章节里有几条已被后来的实测推翻，
> 而它们写成了祈使句、最容易被照抄。已就地标注（原文保留，约束 2），**受影响的是**：
>
> | 章节 | 早期写法 | 实测订正 |
> |---|---|---|
> | §一 | context/设备"❌ 未验证" | 整条链已真机交付；但"HTP 与 CPU 参考不等价"**仍未解决**（§1.3） |
> | §二③ | 量化命令用 `--use_per_channel_quantization` | DiT 上**完全空转**；正解 `--use_per_row_quantization`（§11.1） |
> | §三 3.3 | 分段按"手机内存/进程 RSS" | 真正的约束是 **PD 红线**；进程 RSS **看不见 context**（§12、§37） |
> | §五 级别 2 | 用 PSNR 判量化好坏 | PSNR 只量**一致性**；量**画质**会**反相关**（§十七） |
> | §五 2.5a、§六 | "不要加 `--use_native_*`" | 真规则是**标志随 `.raw` 实际 dtype 走**（§45.8） |
> | §九 第 2 条 | overrides 通道"撞校验失败" | **已打通**，真因是用了 DLC 内部名（§15.15、§十） |

## 一、当前验证状态（先看这个，决定你能信到哪一步）

### 1.1 现状（2026-09-20 重写，**以此为准**）

**整条链路已走通并真机交付**：Z-Image Turbo 跑在 SM8750 的 HTP 上，五个比例全部出图，
用户已亲自导入验证（HANDOVER §15.50.1）。

| 阶段 | 验证状态 | 证据 |
|---|---|---|
| ONNX 导出 / 图手术 | ✅ 已验证 | FP32 ONNX 全流程出完美图；基座导出对官方 step-0 相对 L2 **1.79%**、余弦 0.999841（§15.44） |
| `qairt-converter`（ONNX→FP32 DLC） | ✅ 已验证 | vs FP32 ONNX 相对误差 0.0074%（浮点噪声量级） |
| `qairt-quantizer`（FP32→A16W8 DLC） | ✅ 已验证 | 但**配方已换**：per-row + 选择性 FP16，见 §1.2 |
| `qnn-context-binary-generator`（DLC→.bin） | ✅ **已验证** | 八段 + 五图 mg context 全部建成并在设备上跑出图；**前提是带 `--config_file` 与 `--htp_socs`**（§十二、§二十五） |
| 设备 HTP 运行 | ✅ **已交付** | 五比例出图；P2 提速后用户实际等待 约 350 s → 约 148 s；出图 sha256 逐字节可复现（§15.50） |
| 设备 HTP **数值与 CPU 参考等价** | ❌ **仍不等价** | 这是**另一个问题**，见 §1.3。已交付 ≠ 数值等价 |

### 1.2 🔴 交付所用的量化配方（照抄这个，**不是** §二③ 那条）

```
A16W8 + --use_per_row_quantization + 手术式选择性 FP16（+ 显式 Clip ±65504）
```
- per-row：§十一（E_all 19.24% → 7.87%，**唯一被实测证明有效的 HTP 精度手段**）
- 选择性 FP16：§二十九（端到端再改善 12.73 pp），最优配置可照抄 §29.8
- 🔴 **§二③ 里那条命令是 2026-08-13 的第一版**，其中 `--use_per_channel_quantization`
  在 DiT 上**完全空转**（零 Convolution），照抄等于白开。详见 §二③ 的订正框与 §11.1。

### 1.3 ⚠️ 已交付，但这两件事至今**未解决**（换模型时要知道）

| 问题 | 状态 | 出处 |
|---|---|---|
| HTP 跑 `.bin` 与 SNPE CPU 参考跑 `.dlc` **数值不等价**，且差到毁图 | 🔴 **机制未查明**（现象已充分实测：逐段隔离 part1a 2.35% / part1b 19.47% / part2 43.04%） | §十五 |
| 与官方实现的一致性：Tier 2 的 caption 32→80 序列手术引入 **15.52%** | 🔴 根因已定位（padding **个数**参与注意力），按用户指示**挂起** | §四十四、§15.44 |

⇒ **可信度边界**：本指南的**流程**是可复用的（已端到端交付一次）；
但"量化后与 FP32 的数值差距"这件事**没有被解决，只是被降到了可接受**。
换新模型时不要指望照做就能拿到数值等价。

---

> <details><summary>🔴 <b>原文（2026-08-13 写，已全面过期，保留供溯源）</b> —— 点开前先读上面</summary>
>
> | 阶段 | 验证状态 | 证据 |
> |---|---|---|
> | `qnn-context-binary-generator`（DLC→.bin） | ❌ **未验证** | 见第九节；已知 part2 编译期报过 requant inf 错误 |
> | 设备 HTP 运行 | ❌ **未验证** | 设备输出与 CPU 参考不一致，根因排查中 |
>
> **结论**：转换 + 量化这两步的配方是可信的、可直接复用的；
> context binary 生成与设备运行这两步还没有被证明正确，换新模型时要独立验证。
>
> **为什么过期**：写于项目第一天，此后整条链路已交付（§15.50）。
> 但注意 —— 它说的"设备输出与 CPU 参考不一致"这一条**至今仍然成立**（见 §1.3），
> 过期的是"未验证"这个状态，不是那个现象。
> </details>

## 二、完整流程

```
.onnx ──①图手术──> _fixed.onnx ──②qairt-converter──> _fp32.dlc
                                      ──③qairt-quantizer(+校准数据)──> _quantized.dlc
                                      ──④qnn-context-binary-generator──> _ctx.<SOC>.bin
```

### ② 转换命令（本项目实际使用、已验证）

```bash
qairt-converter \
  --input_network <model>_fixed.onnx \
  --output_path   <model>_fp32.dlc \
  --target_backend HTP \
  --onnx_skip_simplification \
  -d <input1> <dim1> -d <input2> <dim2> ...      # 每个输入都要给固定 shape
```

> **找回历史命令的方法**（本项目曾因找不到原命令阻塞很久，9.4 节）：
> `snpe-dlc-info -i <dlc>` 的输出里**嵌有完整的 Converter command 和 Quantizer command**。
> 不需要凭猜测重建。

### ③ 量化命令

> 🔴 **【2026-09-20 订正】下面这条是 2026-08-13 的第一版，标题原写"已验证"。
> 它能跑通，但 `--use_per_channel_quantization` 在 DiT / Transformer 上是空转的**
> （官方帮助原文：*per-channel quantization for **convolution-based op** weights*；
> 本项目零 Convolution，开了几个月**一个张量都没影响到**，§11.1）。
> **照抄它 = 白白错过本项目唯一实测有效的精度手段**（per-row：E_all 19.24% → 7.87%）。
> 原文保留（约束 2），但**新模型请用 §1.2 的配方**。

```bash
# ⚠️ 第一版，仅供溯源。新模型请看下面的"当前配方"
qairt-quantizer \
  --input_dlc  <model>_fp32.dlc \
  --output_dlc <model>_quantized.dlc \
  --input_list <校准数据清单>.txt \
  --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
  --act_quantizer_calibration min-max --param_quantizer_calibration min-max \
  --act_quantizer_schema asymmetric --param_quantizer_schema asymmetric \
  --use_per_channel_quantization \          # ← 🔴 DiT 上空转，换成 --use_per_row_quantization
  --target_backend HTP
```

**当前配方（交付所用，A16W8 + per-row）**：

```bash
qairt-quantizer \
  --input_dlc  <model>_fp32.dlc \
  --output_dlc <model>_quantized.dlc \
  --input_list <校准数据清单>.txt \
  --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
  --act_quantizer_calibration min-max --param_quantizer_calibration min-max \
  --act_quantizer_schema asymmetric --param_quantizer_schema asymmetric \
  --use_per_row_quantization \              # ← ✅ 作用于 MatMul / FullyConnected
  --target_backend HTP
```

选择理由与代价：
- **A16W8**：W4A8 / W8A8 质量不足；16 位激活对扩散模型的中间张量是必要的。
- **per-row 而不是 per-channel**：见 §十一。代价只有 DLC +14.5 MB，不改图、不重新转换。
- 🔴 **一旦启用 per-row，就失去 SNPE CPU 参考这条离线校验路径**
  （`error_code=202; Dequantization of axis-quant tensor is not supported for FullyConnected`，§11.3）。
  **在开 per-row 之前先把需要的 CPU 参考跑完并落盘。**
- 要叠加选择性 FP16 时，命令形态不同（`--enable_float_fallback` + overrides + **不给** `--input_list`），
  见 §29.3。

### ④ context binary 生成

```bash
qnn-context-binary-generator \
  --backend <SDK>/lib/x86_64-windows-msvc/QnnHtp.dll \
  --dlc_path <model>_quantized.dlc \
  --binary_file <model>_ctx_<soc> \
  --output_dir <dir> \
  --htp_socs sm8750 \
  --config_file backend_ext.json
```

`backend_ext.json` 必须是**两层结构**（这个格式踩了很久，见 HANDOVER 12.4）：

```json
// 外层：--config_file 指向它
{ "backend_extensions": {
    "shared_library_path": "<SDK>/lib/x86_64-windows-msvc/QnnHtpNetRunExtensions.dll",
    "config_file_path":    "<绝对路径>/backend_config_detail.json" } }
```
```json
// 内层：真正的 devices 配置放这里
{ "devices": [ { "soc_model": 69, "dsp_arch": "v79" } ] }
```
`soc_model` 用 `QnnTypes.h` 的枚举值（SM8750=69、SM8550=43）；`dsp_arch` 是字符串（"v79"/"v73"）。
`cores`/`perf_profile`/`rpc_control_latency` 只对 `qnn-net-run` 有效，写进来会报 Unknown Key。

## 二·补、PyTorch → ONNX 导出（本次未亲自执行，以下为从产物元数据反推）

> **诚实标注**：本文件作者接手时 ONNX 和 DLC 已经存在，**导出脚本在磁盘上已找不到**
> （`find` 全盘搜 `*export*` / `*onnx*.py` 无结果）。以下内容分两类：
> 【实证】= 从 ONNX 文件自带元数据读出；【推断】= 由实证推出但未复现。

### 2.1 【实证】环境版本（`ZImage_QNN_Evidence/environment.txt` + ONNX producer 字段）

```
python 3.12.3 / torch 2.13.0+cpu / diffusers 0.39.0 / transformers 5.14.1
onnx 1.22.0 / onnxruntime 1.28.0
模型来源: Tongyi-MAI/Z-Image-Turbo (HuggingFace)，各文件 sha256 记录在 model_revision.txt
```
ONNX 的 `producer_name='pytorch'`、`producer_version='2.13.0+cpu'` 与之吻合。

### 2.2 【实证】用的是新版 dynamo 导出器，不是 TorchScript tracer

所有 ONNX 的节点 metadata 里都带这些键：
```
pkg.torch.onnx.fx_node        pkg.torch.onnx.stack_trace
pkg.torch.onnx.class_hierarchy pkg.torch.onnx.name_scopes
pkg.onnxscript.rewriter.rule_name        <- onnxscript 重写器
```
`fx_node` / `onnxscript.rewriter` 是 **`torch.onnx.export(..., dynamo=True)`**（FX 路线）
的特征，旧的 TorchScript 导出不会产生。**转新模型时应沿用 dynamo 路线**，否则下游遇到的
算子形态会完全不同，本文件记录的坑未必适用。

### 2.3 【实证】模型分段是用 `onnx.utils.extract_model` 做的，不是重新导出

```
text_encoder_part1..4 :  producer_name = 'onnx.utils.extract_model'
transformer_part1a/1b  :  producer_name = ''（被 opset 修复脚本重写过）
transformer_part1/2    :  producer_name = 'pytorch'（原始导出产物）
```

即流程是：**整模型导出一次 → 用 `onnx.utils.extract_model` 按张量名切分**。
这对 Flux.2 Klein 同样适用——先整体导出，再按内存预算切段，切点选在
**张量数量少、shape 规整**的位置（本项目切在残差流上：`add_2452`/`add_4828`/`add_7204`）。

```python
import onnx.utils
onnx.utils.extract_model(src, dst, input_names=[...], output_names=[...])
```

### 2.4 【实证】opset：导出为 18，qairt-converter 需要降到 17

```
transformer_part1.onnx        opset 18   (原始)
transformer_part1_fixed.onnx  opset 17   (降级后)
transformer_part2_fixed.onnx  opset 17
```
> 🔴 **【2026-09-20 实测订正】"只支持到 16/17"这个说法不准，别据此做整体降级。**
> 实查本项目产物：`text_encoder_part1~4`、`vae_decoder` **全部是 opset 18**，
> 且都已转换成功、已交付跑通。真正需要降的是**特定算子**（下面第 2 条那个单元素 `Split`），
> 不是整个 opset。
> ⇒ **做法**：先按原 opset 转，报错了再按报错的算子处理。
> 无差别降级既浪费时间，又可能引入新问题（§34.2：转换器不报错 ≠ 图是对的）。

`qairt-converter 2.48` 只支持到 opset 16/17。降级脚本与 Split 节点修复详见
`../archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md` 4.1 节（`fix_transformer_opset.py` / `fix_transformer_part2_opset.py`）。

**已知需要修的两类问题**（HANDOVER_DOC 4.1）：
1. opset 18 新特性算子 → 降级改写
2. `torch.split_with_sizes([full_size])`（单元素 split）导出成"单输出 Split 节点"，
   qairt-converter 无法注册其输出 buffer，报 `KeyError: 'split_with_sizes_X_split_0'`

### 2.5 【缺口】没有记录、需要下次补上的

- 导出时的具体 `dynamic_shapes` / 示例输入设置
- text_encoder 取的是哪一层隐藏状态（`hidden_states[-2]`？计划文件里一直标着"未验证"）
- 各段切点是如何选定的（只知道切在哪，不知道选择依据）

**转 Flux.2 Klein 时建议**：把导出脚本**保存进版本库**，并在产物旁写一份
`export_manifest.json`（记录 torch/diffusers 版本、opset、dynamic_shapes、切点张量名、
每个产物的 sha256）。本项目就是因为没做这件事，导出环节无法复现。

## 三、ONNX 侧的前置处理（①图手术）

### 3.1 固定动态 shape（**卡 B 第一步；本节是完整配方，含 5 个通用坑**）

QAIRT 要求 `Slice` 等算子的索引是**图中的 initializer**（常量）。而 dynamo 导出的文本编码器
普遍带 `Shape → Gather → Slice` 的动态链（序列长度依赖），必须先把它固化掉。

> 📌 **实现出处**：`scripts/legacy_export/convert_qairt_zimage.py::freeze_dynamic_shapes`（第 234~389 行）。
> 那是**全项目唯一的实现**，2026-09-20 前只存在于仓库外，现已存档（见该目录 README）。
> 本节是从它提炼的算法与坑，**正常情况下读本节就够，不必回去读代码**。

**七步算法**：

1. **只加载图结构**：`onnx.load(path, load_external_data=False)`
   🔴 **坑 1**：大模型连权重一起加载会爆内存，并撞 **protobuf 2 GB 上限**。
2. **把动态输入维度覆盖成固定值**：写 `dim_value`，并**清空 `dim_param`**
   （`tensor_type.shape.dim[i].dim_param = ""`）。
   🔴 **坑 2**：只写 `dim_value` 不清 `dim_param`，维度仍被当成符号，后面的 shape inference 推不出常量。
3. **`GatherND` → `Reshape` 替换**（QNN 兼容性）：本项目把它换成 `Reshape(data, [1,1,1,-1])`。
4. **找出动态 shape 依赖**：扫描所有节点，凡下列输入**不在 initializer 集合里**的，登记为待求值张量：
   - `Expand` / `Reshape` / `Tile` / `ConstantOfShape` 的 `input[1]`
   - `Slice` 的 `input[1]`（starts）与 `input[2]`（ends）
5. **用 ORT 求值**：把这些张量**临时加成 `graph.output`**，存成探针模型，`InferenceSession` 跑一次。
   🔴🔴 **坑 3（最隐蔽）**：dummy 输入必须用 **`np.ones`，不能用 `np.zeros`**。
   用 zeros 时 `attention_mask` 的 `ReduceSum` = 0 ⇒ 依赖 mask 长度的 `Slice` 算出 **end=0**
   ⇒ 空切片错误。**这个错误发生在求值阶段，报出来的却是切片形状问题，极难往回追。**
6. **注入并切断依赖**：把求得的值做成 INT64 initializer，然后**遍历全图替换所有引用**。
   🔴 **坑 4**：只 append initializer 而不替换引用，等于没做 —— 原来的动态链（`Shape`/`ReduceSum`）
   还在图里、还会被转换器看到。**替换引用才是真正切断依赖的那一步。**
   ⊕ 求值完记得把第 5 步临时加的 output 弹回去（`while len(graph.output) > orig_len: pop()`），
   否则图的输出签名变了，下游契约全部对不上。
7. **形状传播**：`onnx.shape_inference.infer_shapes_path(..., data_prop=True)`。
   🔴 **坑 5**：`data_prop=True` + `check_type=True` 在某些图上会抛异常。
   **要有回退**：捕获后改用 `check_type=False, data_prop=False` 再跑一次基础推断
   —— 但**回退成功不等于固化成功**，回退后必须回到门 B2 重新确认没有残留动态索引。

**验收（门 B1）**：`_fixed.onnx` 与原图在 FP32 下**逐位相同**（ORT，约 1 分钟）；
且转换器不再报 `Slice` 索引非常量一类的错误。
⚠️ **"转换器不报错"不等于图是对的**（§34.2）—— 逐位比对才是判据。

### 3.2 【重要·通用】把注意力掩码的 -inf 常量换成温和负值

**这是本项目踩到的最隐蔽的坑，任何带 attention mask 的模型都会遇到。**

PyTorch 导出 ONNX 时，掩码的"负无穷"通常写成 **float32 最小值 `-3.4028235e+38`**。
量化器会把它当普通张量校准，得到：

```
val_104:  range=[-3.40282e+38, 0]   scale=5.19e+33     <- 退化
```

后果：与下游张量的 scale 比值达 **3.4e42**，HTP 编译直接报
`scale too large for requant qu16->qu16: inf`。本项目实测：全流水线 4,940 个张量里
**只有这一个**退化，但它经 `Where` → 加性掩码 → 污染了 part2 **全部 15 个注意力块**
（16 条极端边，见 HANDOVER 14.13）。

**做法（第一步，必要但【不充分】）**：转换前把该常量改成 `-100.0` 之类的温和负值。
`exp(-100)` 在 float32 下即为 0，掩码语义完全等价，而 scale 从 5.19e33 降到 1.53e-03。

```python
for t in model.graph.initializer:
    if t.name == "<mask_const>":           # 用 snpe-dlc-info 找 range 含 ±FLT_MAX 的张量
        t.CopyFrom(numpy_helper.from_array(np.array(-100.0, np.float32), name=t.name))
```

#### 3.2.1 【实测·2026-08-14】只改常量**不够**，必须同时看 `Where` 的**输出** encoding

本项目做完上面这一步后，**同一个 OpID 仍然报错**，只是数值从 `inf` 变成 `1000000.0`：

```
原始    linearclip.h:248 ... requant qu16->qu16: inf ;              OpID: 0x6994000000a2
maskfix linearclip.h:248 ... requant qu16->qu16: 1000000.000000 ;   OpID: 0x6994000000a2   <- 同一个 OpID
```

原因：改的是**输入**常量，而报错的比值是 **`scale(掩码常量) / scale(Where 输出)`**。
量化器给 `Where` 输出定的区间是从"非掩码"分支继承来的：

```
输入  val_103  range=[0, 1e-4]     <- 非掩码分支（≈0）
输入  val_104  range=[-100, 0]     <- 掩码分支（已改成 -100）
输出  val_105  range=[0, 1e-4]     <- 【病根】offset=0 的无符号 u16，一个负数都表示不了
```

`(100/65535) / (1e-4/65535)` 正好 = **1e6**，与日志逐位吻合。

> **通用教训**：`Where(mask, 0, -BIG)` 这种加性掩码，量化器很容易只按**多数分支**
> （全是 ≈0）来定输出区间，把 `-BIG` 那一侧**整个排除在表示范围之外**。
> 掩码分支越"稀有"，越容易发生。**换任何新模型都要单独检查这个输出张量。**

**判定式**（一句话）：`min(Where 输出) <= min(掩码分支)` 必须成立，否则掩码会被钳掉。
本项目实测对照：

| 图 | 掩码分支 min | 输出 min | 判定 | 编译 |
|---|---|---|---|---|
| part1a `node_where_1` | -1.68 | -18.54 | ✅ 可表示 | 干净 |
| part2 `node_Where_105` | -100 | **0** | ❌ 被钳到 0 | **报错** |

真正的修法有两条（本项目**尚未验证**哪条有效，因为该路径在当前 prompt 下是死的）：
① 让校准数据**真的覆盖掩码分支**（`unified_mask` 里必须有 0），
   这样量化器自然会把输出区间开到负侧——这也是 §4.2「校准必须覆盖所有分支」的又一实例；
② 用 `--quantization_overrides` 手工指定该输出张量的 encoding。

#### 3.2.2 检查方法（**不要**再用 `scan_requant_ratios.py`）

```bash
python scripts/map_opid.py          # 解析 snpe-dlc-info 转储，算子内全部张量两两取比值
```

`scripts/scan_requant_ratios.py` 有两个会**静默漏报**的盲区，本项目已因此漏掉一整轮：

1. 正则不允许张量名含点，`layers.*.weight` 这类全部漏掉；
2. **更致命**：它只算「激活输入 → 输出」并显式排除 `STATIC`。
   而掩码常量正是 `STATIC`，`Eltwise` 两个操作数之间的对齐（input↔input）它也从不算。
   ⇒ **它恰好扫不到本节这个坑。**

`map_opid.py` 对算子内全部张量两两取比值，并**按 float32 复算**
（超过 3.4028235e38 即 inf，与 SDK 内部表示一致——原始那个 3.4e42 就是这样变成 `inf` 的）。

> **转储精度陷阱**：`snpe-dlc-info` 的 scale 只打印 12 位小数。
> 直接拿打印值做比值会得到 999936 而不是整 1e6，看起来"对不上"。
> 要做等式检验就用精确值（`100/65535`、`1e-4/65535`）重算。

#### 3.2.3 【实测】这个报错**不阻断构建**，别把它当致命错误

`qnn-context-binary-generator` 打印该 `[ERROR]` 后仍然 `EXIT=0`，
`.bin` 正常产出（本项目 2.9 GB）。它是否影响数值输出**尚未验证**。
所以：**看到它要查，但不要因为它就认定当前的失败由它引起**——
本项目实测该路径在当前 prompt 下压根没被走到（`unified_mask` 全为 1，
掩码分支一次都没被选中），它是一个真实但**未被触发**的潜在缺陷。

### 3.3 模型分段（**按 PD 红线反推**，不是按手机内存）

Z-Image Turbo 被拆成 8 段（text_encoder ×4、transformer part1a/1b/2、vae_decoder）。

🔴 **【2026-09-20 订正·分段依据】** 本节原写「拆分动机是设备内存……实测进程涨到 7.3 GB 即被杀，
设备侧按阶段加载/释放」。**这个依据用错了口径，做法也已被推翻**：

| 原文说法 | 实测订正 | 出处 |
|---|---|---|
| 依据 = 手机内存 / 进程涨到 7.3 GB 被杀 | 🔴 **真正的硬约束是 DSP 侧 unsigned PD 容量**。失败时 `lowmemorykiller` 同时报 *device has enough memory 8047228Kib* ⇒ **不是内存不足** | §12.1 |
| 用**进程内存**估算 | 🔴 **进程 `VmRSS` 看不见 context**（后端全程仅 ~200 MiB；context 常驻 DSP/ION 侧）。要看 `/proc/meminfo` 的 **`IonTotalUsed`** | §37.1/37.2 |
| 设备侧**按阶段加载/释放** | 🟢 已改为**常驻**：把 `loadGraph` 提出步循环 −26.9%，四段常驻再 −21.2% | §三十九 |

**当前的正确做法**：

1. **切口按 PD 红线反推**：单个 context 的 `contextBlob + spillFill + opData` 合计
   **≤ 3.3 GB 留余量**，超过 **3.5 GB 就在悬崖边**（SM8750/v79 实测上限 3506~3535 MB，
   PD 估算口径 3.62~3.65 GB，§12.2）。⚠️ **红线是芯片相关的，换 SoC 必须重测。**
2. **内存预算用 ③ 口径**：设备实际占用 ≈ **context 文件字节 × 1.42**（实测倍率，§37.1）。
   用文件大小估会**低估约 40%**。
3. **切口选在张量少、字节小的位置**（本项目 part2 切口：6 张量 / 65.6 MB），
   并且**尽量切在段间 encoding 一致处** —— 不一致的话每步要 CPU 逐元素重量化，
   实测 `prep` 从 14 ms 涨到约 690 ms（§47.4 第 5 条）。
4. **切完必须做 DCE**（死代码消除），否则子图会拖着整个原图的 initializer，§十三。
5. 🔴 **别指望用 external weights / spill-fill 绕开 PD 上限** —— 实测四种组合全部失败（§十四）。

⊕ 容量不够时的正解是**继续拆段**或**多图共享权重**（§四十二），不是压缩 context。

## 四、校准数据（最容易被忽略、且会静默劣化质量的一环）

### 4.1 格式
- 每个输入一个 `.raw` 文件，**float32**，`input_list.txt` 每行一个样本：
  `name1:=path1 name2:=path2 ...`
- 本项目用 40 个样本；part2 每样本约 4.5 分钟（CPU 全图前向），总计约 3 小时。

### 4.2 【重要】校准数据必须覆盖所有分支

本项目的实证教训：校准样本的 `cap_pad_mask` **全为 0**，导致：
- 掩码分支从未激活 → `val_105`（Where 的输出）被校准成 `[0, 0.0001]`，scale 1.526e-09
- 这个畸形 scale 又污染了 15 条下游边

**换新模型时**：确认校准数据里各类条件分支（掩码激活/不激活、不同序列长度、
不同 timestep 区间）都有覆盖。只用"典型样本"会让量化器看不到真实动态范围。

### 4.3 timestep 覆盖
本项目 `timestep` 的校准范围是 [0, 0.7]，与 8 步调度器实际取值范围精确吻合
（`1 - sigma`，sigma 从 1.0 降到 0.3）。换调度器/步数时要重新确认。

## 五、验证方法（这是本项目最有价值的产出之一）

**不要只看"没报错"，也不要只看中间张量误差。** 建立三级验证：

### 级别 1：转换保真度（离线，分钟级）
FP32 DLC vs FP32 ONNX 逐张量比对，相对误差应在浮点噪声量级（本项目实测 0.0074%）。

### 级别 2：量化后成图（离线，小时级）——**最关键**
搭一个**混合流水线**：Python 参考实现跑调度器/VAE，只把待验证的段换成量化 DLC
（用 `snpe-net-run` 跑），同 prompt 同 seed 出图，与纯 FP32 基准做像素级对比。

参考实现：`scripts/testB_hybrid_pipeline.py`
本项目实测：量化 transformer → PSNR 23.89 dB；再叠加量化 VAE → PSNR 51.11 dB。

> **为什么这一级不可省**：本项目曾用"中间张量误差 26.9%"判断量化有问题，
> 花了整整一轮做微观定位，最后被本级验证否定——**中间张量误差大 ≠ 成图坏**。
> 实测标定点：噪声预测的 L2 相对误差达 **15.988%**，成图依然完好。

> 🔴🔴 **【2026-09-20 订正·判据】上面那个 "PSNR 23.89 dB" 是在量"一致性"，不是"画质"。
> 别把它当验收门用来在配置之间排序 —— §十七 实测 PSNR 会指向相反方向。**
>
> | 你要回答的问题 | 用什么尺子 | 不能用什么 |
> |---|---|---|
> | **一致性**：量化输出离 FP32 参考多远 | PSNR / 相对 L2 / 余弦 ✅（本节这个数就是它） | — |
> | **画质**：图本身好不好 | **只能人看**（打开图片，§36.3 的 8 个失效模式探针） | 🔴 PSNR / 像素差 / 逐点 L2 |
>
> 本项目实测这两者**反相关**：单变量对照中，出「马赛克色团」的配置 A 三项指标全面优于
> 出「清晰的猫」的配置 B（PSNR 12.46 vs 11.60 dB）——**只看数字会把唯一成功的配置判成倒退并放弃**（§17.1）。
> 机制：量化误差不把图推离数据流形，而是**沿流形移到另一个合理样本**（§36.2）。
>
> ⇒ **级别 2 的正确用法**：① 出图后**必须人打开看**（CLAUDE.md 约束 1 明令，不许转述形容词）；
> ② PSNR 只用于**同一配置内的回归监控**（确认改动没把模型改坏），**不得跨配置排序**；
> ③ 要自动化就用结构/感知类判据（可辨识主体、CLIP 图文相似度、FID），不要标量像素指标。
>
> ⚠️ 顺带：本节的"级别 1 / 2 / 2.5 / 3"是**验证阶段**；另有一套 L1/L2/L3 是**指标层**
> （L1 中间张量 / L2 终点 latents / L3 成图 PSNR，HANDOVER §15.41）。两套命名不相干，别混。
> 已知 **L1 不能换算成 L3**（#121）。

> 🔴 **【2026-08-15 重大更正】级别 2 跑在 SNPE CPU 参考上，【远远不够】。**
> 本项目实测：**同一份量化 DLC，在 CPU 参考上出清晰的猫（PSNR 23.89 dB），
> 编成 `.bin` 在设备 HTP 上跑出橙色色块（PSNR 12.06 dB）。**
> 级别 2 全绿**不能**说明模型在设备上可用。**必须做下面的级别 2.5。**

### 级别 2.5：设备 HTP 上的 `.bin` 逐张量 / 在环验证 —— **不可省，且必须在交付前做**

这是本项目**最大的教训**：转换、量化、级别 1、级别 2 全部通过，
`qnn-context-binary-generator` 也 `EXIT=0`，**但设备上就是出色块**。
差异只在最后一环——**HTP 执行 `.bin`**，而它此前从未被验证过。

**2.5a 逐张量比对**（每段几十秒，最先做）

设备上不需要装 app，用 `qnn-net-run` 直接跑 `.bin`：

```bash
# 设备 /data/local/tmp/x 里放：qnn-net-run + libQnnHtp.so + libQnnHtpV<N>Stub.so
#   + libQnnSystem.so + libQnnHtpPrepare.so + libQnnHtpNetRunExtensions.so
#   + libQnnHtpV<N>Skel.so（来自 lib/hexagon-v<N>/unsigned/，N 由 SoC 决定）
export LD_LIBRARY_PATH=/data/local/tmp/x
export ADSP_LIBRARY_PATH=/data/local/tmp/x
./qnn-net-run --retrieve_context part.bin --backend libQnnHtp.so \
              --input_list in/list.txt --output_dir out --log_level error
```

把输出与级别 2 用的 SNPE CPU 输出逐张量比（相对 L2 / 余弦 / 分位数）。

三个必须遵守的点：

1. 🔴🔴 **`--use_native_*` 标志必须与你写进 `.raw` 的实际 dtype 一致 —— 这是一条双向规则，
   两个方向都错过**（2026-09-18 订正；本节原文只写了其中一半，见下）：

   | 你的 `.raw` 里写的是 | 标志 | 不这么做的后果 |
   |---|---|---|
   | **float32**（本节的做法，与 SNPE 侧同口径便于相减） | **不加** | 加了标志 ⇒ 按原生宽度解析 float32 字节 ⇒ 元素数错 |
   | **原生字节**（`uFxp_16` 等） | **必须显式加两个** | 不加 ⇒ 按「字节数 ÷ 期望 = 0.5」**静默缩小张量**，`rc=0` 无报错 |

   🔴 **后一种情况会骗过字节守卫**：对 `uFxp_16` 张量，原生字节数 ≡ float32 的一半，
   写出的 float32 输出**恰好等于契约里的原生字节数** ⇒ 「输出字节数 == 契约」这条守卫报 **PASS**。
   实测三臂见 §45.8（代价：数小时 + 误判方向，一条历史测速数据的绝对值因此作废）。

   **怎么知道自己处在哪种模式**：拿一个 `Bool_8` 或 `Int_32` 张量当**模式指示器** ——
   它们的原生字节数不等于 float32 的一半（分别是 ×4、×1），一眼能看出来。

   > <details><summary>原文（2026-08-15，只覆盖了 float32 那一半）</summary>
   >
   > 「**不要**加 `--use_native_input_files` / `--use_native_output_files`。
   > 非 native 模式下 qnn-net-run 按 float32 读写，与 SNPE 侧同口径，两边才能直接相减。
   > （实测：喂 128 字节 float32 给一个 `BOOL_8[1,32]` 张量，工具会自行转换，正常运行。）」
   >
   > **为什么要订正**：这句话在它自己的语境（输入写 float32）里是对的，
   > 但被写成了无条件的祈使句。§45.8 后来在"输入写原生字节"的场景下得出**相反**的指令，
   > 两处直接对立。真正的规则是上表：**标志随 `.raw` 的实际 dtype 走**。
   > ⊕ 这本身是 §47.10 模式 4（判据/操作化没被验证）的又一例：
   > 一条只在某个条件下成立的规则，被写成了绝对规则。
   > </details>
2. **必须先比 `.bin` 的 md5**：主机副本 vs 设备 app 里那份，确认在验同一个模型。
   app 私有目录 `/data/data/<pkg>/` 是 `drwx------`，只 chmod 子目录没用；
   正确做法是把主机上同 md5 的 `.bin` 推到 `/data/local/tmp`。
3. **必须做隔离与级联两组**。隔离＝两侧都喂 CPU 参考的上游输出（测该段自身差异）；
   级联＝设备侧全程用自己的上游输出（测真实累积）。只做级联无法归因。

**2.5b 在环出图**（约 10 分钟，决定性）

把级别 2 的混合流水线里的 `snpe-net-run` 换成上面的设备调用，**其余一个字节不改**，
同 prompt 同 seed 出图。参考实现：`scripts/htp_inloop_pipeline.py`。
段间中间张量**全程留在设备上**，每步只推/拉 `latents`（各 1 MB），否则每步要多搬 130 MB。

本项目实测阶梯（同一基准 `zimage_fp32_full_pipeline.png`）：

| 配置 | 平均\|像素差\| | PSNR | 图像 |
|---|---|---|---|
| CPU 参考跑量化 transformer | 9.32 | 23.89 dB | 清晰的猫 |
| **设备 HTP 跑同一模型** | **46.73** | **12.06 dB** | **橙色色块** |
| 设备 app 真机 | 46.40 | 11.56 dB | 橙色色块 |

**成本参考**：8 步 × 3 段 = 24 次 `qnn-net-run`，单次含 1.5~2.9 GB context 加载，
part1a 28s / part1b 12s / part2 26s，全程约 9 分钟。**这个代价必须付。**

> **【实测·2026-08-15】排查 HTP 数值问题时可以跳过"`.bin` 是不是编坏了"这一层。**
> QNN 的 graph prepare **确定性且平台无关**：同一份 `.dlc`，
> ① 在 x86 宿主用 `qnn-context-binary-generator` 离线编成 `.bin` 再执行，
> ② 在 aarch64 设备上用 `qnn-net-run --dlc_path` 在线建图再执行，
> 两者输出 **逐位相同**（本项目 part1b 实测：相对 L2 = 0.0000%，最大差 = 0）。
>
> 两个推论：
> - 数值不对时**不必怀疑离线 prepare**，直接查 HTP 执行语义或量化配置。
> - 离线 `.bin` 是**纯粹的加载提速**，不改数值：本项目实测 16m26s（在线建图）
>   → 12s（加载 `.bin`），**快 80 倍**。
>
> 验证这一点的方法（也是排除法的标准动作）：把 `.dlc` 推到设备，
> `qnn-net-run --dlc_path x.dlc --backend libQnnHtp.so`（需 `libQnnModelDlc.so`），
> 与 `.bin` 路径同输入比对。注意区分两条路径的日志特征——
> `.bin` 打印 `Creating context from binary file`，`.dlc` 打印 `Composing Graphs`/`Finalizing Graphs`；
> 若看到 `Failed in cacheSelection`，说明**没有**命中缓存，正是我们要的干净对照。

### 级别 3：设备真机 app（需要手机）
零成本诊断：C++ 侧 `logStats()` 已经会打印每步速度场的 min/max/mean/std，
抓 native logcat 与级别 2 的参考值对照即可（注意默认 logcat 抓取可能过滤掉 native 日志）。

> 注意级别 2.5 与级别 3 的分工：2.5 **绕开 app**，只测 HTP 执行本身；
> 3 测的是完整 app。本项目正是靠 2.5 才把根因从 C++ 实现身上摘出来——
> 2.5 里根本没用到 C++，照样复现了真机失败。

## 六、宿主侧数据接口（喂错不报错，务必按 `EXECUTION_MODEL.md`）

最危险的一条：**所有输入 `.raw` 一律写 float32**，与 DLC 内部声明的
`Bool_8`/`uFxp_16`/`Int_32` 无关。按声明宽度写会导致 snpe 用
「输入字节数 ÷ 期望字节数」推断批次并**静默缩小整张图**（实测：32/128 → 所有输出缩到 1/4，
全程零报错）。

统一走 `scripts/snpe_runner.py`，它强制四项检查。详见 `EXECUTION_MODEL.md`。

> 🔴 **【适用范围，2026-09-20 补】上面这条只管 `snpe-net-run` 跑 `.dlc`。
> 换成 `qnn-net-run` 跑 `.bin` 时规则不同**，且两边的默认行为看起来像、后果却相反：
>
> | 工具 | 默认解析 | 规则 |
> |---|---|---|
> | `snpe-net-run`（SNPE，`.dlc`） | 按 DLC 声明宽度推断批次 | **`.raw` 一律写 float32**（本节） |
> | `qnn-net-run`（QNN，`.bin`） | 按 **float32** 读写 | **标志随 `.raw` 实际 dtype 走**：写 float32 ⇒ 不加标志；写原生字节 ⇒ **必须**加 `--use_native_input_files --use_native_output_files`（§45.8 / §五 2.5a） |
>
> 两边都会**静默缩小张量且 `rc=0`**，而且 `qnn-net-run` 那条**连字节守卫都骗得过**
> （`uFxp_16` 的原生字节数恰好是 float32 的一半）。
> ⇒ **跨工具对照时，先确认两边处在同一种解析模式**，用 `Bool_8` / `Int_32` 张量当模式指示器。

## 七、成本与资源预算（实测，用于排期）

| 阶段 | 耗时 | 峰值内存 | 产物大小 |
|---|---|---|---|
| 转换 part2（最大段） | ≈ 2 分钟 | — | 10.86 GB |
| 量化 part2（40 样本，**带校准**） | ≈ 3 小时 | ≈ 17.6 GB | 2.72 GB |
| 量化（**带 `--quantization_overrides`，无需校准数据**） | **≈ 0.1 分钟** | 低 | — |
| context binary part1a | Graph Sequencing 单阶段约 3 小时 | — | 2.37 GB |
| 混合流水线出图（8 步） | ≈ 1.5 小时（24 次 snpe 调用） | ≈ 17 GB | — |

本机总内存 23.7 GB，**量化与 context 生成不能并行**，要排队（实测建图峰值 **17.8 / 23.7 GB**）。

> ⚠️ **【2026-09-20 标注】表中 "context binary part1a ≈ 3 小时" 与 §47.1 阶段 5 的
> "15~25 分钟/段；五图 mg 约 2.7 小时" 不一致，且本文档没有留下可判定的对照实测。**
> 两者的**适用条件不同但未被记录**（本表是 2026-08 的早期 part1a：per-tensor、含两个
> 全尺寸 `ScatterElements`、未做 DCE；§47.1 是交付期数据）。
> ⇒ **排期时按 §47.1 估，但对最大的那一段留出数小时的余量**；
> 首次建图时**实测记录一次**，再回来订正本表（③ 未验证：两者差异是否由上述条件差异造成）。

**【实测】排查算子级编译错误时，迭代成本不是整轮构建时间。**
`Graph Optimizations` 阶段的算子错误在第 **310 秒（约 5.2 分钟）** 就会打印，
而完整构建约 34 分钟（part1a 的 `Graph Sequencing` 更是数小时——但那与算子错误的
出现时机无关）。⇒ 复现这类错误**跑到报错出现即可终止**，迭代成本 5 分钟而非 34 分钟。

**更进一步：很多这类问题根本不需要重跑构建。**
`snpe-dlc-info` 的转储里已经包含全部 encoding，比值类问题可以**纯离线复算**
（本项目定位 `OpID: 0x6994000000a2` 就是零构建完成的，见 §3.2.1）。
**先离线算，再花 5 分钟构建，最后才考虑二分裁剪** —— 顺序不要反。

## 八、转换 / 量化前的图体检清单

> 🔴 **【2026-09-20 归位】本文档里有三份清单，各管一段，别互相替代**
> （它们内容重叠但**不等价**，早期读者会不知道该用哪份）：
>
> | 清单 | 管哪一段 | 给谁 |
> |---|---|---|
> | **§47.6 第一天清单**（6 条） | **总入口**：环境、红线重测、overrides 留档 | 刚接手的人，**先读它** |
> | **§47.7 导出硬性要求**（8 条） | **阶段 1**：ONNX 导出时就要满足 | 做导出的人 |
> | **本节**（第 0 项 + 7 条） | **阶段 2~4**：拿到 ONNX 后、转换/量化前的图体检 | 做转换的人 |
>
> ⚠️ 三份都过，才等于 §47.1 阶段 1~4 的门全过。本节不覆盖建图/设备/交付（阶段 5~9），那些在 §47.1。

> 🔴 **第 0 项（2026-08-21 新增，一次踩坑代价：误判 2 天 + 产品长期出劣化图）**
>
> **动手喂任何数据之前，先跑：**
> ```bash
> python scripts/check_graph_io_convention.py <onnx目录>
> ```
> 它 dump 每个图的头尾节点，标出**带常量的逐元素算子**——那些是被烘焙进图里的前/后处理，
> **宿主侧绝对不能再做一遍**。
>
> **本项目扫描 24 个图的实测结果**：
> - 🔴 `vae_decoder.onnx` 头节点 `Div(vae_latents, 0.3611)` ⇒ **真 bug**，宿主与 app 都又做了一遍
> - ✅ `transformer_part1*.onnx` 头节点 `Mul(timestep, 1000)` ⇒ **查过，是对的**：
>   图期望归一化值，宿主 `1-sigma`、app `1-t/1000`、官方 `(1000-t)/1000` 三者一致
> - ⚪ `Pow 2.0` / `Add 1e-5` / `Add 1.0` ⇒ **假阳性**：从网络中间切开的分段头部露出的 RmsNorm 内部
>
> **配套的强制验证（约束 8 的最便宜形式）**：
> **第一个算子的输出**——两侧权重与输入都逐位相同 ⇒ **输出必须逐位相同**。
> 本项目正是靠它抓到根因：实测 std 之比 **2.7595**，而 `1/0.3611 = 2.7693`。
>
> **还要比对**：校准数据的分布 vs 推理时实际喂入的分布（std / min / max）。
> 两者由不同代码产生时必然会漂，本项目就漂了 2.77 倍，输入被量化器硬钳。


1. [ ] 用 `snpe-dlc-info` 找出 range 含 ±FLT_MAX 或 scale 异常的张量（掩码常量陷阱）
2. [ ] 跑 `map_opid.py`，确认没有 >1e4 的边
      （**不要**用 `scan_requant_ratios.py`，它扫不到 STATIC↔输出 这类边，见 §3.2.2）
2b.[ ] 对每个 `Where`/`Select` 单独验 `min(输出) <= min(掩码分支)`，见 §3.2.1。
      这条**光看"有没有 ±FLT_MAX 张量"是查不出来的**——常量改好了它照样成立
3. [ ] 确认校准数据覆盖所有条件分支（尤其掩码激活的情况）
4. [ ] 确认 timestep/sigma 校准范围与实际调度器一致
5. [ ] 用 `dlc_contracts.py` 生成契约，确认 DLC 的输入集合与 ONNX 的差异（会有输入被折叠掉）
6. [ ] 做级别 2 验证（混合流水线出图），不要跳过。
      🔴 **出图必须人打开看**；PSNR 只量"一致性"，不得用来跨配置排序（§五 级别 2 的订正框、§十七）
7. [ ] 估算单段 context 的设备常驻内存，决定分段方式。
      🔴 **按 PD 红线（≤3.3 GB/context）反推，不是按手机内存**；
      内存 = 文件字节 **× 1.42**，且**进程 RSS 看不见 context**（见 §3.3 订正、§12.2、§37.1）
8. [ ] **确定量化配方**：DiT/Transformer 起手式 = A16W8 + `--use_per_row_quantization`
      （**不是** `--use_per_channel_quantization`，后者零 Conv 模型上空转，§11.1）。
      🔴 开 per-row 前**先把需要的 SNPE CPU 参考跑完落盘**——开了之后这条离线路径就没了（§11.3）
9. [ ] **第一次量化就留下 `--quantization_overrides` JSON**：后续重量化 3 小时 → 0.1 分钟（§35.1、§47.6）

## 九、尚未验证 / 已知未解决

1. ~~**`qnn-context-binary-generator` 产物的正确性**：CPU 参考路径（`.dlc`）与 HTP（`.bin`）
   是否数值等价，**从未验证**。本项目当前的失败很可能就在这一段。~~
   🔴 **【2026-08-15 已验证，且是根因】**：两者**不等价**，且不等价的程度足以毁图。
   实测（`EXP_PLAN_HTP_VS_CPU*`、`EXP_PLAN_HTP_INLOOP`）：同输入下逐段隔离相对 L2
   part1a **2.35%** / part1b **19.47%** / part2 **43.04%**，级联噪声预测 **47.07%**；
   在环出图 PSNR **12.06 dB**（CPU 参考同模型是 23.89 dB）。
   HTP 侧**逐位确定性**，是系统性偏差不是运行时噪声。
   ⇒ **换任何新模型都必须做级别 2.5（见 §五），不能拿 CPU 参考的结果当交付依据。**
   **"为什么 HTP 会偏"仍未查明**——候选线索：全部 8 图构建时都打印过
   `<E> Unsupported HTP Arch 1 79` 却仍 `EXIT=0`，该告警的含义**尚未验证**。
2. ~~**`--quantization_overrides` 通道**：`version="2.0.0"` 会触发 SDK 原生崩溃；
   用 `"0.6.1"` 可绕开，但产出的 DLC 在 HTP 编译时会撞 `node_MatMul_333` 校验失败
   （根因未查明）。详见 HANDOVER 13.x。**换新模型若需要 override，先做小规模验证。**~~
   🟢 **【2026-08-15 已打通，本条作废】**（§15.15）：`version="0.6.1"` + **ONNX 张量名**
   ⇒ `Processed N quantization encodings` → `INFO_CONVERSION_SUCCESS`，约 60 秒。
   当时"撞 `node_MatMul_333`"的真因是**用了 DLC 内部名**（`linear_99_fc`）⇒ 静默失效，
   不是通道本身坏。完整配方见 **§十**，必检项见 §10.3。
   ⇒ 🔴 **反过来说：overrides 现在是必经之路**（选择性 FP16、per-axis 注入都只能走它），
   §47.6 第 4 条要求**第一次量化就留下 overrides JSON**（后续重量化 3 小时 → 0.1 分钟）。
3. **AIMET 量化路线**（`--use_aimet_quantizer`）：环境已装好（WSL），**从未真正使用过**。
   ⊕ 截至交付仍未用过；本项目的精度改善全部由 per-row + 选择性 FP16 取得。
4. **`val_105` 一类"校准未覆盖分支"造成的畸形 scale**：目前只能靠改校准数据或 override 解决，
   本项目尚未验证哪种有效。
   ⊕ 补充：该路径在本项目的实际 prompt 下**从未被走到**（`unified_mask` 全 1），
   所以它是"真实但未被触发"的潜在缺陷（§3.2.3）。换模型时若掩码分支会被激活，这条会立刻变成阻塞项。
5. 🔴 **HTP 与 CPU 参考不等价的机制**（第 1 条的后半段）：现象已充分实测，**机制至今未查明**。
   已排除：离线 prepare（`.bin` 与在线建图 `.dlc` 逐位相同）、C++ 实现（级别 2.5 绕开 app 照样复现）。
   ⊕ `<E> Unsupported HTP Arch 1 79` 这条候选线索至今**仍未验证**含义。
6. 🔴 **与官方实现的 15.52% 差距**：根因已定位到 padding **个数**（§四十四），
   但**修复方案未实施**（用户指示挂起）。换模型时若也要做变长序列手术，先读 §四十四再动手。


---

## 十、混合精度 / 逐算子量化控制（**转 Flux.2 Klein 等新模型时大概率会用到**）

> 来源：Z-Image Turbo 排查过程的实测（2026-08-15），HANDOVER §15.14~15.15。
> 这一节是**流程性结论**，与具体模型无关。

### 10.1 唯一入口：`qairt-converter --quantization_overrides`

想让**某些算子/张量**用不同精度（FP16、对称权重、不同位宽），**只有这一条路**。

**三条常见的错路，都已实测排除：**

| 尝试 | 结果 |
|---|---|
| `--enable_float_fallback` 单独用 | ❌ 运行时报错，**必须配合 overrides 或 QAT encodings** |
| `--param_quantizer_schema symmetric` | ⚠️ **全局生效**，会波及所有算子；若某个算子的 OpDef 不接受该数据类型（如 RmsNorm 不收 `SFxp8`），**整张图编不出来** |
| `--restrict_quantization_steps` | ⚠️ 只对 **A16W16** 有效；A16W8 下**静默无效果**（852 个 encoding 一个没变） |

### 10.2 overrides 文件格式（实测可用）

```json
{
  "activation_encodings": {
    "<ONNX张量名>": [{"bitwidth": 16, "dtype": "float"}]
  },
  "param_encodings": {},
  "version": "0.6.1"
}
```

- **`version` 必须是 `"0.6.1"`**。本项目实测 `"2.0.0"` 会触发 SDK 原生崩溃。
- 只做精度切换时**只需点名张量**，不必提供 scale/offset（官方 `converters.html`
  → *Float mixed precision conversion*）。

### 10.3 🔴 必检项：`Processed N quantization encodings`

**张量名写错时转换器不报错**，只是日志里打 `Processed 0 quantization encodings`，
然后 `INFO_CONVERSION_SUCCESS`、EXIT=0 —— **静默失效**。

```bash
grep "Processed .* quantization encodings" converter.log
```
**N == 0 一定是名字写错了。** 与 `snpe-net-run` 静默缩批（`EXECUTION_MODEL.md` 规则 1）
属同一类陷阱：工具在输入不合法时不失败，而是安静地做一件别的事。

#### 10.3 补 ⚠️ **N 不等于点名数**（2026-08-15 实测订正，别照搬旧写法）

本指南先前写"N 必须等于点名张量数"，**只对 `activation_encodings` 成立**。
实测 `param_encodings`（权重）时 N 会成倍放大：

| 点名 | 其中 Gemm 权重 | 日志 N |
|---|---|---|
| 2 个 | 1 个 | **5** |
| 59 个 | 7 个 | **125** |

拟合：`N = 2 × 点名数 + Gemm 权重数`（59×2+7 = 125，2×2+1 = 5）。
即**每个权重张量算 2 次，经 `_permute` 的 Gemm 权重再多算 1 次**。

⇒ **可靠的检查不是数日志，而是数产物**：转换后 `snpe-dlc-info` 里
`encoding : bitwidth 8` 的行数必须等于点名数：

```bash
grep -cE "encoding : bitwidth 8" <converted>_dlcinfo.txt     # 必须 == 点名数
grep -oE "encoding : bitwidth 8.*offset [-0-9.]+" ... | grep -oE "offset [-0-9.]+" | sort | uniq -c
```
本项目实测：59 个点名 → **59 行、offset 全为 −128**（AIMET 对称写法）。
日志的 N 仍要看（**0 = 名字全错**），但**不能**用 `N == 点名数` 当判据。

### 10.4 名字从哪来：**ONNX 名，不是 DLC 名**

DLC 会给张量加后缀（ONNX `linear_99` → DLC `linear_99_fc`）。
overrides 认的是 **ONNX 名**。取法：

```python
import onnx
m = onnx.load(path, load_external_data=False)   # 不必载权重，秒级
for n in m.graph.node:
    print(n.op_type, n.name, list(n.output))
```

### 10.5 改精度前必须先查 OpDef 支持矩阵

不是所有算子都接受所有数据类型组合。本项目实测（`HtpOpDefSupplement.html`，UFxp16 激活）：

| 算子 | 权重（in[1]）允许的类型 | SFxp8（对称 8-bit）? |
|---|---|---|
| FullyConnected | SFxp16 / **SFxp8** / UFxp16 / UFxp8 | ✅ 收 |
| MatMul | SFxp16 / **SFxp8** / UFxp16 / UFxp8 | ✅ 收 |
| **RmsNorm** | SFxp16 / UFxp16 / UFxp8 | ❌ **不收** |

⇒ 全局 `--param_quantizer_schema symmetric` 会把 RmsNorm 的 gamma 也变成 `SFxp8`，
**直接编不出图**（`QnnBackend_validateOpConfig failed 3110`）；
而"只对 FC 用对称"在 OpDef 层面是合法的。

用 `scripts/check_opdef_support.py` 先查，**不要靠试错**——
一次失败的量化 + 建图是 1~2 小时，查表是 2 分钟。

### 10.6 🔴 用 overrides 时**必须显式写 `--float_bitwidth 32`**（否则整张图被降成 FP16）

**①实测（2026-08-15）**，同一份 `transformer_part1b.onnx`、同一条命令，只差这一个参数：

| 命令 | 产物 DLC 大小 | 张量数据类型 |
|---|---|---|
| 加 `--quantization_overrides`，**不写** `--float_bitwidth` | 2.77 GB | **1607 个 Float_16 / 0 个 Float_32** |
| 加 `--quantization_overrides` + **`--float_bitwidth 32`** | 5.54 GB | **1607 个 Float_32 / 0 个 Float_16** |
| 基线（无 overrides） | 5.54 GB | 全 Float_32 |

官方 `converters.html` 原文（**连同适用范围引**）：
> Float bitwidth 32 is the default bitwidth for float source model conversion.
> **Float bitwidth 16 is the default bitwidth for source model with quantization
> encodings or overrides.**

⚠️ **最坑的一点**：DLC 里内嵌的命令行记录**仍写着 `float_bitwidth=32`**
（那是 argparse 的默认值，不是实际行为）—— **记录值不能当证据，必须去数张量类型**。

> 🔴 **【2026-09-21 补】要复现/验证这条，样本必须含「未被 overrides 覆盖的浮点张量」。**
> 本轮试图用单算子探针 `fc99`（3 个张量，**全部被 overrides 显式指定**）验证，
> 两臂 dtype 分布**完全相同**，看起来像"这条不成立" —— 实际是**样本里没有被测现象**：
> 没有未覆盖的浮点张量，`--float_bitwidth` 自然无从作用。
> ⊕ 顺带实测到一条：**overrides 条目里显式写的 `bitwidth` 优先**
> （标 `{"bitwidth":16,"dtype":"float"}` 的张量，加不加 `--float_bitwidth 32` 都是 `Float_16`）。
> ⇒ **选样规则**：验这个门要用**真实分段**（本项目 part1b 有 1607 个浮点张量），
> 不能用"所有张量都被覆盖"的最小探针。这是 §47.10 模式 4 的一个变体：
> **判据本身没问题，但样本不含被测变量，得到的"通过"是假的。**

后果：不写这个参数，你以为只改了几个张量的量化 schema，
实际上把**整张图的浮点权重先降成了 FP16**，单变量对照被破坏；
RmsNorm 里的 `x²` 之类还可能在 FP16 下溢出（本模型激活量程到 6265，平方即 3.9e7 ≫ 65504）。

### 10.7 想"只对某类算子改量化 schema"的通用做法（**别自己推量化器的取整公式**）

需求：只让 FullyConnected 的权重对称，RmsNorm 保持非对称（因为它的 OpDef 不收 SFxp8）。

**做法（三步，成本 = 一次多余的量化）**：
1. 先用**全局**开关跑一版（`--param_quantizer_schema symmetric`），哪怕它编不出 `.bin` 也没关系
   —— 要的只是 **SDK 自己算出来的对称 encoding**；
2. 从这版 DLC 的 `snpe-dlc-info` 转储里，把目标算子权重的 `scale` 抄出来，
   写成 overrides 的 `param_encodings`（`min = -128·s`、`max = 127·s`、`offset = -128`、
   `is_symmetric = "True"`）；
3. 用 overrides 重新转换 + 量化，**没被点名的张量保持默认**。

**为什么不自己算 `max|w|/127`**：实测量化器的对称取值**不等于**这个公式
（`val_1788`：SDK 给 `scale = 0.020669290796`，而 `max|w|/127` 的估算是 `0.0206635`，
且 SDK 的 `max` 全落在**二进制小数**格点上，像是被舍入到了某种低精度表示）。
**规则不明时，抄 SDK 自己的输出，零猜测。**

参考实现：`scripts/make_sym_fc_overrides.py`（含 ONNX 名核验与 `_permute` 后缀剥离）。

### 10.8 尚未验证（③，不得当事实用）

- overrides 指定 FP16 后，**HTP 上是否真以 FP16 执行**——本项目只验证了转换成功。
  验证方法：`qnn-context-binary-utility --json_file` dump context binary 元数据，
  比对该张量的实际数据类型（与本指南级别 2.5 的 encoding 核对同法）。
### 10.9 ✅ 已验证：**选择性对称权重（FC-only）在 HTP 上编得过图，但精度更差**

**①实测（2026-08-15，`scripts/EXP_PLAN_SYM_FC_OVR.md`）**：

- **可行性**：只对 59 个 FullyConnected 权重指定对称 `sFxp_8`、RmsNorm 的 gamma 不点名
  ⇒ 量化产出 **59/59 `sFxp_8` 且 offset==0**、**44/44 gamma 与基线逐字段相同**，
  `qnn-context-binary-generator` **EXIT=0**（对照：全局 symmetric 在 200 ms 挂在 RmsNorm）。
  ⇒ **"按算子选择性改量化 schema"这条工程路径是通的**，可直接复用到新模型。
- **精度结论（本模型，不可外推）**：设备实测主体余弦 **0.6033 → 0.5507**，**变差**；
  同一改动在 SNPE CPU 参考上只损失 0.0014，在 HTP 上损失 0.0526（**37.6 倍**）。

⇒ 换新模型时：**这条路径可用，但"对称权重必然更好"不成立，必须自己实测**。
官方 `htp_backend.html` 的通用建议
（*"recommended to always use symmetrical quantization of weights"*）
在 **A16W8 + 零 Convolution 的 DiT** 上被实测推翻。

**副产品（可复用的判据）**：想知道一次 encoding 改动"值不值得上设备"，
先在宿主上算两件事——① 权重自身量化表示误差的变化、② scale 的变粗/变细倍数
（`scripts/repr_cost_sym.py`）。本项目实测：scale 中位变粗 1.073×、表示误差 3.73%→3.97%，
就足以让设备侧余弦掉 0.0526。**HTP 对权重量化步长的敏感度远高于 CPU 参考。**

### 10.10 尚未验证（③，不得当事实用）

- overrides 指定 FP16 后，**HTP 上是否真以 FP16 执行**——本项目只验证了转换成功。
  验证方法：`qnn-context-binary-utility --json_file` dump context binary 元数据，
  比对该张量的实际数据类型（与本指南级别 2.5 的 encoding 核对同法）。
（本节暂无未决项；per-row 的结论见下面第十一节。）

## 十一、🔴 **Transformer / DiT 类模型必看：per-row 而不是 per-channel**

> 来源：Z-Image Turbo 排查 2026-08-15 的实测，HANDOVER §15.17。
> **这是本项目迄今唯一被实测证明有效的 HTP 精度改善手段**，换新模型时应作为默认起手式。

### 11.1 两个开关是两回事，别用错

| 开关 | 官方帮助原文 | 对 DiT / Transformer |
|---|---|---|
| `--use_per_channel_quantization` | *"per-channel quantization for **convolution-based op** weights"* | ❌ **零 Convolution ⇒ 完全空转**（本项目开了几个月，一个张量都没影响到） |
| `--use_per_row_quantization` | *"rowwise quantization of **Matmul and FullyConnected** ops"* | ✅ **正解**。本项目 99.99% 的 8-bit 权重在 FullyConnected 上 |

配套还有 `--enable_per_row_quantized_bias`（本项目未启用，保持单变量）。

### 11.2 实测收益（part1b，A16W8，设备 SM8750 / Hexagon v79）

| | E_all（vs FP32） | 主体余弦 | slope |
|---|---|---|---|
| per-tensor（原配置） | 19.24% | 0.6033 | 0.8622 |
| **per-row** | **7.87%** | **0.7986** | **1.0435** |

- 权重量化步长中位细化约 **3.9 倍**，权重表示误差中位 **3.73% → 1.02%**
- 代价：DLC 只大 **+14.5 MB**（多出来的 scale 数组），**不改图、不重新转换**
- 附带修好了"HTP 系统性把幅值算小"（slope 0.86 → 1.04）

### 11.3 用之前必须知道的三件事

1. **HTP 强制对称**：`HtpOpDefSupplement.html` → FullyConnected → INT16 → in[1]：
   *"Given a 2D weight of dimensions [m n], AXIS_SCALE_OFFSET ... is supported **with only
   axis 'm'** and the values are expected to be **signed and symmetrically quantized**"*，
   且权重 **must have rank 2**。⇒ 产物必然是 `sFxp_8` / offset 0。
2. **SNPE CPU 参考跑不了**：`error_code=202; Dequantization of axis-quant tensor is not
   supported for FullyConnected` ⇒ **一旦启用 per-row，就失去 CPU 参考这条离线校验路径**。
   若你的验证流程依赖 CPU 参考（本指南级别 2），要提前规划替代方案。
3. **转储格式变了**：per-axis 打的是 `encoding for channel_0:` + 单独一行
   `axis-quant: axis: 0, num_elements: N`，**不是** `encoding :`。
   按老正则写的检查脚本会**匹配不到、显示成 0 个**——又一个静默失效点。

### 11.4 一条可复用的判据：**表示误差不是 HTP 误差的预测量**

本项目实测两次，方向相反但结论一致：

| 改动 | 权重表示误差 | 设备主体余弦 |
|---|---|---|
| 对称化（scale ×1.073 变粗） | 3.73% → 3.97%（+6.5%） | **−0.0526** |
| per-row（scale ≈×0.26 变细） | 3.73% → 1.02%（×0.27） | **+0.2479** |

同一改动在 SNPE CPU 参考上只损失 0.0014，在 HTP 上损失 0.0526 ⇒ **放大 37.6 倍**。

⇒ **"量化表示误差小"/"MSE 最优"不能推出"HTP 上精度好"。**
   评估一个量化配置该不该上设备时，**看量化步长（scale）的变化倍数**比看表示误差更有预测力。

## 十二、🔴 **context binary 有硬性容量上限（unsigned PD）—— 分段大小必须按它来定**

> 来源：Z-Image Turbo 2026-08-16 实测，HANDOVER §15.18 / 台账 #57。
> **转新模型（Flux.2 Klein 等）时这是第一优先要算的约束**，比精度问题更早撞上。

### 12.1 现象与真实原因

给 part2 加了 `--use_per_row_quantization` 后，`.bin` 只大了 5.5 MB，
但设备上**加载直接失败**，qnn-net-run 只打印一句无信息量的：

```
Could not create context from binary
Create From Binary failure
```

**真实原因要开 FARF 才看得到**（见 12.3）：

```
QnnDsp <E> Failed to find available PD for contextId 1 on deviceId 0 coreId 0
           with context size estimate 3652345600
QnnDsp <E> context create from binary failed on contextId 1, err = 1002
```

⇒ **不是手机内存不足**（同一时刻 `lowmemorykiller: device has enough memory 8047228Kib`），
而是 **DSP 侧 unsigned 保护域（PD）装不下这个 context**。

### 12.2 实测边界（SM8750 / Hexagon v79 / unsigned PD）

| 配置 | contextBlob | spillFill | opData | 合计 | 能否加载 |
|---|---|---|---|---|---|
| part1a per-row | 2371 MB | 256.8 | 201.7 | 2829 | ✅ |
| part1b per-row | 1473 MB | 222.8 | 157.0 | 1853 | ✅ |
| part2 baseline | 2929 MB | 242.3 | 334.9 | **3506** | ✅ |
| **part2 per-row** | 2934 MB | 258.7 | 342.5 | **3535** | ❌ **PD 估算 3652 MB** |

⇒ **上限落在合计 3506~3535 MB 之间（PD 估算口径约 3.62~3.65 GB）。**
⇒ **实践规则：单个 context 的合计占用应控制在 3.3 GB 以内留余量；
   超过 3.5 GB 就是在悬崖边上。** 本项目的 part2 原本就贴着上限，
   **任何让它变大的改动都会直接打不开**——包括本来无害的量化改良。

### 12.3 怎么拿到真实错误（否则只能瞎猜）

设备工作目录里放一个与可执行文件同名的 `.farf` 文件即可打开 DSP 侧日志：

```bash
adb shell "printf '0x1f\n' > /data/local/tmp/<dir>/qnn-net-run.farf"
adb logcat -c
adb shell "cd /data/local/tmp/<dir> && export LD_LIBRARY_PATH=. && export ADSP_LIBRARY_PATH=. \
           && ./qnn-net-run --retrieve_context x.bin --backend libQnnHtp.so ..."
adb logcat -d | grep -E "QnnDsp|context size estimate"
```

**没有这一步，`Could not create context from binary` 这句话不含任何可用信息。**

### 12.4 已实测无效的"缩小 context"手段（别再试）

| 手段 | 为什么无效 |
|---|---|
| `--vtcm_override` | 实测三份 bin 的 `vtcmSize` 全是 **8 MB**，已是 v79 上限 |
| `--optimization_level_override` | 默认已是 0；实测 O=3 只会让体积**变大**（1471.5 → 1493.8 MB） |
| `--dlbc_override` | 文档原文：DLBC 压缩的是 **inputs**（降带宽），不缩 context 存储 |
| Sparse Weights Compression | 只对**稀疏**权重有效；DiT 权重稠密 |
| signed PD（空间更大） | 官方要求 *"client also needs to push a **signed** dsp image to target"* |

### 12.5 可行的方向

1. **分段时就按 3.3 GB 上限规划**（最省事，换新模型时一开始就做对）
2. 降低单段权重量：更少的层 / 更低位宽
3. 只对部分算子启用会增大 context 的量化特性（如逐行量化）

## 十三、🔴 **切分 ONNX 子图：必须先做死代码消除（DCE）**

> 来源：Z-Image Turbo 2026-08-16/17 实测，代价约 6 小时返工。台账 #62。
> **转 Flux.2 Klein 等模型时只要做图切分，就一定会遇到。**

### 13.1 现象

把 `transformer_part2`（1724 节点）在残差流处切成两段后，
`qairt-quantizer` 的内存**无界线性增长**（约 8.5 GB/分钟、无平台），
涨到 **134 GB** 后 `MemoryError: bad allocation`。
而**完整的同一张图**量化正常（历史上成功产出 2.75 GB 的量化 DLC）。

**"更小的子图反而比完整图更吃内存"** —— 这就是 DCE 缺失的典型信号。

### 13.2 原因

原图里有 **84 个死节点**（输出不可达图输出）：

```
IsNaN 15、Where 15、Gather 10、Reshape 14、Cast 6、Sign/Equal/Not/Mod/And/Sub 各 3、Div/Mul/Concat 各 2
```

它们是注意力后的 NaN 清洗路径，**消费 30 个 `[1,30,4128,4128]` 的注意力分数矩阵
（fp32 单个 1.9044 GiB，合计 57.1 GiB）**。

- **转换器对完整图会自己做 DCE** —— 生产 part2 DLC 里 `IsNaN` 计数为 **0**
- **但 `onnx.utils.Extractor` 按你给的输入/输出集切分时，会把死节点保留下来**，
  于是它们"复活"成子图的必需部分，量化时那些巨型张量全部要参与校准

**实测指纹**：量化内存的增长步长恰好 **1.90~1.91 GB = 单个 `[1,30,4128,4128]` fp32 张量**。

### 13.3 做法与效果

先算"从图输出反向可达"的活节点集，只保留它们，再切：

```python
live, stack = set(), [o.name for o in g.output]
prod = {o: i for i, n in enumerate(g.node) for o in n.output}
while stack:
    t = stack.pop(); k = prod.get(t)
    if k is None or k in live: continue
    live.add(k)
    for x in g.node[k].input:
        if x and x not in init_names and x not in graph_input_names:
            stack.append(x)
```

| | 切割集 | 量化内存 |
|---|---|---|
| 不 DCE 直接切 | **14 个张量、16.4 GB**（8 个巨型矩阵被死节点拽过切口） | 线性涨到 134 GB → OOM |
| **先 DCE 再切** | **6 个张量、65.6 MB** | **峰值 32.36 GB 封顶，5.5 分钟完成** |

参考实现：`scripts/split_part2_dce.py`。

### 13.4 切分的其他三个必检项

1. **源文件必须核实来源**：从生产流水线日志的 `--input_network` 取，
   **不要按文件名猜**。本项目误用了 `transformer_part2.onnx`（原始导出）
   而非生产用的 `transformer_part2_fixed.onnx`，两者差 **108 个节点**，整轮作废。
   > 症状识别：如果你在切分后"重新发明"了一个修复（例如手工固化 `latents_shape`），
   > **立刻怀疑是不是用了旧版本**——那个修复很可能上游早就做过了。
2. **切割集必须用反向可达性算，不能按节点序号**
   （约束 8：Id 顺序 ≠ 依赖顺序）。早期节点的输出会被很后面的节点消费。
3. **切完必须先做数值等价验证再往下走**：同一输入下
   `完整图输出` vs `子图A→子图B 输出`，报相对 L2 / 最大绝对差 / 余弦。
   本项目实测可以做到**逐位相同**（相对 L2 = 0.00000000%）。
   参考实现：`scripts/verify_split_equiv.py`。

### 13.5 工具坑

- **不能用 `onnx.utils.extract_model`**：它会内联外部权重，模型 >2 GB 时撞
  protobuf 上限（`EncodeError: Failed to serialize proto`）。
  改用底层 `onnx.utils.Extractor` 在 `load_external_data=False` 的 proto 上切，
  产物存到与 `.onnx.data` **同目录**，外部引用照样解析，且**不复制**权重。

## 十四、external weights / spill-fill buffer —— ❌ 不能用来绕开 PD 容量上限

> 台账 #60。**结论是否定的，但过程里得到的几个数值和坑很有用。**

想法：既然 context 装不进 PD，把 weights / spill-fill 放到外部 DMA buffer，
是否就能腾出空间？（`QNN_HTP_MEM_WEIGHTS_BUFFER` / `QNN_HTP_MEM_SHARED_SPILLFILL_BUFFER`）

**实测四档（自建最小 probe，仅加载不推理）**：

| 档 | createFromBinary | memRegister | contextFinalize |
|---|---|---|---|
| 普通 | ❌ 0x3ea | — | — |
| DEFER + spill-fill | ✅ | ✅ 260.0 MB | ❌ 0x3ea |
| DEFER + weights | ✅ | ✅ 2730.5 MB | ❌ 0x3ea |
| DEFER + 两者 | ✅ | ✅ 两个都成功 | ❌ 0x3ea |

**FARF 的 `context size estimate` 在外置前后完全相同（3652345600）**
⇒ 外置**不减少** PD 的容量估算。

**但过程中确立了几条有用的事实**：

1. `QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT` **确实能让 `createFromBinary` 通过**
   （同一份 2934 MB 的 bin，普通模式直接 0x3ea），
   但真正的 PD 检查在 `contextFinalize`，**只推迟不豁免**。
2. 查 buffer 需求的两个 property 可用，返回值可信：
   `QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE`、
   `QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE`。
3. ⚠️ **`rpcmem_alloc` 的 size 参数是 `int`**，分配 >2 GB 直接失败（32 位溢出）。
   设备 `libcdsprpc.so` 导出 64 位的 **`rpcmem_alloc2`**，大 buffer 必须用它。
4. ⚠️ **`qnn-net-run` 2.48 无法使用 `context_configs/spill_fill_buffer|weights_buffer`**：
   JSON schema 要 integer、下游解析器要 string，三种取值全试过。
   已核对设备端五个关键库与 SDK md5 完全一致，**排除版本错配，确认是工具缺陷**。
   ⇒ 要用这套 API 只能自己写程序（参考 `scripts/extbuf_probe/extbuf_probe.cpp`）。

## 十五、建 context binary / 建对照实验时的三个静默陷阱（2026-08-17 实测）

> 台账 #64 / #65 / #63。三条都属于**不报错、但结论悄悄变错**的类型，
> 转 Flux.2 Klein 等新模型时同样会踩。

### 15.1 🔴 `qnn-context-binary-generator` 漏 `--htp_socs` ⇒ 产出的**不是**设备用的离线缓存

```bash
# ❌ 少了 --htp_socs：产物 foo_ctx.bin，rc=0，无任何警告
qnn-context-binary-generator --backend QnnHtp.dll --dlc_path foo.dlc \
    --binary_file foo_ctx --output_dir out --config_file backend.json

# ✅ 正确：产物 foo_ctx.SM8750.bin
qnn-context-binary-generator ... --htp_socs sm8750
```

帮助原文：*"Specify SoC(s) to generate **HTP Offline Cache** for"*。

**判别方法（10 秒）**：
1. **看文件名有没有 SoC 后缀**（`*.SM8750.bin`）——最快
2. 看建图日志末尾有没有 `Cache binary for SM8750 copied to ...` 那一行

⚠️ **即使漏了它，backend extension 配置仍然会被读进去**
（日志照样打 `Unsupported HTP Arch 1 79`，那是宿主无 HTP stub 的正常告警），
所以**不能靠日志里有没有 arch 信息来判断**。

**为什么重要**：一条流水线上所有 `.bin` 必须走同一条生成路径，
否则新建的探针/对照**测的不是同一个东西**，而这个差异不会以任何报错的形式出现。

### 15.2 🔴 A16W8 下"正确定点"自身的等效位宽只有 ~7.4 bit（别拿位宽跨实验比）

单 MatMul 探针（`y = x @ W`，`W[K,64]`，K = 64/1024/3840，32 个测试样本，
用与 HTP 完全相同的 encoding 做**正确的**定点：整数累加 + 末端一次 requant）：

| K | `E_fxp`（vs float64 真值） | 等效位宽 |
|---|---|---|
| 64 | 1.2289% | **7.35** |
| 1024 | 1.3448% | **7.04** |
| 3840 | 1.0326% | **7.53** |

**机制**：权重只有 8 bit，经 K 次累加后**权重量化误差主导**总误差，
与激活是不是 16 bit 无关。⇒ **A16W8 的端到端等效位宽天然就在 7 bit 量级，
这不是后端的缺陷。**

⚠️ **由此得到的表述纪律**：说"某某只有 N bit"时**必须写清分母是什么**——
"HTP 误差相对某张量自身量程"与"含 W8 量化误差的端到端等效位宽"
**是两个不同的量，不得互相比较**。本项目差点用前者的 11.34 bit 去对照后者的 7.4 bit。

### 15.3 ⚠️ 包装 adb / 子进程的辅助函数**不要吞掉 stderr 和返回码**

实测：在环脚本里的 `sh()` 只 `return r.stdout`，设备中途 USB 掉线时它返回**空串**，
上层于是报 `qnn-net-run failed [part1a/s0]` —— **报错指向模型，真实原因
（`adb: device not found`）全在被丢弃的 stderr 里**。

```python
def sh(cmd):
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True, text=True)
    if r.returncode != 0:                      # ← 必须有
        raise RuntimeError(f"adb shell 失败 (rc={r.returncode})\nstderr: {r.stderr}")
    return r.stdout
```

并在每段模型执行**之前**显式确认设备在线，让掉线在"跑模型"之前就被认出来。
**排查闭源后端时，"报错说谎"比报错本身贵得多。**

### 15.4 一条通用做法：**判据的有效性自检，能在宿主做的一定先在宿主做完**

单 MatMul 探针的有效性自检是"`E_fxp` 必须显著小于 `E_htp`"。
`E_fxp` 只依赖 encoding + 权重 + 测试输入，**与设备无关** ⇒ 可以在上机前算完。
本项目据此在**没占用任何设备时间**的情况下，就确认了：
① 主判据有判别力（`E_fxp` 1.0~1.3% ≪ 5% 无效线）；
② 方案里的**位宽子判据在该设计下没有分辨力**（见 15.2）。

⇒ **凡是判据里有"参考侧"的量，先问一句：它需要设备吗？** 不需要就先算掉。
设备时间往往是最稀缺的资源（本项目的设备随用户上下班走）。

### 15.5 工具坑：SDK 2.48 的 python 工具与 onnx >= 1.19 不兼容

`qairt-converter` / `qairt-quantizer` / `qairt-dlc-info` 内部调用
`onnx.version.version`，而 onnx 1.22 已不再提供 `onnx.version` 子模块 ⇒
`AttributeError: module 'onnx' has no attribute 'version'`。

通用启动器（`scripts/qairt_tool.py`）：先补上 `onnx.version` 再 `runpy` 原工具，
并且**必须设 `PYTHONPATH=<SDK>/lib/python`**，否则 `ModuleNotFoundError: No module named 'qti'`。

```bash
PYTHONPATH="$SDK/lib/python" python scripts/qairt_tool.py qairt-converter --input_network a.onnx --output_path a.dlc
```

## 十六、🔴 权重和激活是两类东西：别把同一个校准手段两边用

> 台账 #66 / #67 / #68。**这一节能直接省掉转新模型时的多轮无效量化。**

### 16.1 实测对比（Z-Image Turbo，A16W8，part2 全部 121 个 FC 权重张量）

| 对象 | 手段 | 被动元素占比 | 它们占能量比 | 结果 |
|---|---|---|---|---|
| **激活** `unified` | percentile p99.99 钳量程 | 0.0261% | **98.75%** | ❌ **灾难**（设备主体余弦 0.7986 → 0.4793） |
| **权重** FC | mse 校准裁量程（可达上界） | 0.0520% | **1.12%** | ⚪ 无害但**几乎无收益**（表示误差 1.1168% → 1.0990%，仅 1.6%） |

🔴 **结论：权重不是离群值主导的，激活才是。**

这一条直接解释了本项目两个看似矛盾的实测：
- `--use_per_row_quantization`（**权重**，每行自己的量程，**范围不丢**）⇒ **有效**，唯一有效手段
- `--act_quantizer_calibration percentile`（**激活**，**砍量程**换分辨率）⇒ **净损害**

⇒ **"细步长有收益"只在动态范围完整保留时成立**，而"能不能砍量程"
取决于**该张量的能量是不是集中在尾部**——权重不集中，激活极度集中。
**换模型后必须重新测这个占比，不得沿用。** 3 行代码：

```python
thr = np.percentile(np.abs(a), 99.99)
clipped = np.abs(a) > thr
print(clipped.mean(), (a[clipped]**2).sum() / (a**2).sum())   # 占比, 占能量
```

### 16.2 🔴 8-bit 下 min-max 往往**就是** MSE 最优 ⇒ 换权重校准方法基本无收益

**机制**：N 个样本的高斯极值约 `σ√(2 ln N)`（N=4096 时 ≈ 3.7σ），
而 8-bit 均匀量化的 MSE 最优裁剪点约 **3.9σ** ⇒ **min-max 取到的量程已经在最优点附近**。

已知样本实测（逐行在裁剪比例 α 上网格搜索，取 MSE 最优）：

| 分布 | 最优 α | MSE 收益 |
|---|---|---|
| 高斯 N(0,1) | 0.976 | 仅 2.5% |
| Student-t df=2 / Cauchy / 对数正态 | **1.000** | **0**（min-max 即最优） |

⚠️ **反直觉但重要**：**重尾分布反而更不该裁**。
裁剪代价是 `(V−T)²`，随偏离量**平方**增长；收益只来自步长减小（∝ step²）。
一个 200 倍的离群点裁到 4σ，代价约 256，而 4096 个主体元素的量化误差总和才 3。
⇒ **"有离群点所以该裁掉"这个直觉在 MSE 意义下是错的。**

⇒ **`--param_quantizer_calibration mse/sqnr/entropy/percentile` 在 8-bit 权重上
基本不用试**（本项目实测可达上界只比 min-max 好 1.6%）。
把预算花在**粒度**（per-row）而不是**校准方法**上。

### 16.3 方法论：**"可达上界"预筛 —— 单向否决，成本极低**

不要直接跑 SDK 的 mse 校准（一次量化 70 min + 建图 15 min + 设备时间）。
先在宿主用**理想化的网格搜索**算出该方法的**可达上界**：

- **上界都没有收益 ⇒ 真实实现更不可能有 ⇒ 直接否决**（逻辑成立，单向）
- 上界有收益 **⇒ 不能反推真实实现能拿到**，更不能反推设备上有收益

本项目用它在**不占用设备**的前提下否决了一个候选，省下约 85 分钟 + 设备时间。

⚠️ **预算提醒（实测失准）**：这类逐元素网格搜索按
**张量数 × 网格档数 × 张量元素数** 估，不要凭感觉——
本项目估 <10 分钟，实际 **49 分钟**（121 张量 × 51 档 × 最大 3840×10240），低估 5 倍。

### 16.4 ⚠️ 网格搜索的分辨率必须覆盖真实最优区间

第一版网格 `linspace(0.40, 1.00, 25)`（步长 0.025）**会整个跳过 α≈0.976 那个真实最优点**，
等于**没给候选公平机会**。近 1 处必须加密（本项目用到 0.0025）。
**否决一个候选之前，先确认你的搜索有能力找到它的最优点**——
否则否决的是你的网格，不是那个候选。

## 十七、🔴 验收判据：**逐像素 / L2 类指标不能用来评判生成模型**

> 台账 #69 / #70。**这一节是本项目最贵的一条方法论教训，转任何生成模型都适用。**

### 17.1 实测：三把标量尺子同时与真实结果反相关

Z-Image Turbo 在 HTP 上，单变量对照（只换 part2 的量化配置，其余逐字节相同）：

| 指标 | 配置 A（图：马赛克色团，**认不出任何物体**） | 配置 B（图：**清晰的猫**） | 指标方向 |
|---|---|---|---|
| step-0 噪声预测相对 L2 | 43.41% | **44.80%** | ❌ 说 B 更差 |
| 对 FP32 参考图平均像素差 | 41.37 | **44.52** | ❌ 说 B 更差 |
| PSNR | 12.46 dB | **11.60 dB** | ❌ 说 B 更差 |

**三个指标一致地指向错误方向。** 只看数字会把唯一成功的配置判成倒退并放弃。

### 17.2 为什么会这样（机制，不是玄学）

1. **生成模型允许"画对了但内容不同"**。量化后的模型出的是**另一个合理场景**
   （本项目：背景由"完整房间"变成"空白墙"），逐像素比较把这种差异当成巨大误差。
2. **失败输出反而可能在 L2 上更近**。色块是平滑渐变，接近照片的**均值**；
   而一只结构清晰、纹理丰富的猫，与参考图逐像素做差反而更大。
3. **中间张量指标同理**：本项目实测过 per-row 把中间张量 `unified` 的主体余弦
   从 0.6033 拉到 0.7986（大幅改善），但"噪声预测"指标几乎不动
   —— **中间张量指标与最终画质之间不是线性关系**。

### 17.3 做法

- **验收必须看图**，而且是**人看**。把"打开图片确认"写进流程，不许用形容词转述。
- 标量指标只能用于**同一配置内部的回归监控**（例如确认改动没把模型改坏），
  **不能用于跨配置排序**。
- 若一定要自动化，用**结构/感知**类判据（是否出现可辨识主体、CLIP 图文相似度、
  FID 之类分布级指标），**不要用 PSNR / 像素差 / 逐点相对 L2**。
- ⚠️ **标定点必须来自"只变一个变量"的对照**。本项目原标定
  （"15.99% ⇒ 成图完好、47.07% ⇒ 色块"）的两个点分别来自**不同流水线**
  （全 CPU 参考 vs 全 HTP），**metric 与 configuration 一起变了**
  ⇒ 那是**共变，不是因果**，第三个点出现就崩了。

### 17.4 一条可直接复用的检查

**每次用某个标量指标做"放弃/降级"决定前，先问：这个指标在【当前区间】还有分辨力吗？**
验证方法很便宜：找两个**已知好坏差别巨大**的样本，看该指标能否把它们分开。
分不开就说明它在这个区间是盲的，**此时它给出的任何排序都不可用**。

## 十八、🔴 跨实现对照：先核对随机数生成方式，否则比的是两个不同采样

> 台账 #74。**转任何生成模型、比较两套实现（Python 参考 vs C++ 部署）时都会踩。**

### 18.1 现象

同一个 `seed=42`、同一个 prompt、同一批 `.bin`，两侧生成的图**完全不同**，
像素差 127.99（PSNR 4.69 dB）。看上去像"部署实现有严重 bug"。

**真相**：两侧的随机数生成器不同 ⇒ **初始噪声完全不同** ⇒ 两张图是同一模型的
**两个不同采样**，像素比较**毫无意义**。

| 侧 | 代码 | 引擎 |
|---|---|---|
| Python 参考 | `np.random.default_rng(seed).standard_normal()` | **PCG64** |
| C++ 部署 | `std::mt19937 g(seed); std::normal_distribution<float> d;` | **MT19937 + Marsaglia 极坐标** |

⇒ **"同 seed" 不等于 "同噪声"。** 扩散模型的初始噪声决定整张图的内容。

### 18.2 正确做法：让部署侧生成噪声，不要在参考侧复现

复现 C++ 的 `normal_distribution` 是**实现细节**（不是标准保证），很难逐位等价。
本项目实测：MT19937 引擎能通过官方测试向量（第 10000 个输出 = 4123659995），
但 `normal_distribution` 的复现只有 **73% 逐位相同**
（libc++ 全程 float32 算 `sqrt(-2*log(s)/s)`，Python 用 double 再收窄，差 ~1e-7）。

⇒ **正解：用部署侧的编译器/标准库编一个 20 行探针，把噪声写成 raw，喂给参考侧。**

```cpp
std::mt19937 generator(seed);
std::normal_distribution<float> normal(0.0f, 1.0f);
for (float& v : latents) v = normal(generator);
fwrite(latents.data(), sizeof(float), n, f);
```

NDK 交叉编译 + `adb push/pull` 即可，**等价性由构造保证**。
参考脚本加一个"从文件读初始噪声"的开关。

### 18.3 连带的检查项

- **参考图也必须用同一份噪声重新生成。** 本项目手上那张 FP32 参考图是用 numpy 噪声出的，
  拿它去比 C++ 侧的图，得到的 117 / 126 两个数**同样无效**——
  修正了一处却在另一处重犯。
- 更一般地：**任何"只改一个变量"的对照，都要把随机源列进变量清单**。
  它不显眼，但决定整个输出。

### 18.4 这条为什么值钱

它把"部署实现是不是有 bug"这个悬了很久的怀疑一次性关闭了：
噪声对齐后，C++ + 量化文本编码/VAE 与 Python + FP32 文本编码/VAE
**产出同一只猫、同姿态、同构图**（像素差 34.26），
⇒ 实现层与文本编码/VAE 量化均无害，**画质问题可以完全归到 transformer 量化上**。
在此之前，这三者是纠缠的。

> 🔴 **【2026-08-21 补·适用范围】上面这句"画质问题可以完全归到 transformer 量化上"必须限定读法。**
> 该对照的两臂**用的是同一份我们自己导出的 VAE**（一臂量化、一臂 fp32），
> ⇒ **VAE 导出本身的缺陷在这个对照里是共模的，会被完全抵消。**
> 2026-08-19 与官方实现对照后发现：我们的 VAE 导出另有一个 **×1.33 的全局增益缺陷**（见 §十九）。
> ⇒ 正确读法是：**"相对我们自己的 FP32 流水线"**，画质差距归于 transformer 量化；
> **相对官方实现**则还有一份独立的 VAE 导出缺陷。**不要把前者外推成后者。**


---

## 十九、🔴 感知类标尺（梯度能量 / 高频 / 锐度）**必须先做增益归一**，否则量的是对比度

**这是 §十七的同类陷阱，但更隐蔽**：§十七说的是"L2 类指标不适合生成模型"，
本节说的是**你自以为在量"锐度"，其实在量"对比度"**。

### 19.1 机制（三行数学，必看）

梯度算子是线性的 ⇒ 把一张图整体乘以 `k`，它的梯度幅值也整体乘以 `k`。
⇒ **一个 ×1.33 的全局增益，会原样表现为"高频能量 +33%"，而图中一根边都没有变硬。**

### 19.2 本项目实测的代价（2026-08-21）

我们的 ONNX VAE 相对官方 PyTorch VAE：

| 量 | 值 |
|---|---|
| 梯度能量比（原口径） | **1.305（"+30.5% 高频"）** |
| 像素 std 之比 | **1.2815** |
| 钳位前 float 的全局仿射增益 | **1.3324**（R² = 0.9903） |
| 梯度能量比（**mean/std 对齐后**） | **1.018** |

⇒ 那个"+30.5% 高频"里 **约 94% 是增益**。项目据此把缺陷误述为"细节被推成硬边（锐化）"
并按此设计逐层探针，**误导约 2 天**。

### 19.3 硬性要求（写进检查清单）

1. **凡报告高频/梯度/锐度类指标，必须同时报告两臂的 mean 与 std**，
   并给出**对齐 mean/std 之后**的同一指标。两个数一起看才有意义。
2. **凡声称"更锐/更糊"，必须先排除全局增益**：做一次最小二乘仿射拟合 `b ≈ a·x + c`，
   报 `a` 与 `R²`。`R²` 很高 ⇒ 差异主要是仿射，**不是锐度**。
3. **判据的阈值不能用绝对像素数拍**。1024² 照片上一次肉眼可见的结构改变
   （3×3 均值模糊）只产生 **1.16** 的平均 |像素差| ——
   若你把阈值定在 3.0，会把"模糊"判成"没差别"。**必须用已知样本标定阈值**（约束 8）。
4. **控制组里必须包含竞争假设本身**。本项目的 G-check 用了三组：
   C1 纯增益、C2 模糊、**C3 锐化（← 竞争假设）**。
   只有 C3 也返回低分，才能说这把尺子区分得开"增益"与"锐化"。

### 19.4 顺带：全局增益缺陷的典型来源（转新模型时优先查）

`×1.33` 且**近各向同性**（逐通道 1.311/1.343/1.323）、**传递曲线是直的**
（64 段自由度只多解释 0.05 pp）⇒ 典型来源是**归一化层的方差被系统性低估**：
- `GroupNorm` 被导出为 `Reshape→InstanceNormalization→Reshape→Mul→Add` 时，
  若运行时用单遍 `E[x²]−E[x]²` 求方差，在大空间尺寸（1024² 下每组上百万元素）会因抵消而低估
  ⇒ 除以偏小的 `sqrt(var+eps)` ⇒ **输出被放大**
- N 个归一化层复合：单层只需 `1.33^(1/N)`（N=30 时仅 **+0.95%/层**）就能累积出 ×1.33
  ⇒ **单层看几乎正常，必须逐层看 std 之比才抓得到**

⚠️ 🔴 **2026-08-21 订正：本项目这一例的真实原因不是归一化精度，是【反缩放被做了两遍】（见 §二十）。**
上面这段推测**在本例中已被证否**（按约束 2 保留原文）。
它作为"全局增益的候选来源"仍然成立，但**排查顺序必须放在 §二十 之后**——
先核对"图内到底做了哪些前后处理"，再谈数值精度。**本项目正是把这两步的顺序搞反，多花了两天。**


---

## 二十、🔴🔴 **导出的图可能把前/后处理烘焙进去了 —— 喂数据前必须先看图的头尾节点**

**这是本项目代价最大的一个坑（2026-08-21 定位，此前误判两天）。**

### 20.1 现象

`vae_decoder.onnx` 的**第一个节点**是：

```
node_div   Div   in=['vae_latents', const=0.361100]  ->  ['div']
```

即**反缩放 `latents / scaling_factor` 已经烘焙在图里**。
而调用方（6 个宿主脚本 + app 的 C++ 流水线）都按 diffusers 的常规写法，
在喂进去之前**又做了一遍** `latents / 0.3611 + 0.1159`。

⇒ **除以了两遍。** 三臂实测（对照官方 PyTorch）：

| 喂法 | 平均\|像素差\| | PSNR | 高频能量比 |
|---|---|---|---|
| `lat/s + shift`（现行） | 13.326 | 23.51 dB | 1.3049× |
| `lat`（裸） | 0.751 | 46.52 dB | 0.9976× |
| **`lat + shift·s`（正确）** | **0.001** | **79.92 dB** | **1.0000×** |

### 20.2 为什么它极难被发现（三条，全部在本项目真实发生）

1. **图结构与权重的静态检查会全部通过。** 我们逐项核对过 35 Conv / 30 GroupNorm /
   29 SiLU / 3 Resize / 1 attention / eps / **73 个权重逐位吻合** —— 全对。
   **因为导出确实是对的，错的是调用方。**
2. **成图看起来"差不多但质感不对"，不会崩。** VAE 里的归一化层会把尺度大部分吃掉，
   一个 2.77 倍的输入误差最终只表现为约 1.33 倍的对比度偏移 ⇒
   **容易被误读成"锐化/蜡质/量化损伤"这类玄学画质问题。**
3. **所有"我们 A vs 我们 B"的对照都测不出它** —— 两臂共用同一个错误喂法，**共模抵消**。
   只有跟**官方实现**对照才暴露。

### 20.3 第二重损害：量化量程被打穿（转量化模型时尤其致命）

VAE 的量化校准用的是**裸 latents**（与图内 Div 的期望一致）：

| | std | 范围 |
|---|---|---|
| 校准样本 | 0.86 ~ 1.08 | ±3.4 ~ ±4.5 |
| **实际喂入**（`lat/s+shift`） | **3.045** | **±12.2** |

⇒ **输入超出已标定激活量程 2.77 倍，被量化器硬钳。**
这部分损害此前被完全记在"量化"账上（+8.5 pp），实际是喂法错误。

⇒ 🔴 **推论：校准数据的分布，必须与推理时实际喂入的分布来自同一条代码路径。**
两者由不同代码产生时，**必须实测比对 std / min / max**，这是 10 秒的检查。

### 20.4 硬性要求（写进换新模型的检查清单）

1. **喂任何导出图之前，先 dump 它的头 N 个和尾 N 个节点**，确认前/后处理边界在哪：
   ```python
   for n in m.graph.node[:8]:  print(n.name, n.op_type, n.input, '->', n.output)
   for n in m.graph.node[-8:]: print(n.name, n.op_type, n.input, '->', n.output)
   ```
   看到 `Div`/`Mul`/`Add`/`Sub` 带常量 ⇒ **该步骤图内已做，宿主不要再做**。
2. **必须做一次"已知答案的样本"验证**（约束 8）。最便宜的一种：
   **第一个算子的输出**。它的权重与输入两侧都逐位相同 ⇒ **输出必须逐位相同**。
   本项目正是靠这条抓到根因：实测 std 之比 **2.7595**，而 `1/0.3611 = 2.7693`。
   ⇒ **这个数字直接把根因指出来了。**
3. **校准数据与推理输入必须比对分布**（std / min / max），见 20.3。
4. **只要是"我们 vs 官方"的对照被污染，所有依赖它的结论一律重做**，
   不要因为"我们 vs 我们"的对照还成立就以为没事——**两类结论的适用范围不同**。

---

## 二十一、🗂️ **整理纪律：转下一个模型时从第一天就这么放**

本项目到 2026-08-21 为止，根目录平铺了 **60 个文件、293 MB**，其中 **47 个无人引用、285 MB 是过程垃圾**
（设备全量 logcat 272 MB、构建日志、UI dump、SSE 原始响应）。
新线程被告知"读交接文档"时，很容易**打开错的那一份**——本项目就有两个 `HANDOVER*` 前缀的文件，
其中一份是过期且字节损坏的旧稿（已归档为 `archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md`）。

### 21.1 目录约定

| 目录 | 放什么 | 判断标准 |
|---|---|---|
| **根目录** | **只放"下一个人开工要看的东西"** | 本项目最终只剩 2 个：`CLAUDE.md`（约束）+ `MAINLINE.md`（开工必读） |
| `docs/` | 主文档（排查记录、转换指南、执行契约、已关闭台账） | 会被反复引用的 |
| `docs/reviews/` | 外部审查、诊断报告 | 一次性但要留证 |
| `archive/` | **已作废/已冻结**的旧版本 | 保留（约束 2：不静默删除），但**必须在文件头写明错在哪** |
| `scripts/` | 脚本 + `EXP_PLAN_*.md` 实验方案 | — |
| `reference/` | 从 SDK 文档提取的关键小节 | 省得每次重新抽 |
| `evidence/` | **被文档当证据引用**的产物 | 判断标准：删了会让某条结论失去依据 |
| `scratch_runs/` | 实验产物 | 大但有用 |
| `logs/` | **过程垃圾** | 判断标准：**产生一小时后就只剩体积** |

### 21.2 三条硬规则

1. **过程产物从第一天就写 `logs/`，不要写根目录。**
   全量 logcat、构建日志、UI dump、SSE 原始响应——它们在产生那一刻有用，之后只剩体积。
2. **同类文档不要用同一个前缀。**
   `HANDOVER_DOC.md` 与 `HANDOVER_2026-08-13_*.md` 并存，直接导致"读交接文档"读错文件。
   作废的旧稿要**改名到不可能被误认**（加 `ARCHIVE_`/`OBSOLETE_` 前缀），并在头部写明错在哪。
3. **不要维护两份需要人工同步的状态文档。**
   本项目的 `STATUS_FOR_REVIEW.md` 与 `MAINLINE.md` 内容重叠、靠人工同步 ⇒ **必然过期**，
   事实上过期了 4 天还在被当成开工必读。**要么单一来源，要么按需现生成。**

### 21.3 搬文件必须连引用一起搬

根目录 60 个文件里有 15 个被 40 余处引用（文档互引、`CLAUDE.md` 引用、实验方案引用）。
**手搬必然留死链。** 用 `scripts/tidy_repo.py`：先建依赖图 → 空跑打印计划 → 改引用 → 移动 → 死链复核。

⚠️ 实测踩到的两个坑：
- **脚本会改写自己**：`scripts/*.py` 在扫描范围内，脚本里的文件名字面量被当成引用替换了 ⇒ **必须自排除**。
- **绝对路径写法躲过了改写**：`D:\LocalDreamZImage\xxx.png` 因为前面是路径分隔符，被"只替换裸文件名"的
  正则跳过 ⇒ **改写要同时覆盖裸名与绝对路径两种写法**。

### 21.4 🔴 **什么时候可以删大产物**（2026-09-21 定，磁盘是转新模型的硬门槛）

一个模型转完，磁盘上会堆起几百 GB 的实验产物。**转下一个模型前必须清，而清的判据容易搞错。**

**判据不是"它是不是证据"，而是"这个结论还需要被重新验证吗"。**

| 类别 | 处置 | 理由 |
|---|---|---|
| **大产物**（GB 级 `.dlc` / `.bin` / `.raw`） | 支撑的结论**已严谨验证且已关闭** ⇒ **删** | 结论不需要重跑，为它供养几十 GB 是纯成本 |
| **当前交付件 + 回滚源** | 一律留 | 要对拍、要回滚 |
| **小文件**（文档、清单、配方、manifest） | **一律留**，哪怕已过时/已被推翻 | 价值密度极高：几 KB 记住"这条路为什么被排除"；删了才是真损失 |

⚠️ **别把「结论被推翻时保留原文」（约束 2）套到二进制上** —— 那条讲的是**文档**纪律，
从不要求养着中间产物。本项目作者一度用"这些是台账证据"主张不删 53 GB，是过度泛化。

#### 四条操作纪律（每条都对应本项目的一次实际风险）

1. 🔴 **删除走白名单，不走黑名单**。
   "不在保护名单就是候选"漏写一条就删错；"只删明确知道能删的"漏写只是少删。
   本项目按黑名单跑时漏过**两次**：① #171 的 single 形态回滚源（名字与交付件完全不同，
   谱系白名单里自然不出现）；② TE 四段的来源 DLC。
2. 🔴🔴 **谱系断链 = 清理盲区**。上面第 ② 条的根因是 TE 目录里**没有建图配置 json**
   ⇒ 自动谱系认不出它的来源 DLC ⇒ 被判成可删。
   ⇒ **转新模型时每建一个 context 就把配置与日志留在产物旁**（本项目 transformer/vae 都留了 `d.json`/`e.json`，
   所以谱系完整；TE 没留，就出了盲区）。
3. 🔴 **删前写 manifest 并永久保留**：路径、字节数、属于哪个实验、**结论在哪**，
   以及 DLC 里**内嵌的 Converter/Quantizer 命令**（§二：删了就再也拿不回来）。
   同一实验组的配方相同，抽一个代表即可 —— 逐个抽 34 个文件要半小时以上，代价与收益不成比例。
4. 🔴 **删后核对**交付白名单逐个仍在（约束 11 铁律 3 的精神：改了状态就验）。

⊕ 工具：`scripts/cleanup_by_lineage.py`（白名单式 + manifest + 删后核对），
依赖 `scripts/build_lineage.py` 从设备 sha256 反查出的交付白名单。
**不要用通配符**：`ctx_part1a_fp16`（L32，可删）与 `ctx_part1a_fp16_L80`（回滚源，必留）只差一个后缀。

---

## 二十二、🔴 **HTP 上权重量化的机制结论：offset 项是最大单一误差源**（2026-08-22 算子级实测）

### 22.1 结论（可直接用于新模型的配置选择）

在 A16W8 + DiT（零 Conv）配置下，对一个真实的 K=3840 注意力投影层做**单算子级**三臂对照
（唯一变量 = 权重 encoding；每臂的误差都对**各自的**正确定点计算 ⇒ 表示误差被约掉，
量的是纯粹的"HTP 比正确定点差多少"）：

| 权重 encoding | 表示代价（输出域 vs FP32） | **HTP 超出正确定点** |
|---|---|---|
| 非对称 per-tensor（offset −126） | 1.3323% | **1.5544%** |
| 对称 per-tensor（offset 0） | 1.3550% | **0.8310%（−46.5%）** |
| **对称 per-row**（每列一个 scale） | **0.5050%** | **0.6591%（再 −20.7%）** |

⇒ ②**非对称权重的 offset 项占 HTP 算子级超出误差的 46.5%**，是最大单一来源。
机制：定点展开 `y = Σq_x·q_w − z_w·Σq_x`，本项目实测 **offset 项/主项 = 0.891~0.979**，
两个近乎等大的数相减 ⇒ 中间精度稍差就吃掉有效位。

### 22.2 🔴 但**不要**用"per-tensor 对称化"去修它

这是个陷阱，本项目踩过（端到端余弦 0.6033 → **0.5507**，更差）：

- 对称化**消除 offset 项** ⇒ 有收益（−46.5%）
- 对称化**同时粗化 scale**（本项目实测中位 **1.073×**）
- 而 **HTP 对 scale 粗化的敏感度 = CPU 的 37.6 倍**（同一改动：CPU Δcos −0.0014，HTP −0.0526）
- ⇒ 两者相抵，**净负**

⇒ ②**正确做法是「消除 offset 项而不粗化 scale」**，也就是 **per-row 对称量化**
（`--use_per_row_quantization`）——它两头都占。这解释了为什么它是本项目唯一有效的手段。

### 22.3 换新模型时的建议顺序

1. **先开 `--use_per_row_quantization`**（对 DiT/Transformer 类；#44：`use_per_channel_quantization` 只作用于 Conv，零 Conv 模型上是空转）
2. 想验证 offset 项在你的模型上占多大，用 §22.4 的单算子重放台，**不要**直接端到端换对称——
   端到端把两个效应混在一起，得到的结论会与机制相反

### 22.4 🛠️ 可复用的诊断台：**单算子精确重放**

这是本项目分离"算子级 vs 图层面"效应的工具，脚本可直接改用：
`scripts/p0b_extract_encodings.py` → `p0b_build.py` → `p0b_run_analyze.py`

做法：
1. 从量化 DLC 里抽出目标层的**精确 encoding**（`snpe-dlc-info -d -s <csv>`，
   **`-d` 才给 per-axis 全部通道**；默认只打 channel_0）
2. 建单算子 ONNX（`y = x @ W`，FP32 权重取自源 ONNX）
3. 用 `--quantization_overrides` 把**真实 encoding 原样注入**，**不重新校准**
4. 建 context 上机，喂**逐字节相同的真实输入**
5. 与"正确定点模拟"比 ⇒ 得到"HTP 超出正确定点"的纯量

**四道必须有的门**：
- **V1**：读回 DLC 确认 encoding 逐位落地（**不能靠 `Processed N`**，那个数不是张量数）
- **V2**：设备输出字节数（约束 3）
- **V3**：输入与图内那次同一份文件（md5）
- 🔴 **V4 已知答案门**：用同一套模拟器复算**图内**误差，必须复现历史记录的数字。
  复现不出 ⇒ 模拟器或口径不对，**standalone 的数字一律不可采信**

### 22.5 ⚠️ 两个操作化陷阱（都踩过）

1. **判据锚在哪个配置上，重放就必须用哪个配置。**
   本项目差点拿 per-row encoding 去重放，而锚点（历史误差数字）来自 per-tensor 基线 ⇒ 判据会落空。
2. **代价门必须与被比较量同量纲。**
   约束 4·补 的 G0-cost 写的是"纯表示误差 vs 端到端误差"，但**表示误差算在权重上、
   端到端误差算在输出上**，直接比会把基线自己也否决掉。
   正解：把 encoding 代价**换算到输出域**（用正确定点模拟算 vs FP32 的输出误差）再比。

---

## 二十三、⚠️ **分块量化（BQ）在 QAIRT 2.48 上只支持 int4 静态权重**（2026-08-22 实测）

转新模型时若想用比 per-channel/per-row 更细的 scale，先看这条，能省一轮：

### 23.1 事实

- `qairt-quantizer` **没有** BQ 的 CLI 开关（完整 `--help` 已读）⇒ 只能经
  `--quantization_overrides` 注入 **schema 2.0.0** 的 BQ 格式
  （`y_scale` 二维 `[通道][块]` + `axis`（= 分块所沿的轴）+ `block_size`）
- **单变量试探（只改 `output_dtype`）**：

| `output_dtype` | 结果 |
|---|---|
| `int4` | ✅ converter rc=0 |
| `int8` | ❌ `modeltools::StaticTensorQuantizer::quantizeAndPackStaticTensors: Unhandled quantization encoding type` |

⇒ **不是格式错，也不是后端不支持 BQ 本身——是静态权重量化器只实现了 int4 的分块 encoding。**
（两次都打印 `Processed N quantization encodings`，说明 JSON 被正常解析，问题在其后的打包环节。）

### 23.2 int4 + 细分块值不值得？本项目实测：**不值得**

输出域表示误差（DiT 的 K=3840 注意力投影层）：

| 配置 | 表示误差 | 权重体积 |
|---|---|---|
| **int8 per-row** | **0.5051%** | 100% |
| int4 BQ block=128 | 6.2933% | 53% |
| int4 BQ block=32（最细） | **5.0470%** | 62% |

⇒ **位宽起支配作用，分块细化补不回来**（差一个数量级）。
除非你的模型本来就打算走 int4，否则不要为了"更细的 scale"去换 BQ。

### 23.3 一个会浪费时间的格式坑

给二维 `y_zero_point` 会让转换器在 `contain_decimal_num()` 里对 ndarray 调 `round()` 而
`TypeError: type numpy.ndarray doesn't define __round__ method`。
`y_zero_point` 是**可选项（默认 0）**，对称量化时**直接省略**即可。

### 23.4 方法论：把"支持性"和"值不值得"拆成两问，先问便宜的那个

本项目就是靠这个在 **20 分钟内、零设备占用**关掉了这条线：
先用一次 `int4` vs `int8` 的单变量试探定性支持范围，再用宿主上的表示误差算值不值。
**不要一上来就建 context 上机。**

---

## 二十四、🔴🔴 **HTP 后端选项为什么"设了没用"——三个必须同时满足的条件**（2026-08-22 实测）

本项目有三条线索（`HIGH_PRECISION_SIGMOID`、`advanced_activation_fusion`、VTCM 分块）
**卡在"设了但产物 md5 逐位相同"上长达一周**，最后查明**全部是配置写法问题，不是后端不支持**。
转新模型时按下面三条对照，能省一整轮。

### 24.1 条件一：配置文件必须有 `graphs` 段

`qnn-context-binary-generator --config_file` 指向的是**外层** JSON：

```json
{ "backend_extensions": {
    "shared_library_path": "<SDK>/lib/x86_64-windows-msvc/QnnHtpNetRunExtensions.dll",
    "config_file_path": "<detail>.json" } }
```

而**真正的选项在 `<detail>.json`**，官方 schema（`htp_backend.html`）是：

```json
{ "graphs": [ { "graph_names": ["..."], "vtcm_mb": 8, "O": 3, "dlbc": 1 } ],
  "devices": [ { "soc_model": 69, "dsp_arch": "v79" } ] }
```

🔴 **本项目踩的坑**：detail 文件里**只有 `devices`，完全没有 `graphs` 段** ⇒
选项写在任何别的地方都等于没写，而且**不报错**。

### 24.2 条件二：`graph_names` 必须填对，否则整段落空且不报错

图名**不是** ONNX 图名，**也不是**量化 DLC 的名字，而是
**converter 产出的 fp32 DLC 的文件名主干**：

| DLC | 图名 |
|---|---|
| `fc99_quantized.dlc`（由 `fc99_fp32.dlc` 量化） | **`fc99_fp32`** |
| part1b 的 context | **`transformer_part1b_fp32`** |

**怎么读**：
```bash
qnn-context-binary-utility --context_binary <ctx.bin> --json_file out.json
# 看 info.graphs[].info.graphName
```

⚠️ 填错时**不报错**。本项目实测：填 `fc99`（错）与不给 config 相比 md5 确实变了，
**但四个不同选项之间产物完全相同** ——「看起来生效了其实没有」，
**必须用"两个不同取值产出不同 md5"来验证，而不是"和不给 config 相比变了"**。

### 24.3 条件三：`snpe-dlc-graph-prepare` 的开关**带不进 `.bin` context**

`snpe-dlc-graph-prepare` 把同样的选项做成了一级 CLI 参数
（`--vtcm_override` / `--htp_high_precision_sigmoid` / `--htp_advanced_activation_fusion` /
`--optimization_level` / `--htp_dlbc(_weights)` / `--htp_slc_alloc` / `--htp_cached_weights` /
`--num_hvx_threads`），**而且确实生效**（8 个配置 8 个不同 md5，体积 30.4~37.6 MB 明显不同）。

🔴 **但它产出的是"内嵌 HTP cache 的 DLC"，不是 `.bin`。实测**：
把三份**不同的** prepared DLC 分别喂给 `qnn-context-binary-generator --dlc_path`，
**产出的 `.bin` 三者逐位相同，且与直接从未 prepare 的量化 DLC 生成的那份也相同**
⇒ **context-binary-generator 丢弃内嵌 cache，按自己的默认重新编译。**

⇒ **要用 graph-prepare 的开关，必须走 DLC 执行路径（`snpe-net-run`），不能走 `.bin`。**
如果你的 app 部署用的是 `.bin`（本项目就是），这是一条**产品化硬约束**：
只有能经 `--config_file` 的 `graphs` 段落地的选项才用得上。

### 24.4 实测生效性对照（单算子 MatMul，图名正确）

| 选项 | 经 `.bin` 路径 |
|---|---|
| **`O`（optimization level）** | ✅ 生效（O=1/3/默认 三者 md5 与体积均不同） |
| **`dlbc`** | ✅ 生效 |
| `vtcm_mb` | ❌ 对本图无效果（2/4/8 产物完全相同） |

⚠️ 划界：这是**单算子**的结论。`vtcm_mb` 在大图上是否有效未测。

### 24.5 一条通用纪律

**"选项生效"必须用"两个不同取值 ⇒ 两个不同产物"来证明**，
不能用"加了选项 ⇒ 产物变了"——后者会被"配置被解析但落到空处"骗过去。
本项目在这一点上被骗了一周。

### 24.6 这两个 HTP 开关**对量化图无效**（实测，别浪费时间）

| 键名 | 结论 |
|---|---|
| `advanced_activation_fusion` | ❌ 官方 schema 注释明写 ***"no effect on quantized graphs"***，实测 true/false 产物 md5 相同 |
| `use_high_precision_fp16_sigmoid` | ❌ **fp16 专用**；量化图（`uFxp_16` 激活）上实测 true/false 产物 md5 相同<br>⚠️ 注意**正确键名不是** `HIGH_PRECISION_SIGMOID` |

🔴 **验证时必须带一个"已知生效"的对照**（本项目用 `O=1`，产物 md5 确实变）。
没有对照，你只能说"又没反应"，**不能**说"确实不适用"——这两者在本项目里差了一周。

---

## 二十五、🔴🔴 **`--htp_socs` 不设置 DSP 架构——只给它会静默编译成 v68**（2026-08-22 实测，代价：一整轮结论作废）

### 25.1 现象

```bash
qnn-context-binary-generator --dlc_path m.dlc --binary_file ctx \
    --output_dir out --htp_socs sm8750          # ← 看起来完全正确
```

产物：`ctx.SM8750.bin`，rc=0，无任何告警。**读回它的元数据**：

```json
"contextMetadata": { "info": { "dspArch": 68 } }        // ← v68，不是 v79
"graphBlobInfo":   { "info": { "vtcmSize": 4, ... } }   // ← 4 MB，不是 8 MB
```

**加上 `--config_file`（detail 里含 `devices`）后**：`dspArch 79` / `vtcmSize 8`。

### 25.2 代价（本项目实测）

同一份量化 DLC、同一份真实输入、同一套 encoding，只差这一个配置：

| 编译画像 | 单算子 HTP 超出正确定点的误差 |
|---|---|
| dspArch **68** / vtcm **4 MB** | **1.5544%** |
| dspArch **79** / vtcm **8 MB** | **0.1034%** |

**15 倍。** 本项目据此得出的一整轮机制结论全部作废并翻转。

### 25.3 正确写法

```json
// detail.json —— 这一段是必须的
{ "devices": [ { "soc_model": 69, "dsp_arch": "v79" } ] }
```
```json
// --config_file 指向的外层
{ "backend_extensions": {
    "shared_library_path": "<SDK>/lib/x86_64-windows-msvc/QnnHtpNetRunExtensions.dll",
    "config_file_path": "<detail.json 的绝对路径>" } }
```

⚠️ 实测：**`graphs` 段对本例毫无作用**（带与不带 `graphs`，产物逐位相同）；
**全部效果来自 `devices`**。

### 25.4 🔴 判别方法：**读元数据，不要看文件名**

本项目原来的检查是"看产物名有没有 `.SM8750` 后缀"——
**错误产物的名字就是 `c_NOCFG.SM8750.bin`，完全通过该检查。**

**正确做法**：
```bash
qnn-context-binary-utility --context_binary <ctx.bin> --json_file info.json
```
断言两项：
- `info.contextMetadata.info.dspArch` == 目标架构（本项目 **79**）
- `info.graphs[].info.graphBlobInfo.info.vtcmSize` == 目标 VTCM（本项目 **8**）

### 25.5 更普遍的纪律：**任何要与部署产物对照的实验，先 diff 两者的元数据**

本项目这次的根本失误不是"忘了一个参数"，而是：

- 给**模拟器**设了很强的已知答案门（并且**通过了**，给出虚假信心）
- **却没给装置本身设门**——核对了输入逐字节相同、encoding 逐位相同、输出字节数正确，
  **唯独没核对两边是用同一套配方构建的**

⇒ **两条硬规则**：
1. **照抄部署用的构建命令，不要另写一条。** 分歧总是在"自己拼一条新命令"时进来。
2. **建完立刻 diff 产物元数据**（5 秒），把它做成实验的 V 门之一。

⚠️ 附带说明本次是**怎么发现的**：靠一个无关实验的对照臂恰好带了 config，出现 15 倍落差大到无法忽略。
**这是运气，不是流程。** 上面两条规则就是把它变成流程。

---

## 二十六、🔴🔴 **抽象出来的教训：对照实验的「装置等同性」——本项目栽过四次的同一个类别**

§二十五讲的是一个具体参数。但那不是根本问题。**根本问题是一个反复出现的类别**，
本项目在完全不同的场景下栽过四次，每次都是"一个我没想到要检查的维度上，两臂不同"：

| # | 表面现象 | 真实差异所在的维度 |
|---|---|---|
| **#61** | 整轮返工 | **源产物身份**：用了 `base` 而非 `_fixed` 的 ONNX |
| **#74** | 像素差 127.99，像"实现有大 bug" | **输入数据来源**：numpy PCG64 vs `std::mt19937`，同 seed 不同噪声 |
| **#72** | "app 能跑"的结论全部不成立 | **执行环境**：`run-as` 的 SELinux 域是 `runas_app`，真实 app 是 `untrusted_app` |
| **#95** | 一整轮机制结论作废并翻转 | **构建配方**：漏 `--config_file` ⇒ 静默编译成 v68/4MB |

### 26.1 根因不是"粗心"，是**门设错了地方**

四次都有一个共同结构：**我给"流过装置的数据"设了门，没给"装置本身"设门。**

以 #95 为例，我设的门是：
- 输入 md5 逐字节相同 ✅
- encoding 注入逐位精确 ✅
- 输出字节数正确 ✅
- 模拟器已知答案门（复现历史数字）✅ ← **甚至通过了，给出虚假信心**

**唯独没有**：两臂是否用**同一套配方构建**。
⇒ 数据全对、装置不同，结论照样全错。

### 26.2 装置等同性检查表（换新模型时逐条过）

**做任何"两臂对照"之前**，逐条写出断言。每条都是 10 秒以内的检查：

| 维度 | 具体断言 | 反例 |
|---|---|---|
| **源产物身份** | 从**流水线日志**核实来源，**不由文件名推断** | #61 |
| **构建命令/配置** | **照抄参照那一臂的命令**，不要另写一条 | #95 |
| **目标硬件画像** | 读回 `dspArch` / `vtcmSize` 断言相等（**不是看文件名后缀**） | #95 |
| **量化配置** | 从 DLC 读回 encoding（per-tensor / per-row / 位宽 / 对称性）逐项比 | P0-B 差点栽（per-row vs baseline） |
| **执行环境** | 同一 runtime、同一 SELinux 域、同为 HTP 或同为 CPU | #72、#35 |
| **输入数据来源** | 同一份文件的 md5；跨实现时**先对齐随机源与预处理约定** | #74、#80 |
| **度量口径** | 全量 vs 主体、分母是谁、能量集中度 | 约束 7 |

### 26.3 可跑的检查器（把纪律变成工具）

```bash
# 断言画像
python scripts/check_ctx_identity.py --expect-arch 79 --expect-vtcm 8 <ctx.bin> ...
# 对照实验必做：与参照产物 diff 装置画像
python scripts/check_ctx_identity.py --diff <参照.bin> <待检.bin>
```

它把 `dspArch` / `vtcmSize` / `optimizationLevel` / `htpDlbc` / `numHvxThreads`
划为**装置字段（必须相同，不同即实验不成立）**，
把 `graphNames` / `spillFillBufferSize` 划为模型字段（不同模型间允许不同）。

用它复跑本次事故，**当场拦下**：
```
🔴 c_NOCFG.SM8750.bin **装置画像不同 ⇒ 对照实验不成立**：
     dspArch    参照 79   待检 68
     vtcmSize   参照 8    待检 4
```

### 26.4 两条应写进实验方案模板的硬规则

1. **实验方案里必须显式列出「两臂之间所有可能不同的维度」，并逐一给出断言。**
   只写"唯一变量是 X"是不够的——本项目四次事故，每次都自认为"唯一变量是 X"。
2. **对照臂的构建必须复用参照那一臂的命令**，不要从文档碎片里重新拼装。
   四次里有两次（#61、#95）的分歧就是在"自己拼一条新命令"时进来的。

### 26.5 ⚠️ 还有一条不能自我安慰的

本次能发现，是因为一个**无关实验**的对照臂恰好带了 config，出现 15 倍落差大到无法忽略。
**这是运气，不是流程。** 上面的检查表和检查器，就是把这份运气换成流程。


---

## 二十七、🔴🔴 **选择性 FP16 的真实形态：`wFxp_actFP` —— "算子转 FP16" ≠ "权重转 FP16"**（2026-08-22，文档实查）

**转新模型时如果考虑"把敏感算子放到浮点"，先读本节，能省掉一次基于错误前提的否决。**

### 27.1 本项目因错误前提差点否掉整条路

`EXP_PLAN_FC_VS_REST.md` 写着：三段合计 8-bit 权重 6154 M，其中 **FullyConnected 占 99.99%**
⇒ *"浮点化 FullyConnected 则必然出局"*（+6154 MB，撞 unsigned PD 的 3.3 GB 红线）。

**该否决基于一个未经查证的前提：「算子转 FP16」= 「权重也转 FP16」。**
查 HTP OpDef supplement 后确认：**不是。**

### 27.2 官方依据（`docs/QAIRT-Docs/QNN/OpDef/HtpOpDefSupplement.html`）

列语义：`Configuration | in[0] 激活 | in[1] 权重 | in[2] bias | out[0] 输出`

| 算子 | 关键行 | 结论 |
|---|---|---|
| **FullyConnected** | `FP16 : FLOAT_16 \| **SFIXED_POINT_8** \| FLOAT_16 \| FLOAT_16` | ✅ **激活 FP16 + 权重 8-bit 定点**（= wFxp_actFP）⇒ **权重字节数不变** |
| RmsNorm | `FP16 : FLOAT_16 ×4` | ✅ 可全 FP16（gamma 很小） |
| ElementWiseMultiply / Add | `FP16 : FP16 ×3` | ✅ 全 FP16，无权重 |
| Softmax | `FP16 : FP16 ×2` | ✅ 全 FP16，无权重 |

⚠️ RmsNorm / ElementWise **没有**「定点权重 + 浮点激活」的混合行 ——
它们不需要，因为其权重本来就小到可以直接全浮点。

### 27.3 对应的 quantizer 选项（`SNPE/general/tools.html`，逐字）

> `--keep_weights_quantized`
> Use this option to keep the weights quantized even when the output of the op is in floating point.
> Bias will be converted to floating point as per the output of the op.
> **Required to enable wFxp_actFP configurations** according to the provided bitwidth for weights and activations
> **Note: These modes are not supported by all runtimes. Please check corresponding Backend OpDef supplement if these are supported**

🔴 **引用这句话时必须连同后半句一起引**（约束 5）：
「Required to enable wFxp_actFP」是**必要条件不是充分条件**，
**支不支持要逐算子查目标后端的 OpDef supplement**。

### 27.4 🔴 `--enable_float_fallback` 在 QAIRT 2.48 已是 no-op（`QNN/general/quantization.html`，逐字）

> *"The float fallback feature controlled via command-line option `--enable_float_fallback`,
> present as `--float_fallback` in legacy quantizers **is also a no-op for qairt-quantizer** and can be skipped."*
> *"... `--ignore_quantization_overrides`, and `--enable_float_fallback` are now no-op,
> and are applied by default during **qairt-converter** step itself."*
> *"`--enable_float_fallback` and `--input_list` are **mutually exclusive** options. One of them is mandatory argument for quantizer."*

⇒ **2.48 的选择性浮点机制是**：`qairt-converter` 施加 overrides/encodings，
**凡是缺 encoding 的张量就落成浮点**。所以做选择性 FP16 的正确路径是
**给一份「故意漏掉某些张量」的 encodings**，而不是加某个开关。

⇒ 与 `--dump_encoding_json` 配合的可行流水线（**尚未实测，标注为未验证**）：
① 正常量化一次并 `--dump_encoding_json` 导出全量 encodings
② 删掉想走 FP16 的那些张量的条目
③ 用剩下的作为 `--quantization_overrides` 重新 convert（注意 #45：必须用 **ONNX 张量名**，
   且每次必查 `Processed N encodings` 那一行；#46：必须显式 `--float_bitwidth 32`）

### 27.5 可复用纪律

1. **「体积/可行性」类否决，必须先查 OpDef 支持表再下**——否则会用错误的数据类型模型算出错误的体积。
2. **查支持表要看列语义**（in[0] 是激活、in[1] 是权重），不能只 grep 数据类型名。
   本项目第一次解析就抓错了段落（抓到 Convert 算子里提及 FullyConnected 权重的文字）。
   **正确做法是用 HTML 锚点 `id="<小写算子名>"` 定位算子段。**
3. **同一份文档里，选项说明与后端支持表是两件事**，两边都要查：
   选项说明告诉你"用什么开关"，OpDef supplement 告诉你"这个后端认不认"。

### 27.6 ✅ 实测可用的 wFxp_actFP 配方（2026-08-22 单算子验证通过）

```
qairt-converter  --input_network <op>.onnx --output_path <op>_fp32.dlc                  --quantization_overrides <只含 param_encodings 的 json>                  --float_bitwidth 16

qairt-quantizer  --input_dlc <op>_fp32.dlc --output_dlc <op>_quantized.dlc                  --weights_bitwidth 8 --bias_bitwidth 32                  --param_quantizer_calibration min-max --use_per_row_quantization                  --keep_weights_quantized                  --enable_float_fallback          # ← 🔴 不要给 --input_list
```

🔴 **最容易踩的一步**：如果给了 `--input_list`，quantizer 会**拿校准集把「故意不给 encoding」
的激活重新量化回去**，产物是 `x=uFxp_16 / out=uFxp_16`，wFxp_actFP **不成立**。
文档原文：这两个选项**互斥且必居其一** ⇒ **不给校准集就是"不要量化激活"的官方表达**。

**验证方式（不要看命令行，要读回产物）**：
```
python scripts/qairt_tool.py snpe-dlc-info -i <op>_quantized.dlc -d -s enc.csv
grep -o "<tensor> (data type: [A-Za-z_0-9]*" enc.csv
```
期望：激活 `Float_16`、权重 `sFxp_8`。本项目实测产物体积 14.88 MB vs 全定点 15.10 MB
⇒ **权重确实留在 8-bit**。

⚠️ **代价提示**：单算子上 `spillFillBufferSize` 从 **1.97 MB 涨到 88.67 MB（45 倍）**。
转大模型时**必须先量这一项**是否撞 PD 上限（本项目红线 3.3 GB，见 §容量约束）。

⚠️ **收益提示**：本项目单算子实测收益很小（主体相对 L2 0.5006% → 0.4911%，比值 0.981），
因为**残余误差基本全是权重 8-bit 的代价**。⇒ **wFxp_actFP 解决的是激活量化，
如果你的瓶颈在权重，它帮不上忙。先归因再选杠杆。**

### 27.7 🔴 注入 per-axis encoding 时的两个硬性要求（2026-08-22 实测，各踩一次）

**要求一：per-axis 的 encoding 必须是对称的，转换器会强制检查。**

```
ERROR - Encountered Error: PyQnnModelIrGraphSerializer::fillQuantInfoForPerAxis:
        Axis quantization is required to be symmmetric.
        Ensure all encodings are symmetrically quantized
```

⚠️ 注意这条**在 `Processed N quantization encodings` 之后才炸** ——
也就是说 **`Processed N > 0` 只证明名字匹配上了，不证明 encoding 能用**。
判断"注入成功"必须看到最终产物，不能看这一行就放心。

**要求二：🔴「对称」在有符号表示下是 `offset == 0`，不是 `offset == -128`。**

本项目在 CLAUDE.md 约束 8 里早就记了这条教训（*"量化器改用有符号表示，对称是 offset == 0"*），
**2026-08-22 仍然又踩了一次**，原因是照抄了另一个脚本里 `offset == -128` 的写法。

实查证据（`snpe-dlc-info -d -s csv` 转储 per-row DLC）：
**517,120 个 per-channel encoding 的 offset 全部为 0.0**，数据类型 `sFxp_8`。

⇒ 生成 overrides 时应当**断言**而不是猜：

```python
assert all(c[i]["offset"] == 0 for i in range(len(c))), "per-axis offset 非 0，与对称假设不符"
lst = [enc1(c[i], True) for i in range(len(c))]      # per-axis: is_symmetric=True
lst = [enc1(pt[w], False)]                            # per-tensor: 保持 asymmetric
```

**可复用纪律**：**照抄别的脚本时，抄的是「写法」不是「事实」。**
那份脚本的 `-128` 对它自己的场景可能成立，但**目标 DLC 里的真实 offset 必须自己转储确认**。

### 27.8 🔴 做「选择性 FP16」时，只钉输出是不够的——必须连输入一起钉

**场景**：想让大部分激活走 FP16，但某些张量的动态范围超过 **FP16 上限 65504**，必须留在定点。

**直觉做法（错的）**：在 overrides 里给这些张量的**输出**一个 16-bit encoding，其余不给。

**实测结果（2026-08-22，part1b）**：失败。读回产物发现同一个张量的两个名字分裂了：

```
linear_105        uFxp_16     <- 我钉住的（ONNX 名）
linear_105_fc     Float_16    <- 转换器融合出来的 DLC 内部名，**没钉住，而且会溢出**
                                 该张量 max = 175619 >> 65504
```

**根因（查 OpDef 才看得出来）**：`FullyConnected` 只有两种可用组合 ——
`FP16 激活 + SFxp8 权重` 或 `uFxp_16 激活 + SFxp8 权重`，
**没有「FP16 输入 + 定点输出」这一行**。
⇒ 只钉输出时，转换器让 FC 本身跑 FP16、然后在后面**插一个 Convert** 转成定点。
**溢出发生在 Convert 之前的 FC 内部**，钉输出救不了。

**正确做法**：把这些算子的**激活输入**也钉成定点，整个算子才落到 INT16 行。

```python
# 从 ONNX 反查这些算子的激活输入（不是从 DLC 名猜）
prod = {o: n for n in graph.node for o in n.output}
inits = {t.name for t in graph.initializer}
for t in OVERFLOW_TENSORS:
    n = prod[t]
    act_in = [i for i in n.input if i not in inits]   # -> 一并加入钉死清单
```

实测修正后：12 个名字全部落回 `uFxp_16`，620 个激活为 `Float_16`，权重仍 `sFxp_8` per-row。

**可复用纪律三条**：
1. **DLC 里一个 ONNX 张量可能分裂成两个名字**（`X` 与 `X_fc`）。
   overrides 只认 ONNX 名 ⇒ **融合产生的内部名钉不住**，只能靠约束整个算子的数据类型来间接控制。
2. **决定「哪些张量必须留定点」的判据是 `|a|max > 65504`**，
   而 `|a|max` **正是 min-max 校准记录在 encoding 里的量** ⇒ 可从现成的 `snpe-dlc-info -d -s csv`
   对全模型直接筛出来，**不必重跑模型**。
3. **凡是被迫留定点的算子，它的输入也一起被迫留定点** ⇒ 统计"能省多少"时要按**算子**算，不是按张量算。

### 27.9 🔴🔴 选 FP16 张量时，**只看张量自身的动态范围是不够的**

**2026-08-22 实测代价：一次完整的量化 + 建图 + 上机（约 1 小时机器时间），输出 38% 是 inf/nan。**

**错误判据**：`|a|max <= 65504`（FP16 上限）⇒ 可以走 FP16。
按此筛 part1b：655 个 16-bit 激活里只有 6 个超标，看起来 98.2% 都能转。

**实际结果**：转完之后设备输出 **38% 是 inf/nan**。

**根因（🔴 2026-08-22 当日订正，见本节末「订正」）**：
~~RmsNorm 在图里是拆开的算子，`x²` 是一个独立的 FP16 张量~~ —— **该表述已作废**。
**实查 DLC：转换器把整个 RmsNorm 模式融合成单个 `RmsNorm` 算子**
（part1b / part2a 里 `pow_*`/`mean_*`/`rsqrt_*` 张量数**均为 0**；part2a 显示 `RmsNorm: 46` 个）
⇒ **图里没有独立的 FP16 `x²` 张量**。

**目前能站住的只有**：①FP16 版设备输出 **38% inf/nan**（实测）；
②RmsNorm 输入 `|a|max` 达 238418、其平方 5.68e10 **远超 FP16 上限 65504**（实测）。
③「溢出发生在融合算子内部的平方和累加上」是**未验证假设**——
融合算子内部是否用更高精度累加，本项目**未查证**。

⇒ 若内部确实以 FP16 累加，则对 RmsNorm 输入的真正约束是 **`|x| <= sqrt(65504) ≈ 256`**，不是 65504。
实测 part1b：**44 个 RmsNorm 输入里 37 个（84%）超标**，part2 是 **81/90（90%）**，最极端 `x² = 5.68e10`。
⚠️ **这是一条保守筛选规则，不是已证实的机制**。

**正确做法**：对每个候选张量，**沿其消费者算子推算「最大中间量级」**，再与目标格式上限比。

```python
# 至少要覆盖这几类放大
#   Pow(x, 2) / Mul(x, x)      -> 需要 |x| <= sqrt(FMT_MAX)
#   累加 / ReduceSum           -> 需要 |x| * N <= FMT_MAX
#   连乘                       -> 逐级推算
consumers = {}                     # 从 ONNX 反查每个张量的消费者
for n in graph.node:
    for i in n.input: consumers.setdefault(i, []).append(n)
```

### 订正记录（2026-08-22 当日）

本节初稿把死因写成「独立的 `Pow` FP16 张量溢出」，**是从 ONNX 结构推理 DLC 行为**，
而 DLC 已把 RmsNorm 融合。**实测事实不变（38% inf/nan、平方超上限），但机制解释降级为未验证假设。**
⇒ 教训：**要解释后端行为，必须查后端制品（DLC/OpDef），不能拿源模型结构代替。**

**更一般的教训**：**浮点格式的「范围」优势要按算子链算，不能按张量算。**
FP16 的 5 bit 指数在深层网络里比想象中容易撑爆。
若确实需要大动态范围，**BF16 的指数范围与 FP32 相同**（尾数 8 bit），
往往是比 FP16 更合适的选择 —— 前提是目标后端的 OpDef 里有 BF16 行（本项目实查：
RmsNorm / ElementWise / Softmax / FullyConnected 都有）。

### 27.10 🔴🔴 **BF16 与量化互斥** —— 转新模型时先知道这条，能省掉一整条思路

官方文档 `QNN/general/converters.html` §"BF16 Graph generation use cases" 逐字：

> *"BF16 graph generation is supported through QNN and QAIRT for conversion of ONNX models
> **without QDQ nodes or overrides**."*
> *"**Quantization isn't supported for BF16 graphs.**"*

实际报错（qairt-converter 2.48）：

```
ERROR - Encountered Error: Currently, BF16 isn't supported with models having
        quantization overrides or QDQ nodes.
```

⇒ **不存在「权重 8-bit 定点 + 激活 BF16」这种配置。** 要 BF16 就是整张图 BF16。

**这条与 §27.2 的 `wFxp_actFP` 形成鲜明对比**，转模型时要分清：

| 想要的 | 可行吗 | 依据 |
|---|---|---|
| 权重定点 8-bit + 激活 **FP16** | ✅ 可以（`wFxp_actFP`，`--keep_weights_quantized`） | OpDef 有该行；本项目实测产出成功 |
| 权重定点 8-bit + 激活 **BF16** | ❌ **不可以** | 官方明写 BF16 不与 overrides/QDQ 共存 |
| 整张图 BF16（放弃量化） | ⚠️ 可以，但权重 1→2 字节 | 本项目因内存出局 |

**选型建议**：
- 若模型的**动态范围**撑得住 FP16（关键是看 `Pow`/累加等下游放大后的量级，见 §27.9），
  优先 `wFxp_actFP` —— 权重字节数不变。
- 若动态范围撑不住 FP16，**BF16 救不了你**（它不能与量化共存），
  只能退回定点，或者接受整图 BF16 的体积代价。

### 27.11 ⚠️ **SNPE CPU 运行时跑不了 per-row（axis-quant）DLC**

```
error_code=202; error_message=Invalid fixed point parameter.
Dequantization of axis-quant tensor is not supported for FullyConnected
```

⇒ 一旦模型用了 `--use_per_row_quantization` / `--use_per_channel_quantization`，
**`snpe-net-run` 的 CPU 参考就没法跑了**。

**这条会静默污染"CPU 参考"这个概念**：
如果你的流水线里 CPU 参考仍然跑得通，那它跑的**一定是另一份 per-tensor 的 DLC**，
而不是你部署的那份。本项目就是这样——`fc_probe` 的 CPU 参考取自
`dlc_pipeline/<part>/<part>_quantized.dlc`（基线 per-tensor），
而设备跑的是 per-row。**两者不是同一个配置**，引用结论时必须写清楚。

**替代方案**：用 **FP32 ONNX + onnxruntime** 直接算真值。更干净：
不含任何量化误差，也不受 `--enable_cpu_fxp` 那类浮点/定点前提影响（见 §关于 CPU 参考的坑）。
代价是要把中间张量加进 ONNX 的 graph.output（纯宿主操作，不需要重新量化）。

---

## 二十八、🔴🔴 转任何新模型，量化后**第一件事**：跑一次「量化健康检查」

> 本节是 2026-08-23 一次代价高昂的失误换来的。
> 当时我用相对 L2 沿链做归因，得出"误差由 RmsNorm 固有放大主导、**无解**"的结论，
> 并据此准备收口。**结论是错的**：真正的病灶是**一个张量的量化分辨率被摧毁**，
> 而它恰好落在我探针的空隙里。下面这套检查**十分钟就能发现它**。

### 28.1 唯一需要的指标：**量化级/元素**

```
levels_per_element = median(|a|) / scale
```

- `a` = 该张量的 **FP32 真值**（跑一次 FP32 ONNX 前向即可）
- `scale` = 该张量在**量化 DLC 里的 encoding 步长**（`snpe-dlc-info -d -s csv` 读出）

**含义**：主体元素平均分到几个量化级。**< 10 就是危险，< 1 意味着主体被量化成 0。**

### 28.2 为什么这个指标比相对 L2 好用

本项目 part2a 的实测，15 个张量按该指标排序后**干净地分成两类**：

| 张量 | 量化级/元素 | 判断 |
|---|---|---|
| `mul_21`（SwiGLU 输出） | **0.21** | 🔴 主体被抹平 |
| `linear_7`（FFN down） | **1.07** | 🔴 |
| `add_8`（残差） | **3.71** | 🔴 |
| 其余 12 个 | **31 ~ 2661** | ✅ |

**跨 4 个数量级的清晰分界。** 而同样这批张量用相对 L2 看，是 2.5% ~ 54% 的连续谱，
**看不出哪里是断崖**。

**更重要的是它直接指向原因**：`mul_21` 步长 **0.227**、主体中位 **0.0485**
⇒ 量程 `[-9700, +5165]` 被离群值撑开，而主体只有 0.05 ⇒ **量程/主体 ≈ 200,000，而 16 bit 只有 65536 级**。

### 28.3 它能挡住的错误判断

实测：`mul_21` 的**纯表示误差 53.77%**，HTP 实测 **54.40%** —— **几乎完全吻合**。

⇒ **在把误差归因给"硬件缺陷"或"算子固有放大"之前，先算这个指标。**
本项目差一点就把一个**纯分辨率问题**报成"HTP 数值特性，无解"。

**Occam 顺序（写死）**：
表示误差 → 上游累积 → 算子固有传播 → **最后才是**后端算术缺陷。
**每一层都要有数字，不能跳。**

### 28.4 结构上天然高危的算子（转新模型时直接去查这几类）

| 模式 | 为什么危险 |
|---|---|
| **SwiGLU / GeGLU：`silu(x) * y`** | 两个量相乘，动态范围相乘；本项目实测量程达 `[-9700, +5165]` 而主体 0.05 |
| **FFN down-projection** | 上游 massive activation 汇聚；本项目 `\|a\|max` 达 77,704 |
| **残差 Add** | 累加放大量级；本项目 `add_8` 只有 3.71 级/元素 |
| **任何被称作 "massive activation" 的张量** | 少数元素占据全部量程 |

⚠️ **SwiGLU 现在是主流架构的标配**（LLaMA 系、Flux 系都用）
⇒ **转任何现代 transformer，这一类都要优先查。**

### 28.5 🔴 做归因时的探针纪律

**要判定"某算子是放大点"，它的直接输入必须也在探针集内。**

本项目的失误：链条从 `linear_5` 直接跳到 `mul_23`，中间的 `mul_21` 不在名单里，
于是 `mul_21` 的 **×9.14** 被整个记到了下游 RmsNorm 头上。
⚠️ **当时的可比性门（主体余弦 > 0.9）12/12 全过，并没有拦住** ——
**门过 ≠ 探针够密。** 门只保证"测量有意义"，不保证"没漏掉算子"。

### 28.6 另一条：**标定 max 会低估真实 max**

实测：`linear_7` 的 encoding 记 `max = 76492.65`，而同一输入下 FP32 真值的 `|a|max = 77704.5`
—— **超出 1.6%**。校准集覆盖不到全部运行分布。

⇒ 凡是基于「量程」做的决策（尤其**选浮点格式时判断会不会溢出**），
**必须用实测 FP32 真值 + 安全余量，不能用 encoding 里的标定值。**

### 28.7 十分钟的执行清单

1. 量化出 DLC 后，`snpe-dlc-info -i <q.dlc> -d -s enc.csv`，解析出每个激活张量的 `scale`
2. 结构化列出高危候选（§28.4 那四类），用 ONNX 加 `graph.output` 跑一次 FP32 前向
   （**只算 `median(|a|)` 与 `|a|max` 两个标量，不落盘**，分批控内存）
3. 算 `levels_per_element`，**排序**
4. `< 10` 的挑出来 —— 这就是你的手术名单
5. 再看它们的 `|a|max`：**留 2 倍余量**判断能否放进 FP16（65504）

**全程纯宿主、不占设备、不需要重新量化。**

### 28.8 🔴🔴 **张量的数据类型由「生产者算子」决定，不由 encoding 决定**

**这是 §27.8 那条规则的通则形式。写 §27.8 时我只针对 FullyConnected，
结果在 `Mul` 上又栽了一次，代价是一整轮量化+建图+上机。**

**实测证据**（part1b 选择性 FP16 产物）：

```
overrides 里写了：    mul_481                                              <- 意图：保持定点
产物里实际是：        mul_481 = Float_16                                   <- 没钉住
产物里多出来一个：    mul_481_converted_QNN_DATATYPE_UFIXED_POINT_16 = uFxp_16
```

⇒ **给张量写 encoding 并不会把该张量变成定点。** 转换器的做法是：
让该张量保持在**周围算子的类型**里，然后在**消费者边界插一个 Convert 节点**产生定点副本。

**后果（本项目的 inf 来源）**：`mul_481` 是 SwiGLU 输出，`|a|max` 标定 62,259、
实测约 93,000~118,000（校准低估 1.5~1.9 倍）⇒ **它活在 FP16 里 ⇒ inf**，
尽管我"钉"了它。

### 28.9 由此得出的两条操作规则

**规则 A（要保持定点）**：
> **必须让该张量的生产者算子整体保持定点** —— 递归地给**生产者的所有激活输入**也写 encoding。
> 只写该张量自己的 encoding **无效**。

**规则 B（要转浮点）** —— 🔴 **2026-08-23 当日实测推翻，原文见下方删除线**：

> ~~只需让生产者算子跑浮点（不给该算子的输入/输出写 encoding），
> 转换器会自动在下游定点算子的边界插 Convert ⇒ "外科手术式"转浮点可行，浮点不会无限扩散。~~

**实测结果：浮点会向上游无界扩散。**
part2a 试验：白名单 **23 个**张量不给 encoding，**其余 695 个激活全部给了 encoding**，
产物里 `Float_16` 张量却有 **183 个** —— 连 `linear_1_fc`、`view_*`、
以及 RmsNorm 的 `layers.NN.*_norm1.weight` 都被拖成浮点。

**机制**：把张量 T 设为浮点 ⇒ T 的生产者算子跑浮点 ⇒ 该算子的输入也须浮点
⇒ **沿输入锥（input cone）一路倒推**。给上游张量写 encoding **拦不住**
（与 §28.8 同源：**encoding 不决定张量类型，算子选的配置才决定**）。

**后果**：浮点扩散到注意力块后，`node_MatMul_106` 校验失败：
```
<E> [4294967295] has incorrect Value -32036, expected equal to -32768.
<E> Failed to validate op node_MatMul_106 with error 0xc26
```
（16-bit MatMul 在该配置下要求**对称**量化 offset=-32768，而部署那份 asymmetric 的 -32036 本来是合法的
—— **是浮点扩散改变了该算子选中的配置行**。）

⇒ ⚠️ **"只把个别张量转浮点"在 QAIRT 2.48 的 overrides 机制下并不成立。**
要做选择性浮点，只能接受**整个输入锥**跟着浮点，并据此评估体积/范围风险。

**方法论教训**：我在同一天里写下 §28.8（正向：encoding 不决定类型）之后，
**立刻从它反推出一条未经验证的规则 B 并当成结论写进指南**。
🔴 **从一条实测规律"顺手"推出的逆命题，必须单独验证后才能写。**

### 28.10 🔴 与之配套：**筛选阈值必须按「实测量程」定，不能按标定值**

本项目实测，校准集给出的 `max` 相对真实运行值的低估倍数：

| 张量 | 标定 max | 实测 \|a\|max | 低估 |
|---|---|---|---|
| `mul_21`（SwiGLU） | 5,164.6 | **9,818.4** | **1.90×** |
| `mul_171` | 1,421.9 | 2,774.3 | 1.95× |
| `mul_71` | 15,715.0 | **28,777.3** | 1.83× |
| `linear_7` | 76,492.6 | 77,704.5 | 1.02× |

🔴 **【2026-08-23 当日订正】上表只有 4 个样本，"低估近 2 倍"这个规律是错的。**
扩到 11 个被排除张量实测后，**实测/标定 的比值从 0.09× 到 1.90× 都有**：

| 段 | 实测/标定 |
|---|---|
| part1b（6 个） | 0.93× / 0.87× / 0.94× / 0.97× / 0.81× / 1.04× |
| part2a（2 个） | 1.05× / 0.96× |
| **part1a（3 个）** | **0.09× / 0.16× / 0.20×**（`mul_131` 标定 316,967、实测仅 **27,819**，高估 11 倍） |

⇒ ②**标定值与真实量程的关系逐张量不可预测，既会低估也会大幅高估。**
**没有可用的倍数系数** —— 判断量程**只能实测**（跑一次 FP32 前向，几分钟）。
⚠️ 我曾据 4 个样本外推出「×4 保守代理」并用它建了三段图，
**方向上保守（不会放进危险张量），但因此漏掉了 8 个本可入选的张量**。

**本项目 #102 失败的完整归因（两个独立失误叠加）**：
1. 筛选用「标定 max > 65504」，只抓到 6 个；**按 1.9 倍低估，有效阈值应是 ~34,000**
   ⇒ 漏掉 `linear_97`(38,512)、`linear_113`(55,449)
2. 被"钉"的 `mul_481`(62,259) 因**规则 A 未被遵守**而实际仍是 FP16

⇒ **凡是基于量程的决策，一律用「FP32 前向实测的 `|a|max` × 安全系数 2.0」。**

### 28.11 🔴🔴 `--enable_float_fallback` 跳过的不只是校准，**还有后端专用的图修正**

**症状**：用 overrides 注入全部 encoding + `--enable_float_fallback` 重建模型，
`qnn-context-binary-generator` 在 **175 毫秒**内就失败：

```
<E> [4294967295] has incorrect Value -32036, expected equal to -32768.
<E> QnnBackend_validateOpConfig failed 3110
<E> Failed to validate op node_MatMul_106 with error 0xc26
```

**关键**：这个失败与浮点**毫无关系**。
用**空白名单**（零个浮点张量、纯复现部署 encoding）重建，**报同样的错**。
—— 这个 5 分钟的对照实验是唯一能把"配方问题"与"浮点问题"分开的手段，
**我在做它之前已经错误归因了两次**（"残差 Add 拖累"、"浮点无界扩散"），各浪费一轮构建。

**根因（实测）**：部署 DLC 里存在 **31 个 `*_converted_unsigned_symmetric` 张量**（`offset = -32768`），
每个注意力块一对：

```
val_102_converted_unsigned_symmetric        offset -32768
transpose_2_converted_unsigned_symmetric    offset -32768
```

⇒ **量化器在校准路径上会自动为「双激活 MatMul」的输入插一个「转无符号对称」的 Convert**
（HTP 的 16-bit MatMul 要求对称量化）。
走 `--enable_float_fallback`（无校准）注入 encoding 时，**这一步修正被整个跳过**
（我的产物只有 15 个这类张量，部署有 31 个）。

**修法**：把部署 DLC 里那份对称 encoding **直接注入到基础张量名上**，复现量化器本该做的事：

```python
SYMSUF = "_converted_unsigned_symmetric"
for k, e in deployed_encodings.items():
    if k.endswith(SYMSUF):
        base = k[:-len(SYMSUF)]
        if base in acts:
            acts[base] = [enc(e, is_symmetric=True)]   # offset -32768
```
实测：修正后两臂 context 均正常产出（1443.5 / 1440.3 MB），G2/G3 门全过。

### 28.12 由此固化的一条通用做法

> **凡是"用 overrides 复现一份已量化模型"的场景，
> 先把部署 DLC 的张量名清单与你的产物做集合差。**
> 出现在部署里、你产物里没有的 `*_converted_*` 张量，
> 就是**量化器自动做过、而你的路径跳过了的修正**。

这比读报错高效得多 —— 报错只告诉你**哪个算子**不合法，
集合差直接告诉你**少做了什么**。

⚠️ 已知会被自动插入的 Convert 至少有两类：
`*_converted_unsigned_symmetric`（16-bit MatMul 对称化）、
`*_converted_QNN_DATATYPE_FLOAT_16`（定点↔浮点边界）。

---

## 二十九、✅ 完整配方：**手术式选择性 FP16**（2026-08-23 实测有效，端到端改善 12.73 pp）

> 本章是 §二十八「量化健康检查」的落地手术。
> 本项目实测：端到端 `E_all` **43.41% → 30.68%**，到浮点参考地板的差距**关掉 46.4%**；
> 成图从"毛发尖刺过锐"变为"质感自然、多出背景细节"。

### 29.1 适用判断（先确认你的模型有这个病）

跑 §28.7 的健康检查。若出现**「量化级/元素 < 10」的张量**，本章适用。
本项目四段实测：**每段的 SwiGLU 输出全部中招**（0.06~0.82 级/元素）。

### 29.2 选择规则（三条，缺一不可）

| # | 规则 | 为什么 |
|---|---|---|
| 1 | 量化级/元素 **< 10** | 分辨率确实被摧毁（§28.1） |
| 2 | **实测** `\|a\|max` × **2.0** ≤ 65504 | 标定值低估 1.83~1.95 倍（§28.10）。**没有实测时**用「标定 × 4.0」作保守代理 |
| 3 | **输入锥有界** ⇒ **残差流张量不得入选** | 残差的输入锥 = 它之前的整个网络，会把浮点铺满全图 |

⚠️ 规则 3 是实测出来的：含 15 个残差 Add 时 `Float_16` 达 **183 个**（铺满全图）；
去掉后降到 **153 个**，可正常建图。

### 29.3 命令配方

```
qairt-converter  --input_network <seg>.onnx --output_path <seg>_fp32.dlc                  --quantization_overrides <ovr>.json --float_bitwidth 16                  --source_model_input_shape <每个输入>

qairt-quantizer  --input_dlc <seg>_fp32.dlc --output_dlc <seg>_q.dlc                  --weights_bitwidth 8 --bias_bitwidth 32                  --param_quantizer_calibration min-max --use_per_row_quantization                  --keep_weights_quantized --enable_float_fallback

qnn-context-binary-generator ... --htp_socs <soc> --config_file <ext>.json
```

**overrides 的构造**（这是全部关键）：

1. **权重**：全部逐字注入（per-axis 必须 `is_symmetric=True` 且实际 offset==0，见 §27.7）
2. **激活**：**除白名单外全部注入**；白名单**故意留空** ⇒ 其生产者算子跑浮点
3. 🔴 **必须打对称修正**（§28.11）：把部署 DLC 里 `*_converted_unsigned_symmetric`
   的 encoding **注入到基础张量名上**，否则 16-bit MatMul 校验失败、根本建不出 context
4. **bias 不注入**（`--keep_weights_quantized` 会自动转浮点）

### 29.4 验证四门（每一门都拦下过真实错误）

| 门 | 检查 | 曾拦下 |
|---|---|---|
| **G1 注入门** | 读回 DLC：白名单为 `Float_16`、权重仍 `sFxp_8`/per-row | 名字没匹配上、per-axis 非对称 |
| **G2 容量门** | context 建得出、无 `Failed to find available PD` | —（本项目四段均通过） |
| **G3 装置门** | `dspArch`/`vtcmSize` 与部署一致（**不看文件名**） | 漏 `--config_file` 编成 v68/4MB |
| **G4 数值门** | 设备输出无 inf/nan | 选择规则漏掉超范围张量 ⇒ 38% inf |

### 29.5 代价

- **context 体积基本不变**（本项目四段 1440~2368 MB，与全定点版同量级）
- **速度**：单算子 1.09×、四段端到端每步约 84 s，**远在"翻倍出局"的红线内**
- **浮点会沿输入锥扩散**：白名单 8 个 ⇒ 实际 153 个激活变浮点（约 21%）。
  ⇒ 汇报时措辞必须是「**这一组改动**」，不得说成「只改了 N 个张量」

### 29.6 🔴 本方法找到的过程（比结论更值得抄）

**三次差点收口，每次都是手上已有的数据能反驳，只是没读懂**：

1. 用相对 L2 沿链归因 ⇒ 结论"RmsNorm 固有放大、无解"。
   **错因：探针太稀**，真正放大 ×9.14 的 SwiGLU 落在探针空隙里。
   ⚠️ 可比性门 12/12 全过也没拦住 —— **门过 ≠ 探针够密**。
2. 归因给"HTP 数值特性" ⇒ 结论"已拿到全部"。
   **错因：跳过了 Occam 第一层**。回头算纯表示误差 **53.77%**，与实测 **54.40%** 吻合。
3. 首次建图失败 ⇒ 结论"选择性浮点方向不可行"。
   **错因：没做对照实验**。5 分钟的空白名单对照证明失败与浮点**无关**，是配方缺一步。

⇒ **可复用纪律**：
**"当前尝试失败" ≠ "这条路不通"。判定方向错误之前，必须先做一次能把"配方问题"
与"方向问题"分开的对照实验。**

### 29.7 🔴🔴 **转更多张量到 FP16 不是单调更好的 —— 存在最优点**

**实测（同装置、确定性执行、四段端到端 `E_all` vs FP32）**：

| 配置 | 白名单张量数 | **E_all** | 主体余弦 |
|---|---|---|---|
| 全定点 per-row（基线） | 0 | 43.41% | 0.9031 |
| **手术式 FP16** | **26** | **30.68%** | **0.9488** |
| 手术式 FP16（扩充） | 32 | **39.11%** | 0.9177 |

⇒ **多加 6 个张量，`E_all` 退回 8.4 pp。**

**机制（假设，与实测一致但未完全隔离）**：
FP16 的相对精度地板约 **0.02%**，而 uFxp_16 对**分辨率本来就好**的张量可以做到 **0.007%**
（本项目实测 `mul_314`：uFxp_16 0.0069% vs FP16 0.0203%，**FP16 差 3 倍**）。
而浮点会**沿输入锥扩散**（§28.9）：白名单每多一个，就有约十几个张量被拖成浮点。
⇒ 扩大白名单 = 把大量**良态**张量从"好的定点"换成"较差的浮点"。

**⇒ 可操作的结论**：
1. **只转真正坏的张量**（量化级/元素 < 10），**不要贪多**。
2. **每次扩充白名单都要重测**，不能假设"加了更多总不会更差"。
3. 因为浮点会扩散，**白名单的边际成本随规模上升** —— 越往后加，被拖下水的良态张量越多。

⚠️ **成图侧的观察**：`E_all` 30.68% 与 39.11% 两版的成图**肉眼难分优劣**（都是干净的猫照，
构图不同）。⇒ 标定尺给出的排序**不一定对应画质排序**（本项目已多次记录该现象）。
**做最终取舍时必须人工看图，不能只看标量。**

### 29.8 📋 本项目的最优配置（可照抄复现）

**Z-Image Turbo / SM8750 / QAIRT 2.48 / A16W8 per-row，四段布局。**

| 段 | 白名单（转 FP16 的张量） | 数量 |
|---|---|---|
| part1a | `mul_31 / mul_86 / mul_107 / mul_181 / mul_206 / mul_231 / mul_256 / mul_281` | 8 |
| part1b | `mul_306 / mul_356` | **2** |
| part2a | `mul_21 / mul_46 / mul_71 / mul_96 / mul_121 / mul_146 / mul_171 / linear_55` | 8 |
| part2b | `mul_196 / mul_221 / mul_246 / mul_271 / mul_296 / mul_321 / mul_346 / mul_371` | 8 |

**全部是 SwiGLU 输出**（`linear_55` 是 FFN down）。合计 **26 个**。

**效果（三层指标，与 FP32 参考同噪声同 caption）**：

| | L1 `E_all` | L2 终点 latents | **L3 图像 PSNR** |
|---|---|---|---|
| 全定点 per-row | 43.41% | 69.34% | 11.93 dB |
| **本配置** | **30.68%** | **57.70%** | **15.53 dB** |

⚠️ **不要贪多**：part1b 加到 5 个（`mul_406/431/456`）⇒ L1 退回 39.09%、L3 退到 14.32 dB。
part1a 加到 11 个则几乎无影响。⇒ **每次扩充必须重测**（§29.7）。

**未治的（当前工具链下无解）**：
量程超 FP16 的张量 —— part1b 的 `mul_331`(34,296)/`mul_381`(34,340)/`mul_481`(64,474)、
以及整类 FFN down 输出（`linear_7` 量程 77,704）。
BF16 本可容纳，但**与量化互斥**（§27.10）。

### 29.9 为什么"更多 FP16"会变差：相对精度 vs 绝对精度

| | 小值元素（\|a\|≈0.05） | 大值元素（\|a\|≈25,000） |
|---|---|---|
| uFxp_16（步长 0.77） | 误差 ~0.22 → **极差** | 误差 ~0.22 → **极好** |
| FP16（相对 0.05%） | 误差 ~2.5e-5 → **极好** | 误差 ~12.5 → **差 57 倍** |

⇒ ②**FP16 救主体、毁离群值。**
只有当「主体收益 > 离群损失」时换 FP16 才划算，
而这取决于**下游对离群值的敏感度** —— 无法先验判断，**只能实测**。

---

## 三十、🔴🔴 **正确的混合精度做法：encodings JSON 里的 `dtype` 字段**

> 官方文档 `QNN/general/quantization.html` §"Mixed Precision and FP16 Support"。
> **本项目此前一直用未文档化的旁路（"不给 encoding + `--enable_float_fallback`"），
> 导致浮点无界扩散（白名单 8 个 ⇒ 产物里 183 个浮点张量）。**

### 30.1 机制

encodings JSON 里每个张量可显式声明类型：

```json
{"version": "0.6.1",
 "activation_encodings": {
   "a": [{"bitwidth":16,"dtype":"int","scale":...,"offset":...}],
   "m": [{"bitwidth":16,"dtype":"float"}],        <- 只有它是浮点
   "y": [{"bitwidth":16,"dtype":"int","scale":...,"offset":...}]},
 "param_encodings": {
   "W": [{"bitwidth":8,"dtype":"int","scale":...,"offset":...}]}}
```

配套命令：

```
qairt-converter  --quantization_overrides <json> --float_bitwidth 16
qairt-quantizer  --input_dlc ... --input_list <校准集>                  --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 --float_bitwidth 16
```

🔴 **必须走 `--input_list` 校准路径**（不要用 `--enable_float_fallback`）。

### 30.2 实测验证（两算子探针：`Mul` 然后 `MatMul`，只让 Mul 输出为 FP16）

| 张量 | 声明 | converter 后 | **quantizer 后** |
|---|---|---|---|
| a / b | int 16 | uFxp_16 | **uFxp_16** ✅ |
| **m** | **float 16** | Float_16 | **Float_16** ✅ |
| y | int 16 | Float_16（被 MatMul 规则带浮） | **uFxp_16** ✅ 被纠正回来 |
| W | int 8 | Float_16 | **uFxp_8** ✅ |

**浮点张量总数 = 3**（目标 1 个 + 2 个自动插入的 Convert）⇒ **零扩散**。

⚠️ 中间过程值得注意：converter 阶段因 MatMul 的「三者同类型」规则把 `y`/`W` 也标成了浮点，
**是 quantizer 的校准路径按显式 encoding 把它们纠正回定点的**。
⇒ **不能只看 converter 产出下结论，必须看 quantizer 之后的最终 DLC。**

### 30.3 这一个改动同时解决三件事

| 问题 | 为什么被解决 |
|---|---|
| 浮点沿输入锥无界扩散（§28.9） | 每个张量显式声明类型，不靠推断 |
| `--enable_float_fallback` 跳过后端图修正（§28.11） | 走校准路径，`*_converted_unsigned_symmetric` 等修正**自动应用**，无需手工注入 |
| 依赖未文档化行为 | 改用官方记载机制 |

### 30.5 🔴 **反例：机制正确不等于设备上兑现收益**（2026-08-25 实测，本项目 #123）

`dtype` 机制**确实**做到了零扩散（30.2 已验证），但**把同一批张量用 dtype 声明成 FP16 之后，
设备端一致性并没有比旧机制更好**——旧机制那份"脏"的、扩散出 435 个浮点张量的产物**反而更好**。

**已确认的解释**：旧机制扩散出来的浮点**不是噪声，是承重的**。
①实测（#128）：只保留显式钳位、去掉扩散锥 ⇒ 只值 **1.44 pp**；
而扩散锥本身贡献远大于此。⇒ **"沿输入锥扩散"在数值上等价于给整条子图升精度**，
它的副作用（不可控、体积大）是真的，但它的**收益也是真的**。

⇒ **转新模型时的正确做法**：用 `dtype` 机制做**定位**（零扩散 ⇒ 单变量干净），
但**不要假定把定位结果照搬成 dtype 白名单就能拿到同样收益**——
锥内那些"顺带被抬上去"的张量可能才是主要贡献者。**两者必须各自实测。**

### 30.4 版本与坑

- `version` 只接受 **`"0.5.0"` / `"0.6.1"`**；`1.0.0` 报 *Name must be present in every encoding*，
  `2.0.0` 报 *'encodings'*
- 只写 `{"bitwidth":16,"dtype":"float"}` 而**不给 `--float_bitwidth 16`** ⇒ 产出是 **Float_32** 不是 Float_16
- 文档明写：**未出现在 JSON 里的张量按定点处理**（用 `--act_bw/--weight_bw` 默认值），
  **不是**落成浮点 —— "缺 encoding ⇒ 浮点"只在加了 `--enable_float_fallback` 时成立

## 三十一、🔴🔴 **误差归因：先在宿主上用 QDQ 模拟把误差拆开，再决定上不上设备**（2026-08-25/26 实测）

> **这一节解决的是转新模型时最贵的问题：不知道该修哪里，就只能一个个试，每次几小时设备时间。**
> 本项目靠它把一个卡了 60 多条台账的问题（"权重量化还是激活量化占大头"）在**宿主上 1 天**内关闭。

### 31.1 为什么必须在宿主做

设备上你**只能测端到端**：改一处 → 重量化 → 重建 context → 推到手机 → 跑 8 步 → 看图。
一轮 2~4 小时，而且**同时变了很多东西**，拆不开。

宿主上用 onnxruntime 跑 FP32 原图，**在图里手工插入量化-反量化（QDQ）节点**，
就能**任意组合**"只量化权重 / 只量化激活 / 两者都量 / 都不量"，每轮约 6 分钟，且**完全单变量**。

### 31.2 怎么插 QDQ（opset 17 没有 uint16 的 QuantizeLinear）

用四个算子手工拼：

```
Div(scale) → Round → Clip(0, 2^bw - 1) → Mul(scale)      # 对称/零点自行加减 offset
```

- **激活**：per-tensor，scale/offset 直接从量化器产出的 encodings JSON 里读，**不要自己重算**
  （自己重算 = 又造了一把没标定的尺，见约束 7）
- **权重**：per-channel。🔴 **轴必须验证，不能假定**——
  本项目在 21 个无歧义样本上逐个核对，确认全是 `axis=1`，**然后才**把这条规则用到有歧义的张量上
- **自检是强制的**：必须有一个"什么都不量化"的模式（本项目的 mode Z），
  **要求与直接跑 FP32 原图逐位相同**。这一步过不了，后面所有数都不能信

### 31.3 本项目的分解结果（示范这张表长什么样、能推出什么）

| 配置 | 首步误差 `E_all` | 读法 |
|---|---|---|
| FP32 原图 | 0% | 基准 |
| **只量化权重**（int8 per-row） | **5.25%** | ⇒ 权重不是问题 |
| **+ 激活 uFxp16** | **35.25%** | ⇒ **激活量化独占 30 pp，是绝对大头** |
| **+ 真实 HTP 执行** | 43.41% | ⇒ 后端本身再加 8 pp |

⇒ 一张表同时否掉了两个流行猜想（"8-bit 权重不够""HTP 有 bug 是主因"），
并**直接指出唯一值得投入的方向**。**转新模型时这应当是量化后的第一张表。**

### 31.4 🔴 **配套的必做项：massive activation 溢出筛查**

激活占大头时，第一个要查的**不是**"位宽够不够"，而是**有没有张量的动态范围本身就超出目标格式**。

**FP16 最大值 65504。** 本项目实测：转 FP16 的张量里有一批 `|a|max` 远超 65504
（massive activation / 激活离群点），转过去直接 **inf**。

**做法**：
1. 实测**每个**候选张量的 `|a|max`（不是估，不是按类型猜）
2. 超 65504 的，在图里**显式插入 `Clip(±65504)`**
3. 收益实测：本项目 part1a 7 / part1b 7 / part2a 4 / part2b 0 个张量插 Clip ⇒
   **端到端图像 PSNR 15.53 → 18.67 dB（+3.14）**，代价是生成时间 +13.5%

⚠️ **不要用 percentile 校准去"收紧量程"代替钳位**——那是砍掉动态范围换分辨率，
本项目实测是**净损害**（见 §约束 4·补 / 台账 G0-cost）。
**钳位（只削掉真正溢出的极值）与 percentile（按分位数削）不是一回事。**

### 31.5 🔴 **宿主侧的陷阱：一次声明几百个图输出会把机器压死**

要拿到几百个中间张量的统计量时，最自然的写法是把它们全部加进 `graph.output` 跑一次。
**不要这么做。**

**机制**：张量一旦成为 graph output，ORT 的内存规划器就**不再为它复用缓冲区**。
本项目实测：一次声明 **606** 个输出 ⇒ 宿主可用内存掉到 **0.3 GB**，8 分钟连一批都没跑完。

**正确写法**：**分批**，每批**单独建 session、只声明该批的输出**（本项目 `NMAX=32`），
并加一道 `free_gb() < 3.0 ⇒ 中止` 的闸门。

---

## 三十二、⚡ **测「加载 vs 计算」的标准手法，以及那个几乎必踩的分母陷阱**（2026-08-26 实测）

转完模型要上产品时，第一个性能问题总是"时间花在哪"。

### 32.1 手法（工具自带，不用改代码、不用重建产物）

```
qnn-net-run --retrieve_context <seg>.bin --num_inferences N ...
```

同一段分别跑 **N=1** 与 **N=5**，对耗时做线性拟合：

- **斜率** `(T5 − T1) / 4` = **纯计算/次**
- **截距** `T1 − 斜率` = **加载 + 初始化**

成本：每段约 3 分钟，零风险。

### 32.2 🔴🔴 **分母陷阱：`qnn-net-run` 的总耗时不是 app 的每步耗时**

本项目实测四段"计算"合计 **74.4 s**，而 app 跑完一整步只要 **32.4 s**——
**离线工具比真实 app 还慢一倍多**。

**原因**：`qnn-net-run` 要**把每个输出张量写盘**（本项目 part1a 有 8 个输出、100+ MB），
**app 完全不付这个成本**（张量留在内存里直接喂下一段）。

⇒ 若拿 `qnn-net-run` 的总耗时当分母算"加载占比"，本项目得到 **11~19% ⇒ 放弃**；
换成 app 实测的每步耗时当分母，得到 **41% ⇒ 值得做**。**结论完全相反。**

**规则**：
- **分母必须是你实际要优化的那个时间**（真实链路的每步/每次耗时，从 logcat 或应用埋点拿）
- 截距（加载）可以跨环境迁移，**斜率（计算）不行**——它含着离线工具特有的 I/O

### 32.3 顺带一条经验数据

**加载时间与 context 体积不成正比**：本项目 2.37 GB 的段加载 3.7 s，
1.49 GB 的段加载 3.4 s ⇒ **固定开销（图反序列化、HTP 上下文准备）占大头**。
⇒ 想靠"把大段拆小"来降低加载开销，方向是错的；**要降的是加载次数，不是单次体积**。


---

## 三十三、🔴🔴 **序列长度会被 `torch.onnx` 烘焙成常量 —— 导出时用的示例 prompt 就是产品的硬上限**（2026-08-27 实测）

### 33.1 事故形态

Z-Image 的 app 一直把 prompt **静默截断到 20 个 token**（可用正文仅 ~12），
用户长 prompt 的尾部完全不起作用，删掉尾部再生成**出图一模一样**。

**「20」不是任何人的设计决定。** 解码当初的校准样本得到：

```
<|im_start|>user\nA cinematic shot of a cute cat sitting on a wooden desk<|im_end|>\n<|im_start|>assistant\n
= 8 个模板 token + 12 个正文 token = 恰好 20
```

⇒ **导出时喂的那句示例 prompt 有多长，产品的 prompt 上限就是多长。**

### 33.2 机制（这是通用陷阱，不是本模型特有）

`torch.onnx.export(..., dynamic_axes=...)` 只把**图输入**标成符号维
（本项目的 TE ONNX 输入确实是 `[?, ?]`、中间是 `[1, seq_len, 2560]`），
但模型内部的 `.view()/.reshape()/.expand()` 拿到的是**追踪期的 Python 整数**，
会被**烘焙成 initializer 常量**：

```
(1, 20, -1, 128)   [B, seq, heads, dim]
(1, 32, 20, 128)   [B, heads, seq, dim]
(-1, 20, 128)      [B*heads, seq, dim]
(1, 32, 128, 20)   转置后 [B, heads, dim, seq]
```

⇒ **看输入是符号维就以为模型支持变长，是错的。** 必须扫 initializer。

### 33.3 检查方法（转任何新模型时做，10 秒）

```python
# 扫出所有"小整数常量里含追踪长度 L"的 initializer
for t in graph.initializer:
    if t.data_type in (INT64, INT32):
        a = numpy_helper.to_array(t)
        if 0 < a.size <= 16 and L in a.reshape(-1).tolist():
            print(t.name, a.reshape(-1).tolist())
```

本项目实测：TE **每段恰好 22 个**；transformer part1a 19 个、part1b/2a/2b 各 5 个。

### 33.4 事后补救：**改常量比重导便宜得多，而且保住张量名**

若已经导错，**不必从 PyTorch 重导**：逐元素把常量里的 `L_old` 改成 `L_new`，
重新转换 + 量化 + 建 context 即可。本项目 TE 四段实测：
**转换分钟级、量化约 20 秒/段、建 context 约 10 秒/段。**

🟢 **不重导的最大好处：张量名不变** ⇒ 按张量名钉死的配方
（混合精度白名单、Clip 注入点、encoding overrides）**全部继续适用**。

### 33.5 🔴 两个必须先验证的歧义（**都真实发生过**）

**替换规则必须是「逐元素把 L_old 换成 L_new」，绝不能按位置或整条形状替换**，
因为 `L_old` 常与别的维度撞值：

| 模型 | 撞值 | 后果 |
|---|---|---|
| 本项目 TE | 目标 seq=**32** 与注意力**头数 32** 撞 | 按位置替换会把头数改坏 |
| 本项目 transformer | `32×128 = 4096`，而 **4096 又是图像 token 数** | 同一字面量两种含义 |

**定稿前必须逐个追消费者**（`[nd for nd in graph.node if const_name in nd.input]`），
按「它出现在形状的哪一维、上游是谁」判定归属，**不许假设**。
本项目实测结论：所有 `4096` 都在序列位且属图像流（不动），`32` 全属 caption 长度（要改）。

### 33.6 还要一起改的两类东西

1. **图尾的桥接结构**：若导出时用 `Concat(hidden[:,0:L], Expand(hidden[:,L-1:L], [1,pad,1]))`
   把 L 桥到下游要求的宽度，改完要把它换成直接切片。
2. **依赖序列长度的数据张量**：先看它们规不规则。本项目 transformer part1a 有 8 个，
   实测**全是 0 或等差 `1..L`** ⇒ 可程序化扩展；**若是学到的参数就只能重导**。

### 33.7 别忘了下游契约（本项目为此栽了一次）

改完模型必须同步**交付契约里的张量规格**，不只是 `sha256`/`size_bytes`。
本项目漏了 `inputs/outputs` 的 `fixed_shape` / `exact_bytes`，真机报
`manifest input byte mismatch: input_ids` —— 详见 `scripts/contract_dryrun.py`，
那是一个**零设备成本**、在宿主上就能复现该错误的干跑校验器。

---

## 三十四、🔴🔴 **改序列长度的图手术：三个会让你以为成功的陷阱**（2026-08-28 实测）

§三十三 说明了序列长度为什么会被烘焙成常量。本节讲**怎么改它而不被静默坑掉**。
三条都是本项目当天实际踩到的，且**每一条都表现为「看起来成功了」**。

### 34.1 只改 initializer 和图 IO 是不够 —— **`graph.value_info` 也带着旧长度**

改序列长度的脚本通常扫三处：形状常量（小的 INT64 initializer）、数据张量（dims 含旧长度的
initializer）、图 input/output 的 dim。**漏掉的第四处是 `graph.value_info`** ——
它是导出时保存的**每个中间张量的推断形状**。

本项目实测（caption 槽 32→80，`unified` 4128→4176）：

| 段 | `value_info` 条数 | 其中仍写旧长度 |
|---|---|---|
| part1a / part1b | **0** | 0 |
| part2a_fixed | 1004 | **793** |
| part2b_fixed | 953 | **740** |

⇒ **两段有、两段没有**。只测有 value_info 的那两段之外的段，会得出"方法正确"的错误结论。
症状是 onnxruntime 加载时报：

```
Node (node_mul) Op (Mul) [ShapeInferenceError] Incompatible dimensions
```

—— 报错指向一个**完全无辜**的算子（`Mul(unified, rsqrt)`，两个输入都在正确的锥上），
因为冲突来自 value_info 里声明的 `pow_1 = [1,4128,3840]` 与实际的 `[1,4176,3840]`。
**又一次「报错说谎」**（约束 11·再补）。

**做法**：把 `g.value_info` 和 `g.input`/`g.output` 一起做维度重映射，
并在结尾断言三者都不再含旧长度。
⚠️ 重映射前**必须先查目标值有没有歧义**：本项目实测 part2a/2b 的 value_info 里
**`32` 一次都没出现过**（只有 4128），所以无歧义；换个模型必须重新查，不得照抄。

### 34.2 🔴 **「转换器不报错」不等于「图是对的」**

本项目先前在带着 809 条陈旧 value_info 的图上跑 QAIRT 转换，**八段全部报成功**，
并据此写下"L=80 转换已验证"。而同一批图**连 onnxruntime 都加载不了**。

⇒ ②**ONNX 图手术后的第一道检查应该是「用 onnxruntime 跑一遍 FP32 前向」，不是「送去转换」。**
形状推断是 ORT 加载时的强制步骤，比转换器严格；转换器可能忽略 value_info，
也可能采信它而**静默产出错误的 DLC**（本项目未区分二者，标为③未验证）。
同类：#64（漏 `--htp_socs` 静默产出非定向 context）、#95（漏 `--config_file` 静默编成 v68）。

### 34.3 🔴 **钳位/白名单集合不能跨序列长度复用 —— 必须在目标长度下重测**

如果你像 §二十九/§三十 那样维护一份"哪些张量走 FP16 / 哪些要插 `Clip(±65504)`"的名单，
**那份名单是在某个具体序列长度下挑出来的，换长度必须重挑。**
⚠️ 先查清楚旧名单**每个成员是怎么挑的**：本项目四段里只有 part1b 用了实测 `|a|max`，
part1a / part2a / part2b **全部用标定挑的**（当时没做实测）—— 这一点直接决定下面的可信度。

本项目实测：`linear_23`（`[1,L,3840]`，唯一 caption-only 的被钳张量）
在 L=80 下实测 **`|a|max` = 84,350，已超 FP16 上限 65504 的 1.29 倍**。

🔴 **本节初稿曾据此写下"序列长度会实质性抬高量程"。那是错的，且不只是"无证据"——
它已被同口径实测直接推翻**（保留原文以说明推翻理由）。

当时拿来对比的 L=32 那个 58,406 是**标定**值，与 84,350 之间**同时差着三样**：
口径（标定 vs 实测）、样本（量化校准语料 vs 单个 prompt）、长度。
而 #110 已实测标定/实测比值横跨 **0.09×~1.90×** ⇒ 1.44 这个倍数完全落在标定误差内。

随后补做同口径配对（L=32 与 L=80 **都用 FP32 前向实测**，n 见下）：

| | 全部配对 | 决策相关区间（L32 实测 >5000） |
|---|---|---|
| n | 605 | **59** |
| L80/L32 比值 中位 | 0.9960 | **0.9889** |
| 范围 | 0.2855 ~ 1.5925 | **0.9706 ~ 1.0179** |
| 比值 >1.05 的 | — | **0 个** |

（`linear_153` 247,617→243,031；`linear_113` 47,720→46,722；`mul_131` 27,819→27,396）

⇒ ②**序列长度 32→80 不改变激活量程，方向甚至略降。**
⚠️ 这些配对**同时差着长度、caption 内容、输入链来源**三样，捆绑不可分离；
但对"会不会溢出"这个问题，≤1.8% 的散布**框住了三者合计的影响**。

⇒ 🔴 **那 84,350 的真实来源是「标定不准」**，不是长度。同一次实测里 part1a 的标定/实测比：
`linear_18` **11.67×**、`mul_131` **11.57×**、`linear_10` **8.00×**（高估），
`linear_23` **0.69×**（低估 1.44×）。

⇒ **规程不变，但理由要换成对的那个**：
不是"长度会抬高量程"（已推翻），而是——
②**旧名单里凡是用标定挑的成员，其依据本来就不可靠**（本项目实测高估达 11.67×、低估达 1.44×）。
换长度时你**必然要重跑一次前向**（校准数据也得重生成），
**顺手把 `\|a\|max` 实测出来几乎不额外花钱**，却能把整份名单从"标定推测"升级为"实测"。
本项目正是靠这一步，才发现部署配置里有 3 个多余的钳位、和 1 个标定说安全实则超限的张量。

⇒ **规程**：改序列长度后，在**新长度**下重跑一次 FP32 前向逐张量测 `|a|max`，
把选择规则重新套一遍，与旧集合做集合差。**新成员必须补进钳位/白名单**，
否则它在设备上变 `inf`（本项目实测过这种死法：38% inf/nan，图全毁）。
成本：宿主约 1 小时/段，零设备 —— 与「白建一整套 context」相比便宜两个数量级。

### 34.4 顺带：**权威源解析必须被"用上"，不只是"存在"**

本项目为防止取错源写了 `scripts/canonical_sources.py`，但**手术脚本压根没导入它**，
于是同一个取错源的错误犯了第三次（part2a 取成 `transformer_part2a.onnx`，
而权威 base 是 `transformer_part2a_fixed.onnx`，实测 1159917 vs 1136771 字节）。

⇒ ②**「建了权威模块」不等于「权威模块被用上」。** 落地做法：
让每个消费脚本 `import` 该模块并**断言源主干来自它**，而不是把文件名写在脚本顶部。

---

## 三十五、🔴 改序列长度的**完整成本账**（2026-08-29 实测，八段全部走完）

§三十四 讲了怎么改图不被坑。本节给**真实工时**，以及一个把预估砍掉大半的发现。

### 35.1 🟢 关键发现：**用 `--quantization_overrides` 的段，重量化不需要校准数据**

改序列长度前，最吓人的一项预估是"校准数据要在新长度下重新生成，宿主数小时"。
**先去读部署产物里记录的命令**（§7.6：每个量化 DLC 都存着 `Converter command` 与
`Quantizer command`），本项目实读四段 transformer 的结果是：

```
Quantizer command: ... float_fallback=True; input_list=None; use_per_row_quantization=True;
                       keep_weights_quantized=True; weights_bitwidth=8; bias_bitwidth=32
Converter command: ... quantization_overrides=<seg>_fp16_ovr.json; float_bitwidth=16
```

**`input_list=None`** ⇒ 这些段的 encoding **全部来自 overrides JSON，量化器根本不跑校准**。
而那份 JSON 里只有 per-tensor 的 `bitwidth/min/max/scale/offset`，**没有任何形状信息**
⇒ 换序列长度后**直接复用**。

⇒ ②**"重量化要重做校准"只对真正用 `--input_list` 的段成立**（本项目是 text_encoder），
对走 overrides 的段是零成本。**先读命令，再排期。**

🟢 **而且这一点可以被自证**：把生成 overrides 的脚本在新长度下重跑一遍，
产物应与部署版**逐字节相同**。本项目四段实测：75,862,649 / 50,533,008 / 51,385,307 /
51,024,267 字节，**四段全部逐字节相同**。
⇒ 这同时**独立印证了 §34.3 的结论**（序列长度不改变量程），不靠推理。

### 35.2 实测工时（SM8750 / QAIRT 2.48 / 24 GB 宿主）

| 步骤 | transformer（4 段） | text_encoder（4 段） |
|---|---|---|
| ONNX 图手术 | 秒级 | 秒级 |
| FP32 前向验证（ORT） | 约 3 分钟/链 | 含在链里 |
| 转换 fp32 DLC | **3.5 ~ 6.3 分钟/段** | （已有产物） |
| 量化 | **0.1 分钟/段**（无校准） | 约 20 秒/段 |
| 建 context | **8.9 ~ 19.3 分钟/段** | 约 10 秒/段 |
| **合计** | **约 70 分钟** | **约 2.5 分钟** |

⚠️ 另需一次性的**安全门**：在新长度下逐张量测 `|a|max`（§34.3），本项目约 4 小时/四段。
⚠️ 指南旧处写的"量化约 3 小时/段"是**带校准**的数字，对 overrides 路径不适用。

### 35.3 体积与 PD 红线：几乎不动

context `.bin` 体积由**权重**主导，激活缓冲是运行时分配的
⇒ caption 槽 32→80、`unified` 4128→4176 之后，四段体积变化 **+0.02% / −0.31% / −1.73% / −1.62%**
（有的还变小）。PD 红线（本项目 3506 MB）**毫无压力**。

### 35.4 🔴 交付侧：同一个常量在几处各写一遍，就会有几处说谎

改 `L` 时要同步的**不止模型**。本项目实际踩到的：

| 位置 | 症状 |
|---|---|
| app `Config.hpp` 的 `zimage_text_max_length` | 真相来源 |
| app `PipelineZImage.hpp` 的 `std::array<uint8_t, 32>` | **写死的副本** |
| 契约的 `inputs/outputs` 的 `fixed_shape` / `exact_bytes` | 只改 sha256/size ⇒ 真机报 `manifest input byte mismatch` |
| 契约的 `text_seq_len` | 陈旧 |
| **宿主干跑校验器自己的 `default=32`** | **给出 6 条假 FAIL** |

最后一条最阴：**为防止这个事故而写的检查工具，自己也存了一份写死的副本**，
于是在源头改成 80 之后，它照 32 建模、报了一堆不存在的错误。
⇒ ②**根治只有一条：让副本不存在。** 工具应当**解析**真相来源
（本项目改成从 `Config.hpp` 正则解析 `zimage_text_max_length`），
而不是接受一个默认值。同类见 #73（硬编码图名清单）、#138（手写源 ONNX 清单）。

### 35.5 顺带澄清：内部 QNN 图名变了要不要紧

新建的 context 内部图名会带上你的产物标签（本项目 `part1a_fp16_fp32` → `part1a_fp16_L80_fp32`）。
**先查 app 怎么选图**再决定慌不慌：本项目 app 用契约的 `internal_graph_name` 匹配条目、
**按索引**从 `.bin` 里取图（`(*m_graphsInfo)[graphIdx].graphName`），
契约里的 `qnn_graph_name` **它根本不读** ⇒ 无害。
但仍要在契约里如实更新，**不留假信息**。

---

## 三十六、🔴 方法论：**量测装置必须先被逐字节验证**，以及"一致性"与"画质"是两件事

本节不讲 QNN 选项，讲**怎么让结论站得住**。2026-08-29 一天里，下面每一条都拦下了真实错误。

### 36.1 新建的量测链，先用"已知答案样本"证明它复现旧结果——最好到**逐字节**

转新模型时你会不断新建测量装置（前向链、参考图生成器、契约校验器、指标脚本）。
**装置错了，后面所有数都是废的**，而且错得很安静。

规程：**每个新装置先在一个盘上已有答案的样本上跑一遍**，判据取到能取的最严格档：

| 本项目当天的四个例子 | 判据 | 结果 |
|---|---|---|
| 新建的四段 FP32 前向链 | 与盘上 `latents_fp32_s0.raw` 比 | **逐字节相同** |
| caption 生成链 | 与盘上部署 `caption.raw` 前 20 行比 | **逐字节相同** |
| 8 步 FP32 参考驱动 | 与盘上终点 latents 比 | **逐字节相同**（图像层 110.10 dB） |
| 钳位集合选择规则 | 复现盘上 `<seg>_clipped.json` | **逐元素相同** |

⚠️ **能逐字节就不要用"接近"**。浮点链在同一实现下本就应当逐位可复现；
一旦只能做到"很接近"，说明**有你没意识到的变量**（不同的段划分、不同的输入血统、不同的库版本）。

⇒ 反过来，这套纪律让一个**悬了很久的前提**当天关掉：本项目 #116 怀疑
"某关键测量的段间输入是 CPU 参考链而非纯 FP32 链"，一直没法查，因为**当时没有纯 FP32 链**。
新链验证通过后，做一次单变量对照（同段/同 caption/同候选集/同脚本，**只换输入血统**）：
决策相关区间 n=31，比值 **0.9972~1.0141**，无一跨越阈值 ⇒ 该前提不成立，关闭。

### 36.2 🔴 **"一致性"与"画质"是两个正交问题，用错尺子会得出相反结论**

- **一致性**：同噪声同 caption 下，量化输出离 **FP32 参考**多远（PSNR / 相对 L2 / 余弦）
- **画质**：图本身好不好（**只能人看**）

本项目实测这两者**反相关**（#69/#70/#84）：误差不把图推离数据流形，
而是**沿流形移动到另一个合理样本** ⇒ 表现为"另一张好照片"，不是"同一张照片但更差"。

⇒ ②**两个问题必须分别设计实验**：
- 测一致性：固定噪声 + 固定 caption + FP32 参考，**两臂都用 FP32 文本编码器**
  （把 TE 量化这个变量排除掉）
- 测画质：**多样化 prompt + 真实 app 全链路 + 人看**。注意研究流水线的 PSNR
  **排除了量化 TE/VAE**，描述的**不是用户看到的东西**

🔴 **一致性数字好看不代表画质好，反之亦然。** 本项目 2026-08-29 同日两组结果：
一致性 22.58 dB（仍在"换背景"区间），而 8 个失效模式探针（手/脸/文字/多主体/中文/风格/纹理/低光）
**量化的典型签名一个都没出现**（无暂彩化、无色带、高频细节未被抹平、无 inf 伪影）。

### 36.3 画质评测的探针要按**量化的已知失效模式**设计，不是随便挑 prompt

有效的一组（每条都在考一个具体的失效机制）：

| 探针 | 考察什么 |
|---|---|
| 手指/手势 | 结构性高频细节，量化扩散最经典的崩点 |
| 人脸 + 牙齿 | 小尺度规则结构 |
| **文字渲染** | 最严苛的高频保真，先崩的就是它 |
| 多主体 + 左/中/右 | 属性绑定与空间关系（考的是条件通路不是像素） |
| 非英文 prompt | 文本编码器量化是否伤到语义 |
| 扁平插画/风格化 | 是否只会输出照片（风格遵循） |
| 微距重复纹理 | 高频细节是否被抹平 |
| 低光 + 深阴影 | **暂彩化/色带**——量化的典型签名 |

⚠️ **边界要写清**：没有 FP32 对照时，"这些偏差不是量化引入的"是**推断不是实测**；
每条 prompt n=1 就是冒烟测试，不是评测。

### 36.4 ⚠️ 长会话里的两个自伤操作（都真实发生过）

1. **重跑用同一个产物名，覆盖掉唯一的现场证据。** 本项目当天补跑失败样本时沿用同一
   basename，把崩溃时刻的 logcat 覆盖了（关键行已引进台账，结论存留，但**无法再挖**）。
   ⇒ **任何"复现/补跑"一律换新名字。**
2. **`cmd > log 2>&1; echo "EXIT=$?" >> log` 会让整条命令以 `echo` 的返回码结束**，
   把失败伪装成成功——与 §7.4 记的"管道给 tail 吞返回码"是同一类。
   ⇒ 后台启动长任务时**不要在命令尾部追加任何东西**，让被观测的命令自己决定返回码。


---

## 三十七、🔴🔴 **量 HTP 模型的内存，进程 RSS 是错的口径 —— 要看 `IonTotalUsed`**（2026-08-29 实测）

> 转新模型做内存预算时**第一件要用对的东西**。本项目此前所有内存判断都建立在
> 「context 文件大小」或「PD 分配估算」上，**从未量过设备上真实占用**，
> 三种量还被混在同一个算式里用（见 `EXP_PLAN_SPEEDB.md` §1.2）。

### 37.1 三种量必须分清，它们不相等

| 量 | 怎么取 | 本项目实测（Z-Image，L=80） |
|---|---|---|
| ① context **文件字节** | 宿主 `ls -l *.SM8750.bin` | part1a 2263 MiB／part1b 1398／part2a 1354／part2b 1398 |
| ② **PD 分配估算** | 建 context 失败时报错里的 `context size estimate` | part2a 1887 MB／part2b 1927 MB |
| ③ **设备实际占用** | `/proc/meminfo` 的 **`IonTotalUsed`** | 每个 part2 半段 **1936~1990 MiB** |

⇒ ①**实测倍率：③ ≈ ① 的 1.42×**，而 ③ 与 ② 吻合。
**用文件大小做内存预算会低估约 40%。**

### 37.2 🔴 进程 `VmRSS` / `VmHWM` **看不见 context**

①**实测**：后端进程生成全程 `VmRSS` 只有 **~200 MiB**；唯一的峰值 **2356 MiB**
出现在「把 `.bin` 读进 CPU 缓冲区」那一瞬（≈ part1a 文件大小 2263 MiB），
交给 QNN/DSP 之后 CPU 侧立即释放。

⇒ ②**context 常驻在 DSP/ION 侧，不计入宿主进程的 RSS。**
拿 `ps -o RSS` 或 `dumpsys meminfo <pid>` 做 HTP 模型的内存预算，**量到的是加载缓冲区，不是模型**。

⚠️ 附带一个会误导人的细节：`VmHWM` 可能**小于**同一时刻读到的 `VmRSS`
（实测 2,412,344 < 2,412,992 kB）。内核 `update_hiwater_rss()` 是惰性更新 ⇒
**`VmHWM` 会滞后，不能当唯一的峰值来源**，要配轮询峰值一起看。

### 37.3 正确做法：设备侧 1 Hz 采样 `IonTotalUsed`

```sh
# 关键：启动时把 nsp* 热分区路径缓存一次。每 tick 重扫 85 个分区会把采样率压到 0.5 Hz，
# 而 context 加载是秒级事件，采样率不足会漏掉峰值。
awk '/IonTotalUsed/{print $2}' /proc/meminfo     # 单位 kB
awk '/^MemAvailable/{print $2}' /proc/meminfo    # 判「还塞不塞得下」用这个
```

**不要在宿主上用 `adb shell` 逐 tick 轮询**——每次往返 100~200 ms，实测把采样率压到 0.5 Hz。
把循环推到设备上跑（`scripts/speedb_sampler.sh`），一条长连接把结果流回宿主。

**读法**：分段常驻/释放的流水线，ION 曲线是干净的**方波**，
两个电平之差**就是那一段的真实占用**——这是最可靠的分段内存归因手段，比任何估算都硬。

### 37.4 判「还能不能再常驻一段」（🚫 **本小节的判据已被实测推翻，见 37.6**）

**不要拿峰值去比某条历史「被杀线」**——那条线多半是别的口径（本项目的 7300 MB 就是，
拿它当 ION 阈值会把**已经稳定运行的现网配置**判死）。

正确判据只用同一次测量里的两个数：

```
额外需求（下一段的 ION 占用）  vs  峰值时刻的 MemAvailable
```

①**本项目实测**：额外需求 **1936 MiB** vs 峰值可用 **855~980 MiB** ⇒ 缺口约 1000 MiB ⇒ 否决。
**这个论证自足，不依赖任何历史阈值。**

> 🚫 **上面这条判据是错的（2026-08-30 实测推翻，原文保留在上面）。**
> 它预测「额外需求 1936 MiB > 峰值可用 855~980 MiB ⇒ 装不下”，
> 而实测四段**全部装下了**：ION 峰值 **9100 MiB**，且到顶时仍余 **839 MiB**。
> 
> 🔴 **错在哪：`MemAvailable` 不是一个会被新分配 1:1 消耗的“余额”。**
> 它是内核对“不动用回收能给出多少”的**保守估计**；ION 长大时内核会回收页缓存、压入 zram，
> 于是 `MemAvailable` **大致被维持住**而不是线性下降 —— 本项目实测：
> ION 7202 MiB 时余 855~980 MiB，ION **9100** MiB 时仍余 **839** MiB。
> 把它当余额去减，等于把可回收内存重复计算一次。
> 
> ⚠️ 这是本项目**第二次**在同一个地方犯错：37.2 已经记了“用错口径（RSS）”，
> 而这里是“口径对了、**阈值的语义又想当然了**”。③**代理量未经验证就下否决结论**是反复犯的错。

### 37.5 配套：可复现的**速度**测量协议（实测复现性 0.152%）

改任何东西之前先把尺子标定好，否则收益会被淹没：

1. `am force-stop` —— 让后端成为全新进程
2. **静置到冷机**：所有 `nsph*` 分区 ≤ `T_idle + 3 °C`（本机实测 58.8 → 46.1 °C 约需 **6 分钟**）
3. 重启 app 等端口监听，**此段不计时**
4. 固定 prompt + 固定 seed，测**冷机第一张**；主时钟取 `curl time_total`

①**实测 n=3：202.486 / 204.094 / 204.405 s，`\|B2−B3\|/mean = 0.152%`**，
三张图 sha256 完全相同，ION 峰值逐字节相同。

⚠️ **划界**：这个口径测的是**冷机第一张**，**不是用户连续使用时的稳态**。引用时必须注明。
①实测热行为：**单张图内 NPU 就冲到 103.4 °C**（起点 48~50 °C），步循环内在 60~103 °C 剧烈摆动
⇒ ②「连续多张耗时爬升」累积的不是 NPU 结温（单张已打满），③但累积在哪未实测。

### 37.6 🟢🟢 **正确做法：别推，用一个 200 行的探针直接测**（2026-08-30 实测）

“再常驻一段装不装得下”**不要用任何公式去推**。写一个独立小程序，
把 N 个 context 依次 `QnnContext_createFromBinary` 并**全部保持存活**，每装一个打一次 `IonTotalUsed`。
参考实现：`scripts/quadctx_probe/quadctx_probe.cpp`（NDK 交叉编译，~200 行）。

**为什么这比改 app 便宜得多**：不用改业务代码、不用重编 APK、不用装机，
结论又是决定性的。本项目实测成本：写+编译约 40 分钟，设备跑约 10 分钟。

**三条必须遵守的细节**（否则测出来的不是 app 的形态）：

1. 🔴 **每建完一个 context 立刻释放文件缓冲区**。app 就是“读 .bin 进 buffer → createFromBinary → 释放”；
   把 N 份 buffer 都留着会凭空多占数 GB **宿主**内存，测的就不是同一件事。
2. 🔴 **加一道安全闸**：每装下一段前先看 `MemAvailable`，低于阈值就**干净退出**并释放。
   这是别人的手机，不要去撞那一下。**并且要在输出里区分两种失败**：
   `createFromBinary` 报错 = **真的装不下**；安全闸触发 = **我主动停手**，不构成装不下的证据。
3. 🔴 **先用已知样本验证探针本身**（约束 8）：先只装 1 段，它必定成功，
   且测得的增量应与你在真实 app 里量到的同一段吸合。
   ①**本项目实测：探针 1977 MiB vs app 内 1986 MiB，差 0.5%** ⇒ 装置可信。

**实测输出长这样**（本项目四段 transformer）：

| 步骤 | ION | 逐段增量 | MemAvailable |
|---|---|---|---|
| 起点 | 357 MiB | — | 7239 |
| + part1a | 3285 | **+2928** | 4968 |
| + part1b | 5274 | **+1989** | 3132 |
| + part2a | 7228 | **+1954** | 2433 |
| + part2b | **9040** | **+1812** | **2010** |

①**四段全部加载成功**，且后来在**真实 app** 里复现（ION 峰值 9100 MiB、余 839 MiB、
连续 3 张后端进程未被杀）。

⚠️ **探针结论不等于 app 结论**：探针是 shell 进程（SELinux 域不同、无 JVM 堆/UI、
不受 Android 的 phantom-process 规则约束），且**只加载不执行**。
它的作用是把“要不要为这个方向改代码”从猜变成测；**最终仍必须在真实 app 里验收**。

---

### 37.x 🔴 **拿「MemAvailable 单秒最低」当交付门，门本身没有分辨力**（2026-09-17 设备实测）

本项目交付 mg 时锁了一条 H2「峰值期 MemAvailable 最低 > 800 MiB」，结果**五项门过四项、唯独它不过**。复读每秒原始序列：

| 臂 | ION 峰值 | MemAvail 单秒最低 | 第 2 低 | <800 的秒数 | 平台期中位（ION >= 峰值−300） |
|---|---|---|---|---|---|
| 现网 | 9101 | **742** | 821 | 1 | **1009** |
| 同构型另一次 | 9129 | **397** | 1082 | 1 | 1155 |
| mg + 共享 | 8714 | 703 | 988 | 1 | **1298** |

- **每臂低于阈值的都只有 1 秒**，且多数正落在 ION 峰值那一秒；第 2 低就跳回 800~1200。
- **ION 几乎相同的两次（9101 vs 9129），单秒最低相差 345 MiB** ⇒ 次间噪声 > 要分辨的差距。
- **现网自己当天就过不了这条门**（742 < 800）——定门时没拿现网这个已知样本试一下（约束 8）。

**固化做法**：
1. 内存交付门用**同日现网配对**，不用拍脑袋的绝对阈值（「不比现网更容易被杀」才是本意）。
2. MemAvailable 用**平台期中位数**（ION 高位期间），单秒最低只作参考。ION 峰值本身很稳（同构型两次差 28 MiB），可以直接比。
3. **定门前先拿现网跑一遍这条门**：现网都不过的门，不是门。
4. 采样序列**留在设备上并拉回宿主**（`/data/local/tmp/{ion,avail}_<tag>.txt`），事后才有得复读——本次正是靠它在 10 分钟内定位到尺子问题。

## 三十八、🔴 **context 内存里有一块叫 spill-fill 的开销，它可以跨 context 共享**（2026-08-30 实测 + 官方文档）

转大模型（本项目下一个是 Flux.2 Klein）撞内存时，这是一条**现成的、官方支持的**省量手段。

### 38.1 先把开销拆开看（宿主，零设备，几分钟）

```bash
qnn-context-binary-utility.exe --context_binary <x.bin> --json_file out.json
```

里面 `graphs[0].info.graphBlobInfo.info.spillFillBufferSize` 就是那块开销；
`graphBlobInfoV2` 还会给 `constSize / opDataSize / ioTensorSize / ddrTensorSize`。

①**本项目四段实测**（L=80，A16W8 + 部分 FP16）：

| 段 | context 文件 | **spillFill** | const | 实测 ION | ION/文件 |
|---|---|---|---|---|---|
| part1a | 2263 MiB | **289** | 2159 | 2928 | 1.29× |
| part1b | 1398 | **266** | 1325 | 1989 | 1.42× |
| part2a | 1354 | **273** | 1273 | 1954 | 1.44× |
| part2b | 1398 | **258** | 1327 | 1812 | 1.30× |
| 合计 | 6413 | **1085** | | 8683 | |

⇒ ①**spillFill 占四段总 ION 的 12.5%**，是「ION 比文件大 29~44%」里最大的单项。

### 38.2 官方机制：一组 context 可以共用一份

`docs/QAIRT-Docs/QNN/general/htp/htp_shared_buffer_tutorial.html` 原文：

> **76.** External spill-fill buffers can also be **shared between graphs of multiple contexts**
> by registering the same external spill-fill buffer with multiple contexts.
> **74-75.** the required size for this buffer is the **largest** out of all the spill-fill buffers.
> **81.** Graphs sharing the same external spill-fill buffer **cannot be executed in parallel**.

⇒ N 个 context 从「各自一份」变成「共用最大的那一份」。
①**本项目算下来可省 1085 − 289 = 796 MiB**。

🔴 **两个必须先确认的前提**：

1. **你的段必须是串行执行的**（第 81 行）。扩散模型的步循环天然串行，满足；
   若你想并行跑两个图，这条路直接封死。
2. **它与 `REGISTER_MULTI_CONTEXTS` 互斥**（第 50/95 行明写 *is not supported together with*）。
   官方其实有**两条**共享路径：①外置 buffer（客户端 `rpcmem_alloc` + 把**同一个 fd**
   `QnnMem_register` 给多个 context）；②`QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS`
   + `QnnHtpContext_GroupRegistration_t{firstGroupHandle, maxSpillFillBuffer}`（QNN 内部分配）。
   **选一条，不能叠用。**

⚠️ ~~③「共享后系统 ION 总量真的下降」本项目尚未实测（四段本就装得下，没用上）。
已知的只有：机制在本机注册成功（台账 #60 实测，260 MB 外置 spill-fill 注册 OK）。~~
（**2026-09-17 已被下面 38.3 的实测关闭**，原文保留）

### 38.3 🟢 实测：共享后系统 ION 峰值降 734 MiB，数值逐字节不变，不变慢（2026-09-17，SM8750）

走的是上面第 ② 条路径（`REGISTER_MULTI_CONTEXTS` + `maxSpillFillBuffer` = 组内最大者 302,645,248 B，
组长 = 第一个常驻段）。本仓库里 `PipelineAnima/Sdxl` 早就在用，Z-Image 当初只是**漏接**。

| | ION 峰值 | MemAvail 最低 | app 生成耗时 | 出图 sha256 |
|---|---|---|---|---|
| 同 APK、不共享 | 9129 MiB | 397 | 154.45 s | 与现网相同 |
| 同 APK、共享 | **8395 MiB** | **859** | 153.92 s | 与现网相同 |

- 两臂**只差一个 marker 文件**（严格单变量），ION 每秒采样 314/335 点。
- 宿主公式预测省 806 MiB（ION 折算 854）⇒ 实测是预测的 **86~91%**。**转新模型时可按「公式 × 0.85」做保守预算。**
- ③每臂 1 张，跨次重复性未单独测（同日两 APK 的无共享峰值相差 28 MiB，远小于 734）。
- 🔴 **前提仍是串行**：四段在步循环里严格 1a→1b→2a→2b。若将来某段改为「每步装卸」，**组长不能是被卸的那一段**。

⇒ **转大模型撞内存时，这是第一个该接的手段**：约十行代码、零画质代价、零速度代价、可用 marker 文件做 A/B。

🔴🔴 **组大小必须是「所有成员 context 的所有图」的最大者，而且只认第一个注册者给的值**（2026-09-17 设备实测翻车）

官方原文（`htp_backend.html`）：*Users should figure out the maximum spill fill buffer size needed across all the contexts before proceeding to deserialize.*
`QnnHtpContext.h`：*The value that is passed during the registration of the first context to a group is taken. Subsequent configuration of this value is disregarded.*

本项目把 single 形态实测的最大者（302,645,248 B）**写死进 app**，换成多图 context（mg，一个 context 装 5 个比例的图）后：

| part1a 的图 | spillFill |
|---|---|
| 1:1 | 311,492,608 B（**已超**写死值）|
| 1184x896 / 896x1184 | **331,415,552 B** |
| 1280x720 / 720x1280 | 276,430,848 B |

⇒ 交付后第一张图 `Failed init QNN context: transformer_part1a`，自动回滚。

**固化做法**：
1. 组大小**从要交付的 .bin 元数据现读**（`qnn-context-binary-utility --json_file`，取每个图的 `spillFillBufferSize`），
   范围 = **入组的每个 context × 其中每个图（多图 context 要算全部比例，不只当前启用的那个）**。
2. 这个数**不进代码常数**，随交付物一起下发（本项目写进 marker 文件内容，app 读取；读不出就不共享并报错）。
3. 入组前先确认「谁在组里」：本项目只有四个 transformer 段入组；VAE 的 spillFill 高达 0.93~1.09 GiB，若误入组，组大小会被它撑大、省量几乎归零。
4. 🔴 app 包装层往往只回一句 `Failed init QNN context: <段名>`，QNN 的错误码在 logcat 里 ——
   **抓 logcat 的脚本不要在请求返回后立刻杀 logcat**，否则最关键的几行还在缓冲区里就丢了（本次正是这样丢的）。

⚠️ **别把它与 #60 的结论搞混**：#60 测的是「外置 spill-fill 能否降低 **PD 容量估算**」
（单个 context 能不能装进 DSP 保护域）—— 结论是 **PD 估算一字节未降**。
本节问的是另一个量：**多个 context 同时驻留时的系统 ION 总量**。两者不相干。

---

## 三十九、⚡ **把所有 context 提出步循环：本项目最大的单笔提速**（2026-08-30 交付）

### 39.1 结果

多段扩散模型的典型写法是在步循环里 `loadGraph → run → 释放`，
因为「全部常驻会爆内存」。本项目分两步把它提了出去：

| 阶段 | 做法 | 耗时 | 精度 |
|---|---|---|---|
| 原始 | 四段都在循环内装卸 | 271 s | — |
| 方案 A | part1a+1b 常驻 | 198～204 s | sha256 逐字节相同 |
| **方案 B** | **四段全常驻** | **160.9 s** | **sha256 逐字节相同** |

①**累计 −40.6%，零精度代价**。代码改动就是把 `loadGraph` 移出 `for`，变量改成循环外的 `unique_ptr`。

### 39.2 为什么安全（这一条要自己核）

重复执行同一个已建图不带一次性状态：I/O 张量只建一次后复用，
执行时只往持久 client buffer 里 `memcpy`。**判据就是出图 sha256 逐字节相同**——
同一份 context、同一份输入，常驻与否在数值上就应当一模一样。
🔴 **sha256 不同 ⇒ 那是 bug（状态污染），不是「收益/代价权衡」。不许退而用「指标在噪声内」放行。**

### 39.3 预估模型不准，判据里别写预期值

用「每段加载耗时 × 步数」推收益，本项目**两次都没推准**：
方案 A 预估 47~54 s、实得 **73 s**（偏大）；方案 B 预估 52 s、实得 **43.4 s**（偏小）。
⇒ **判据只写方向 + 下限**（如「快 ≥15% 算有效」），不写预期数字。

### 39.4 验收必须包含「连续多张」

单张成功不够。常驻把内存压力从「来回起落」变成「**全程维持高位**」，风险形态不同。
①本项目验收：连续 3 张均 HTTP 200、**后端进程 PID 全程未变**（没被杀）、三张 sha256 相同。

⚠️ 读 logcat 时注意假阳性：`am_kill ... due to installPackageLI` 是你自己装 APK 触发的；
`fastrpc ... domain_deinit (kill time 155 us)` 是通道拆除的措辞，**都不是进程被杀**。

### 39.5 `O` / `dlbc` 图优化级别：`O:3` 约 −9% 计算，`dlbc` 无效

①**实测**（单段 part1b、`qnn-net-run` 离线、N=1 vs N=9 斜率、正反两趟）：
CTRL 15.903 s/次｜**O3 14.472 = 0.910×**｜DLBC 16.034 = 1.008×。
三次独立测量的 O3/CTRL = 0.939 / 0.890 / 0.930。

三个配套事实：

- ①**默认就是 `O=0`**。只传 `devices` 段、不传 `graphs` 段时，`O`/`dlbc` **根本不生效**（同 §九十三）。
- ①**加 `graphs` 段但只填无效果的 `vtcm_mb:8`，产物与不传时 md5 逐字节相同** ⇒ 可以直接拿部署版当 CTRL。
- ①**`O:3` 的代价是建图时间与体积**：四段建图 17.9~32.4 分钟/段（CTRL 约 10），体积 **+2.34%**
  ⇒ ION 同比上升，**会吃掉常驻方案的余量**，两个优化叠加时必须重测内存。

⚠️ 措辞：以上是 `qnn-net-run` **离线单段**计时，**不等于** app 内每步耗时（同 §三十二的分母陷阱）；
③端到端收益**尚未实测**。

#### 39.5.1 🔴 **但 `O:3` 在本项目最终【不可交付】——两个建图时看不见的代价**（2026-08-30 设备实测）

阶段 1 只测了单段的**计算速度**，没测它对**内存与 PD** 的影响。上设备一试，两条全炸：

**① `O:3` 会把 context 顶过 unsigned PD 红线（#57），而建图时不报错**

①实测：part1a 加 `O:3` 后 `createFromBinary -> 0x3ea`（=1002，即 PD 容量报错）。
失败时是**第一段、什么都还没装、MemAvailable 还有 7717 MiB** ⇒ 与内存无关。

🔴 **这是一个通用陷阱**：`qnn-context-binary-generator` **建图成功不代表能装载**——
真正的 PD 检查在设备端 `contextFinalize`（同 §#60）。
⇒ **建完任何新配置的 context，必须上设备试装一次**，别拿建图日志当验收。
一个 `quadctx_probe`（§37.6）跑一遍就够，不必改 app。

**② `O:3` 的 ION 代价无法从文件大小预测，且可能极大**

①实测四段（文件增长 vs ION 增长）：

| 段 | 文件 | ION | 备注 |
|---|---|---|---|
| part1a | +3.20% | **装不进 PD** | — |
| part1b | +1.43% | **+1.8%** | — |
| part2a | +1.66% | **+3.6%** | — |
| part2b | +2.50% | **+60.4%** | 1812 → **2906 MiB**，两次独立测量**逐字节相同** |

⇒ ②**图优化级别会独立地改变运行时内存，与产物体积不成比例。**
体积 +2.34% 的一组产物，实际 ION 涨了 **+21%**（三段合计 5755 → 6964 MiB）。

**⇒ 实践结论**：`O` 不是一个「零风险的性能开关」。改它之后**必须重测 PD 与 ION**，
而不是只测速度。本项目因此放弃 O3，保留已交付的全常驻方案（§39.1）。

---

## 四十、🔴🔴 **研究「离群值主导」的量时，行采样会把现象整个切掉**（2026-08-30 实测）

### 40.1 事故形态（差点得出相反结论）

测 SmoothQuant 在 MatMul 输出上的收益，样本量大跑不动，于是想学 §#39 的
「取前 512 行」加速。**先用已知样本核对了一下**（约束 8），两组数字如下：

| 口径 | 基线 E 均值 | SmoothQuant 改善 |
|---|---|---|
| **全部 4176 行** | **2.80%** | **+62.9%** |
| 仅前 512 行 | 0.83% | **+0.1%** |

⇒ ②**结论完全相反**。若直接用 512 行，会得出「SmoothQuant 无效，关闭该方向」——
**一个错误的否决**，而且它看起来完全合理（数字干净、alpha 曲线平坦）。

### 40.2 机制

被研究的现象**本身就住在极少数行里**。本项目实测：某个 MatMul 的基线误差
**14.77%**，而同段其他 MatMul 只有 0.6~1.0% —— 那 14.77% 来自少量
massive-activation 行。取前 512 行没抽到它们 ⇒ 基线误差塌到 0.83%，
**于是「没有东西可修」，改善自然是 0。**

### 40.3 规则

🔴 **凡是研究「离群值主导」的量（激活量程、massive activation、钳位、SmoothQuant），
一律不得行采样 / 不得抽样降维**，除非先证明抽样后**基线量本身没有塌**。

判据很简单，三行代码：

```
基线量(全量) vs 基线量(抽样)   —— 两者不同量级 => 抽样非法，立刻停用
```

⚠️ **不要拿别处「用了 512 行也没问题」当依据**：#39 用前 512 行是在比较
两种定点实现的**算术**差异，那件事对行的分布不敏感；本节这件事**全部信号都在行分布里**。
⇒ ②**同一个加速手段在不同问题上合法性不同，必须逐个验证。**

⊕ 同族教训：约束 7（相对 L2 被离群值主导，低估达 48 倍）、
#67（激活钳 0.0261% 丢 98.75% 能量）—— 都是「少数元素携带绝大部分信号」。
**本项目在这一族上已经栽过三次，每次形态不同。**

---

## 四十一、🔴 **代理指标的链条又断一环：MatMul 输出误差不能预测段级误差**（2026-08-31 实测）

本项目已经知道 **L1（step-0 噪声预测）不能换算成 L3（成图 PSNR）**（#121）。
这次发现更低一层同样断裂。

### 41.1 事故形态

评估 SmoothQuant 时设了一道便宜的代价门：对真实的 X 与 W 直接量 **MatMul 输出误差**。
结果非常漂亮：59 个 MatMul 的 **sum E² 降 96.0%**，误差能量集中的几个改善 65~93%。
判据（事前锁定 ≥20%）🟢 通过。

然后在**段级**（同一段的图输出）上单变量实测：

| 段输出 | 基线 | 只重标定 | **+SmoothQuant** |
|---|---|---|---|
| `add_138` | 2.67% | 2.70% | **17.77%** |
| `unified` | 3.82% | 3.05% | 3.59% |
| `add_92` | 1.73% | 1.76% | **16.67%** |
| `latents` | 3.60% | 3.56% | **20.54%** |

⇒ ②**三段差 5~10 倍。代理指标说好 96%，实际差一个数量级。**

### 41.2 为什么会断

MatMul 层的度量只看**那一个算子的输出**，看不见两件事：

1. **权重变粗的代价会传给下游所有依赖该权重的路径**，不止那一个 MatMul 的输出；
2. 段内还有归一化与残差，**局部改善不一定沿链存活**（本项目 #97：RmsNorm 有固有放大）。

### 41.3 规则

🔴 **任何「算子级 / 张量级」的代价门，只能用来【否决】，不能用来【放行】。**
过了门只说明「值得继续花钱测下一层」，**不构成方向成立的证据**。
必须逐层往上验，直到那个被标定过、与产品目标直接挂钩的量（本项目是 L3 成图 PSNR）。

⊕ 本项目已知的断裂点，逐层记下来：
表示误差 ↛ HTP 误差（#52，37.6 倍）；**MatMul 输出误差 ↛ 段级误差（本节，5~10 倍）**；
L1 ↛ L3（#121，判错 2/15 对）；宿主模拟 ↛ 设备（#123，比值 1.826 离群）。

## 四十二、🟢🟢 **一个模型要支持多个输入形状：多图 context + 权重共享**（2026-09-02~04 实测）

> **适用范围**：同一网络的**不同输入形状**（不同分辨率/长宽比/序列长度）。
> 这是高通官方点名的典型用例，不是我们的偏门用法。转 Flux.2 Klein 一类模型时直接照抄。

### 42.1 为什么不能用别的办法（逐条有实测否决）

| 路线 | 结论 |
|---|---|
| 每形状一套完整 context | ❌ zstd **以基座为字典**压 4:3 的 part1b = 1236.3 MB（84.0%），**不带字典也是 1236.3 MB —— 字典零收益**。HTP 编译后按形状重排权重/调度，二进制处处不同 ⇒ 5 个比例 = 34 GB |
| 动态形状 | ❌ HTP 要求静态形状 |
| 在固定画布里做 padding / 信箱裁切 | 🟡 机制可行（SDXL 就这么做），但蒸馏少步模型被强制黑边属训练分布外，画质未知；且需要 VAE encoder |
| **多图 context + 权重共享** | 🟢 **官方正解** |

### 42.2 三个必须同时用对的机制

1. **`--dlc_path a.dlc,b.dlc`**（逗号分隔）把多个图**编进同一个 context**。
   本地 `qnn-context-binary-generator --help` 原文：*"To compose multiple graphs in the context, use comma-separated list of DLC files"*。
2. **`QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED`**（配置键 `context.weight_sharing_enabled`）让多个图共用一份权重。
3. 🔴 **`QNN_CONTEXT_CONFIG_ENABLE_GRAPHS` 是必须的，不是可选的。**
   `QnnContext.h:122-126` 写的是 *"All graphs are enabled by default"* —— 默认全开。
   **本项目实测：不传 ENABLE_GRAPHS 会因 unsigned PD 容量超限而失败（错误码 `0x3ea`）；
   传了则成功。** 已在**两个段、两种几何**上各验一次。
   ⇒ **PD 只统计被 enable 的图**。这条是整条路线成立的生死线。

### 42.3 权重共享是否真生效——怎么验（别看文件大小猜）

看 `qnn-context-binary-utility --json_file` 的元数据，**不要**用「文件是不是接近单图的两倍」来判断：

| 量 | 单图 | 双图共享 | 判读 |
|---|---|---|---|
| `sharedWeightsSize` | —— | **47.5 / 47.5 MiB** | 两图报同一个值，**整个 context 只算一次** |
| `constSize` | 48 MiB | **0.3 / 0.3 MiB** | 权重已从 const 挪进共享区 ⇒ 生效 |
| 文件大小 | 106.3 MiB | 207.5 MiB | ⚠️ **看起来像没共享**，但那是因为共享的只是权重，每图自己的算子/调度数据（`opDataSize` 115.8 vs 172.8 MiB）本来就各占一份 |

⚠️ 官方文档称 Weight Sharing 仅 x86_Linux 可用；**本项目在 Windows 主机上实测生效**
（W3=18.38 MB vs W2=32.52 MB）。**以实测为准，但文档原文一并保留**，换 SDK 版本要重验。

### 42.4 RAM 怎么算（`tools.html` 公式，实测吻合）

```
opDataSize + constSize + ddrTensorSize + spillFillBufferSize + ioTensorSize + vtcmSize
（+ sharedWeightsSize，整个 context 只加一次）× 1.06 ≈ 设备 ION
```

🔴 **`spillFillBufferSize` 是这里最容易爆的一项，而且它随几何非线性跳变。**
本项目实测：1152×864 的 part2a 撑到 **999 MiB**，ION 达基准的 1.38~1.44×；
把几何改成 1184×896（N=4224）后回到正常。
⇒ **筛几何时必须把 spillFill 一起筛**，不能只看 token 数。

### 42.5 🔴 从 Windows 主机往 Android app 私有目录交付：四个实测的坑

本项目 2026-09-04 一次踩全，把设备弄成坏状态（已用设备本地备份逐字节复原）。

| # | 坑 | 固化做法 |
|---|---|---|
| 1 | **Git Bash 的 MSYS 路径转换**把 `/data/local/tmp/x` 改写成 `C:/Program Files/Git/data/local/tmp/x`，push 全失败 | 全脚本 `export MSYS2_ARG_CONV_EXCL='*'`，且**推完比对设备侧字节数** |
| 2 | **`adb push` 失败后照样打印 “1 file pushed … 43.6 MB/s”** | 查返回码 **+** 查 stderr 里的 `error:` **+** 查设备侧字节数。🔴 **绝不要给 push 接 `\| tail -1`** —— 那恰好只留下那句谎话，`set -e` 也因此不触发 |
| 3 | `run-as sh -c 'cat > f'` 在**源不存在时照样建 0 字节文件** | 每步写入后立刻校验字节数；**契约/清单永远最后才换**，前面任一步失败则设备仍是好的 |
| 4 | `adb shell "… cat > f" < 本地文件` 在 Windows 上**不是二进制安全的**（实测 64854 B 传成 62572 B） | 一律 **先 `adb push` 到暂存区，再设备本地 `cat` 管道进 `run-as`** |
| 5 | 🔴 **坑 1 的修法会制造新坑**（2026-09-17 设备实测，交付首步失败回滚）：设了 `MSYS2_ARG_CONV_EXCL='*'` 之后，**`/d/...` 形式的宿主路径也不再被转换**，原样交给 Windows 原生的 `adb.exe` / `python.exe` ⇒ `adb: error: cannot stat '/d/…': No such file or directory` | 传给这类脚本的宿主路径一律用 **`D:/a/b`**（`cygpath -m` 形式）：Git Bash 的 glob/stat/重定向、adb.exe、python.exe **三方都认**。反斜杠 `D:\a` 不行（模式里 `\` 是转义，glob 不展开）；`/d/a` 也不行（原生程序不认）。**用 15 字节探针文件真推一次**即可验证，几秒钟 |

🔴 **坑 5 的元教训**：把外部程序桩掉的「干跑」**结构上查不出「外部程序怎么解释参数」这一类问题**
（路径形式、引号、编码）。改了传给外部程序的参数形式，**必须拿真实外部程序跑一个最小探针**，
不能以干跑全过为准。本项目为此白花了一次插线里的一个来回（实验已过、交付那步翻车回滚、修完续跑）。

| # | 坑 | 固化做法 |
|---|---|---|
| 6 | 🔴 **校验门的「要算哪些文件」写成通配符**（2026-09-17，交付第二次回滚）：设备上 `sha256sum *_ctx.SM8750.bin`，而文本编码器的名字是 `…_ctx_sm8750.SM8750.bin` ⇒ **被通配符漏掉，门判「缺失」**。旧契约恰好不含它们，所以前两次交付没暴露 | **文件清单一律从契约派生**（`check_deploy_sha.py --names <契约>`），不要在脚本里另写一份通配符。宿主上可预演：拿脚本的通配符去套契约清单，纯字符串比较，秒级 |

⊕ **交付前先在设备上 `cp -n` 备份一份**。本项目正是靠这份设备本地备份，在 30 秒内逐字节复原。

### 42.6 宿主侧改比例不必重导出

四个 transformer 段的比例相关常量都在 initializer / value_info 里，可**就地改写**
（见第三十四章的序列长度手术，同一套方法）。两个必须按**名字**而不是按值处理的坑：
`64` / `128` 既是 `axes_dims/2` 也是 `head_dim=3840/30`；VAE 的 `128/256/512` 既是通道数也是空间尺寸。
⇒ **按名字白名单处理，只有明确唯一的值（如 4096/4176）才按值替换**。

### 42.7 🔴🔴 **别拿含 0 值的枚举当「设过没有」的哨兵**（2026-09-04，代价：打死现网 + 约 1.5 小时）

接上 42.2：给 context 传可选配置时，很自然会写一个「把当前启用的配置装进数组」的函数：

```cpp
// 🔴 错误写法
if (m_sfCtxConfig.option == QNN_CONTEXT_CONFIG_OPTION_CUSTOM)   // 判断「设过没有」
    ptrs[k++] = &m_sfCtxConfig;
```

`QnnContext.h:99` 写的是 **`QNN_CONTEXT_CONFIG_OPTION_CUSTOM = 0`**，
而 `QnnContext_Config_t cfg{}` 会把 `option` 零初始化 ⇒ **这个判断恒为真**。
于是一个 `customConfig == nullptr` 的配置被交给 `contextCreateFromBinary`，
HTP 内部直接解引用 ⇒ **设备上 SIGSEGV**（tombstone: `SEGV_MAPERR`, `fault addr 0x0`，
`#00~#03` 全在 `libQnnHtp.so`，`#04+` 才回到自己的 .so）。

⚠️ **失效面比想象大**：本项目里这条路径原本只为多图而加，
但因为交付契约给**每个** transformer 段都写了 `qnn_graph_name`，
凡是走到这个函数的图**全部**中招 —— 包括**已经上线跑了一个月的 1:1**。

**规则**：`{}` 初始化过的 QNN 配置结构，其 `option` 字段**不携带「是否已设置」的信息**
（0 是合法枚举值）。「设过没有」必须用**独立的布尔位**表示。
这与第二十四章「后端选项设了没用」是同一类问题的两面：那边是**设了不生效**，
这边是**没设却被当成设了**。

⊕ 相关纪律（约束 8）：判据里凡涉及**某个具体数值编码**，定稿前必须先用已知样本
验证该编码本身成立。本项目此前已因此栽过两次（对称量化 `offset == -128`；误差沿 Id 单调）。

### 42.8 ⚠️ 契约完整性校验的启动开销随交付规模线性增长

app 每次启动后端都会把契约里每个 .bin 现算一遍 sha256。实测（SM8750，UFS）：

| 交付 | 哈希总量 | 耗时 |
|---|---|---|
| 单比例（现网） | 10 666 MiB | ~170 s |
| 双比例（+1184×896） | 17 286 MiB | ~275 s |

约 **62 MiB/s**。切换分辨率会触发后端重启 ⇒ **再付一次**。
⇒ 多比例交付规模上去之后，这项必须按「只校验当前尺寸引用到的文件」优化，
否则 5 个比例 ≈ 35 GiB ⇒ 每次启动约 **9.4 分钟**。
（本项目按路径去重后再校验，共用同一个多图 .bin 的比例只哈希一次。）

### 42.9 🔴 **图数增加会把权重从 `constSize` 挪进 `sharedWeightsSize`，而共享块比单图的 const 更大**

五图共享（1:1 + 四个比例）实测，`part1a` 的 **1:1 图**在两种 context 里的构成（MiB）：

| | opData | const | shared | 合计 x1.06 |
|---|---|---|---|---|
| 单图 | 201 | **2159** | 0 | **2950** |
| 五图共享 | 202 | 184 | **2315** | **3320** |

⇒ 共享块 2315 比单图的 const 2159 **还大 156 MiB**：HTP 按形状重排权重，
五个图有一部分**共享不掉**，各自留了一份进共享块。

**后果**：单个图的 ION 随 context 内图数**上升**，而 unsigned PD 红线是 **3506~3535 MiB**（#57）。
本项目五图下最高 3320 MiB，余量只有 **5.3%**。
⇒ **图数不是想加多少加多少**。转新模型时必须：
  ① 每加一档比例就重算一次预测 ION（`tools.html` 公式，见 42.4）；
  ② **把最大的那个段单独看**（本项目是 part1a，其余 15 个图都在 1872~3144，毫无压力）；
  ③ 留好**退回「每比例一套单图 context」**的产物 —— 那是唯一不受图数影响的形态。

### 42.10 体积账：五图共享的**实测**增量（别用两图的数据外推）

| 段 | 五图 | 单图 | 共增 | 折合每图 |
|---|---|---|---|---|
| part1a | 3005.3 MiB | 2263.1 | +32.80% | **+8.20%** |
| part1b | 1680.6 | 1397.6 | +20.25% | +5.06% |
| part2a | 1670.8 | 1354.3 | +23.37% | +5.84% |
| part2b | 1679.7 | 1398.0 | +20.15% | +5.04% |
| VAE | 423.2 | 106.3 | +298% | — |

🔴 **两图的数据不能外推到五图**：本项目先用 SHARE5（只在 part1a 上测过）估出 +4.57%/图，
实测是 **+8.20%/图**，低估近一倍。**VAE 更极端**（小模型里权重占比低、per-graph 的算子数据占比高，
五图 423.2 MiB vs 单图 106.3 MiB）。⇒ 报体积前**必须按实际图数建一次**。

### 42.11 🔴🔴 交付形态的一个陷阱：**别把基准图留在旧的单图 context 上**

多图共享做完后，很自然会想「基准（1:1）保持不动，只给新比例加多图 context」——
这样零回归风险。**但那样两份都要留在设备上**，本项目实测：

| 形态 | 设备保有 | 后端启动校验 |
|---|---|---|
| 基准留旧单图 + 新比例走多图 | **18.67 GiB（+79%）** | 308 s |
| **基准也走多图** | **12.30 GiB（+18%）** | ~205 s |

而且前者里**多图 context 内的基准图永远用不到**，纯属死重量 ——
**权重共享的全部意义就是只留一份**。

改基准路由的安全前提（本项目做法）：
① 元数据等价检查（42.3 / M1~M5）确认多图里的基准图与单图**张量名/形状/dtype/量化参数逐位相同**；
② 契约层面再验一次「基准的张量规格与原交付逐字相同」；
③ **旧的单图 .bin 保留在设备上**（交付只增不删）⇒ 回滚只需换回旧契约，**不必重推**。

### 42.12 🔴🔴 方法论：**判据写成绝对阈值，就会把已知良品也判死**

本项目 2026-09-04 第三次栽在同一件事上。做「新比例复用 1:1 的 encoding 会不会被钳」的检查时，
判据写的是「实测 |max| / encoding max > 1.10 判 🔴」。三个新比例在 step 7 全部得 **1.153**，判负。

补做 **1:1 对照臂**后：**现网 1:1 在同一步也是 1.153**（`add_138` 1820.746，
三个新比例 1820.5~1820.9）。⇒ 超量程是 **encoding 本身的性质**，现网带着同样的钳位
一直在正常出图；新比例与它的差在 **0.01% 量级且有正有负**。

⇒ **一个会把已知良品判死的判据，没有决策价值。**
**规则**：凡是「新配置好不好」的判据，一律写成**与已知良品的差值**，
并且**缺对照臂时拒绝下判定**（返回非零的"判不了"，而不是拿绝对值硬判）。
本项目此前两次同类：#143（绝对 PSNR 在三个 prompt 上跨度 9 dB）、D2（因此改用两臂之差）。

⊕ 附带结论：**只测扩散的第 0 步不够**。同一装置下 step 0 的比值是 1.000、step 7 是 1.153 ——
激活分布随 sigma 变化，最大值未必出现在第一步。

### 42.13 🔴🔴 **决定性的一条：共享块在「只启用一个图」时也全部常驻**

42.9 说过共享会让单图的 resident 变大。**真正致命的是它随图数线性增长，而且省不掉**：

| context 内图数 | 权重常驻（shared+const） | 相对单图 |
|---|---|---|
| 单图 | 2159 MiB | — |
| **2 图共享** | **2154 MiB** | **−5，几乎零惩罚** |
| 5 图共享 | 2499 MiB | **+340** |

⇒ **每多一个图约 +115 MiB 常驻**（`(2499−2154)/3`）。
原因：共享块必须装下「HTP 按形状重排后无法合并」的那部分权重，
而 `ENABLE_GRAPHS` 只控制**解开哪个图**，**不控制共享块驻留多少**。

**真机后果（本项目 2026-09-06 实测）**：四段 context 常驻时——

| 形态 | ION 峰值 | 峰值时 MemAvail | 结果 |
|---|---|---|---|
| 单图（现网） | 9111 MiB | 580 | ✅ |
| **5 图共享** | **9486** | **208** | 🔴 **app 被系统杀，两次复现** |
| 单图（五比例交付） | 8885~9158 | 560~739 | ✅ |

⇒ **共享省存储（−73%）但涨常驻，两者反向。**
在一台只剩 580 MiB 余量的设备上，**+375 MiB 就是生死线**。

**规则**：转新模型做多比例/多形状时，
① 先算「四段（或全部常驻段）合计 ION」，不要只看单段；
② **判据用 MemAvailable 而不是 ION 绝对值** —— ION 9111 能活、9486 就死，
   差别在系统还剩多少可回收内存；
③ 图数不是免费的，**先用 2 图量一次惩罚**，再线性外推到目标图数。

### 42.14 ⚡ 交付规模上去后，契约校验必须**按尺寸惰性**做

app 每次启动把契约引用的全部 .bin 现算 sha256。多比例交付会把这个量推到
12.3 GiB（共享）乃至 **35 GiB（单图）**，按实测 **62 MiB/s** 就是 3.4~9.8 分钟。

做法：**启动只校验基准**，首次用到某个尺寸时再校验那一份，且每个文件一进程只算一次。
实测：启动 10667 MiB / **183 秒**（与单比例交付持平），首次切尺寸多约 100 秒，之后为零。
🔴 安全性不打折的前提是：**校验必须发生在该尺寸的任何图被装载之前**。

### 42.15 ⚠️ 找内存峰值**不能用抽稀采样**

本项目 2026-09-06 因此在诊断里绕了一圈：
2 秒采样但每 5 个点打印一次，读到「失败时 ION 7557」，而对照臂（能跑通的现网配置）
是 9111 —— 于是得出「不是内存」的**错误结论**。改回逐点读取后，真峰值是 **9486**。

⇒ 峰值类指标一律**全点扫描取 max**；
⇒ 并且**必须有对照臂**：只有「能跑通的配置」的峰值，才能说明新配置的峰值意味着什么。

## 四十三、⚖️ **Graph Switching：机制成立（省内存），但每次切换 ≈ 一次 `createFromBinary`（不省时间）**（2026-09-06 实测）

> **一句话**：它让「enable 的图」与「已装载的图」解耦，PD 与 ION 都只按已装载的算 ⇒ **突破单 context 容量上限**；
> 但 t_switch 实测 **1.76 s** ≈ 同段 `createFromBinary` 的 1.85 s ⇒ **每步都要遍历所有图的流水线不要用它**（见 §43.7/§43.8）。

> **适用范围**：QAIRT 2.48 / HTP v79（SM8750）/ unsigned PD。
> 官方把 Graph Switching 与 Multi-Graph Switching 都标为 **Beta**，
> 原文写「行为在未来版本可能改变」⇒ 换 SDK 版本必须重验。

### 43.1 它解决的是哪个问题

HTP 的 unsigned PD 有一个**每 context** 的容量上限（本机实测红线 3506~3535 MiB，见 §PD）。
以前的理解是「一个 context binary 里的东西全都要同时装进 PD」，于是模型只能切段、每段一个 context，
而多个 context 又必须**同时常驻**才能避免每步重装 ⇒ 内存被顶满。

Graph Switching 把这件事拆开了：**一个 binary 里可以放多个图，"enable" 的图不等于"已装载"的图**，
运行时只保留一个图处于装载态，执行到别的图时自动卸载旧的、装载新的。

### 43.2 怎么开（`qnn-net-run` 不用改代码就能验）

`--config_file` 的 `context_configs` 里两项，缺一不可：

```json
{ "backend_extensions": { "shared_library_path": ".../libQnnHtpNetRunExtensions.so",
                          "config_file_path": ".../det.json" },
  "context_configs": { "enable_graphs": ["graph_a", "graph_b"],
                       "is_persistent_binary": true,
                       "memory_limit_hint": 1 } }
```

- `memory_limit_hint` **任何非零值**都只表示"进入低内存模式"，具体数值不影响切换行为（官方原文）。
- `is_persistent_binary` 要求宿主侧用 **mmap** 传 binary，且**在 `QnnContext_free` 之前不得 unmap**。
- 不写 `enable_graphs` 时，binary 里所有图都算 enable，但只装载**第一个**；
  写了则只装载列表里的**第一个** ⇒ 用它控制"先装哪个"。
- **C 代码里对应** `QNN_CONTEXT_CONFIG_PERSISTENT_BINARY` 与 `QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT`。

### 43.3 实测：它确实让 PD 只按"已装载"算（单变量，三臂）

装置：把 **part1b 与 part2a 两个不同段**编进同一个 context（`weight_sharing_enabled: false`，
故意不共享权重，两图各带全量 const 1325.4 / 1272.9 MiB），产物 2751.9 MiB。
三臂**只差 context_configs**，其余（binary、输入、执行）完全相同：

| 臂 | enable | 切换 | 结果 | ION 峰值 |
|---|---|---|---|---|
| a | 两图 | ❌ | 🔴 **`RC=16`**，logcat: `Failed to find available PD … context size estimate 4033353472` | 483 MiB（没装上） |
| b | 两图 | ✅ | 🟢 成功 | **2475 MiB** |
| c | 只 part1b | ❌ | 🟢 成功 | **2475 MiB** |

⇒ ①**b 与 c 的 ION 峰值逐 MiB 相同**：两个图都 enable 与只 enable 一个，**常驻完全一样**。
⇒ ①**a 撞 PD、b 不撞**：PD 记账**只算已装载的图**，不算 binary 里的全部图。
⇒ 这条以前是**未验证假设**（官方文档只在多图切换节顺带提过 "provided they all fit into the
current PD's virtual address space"），现在有了单变量实测。

### 43.4 🔴 排错：`qnn-net-run` 的退出码只说"在哪一步"，不说"为什么"

`RC=16` = *Application failure during create from binary*（官方退出码表）。
**真正的原因不在 stdout，在 logcat**：

```
E/QNN: QnnDsp <E> Failed to find available PD for contextId 1 on deviceId 0 coreId 0
       with context size estimate 4033353472
```

⇒ **凡是 create-from-binary 失败，必须同时抓 logcat**，否则会把一次如期而至的容量失败
误判成"其它失败"（本项目就这么误判过一次，多花一轮设备时间）。
`--log_level warn` **不会**把这条打到 stdout —— 它是后端经 Android log 输出的。

### 43.5 🟢 宿主上就能预测 PD 估算（省一次"建完才发现装不上"）

PD 检查发生在**设备端** `contextFinalize`，建图阶段不报（见 §PD/#60）。但设备报的
`context size estimate` 与**宿主元数据算出的 RAM 公式**是可换算的：

| | 值 |
|---|---|
| 设备报的 PD 估算 | 4,033,353,472 B = **3846.5 MiB** |
| 宿主公式（`tools.html` Use Case 1 两图相加） | **3685.0 MiB** |
| **比值** | **1.0438** |

⇒ 经验规则：**PD 估算 ≈ 宿主公式 × 1.04**；红线 3483 MiB ⇒ **宿主公式控制在 ~3340 MiB 以内**。
⚠️ ③**只有一个数据点**，不得当精确换算；换模型/换段结构要重新标定。
（本项目 §3.4 另有两个单段 PD 数字，但原文单位写的是"MB"，按 MiB 还是 10⁶ 读会让比值在
0.985~1.059 之间摆动 ⇒ **单位不明的历史数字不能拿来标定**，这是约束 8 的老问题。）

### 43.6 ⚠️ 什么时候**不该**用它

- 与 `QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES`(5) **互斥**（官方明写）。
  注意它与 `REGISTER_MULTI_CONTEXTS`(2) 是**两个不同选项**，官方**没有**把后者列为冲突项。
- 与"外置权重 DMA 回调"、`DEFER_GRAPH_INIT`、multicore graphs 均不兼容（官方限制表）。
- 切换中的图之间**不支持并发执行**。
- 🔴 **代价要按次数算，不是按内存省了多少算**：见 §43.7。

### 43.7 🔴🔴 **代价实测：一次切换 ≈ 一次完整 `createFromBinary`，「懒加载更便宜」是错的**

这是决定它值不值得用的那个数。**测法（单变量斜率法，不依赖 profiling 是否可用）**：
两臂的 `context_configs` **完全相同**（两图都 enable + 切换开），唯一差别是**执不执行图2**：

```
臂 X: 图1×N 后 图2×N   T_x(N) = init + N·e1 + t_switch + N·e2
臂 Y: 只执行图1×N       T_y(N) = init + N·e1      （图2 的 input_list 写 "__" 跳过）
⇒ T_x(N) − T_y(N) = t_switch + N·e2 ⇒ 对 N 线性拟合，**截距即 t_switch**
```

实测（N=1/3/5，SM8750，part1b→part2a）：

| N | T_x s | T_y s | 差 s |
|---|---|---|---|
| 1 | 13.73 | 7.87 | 5.86 |
| 3 | 30.67 | 16.47 | 14.20 |
| 5 | 47.86 | 25.48 | 22.38 |

拟合 **T_x − T_y = 1.758 + 4.129·N**（三点残差 < 0.03 s）
⇒ **t_switch = 1.76 s**，图2 单次执行 e2 = 4.13 s。

🔴 **关键对照**：同一段 part2a 的**完整 `createFromBinary`** 在真实 app 的 logcat 里实测中位 **1.85 s**。
⇒ **t_switch(1.76) ≈ createFromBinary(1.85)** —— **图切换的「懒加载」并不比重新建 context 便宜。**

**这条推翻了采用它的核心前提。** 它省的是**内存**，不是**时间**；
在「每一步都要用到所有图」的工作负载（扩散模型的去噪循环就是）上，
切换次数 = 图数 × 步数，代价线性堆叠：本项目四段八步 = 32 次 × 1.76 s = **56 s**，
而同样省内存的「把某一段改成每步装卸」只要 **19~43 s**（实测区间）⇒ **图切换反而更贵**。

### 43.8 一句话结论（给下一个模型）

**Graph Switching 适合「多个图但一次只用其中一个、且切换不频繁」的场景**
（典型：一个模型多个输入分辨率／多个 LoRA adapter，用户切一次用很久）。
**不适合「每次推理都要走遍所有图」的流水线** —— 那种情况下它退化成「每步重新加载」，
且没有比直接管理 context 生命周期更便宜。

⊕ 判断方法（动手前就能算）：**切换次数 × t_switch vs 直接省下的常驻内存能换来什么**。
t_switch 可以用本节的斜率法在**不改任何应用代码**的前提下测出来，成本约 10 分钟设备时间。

## 四十四、🔴🔴 **改定长序列时最隐蔽的一类错：padding token 会参与注意力，结果取决于「补了几个」**（2026-09-06~07 实测）

> **一句话**：把模型的固定序列长度从 A 改成 B（为了支持更长的 prompt），
> 即使**每一个形状常量都改对了**，结果仍可能与参考实现差一个数量级 ——
> 因为 padding token **被值替换后照样参与注意力**，而参考实现的 padding **个数**与你不同。
> 本项目为此在现网带了一个 **step-0 相对 L2 15.5%** 的偏差，**三周无人发现**。

### 44.1 现象

Z-Image Turbo 的 caption 槽数从 32 改成 80（为支持更长 prompt）。手术只改常量、不重导 PyTorch。
逐条核对下来**所有被改的常量都正确**（全零的仍全零、等差的正确延续、`unified_mask` 仍全 True）。

但与官方实现对照：

| 装置 | step-0 噪声预测相对 L2（对官方，同 latents 同 caption） |
|---|---|
| **L=32（手术前的基座）** | **1.79%**（余弦 0.999841）|
| **L=80（手术后，现网）** | **15.52%**（余弦 0.987894）|

⇒ ①**基座导出是忠实的；15.5% 全部由「改序列长度」这一个动作引入。**

### 44.2 机制（四条实测互相咬合，不是推理）

| 实测 | 推出 |
|---|---|
| 把 caption 的 padding 区**内容置零** ⇒ step-0 输出**逐字节相同** | padding 的**值**已被图内覆盖，输入内容根本没被用 |
| 把 `cap_pad_mask` 翻转成「全是真 token」⇒ 输出变 **2.71%** | 该 mask **是活的**，确实在做值替换 |
| 图结构：`cap_pad_mask → Gather → Unsqueeze → **Where** → Pad → ScatterElements → RmsNorm/Mul/Add` | 它是**值掩码**，**从未进入任何 Softmax** |
| padding **个数** 10 → 58 ⇒ 第一段输出就差 **1.24%**，逐段放大到最终 **15.80%** | padding token **仍然占着注意力的位置** |

**官方也是同一套机制**（`transformer_z_image.py`）：
```python
feats_cat = torch.where(mask, pad_token, feats_cat)   # 值替换成学出来的 pad_token
...
if all(seq == max_seqlen for seq in item_seqlens):
    attn_mask = None        # ← batch=1 时根本没有注意力掩码
```
⇒ **双方都让 padding 参与注意力。唯一的差别是补了几个。**

| | caption 槽数 | 真实 token | **pad token 个数** |
|---|---|---|---|
| 官方（`ceil(n/32)*32`，动态） | 22 → **32** | 22 | **10** |
| 我们（固定 80） | **80** | 22 | **58** |

多出来的 48 个 `cap_pad_token` 是一批**内容相同**的 key，对每个 image query 给出同样的 logit，
**稀释 softmax** ⇒ 逐层放大。

### 44.3 🔴 由此得到的通用规则

1. **改定长序列长度时，必须同时改「补齐约定」以匹配参考实现，而不只是改形状常量。**
   参考实现若是 `ceil(n/K)*K`，你的固定长度**必须是 K 的倍数**，否则**对任何 n 都对不齐**。
   ⊕ 本项目选了 **80**，而 `SEQ_MULTI_OF = 32` ⇒ **80 不是 32 的倍数 ⇒ 永远对不上**。
2. **先查 padding 是「值掩码」还是「注意力掩码」。** 判别只要三个单变量：
   ①改 padding 内容 → 输出变不变（不变 ⇒ 值被覆盖）；
   ②翻转 mask → 输出变不变（变 ⇒ mask 是活的）；
   ③**改 padding 个数** → 输出变不变（**变 ⇒ 它们在参与注意力**）。
   第③条是关键，也是最容易漏的。
3. **定位方法：差分对拍。** 不要一上来就和参考实现比整链，
   而是**拿自己的「手术前」与「手术后」两个版本比**，逐段落盘、只报**第一个**超阈值的张量。
   本项目就是这样把 15.5% 一刀切到 `part1a` 的输出上的（1.24%），
   两臂都是自己的产物，**成本是与参考实现对拍的十分之一**。

### 44.4 🔴 为什么这类缺陷能长期潜伏

本项目有一套三层指标面板（L1 中间张量 / L2 终点 latents / L3 成图 PSNR），
但它比的是「**量化后 vs 我们自己的 FP32**」——**两臂用的是同一个手术后的图，同时带伤**
⇒ **面板在结构上就看不见导出/手术类缺陷**，它只能测量化误差。

⇒ **通则：任何「自己和自己比」的指标，都测不出自己这一侧的系统性偏差。
   必须定期与参考实现（官方实现）对拍，哪怕只对 step-0。**
本项目的审查报告早在一个月前就写下过这条规则并要求执行，但没有被执行；
一个月后才发现参考系本身偏了。

### 44.5 成本（供下次估算）

| 步骤 | 成本 |
|---|---|
| 官方 PyTorch 单步（bf16，CPU） | 约 **20~25 分钟/次**，内存峰值约 12 GB |
| 我们的 ONNX 链单步 | 约 **5 分钟**，内存峰值约 16.5 GB |
| 逐段落盘 + 比对（自己两个版本） | 两次单步 + 秒级比对 |
| **总计定位到段** | **约 1 小时**，纯宿主 |

⚠️ **必须先做装置门**：用参考实现的**已知输出**（如日志里记过的 `noise_std`）复现一次，
相对差 <=0.1% 才可解读后续数字。本项目该门实测 **0.0022%** 通过。
⚠️ **符号约定**：两侧对「噪声预测」的定义可能差一个负号
（官方 `noise_pred = −model_out` 配 `sample + dt·out`；我们 `nxt = lat − dt·noise`，dt 为负）。
第一版没对齐符号，报出 **198%、余弦 −0.99** 的假警报 ——
**靠「已知样本」才没有报错**：step-0 之后的 `latents_std` 两边一致（0.9516 / 0.9517）。


---

## 四十五、🟢 生图速度诊断：先拆预算、再按算量归一找离群段（2026-09-17，Z-Image P2 实战）

> 场景：模型已跑通、要提速，且**不允许改变输出**（出图 sha256 逐字节不变）。
> 本章是方法与陷阱；本项目的具体数字与结论状态见 HANDOVER §15.46、`scripts/EXP_PLAN_P2_SPEED.md`。
> ⚠️ 本章里**速度收益类**结论目前都是 ③ 假设（设备诊断未做），**结构事实与工具行为**是 ① 实测，下面逐条标明。

### 45.1 第一步：用后端单调时钟拆整张图的时间预算（零设备）

- 用 app 已落盘的 logcat：后端每行自带单调毫秒钟（`Backend: 171294.0ms`），比 logcat 墙钟可靠。
- 逐段计时要在 app 里埋点（本项目 `[segtime]`，只加日志、不碰数值，由出图 sha256 不变自证惰性）。
- 🔴 **装载耗时不能只看 `Initializing → Initialized` 两个标记**：本项目标记内只有 7.97 s，
  标记**之前**的整文件读入另有 12.19 s（①）——只看标记会把装载低估一半以上。
  ⇒ 装载阶段应取「上一段就绪 → 本段就绪」的全程。

### 45.2 第二步：按矩阵乘算量归一，找「单位算量更慢」的离群段

- 从 ONNX **结构**（`onnx.load(..., load_external_data=False)` + shape inference，不载权重，秒级）
  数出每段 MatMul/Gemm 的乘法次数 G；再用段墙钟 ÷ G 得 ms/G。
- **用多个输入尺寸当天然样本**：本项目 5 个宽高比 × 4 段 = 20 个点（①），
  三段稳定在 0.474~0.517 ms/G，唯独一段是 0.709~0.750 ⇒ 离群段一眼可见，且与尺寸无关。
- 🔴 **不要在「只有一个高值点」时做宿主回归**：权重字节、算子数、opData 在离群段上同时偏高、在其余段几乎相等 ⇒
  变量共线，回归系数没有物理意义（本项目曾拟合出「计算量系数 0.01」，与小尺寸明显更快直接矛盾）。
- 排除 IO 与 CPU 侧：比较各段输入输出字节数；比较段间张量生产方/消费方的 scale/offset——
  **不同就意味着宿主每步做一次逐元素重量化**（本项目三处 1604 万元素）。

### 45.3 🔴 导出陷阱：batch=1 的 `pad_sequence` 会变成全尺寸 ScatterElements（① 结构；速度影响 ③）

- 形态：`ScatterElements(data=Expand(fill), indices=Expand(0), updates=Unsqueeze(Pad(x)), axis=0)`。
  沿大小为 1 的维度、索引全 0 ⇒ **FP32 下输出严格等于 updates**，是纯冗余。
- 代价（①）：fill 与 Int32 索引在 DLC 里成为与激活同尺寸的 **STATIC 常量**；本项目 1:1 图 `constSize` 183.84 MiB（其余段 0）。
  速度代价 ③ 未验证（本项目的嫌疑：每步约 1.8 s）。
- **转新模型时的检查**（Flux.2 Klein 等，DiT 类几乎都有 caption/图像 padding）：
  `grep` 转储里 `ScatterElements`，输出元素数 ≥ 1M 的逐个看结构；
  再比**输出与 updates 的 encoding**——**相同**则删掉它下游量化值不变，才有可能保持逐字节一致；不同则删除必然改变数值。
- 量化命令若是 `input_list=None` + 全量 overrides，重量化是确定性的，手术后可按 DLC 内嵌的原命令重建。

### 45.4 装载路径：确认是 mmap 还是「整文件读进内存」（①）

- 本项目同一个 app 里两条路径：`createAndInitModel` → `QnnSampleApp::createFromBinary`（mmap + madvise）；
  `createAndInitContext` → `std::ifstream` 读进 `std::vector<uint8_t>`（构造清零 + 整份拷贝）→ `createFromBuffer`。
  **Z-Image 走的是后者**；台账曾把前者的 mmap 说成「我们已经在用」——**先 grep 确认自己的模型走哪条路径**。

### 45.5 HTP 工具行为（① 宿主探针，SDK 2.48 Windows，SM8750 / soc_model 69）

| 行为 | 实测 |
|---|---|
| `vtcm_mb` 上限 | 图名正确时写 0 / 8 / **64** / 不写 / `--vtcm_override -1` ⇒ 读回**全部 8**、产物 md5 相同 ⇒ **SM8750 上限即 8，调大无空间** |
| 图名写错 | graphs 段整段落空、**不报错**；devices 段仍生效（dspArch 79），VTCM 退回 **4** ⇒ 唯一判据是读回元数据（与 §15.28.5 同一个坑，本项目第二次踩） |
| `hvx_threads` 不设 | 元数据 `numHvxThreads` = **0**（官方文档写「写入默认值 4」，与实测不符）；设 6 ⇒ 6、md5 变。**0 的运行时含义已实测（2026-09-18）：用 6 个线程，与在线建图（文档说用 SoC 最大值）相同 ⇒ SM8750 上限 6，已用满、无杠杆**。读法：设备上 `--profiling_level basic`，viewer 输出里的 `Number of HVX threads used : N  count` |
| 宿主 HTP Performance Estimates | 配置确认生效后 `--profiling_level basic/detailed` 建图，**不产出**估算与带宽统计 ⇒ 对 SM8750 只能上设备 profiling |

### 45.6 设备侧逐算子 profiling 的选择（① 官方文档）

| 档位 | 需要重建 context？ | 给什么 |
|---|---|---|
| `basic` | 否 | 每次推理的主机/加速器执行时间；HVX 线程事件 8001 |
| `detailed` | 否 | 逐算子 **cycles**（文档：并行执行下**不能换算成微秒**，只能做相对比较） |
| `backend` + `linting` | 否 | 主线程 cycles + 每个算子的 Wait + 后台重叠 |
| `optrace` | **是**（建图时也要加 `--profiling_level detailed --profiling_option optrace`） | 时间线 + QHAS 报告（含 DMA 读写） |

🔴 **离线 `qnn-net-run` 测速必须对齐 app 的功耗档**（`--perf_profile burst`）：本项目早先离线单段每次 15.9 s、app 内同段 3.3 s，
③疑为没设功耗档。⇒ 离线诊断前先过**代表性门**：离线测得的段间耗时比必须复现 app 内实测的比值，否则不下归因结论。

### 45.8 🔴🔴 `qnn-net-run` + `.bin` 的 I/O 类型陷阱：喂原生字节会被**静默砍半**，而字节守卫会报 PASS

**默认按 float32 解析输入、按 float32 写输出**；`--use_native_input_files` / `--use_native_output_files` 才按原生
（后者会把输出文件名改成 `<name>_native.raw`）。

**为什么会骗过守卫**：对 `uFxp_16` 张量，**原生字节数 ≡ float32 字节数的一半**。
喂原生字节 ⇒ 运行器按「字节数 ÷ 期望字节数 = 0.5」缩小张量、**rc=0 不报错**；
写出的 float32 输出**恰好等于契约里的原生字节数** ⇒ 「输出字节数 == 契约」这条守卫报 PASS。

**实测三臂（2026-09-18，part1b 单段）**：float32 全尺寸输入 ⇒ 输出 64,143,360 B（对）｜
原生 + 两个 native 标志 ⇒ 输出 32,071,680 B（对）｜**原生 + 无标志 ⇒ 输出 32,071,680 B（只有一半元素）**。

**自证做法（三条一起用）**：① 显式带两个 native 标志；② **逐个**输出张量核字节数；
③ 拿 `Bool_8` / `Int_32` 张量当**模式指示器**（它们的原生字节数 ≠ float32 的一半：float32 模式下分别是 ×4、×1，一眼能看出当前处在哪种模式）。

⇒ 本项目由此订正了一条历史数据的适用范围（离线测速的绝对值不可引用，相对比较仍可能成立）。

### 45.9 读 profiling 输出时的两个坑

1. **总量行与逐算子行长得一样**：`Accelerator (execute) time (cycles) : N  cycles` 是**整段总量**，
   其下缩进的 `node_xxx:OpId_k (cycles) : v  cycles` 才是算子。把总量行当算子会让「单算子占比」被摊薄一半。
   ⇒ 解析只认带 `:OpId_<数字>` 的行；**并且做一次按算子类归并** —— 本项目正是靠「出现一个占 49% 的算子叫 Accelerator (execute) time」当场发现解析错了。
2. **日志 `adb pull` 可能 Permission denied**：先 `chmod -R a+rX`，仍不行用 `adb exec-out cat`（二进制安全）兜底。

### 45.10 ⚠️ 离线 `qnn-net-run` 与 app 内的绝对耗时可能差数倍（本项目 4.5~5.0×）

同一段、同一份 context：离线加速器执行时间 part1a **29.0 s/次**、part1b **14.5 s/次**；
app 内同段每步 **5.83 s / 3.2 s**。**比值一致（1.99 vs 1.72~1.83），绝对值差约 5 倍**，原因未查（③疑 shell 侧拿不到 app 那套 DCVS 电压角）。
⇒ **离线诊断只能用于比值与归因；「能省多少秒」必须回到 app 内测**。定诊断方案时应把「比值落在 app 实测区间」设成**代表性门**，不过门就不下归因结论。

### 45.7 一句话（给下一个模型）

**先拆全图预算，再用「多尺寸 × 多段」的 ms/G 找离群段，再读离群段独有的结构；
凡是要删算子保数值的，先比输出与输入的 encoding；凡是改建图配置的，读回元数据确认生效。**


---

# 四十七、【路线图】把一个新模型从零做到可交付并提速

> **这一章是给"下一个模型"用的入口**（Flux.2 Klein 等）。本项目（Z-Image Turbo，SM8750/v79，QAIRT 2.48，Windows 宿主）
> 从零到可交付 + 三轮提速一共花了约五周，其中**相当一部分是可以避免的返工**。
> 本章按阶段列出：**必过的门**（不过就别往下走）、**该阶段最贵的坑**（都真实发生过）、**实测成本**。
> 细节不在这里重复，每条都指向对应章节。

## 47.0 适用范围（先确认，否则整章都可能不适用）

| 维度 | 本项目实测 | 换模型时要重新确认 |
|---|---|---|
| 芯片 / 后端 | SM8750（v79）、HTP、unsigned PD | PD 红线（本项目 **3.65 GB/context**，§十五）、VTCM 上限（本项目 **8 MB，写 64 也被钳**，§45.5） |
| SDK | QAIRT 2.48.0.260626，Windows x86 宿主 + aarch64-android 设备 | 工具选项会变；**每个新版本先跑一遍 §47.1 的工具自检** |
| 模型类型 | DiT（单流 + 图像/文本双流 refiner），8 步蒸馏 | 结构不同则 §47.3 的手术清单要重来 |
| 量化 | A16W8、per-row、选择性 FP16 + 显式 Clip | 见 §二十九 |

## 47.1 阶段表（顺序不可乱；每阶段的门不过就停）

| # | 阶段 | 必过的门 | 实测成本 |
|---|---|---|---|
| 0 | **开工准备** | 磁盘 ≥ 150 GB 空闲；宿主内存 ≥ 24 GB；`code_lint.py --strict` 干净 | — |
| 1 | **PyTorch → ONNX 导出** | 形状全部固定；**导出后立刻查"烘焙常量"**（§47.2 坑 1） | 本项目未亲自做（§二·补） |
| 2 | **图手术**（序列长度 / 比例 / Clip / FP16 白名单） | 每处手术**单独**在 FP32 下与原图比**逐位**（ORT，约 1 分钟/次） | 每处 10~30 分钟 |
| 3 | **校准数据** | 分布必须与真实推理输入一致（§四） | 收集 + 验证 1~2 小时 |
| 4 | **转换 + 量化** | `Processed N encodings` 不为 0（overrides 闸门，§十五 3）；读回 DLC 确认 dtype 落对（§27.8） | 转换 5 分钟、量化 3 小时（首次）/ 0.1 分钟（有 overrides 时） |
| 5 | **建 context** | **必带 `--config_file`**；建完**读回元数据**核 `dspArch` / `vtcmSize` / 图名（§十二、§45.5） | 15~25 分钟/段；五图 mg 约 2.7 小时 |
| 6 | **宿主侧接口** | 按 `EXECUTION_MODEL.md` 的**三条路径**分别处理；逐个张量核字节数（§47.2 坑 5） | — |
| 7 | **设备装载** | PD 红线、ION 峰值、平台期 MemAvail（与同日现网配对，§37.x） | 每次装载 2~7 s/段 |
| 8 | **数值验证** | 三层面板 + 口径纪律；**与参考实现对拍**（不能只和自己比，§44.4） | 单步 FP32 参考 1~2 分钟（单段）到 5 分钟（整链） |
| 9 | **交付** | 四条铁律（基线先验、备份、真实链路跑一次、真实环境验证） | 一次交付会话 30~60 分钟 |
| 10 | **提速** | 见 §45 与 §47.4 | 见下 |

⊕ **这张表只说"顺序和门的名字"。要照着跑，用下面的 §47.1.1 六张执行卡片**
（命令 + 判据明确到能判通过/不通过 + 该阶段挂载的坑）。

## 47.1.1 六张执行卡片（**把上表展开成可以照着跑的东西**）

> 上面那张阶段表回答"按什么顺序做"，本节回答"每步执行什么、怎么确认做对了"。
> **每张卡片只写三样：命令 / 门（含明确判据）/ 本阶段挂载的坑**，细节一律回指，不重复。
> ⚠️ 命令里的路径、张量名、SoC 都要按你的模型改；标 🧩 的是**本项目没实测过的缺口**。
>
> 🔴 **每个门都标了工具状态**（2026-09-20 逐个实跑得出，见 `scripts/EXP_PLAN_GATE_REPLAY.md`）：
>
> | 标记 | 含义 |
> |---|---|
> | ✅ | **有通用工具**，换模型改个路径就能跑，输出能直接判通过/不通过 |
> | ⚠️ | 有工具但**写死了常量**（SDK 版本、DLC 目录），换模型必须先改文件头 |
> | 🔨 | **需自建**：这件事该做、判据也明确，但本项目**没有可复用的实现**（附实现要点） |
>
> ⊕ **没有标记的门 = 那一轮没验证过**（34 个门里有 20 个需要设备或需要生成新产物，
> 在"零磁盘 / 零设备"约束下测不了）。**它们的工具状态仍是未知的，别默认能跑。**
>
> **那次重放的 14 个可测门里，🔴+🟡 占 42.9%，按事前判据是失败的。**
> 根因：上一轮写卡片时把"这件事该做"和"这件事有工具"混为一谈（§47.10 模式 4 的重犯）。
> ⇒ 现在这三档标记就是那次失败的处置 —— **别把 🔨 当成能直接跑的命令。**

---

### 卡 A｜阶段 0–1：开工准备 + 导出体检

**做什么**：把宿主备好；拿到 ONNX 后**立刻**做一次结构体检（10 分钟，省掉后面数小时）。

```bash
python scripts/code_lint.py --strict                      # 门 A1
python scripts/check_graph_io_convention.py <onnx目录>     # 门 A3-a：烘焙的前/后处理
python scripts/scan_export_residue.py --selftest          # 门 A3-b：先验判据本身（约束 8）
python scripts/scan_export_residue.py <onnx目录>           # 门 A3-b：扫导出残留
```

| 门 | 判据（明确到能判通过/不通过） |
|---|---|
| **A1 宿主** ✅ | 磁盘 ≥ 150 GB 空闲；内存 ≥ 24 GB；lint 干净。<br>🔴 **上一个模型的产物必须先清** —— 转完一个模型磁盘上会堆几百 GB。清的判据不是"它是不是证据"，而是**"这个结论还需要重新验证吗"**；大产物删、小文件（文档/manifest）一律留。**做法与四条操作纪律见 §21.4**，工具 `cleanup_by_lineage.py` |
| **A2 红线**（换 SoC 必做） 🔨 | 实测单 context PD 上限与 VTCM 上限（**需设备**）。本项目 SM8750/v79：**PD ≈ 3.62~3.65 GB**、**VTCM 上限 8 MB（写 64 被静默钳到 8）**。🔴 **这两个数是芯片相关的，不得沿用** |
| **A3-a 烘焙常量** ✅ | 每个图的头尾**带常量的逐元素算子**逐个登记并判定"宿主该不该再做一遍"。本项目 `vae_decoder` 头节点 `Div(0.3611)` 宿主又除一次 ⇒ 解码错 + 输入超校准量程 **2.77 倍被硬钳**。<br>⊕ 实跑：扫 155 图报 **315 处**，输出末尾有红标（`Div val_0 = 0.361100`）与**真嫌疑 vs 假阳性判读规则** —— 🔴 **要看到末尾，别用 `head` 截断**（§47.10 模式 3） |
| **A3-b 导出残留** ✅ | `scan_export_residue.py` 的"判定为恒等/填充"一栏**应为空**；非空的每一条都要在**导出阶段**改掉。本项目两个大的占 part1a **29.1%** cycles。<br>🔴 **两个必看的输出**：① `--selftest` 必须先过（判据自检）；② 若报"shape 信息不完整"，**结果可能漏报** —— 先跑 §3.1 的固化补全 `value_info` 再扫（实测：同一对图，带 value_info 扫出 6 个、不带只扫出 2 个，差异全来自这里） |
| **A3-c 烘焙序列长度** ✅ | `python scripts/scan_baked_seqlen.py <onnx> --len L`：扫 initializer **+ `value_info` 的 shape 维度**（§34.1：只改 initializer 不够），并对每个命中打印**消费者节点**。**导出时用的示例 prompt 就是产品硬上限**。<br>🔴 **命中数不是判据，工具只缩小范围**。自检实测：真实 L=20 的图去扫 32 会命中 **10 个、全是撞值**（头数 32）。必须按 §33.5 **逐个追消费者**判归属，**不许按位置或整条形状替换** |
| **A3-d 算量基线** ✅ | 数 MatMul/Gemm 乘法次数（按段、按比例）。提速阶段的第一把尺是「段墙钟 ÷ 算量」，没有它就只能看绝对耗时。<br>⊕ `p2_zero_device_evidence.py` 直接出 ms/G 表；新模型没有 `[segtime]` 日志时**只有算量列可用**（那列是从 ONNX 结构数的，通用） |
| **A3-e FP32 参考** ✅ | 真实 step-0 的**输入与输出全张量**落盘。后面每次量化/手术/建图对拍都要它；**没有它就只能"和自己比"，而那验不出系统性错误**。<br>**格式定死如下**（照此做就能判通过/不通过）：<br>· 目录 `<产物根>/fp32_ref/<seg>/`，每个张量一个 `<张量名>.raw`，**一律 float32**（与 `EXECUTION_MODEL` 的宿主接口同口径）<br>· 同目录 `manifest.json`：每个张量记 `{shape, dtype, bytes, sha256}`，另记产生它的 ONNX 文件 sha256 与输入 seed/prompt<br>· **门的判据**：对拍前先核 `manifest.json` 里每个 `sha256` —— 有一个对不上就**停**（参考被动过或截断，据此得出的所有结论作废）<br>· 🔴 **参考必须与被测件同源**：`manifest` 里的 ONNX sha256 要能在谱系（`build_lineage.py`）里对上当前交付件的来源 |

**坑**：A3-b 是本项目最贵的一条 —— 事后再删是图级手术，而**同型的两处删一处安全、删另一处毁图**（§47.3）。
🧩 **缺口**：本项目的 ONNX 是别人给的，**"怎么导出"没有实测配方**；§47.7 给的是"下游反过来对导出提的 8 条硬性要求"，
要自己导出时以那 8 条为验收，导出过程本身请另找依据（§二·补 2.5 列了没记录下来的三项）。

---

### 卡 B｜阶段 2–3：图手术 + 校准数据

```bash
# 🔨 门 B2/B3 没有可直接用的工具，见下表。参考实现思路见 scripts/map_opid.py（但它是一次性脚本）
```

| 门 | 判据 |
|---|---|
| **B1 手术等价** ✅ | **每一处手术单独**验等价。**三层，别把第一层当成全部**（2026-09-21 订正）：<br>① **结构性断言**（宿主、秒级，**本项目实际用的就是这层**）：`p2_h1_surgery.py` 的 A1~A6 —— 目标节点类型/轴、四者形状、**indices 全量逐元素检查为 0**（不是抽样）、手术后图 IO 名与形状逐个不变、形状推理通过无悬空引用、删除集合 ⊆ 预期。<br>② **ORT 数值对比**（可选）：⚠️ 大段要加载全部 FP32 权重（本项目 part1a/1b 共享的 `.data` 就 **13.76 GB**，宿主可用内存 15.8 GB）⇒ **有压死宿主的实测先例**，做之前先量内存，或只在小段上做。<br>③ 🔴🔴 **设备实测，不可省**：①②**都不充分** —— §47.3 实测同一张图里两处结构完全相同的实例**结论相反**（删一个 9 输出逐字节相同、删另一个主体误差 125% 毁图）。<br>🔴 **不得由"同型结构"外推**；FP32 等价**推不出** HTP 安全 |
| **B2 极端边** ✅ | `python scripts/scan_requant_edges.py <dump.txt>`：**算子内全部张量两两取 scale 比值**（含 STATIC，张量名允许含点），按 float32 复算（>3.4028235e38 即 inf）。判据：无 > 1e4 的边。⚠️ 光查"有没有 ±FLT_MAX 张量"**查不出**这一类。<br>⚠️ **转储精度陷阱**：scale 只打印 12 位小数，比值会显示 **999936** 而不是整 1e6（§3.2.2）。<br>🔴 **别用 `scan_requant_ratios.py`**（只算"激活输入→输出"且排除 STATIC，恰好扫不到掩码常量那类边）；**`map_opid.py` 也不能直接用**（路径与研究问题全写死，输出 GBK 乱码） |
| **B3 掩码输出** ✅ | `python scripts/scan_where_clip.py <dump.txt>`：判据 `min(输出) ≤ min(所有数据分支)`。常量改好了它照样可能不成立 —— 自检实测：maskfix 把常量改成 -100 后**同一节点仍然 FAIL**（§3.2.1）。<br>⊕ 两个工具共用一份转储解析器（`scan_where_clip.parse_dump`），免得两份各自漂移 |
| **B4 校准覆盖** | 各条件分支都被覆盖（掩码激活/不激活、不同序列长度、timestep 区间）。本项目 `cap_pad_mask` 全 0 ⇒ 畸形 scale 污染 15 条下游边 |
| **B5 校准分布** | 校准数据分布 vs **推理时实际喂入**的分布（std/min/max）。两者由不同代码产生时必漂，本项目漂 2.77 倍、输入被硬钳 |
| **B6 timestep 量程** | 与实际调度器取值范围精确吻合（本项目 [0, 0.7] ↔ 8 步 `1-sigma`） |

**决策点·分段**：切口按 **PD 红线反推**（单 context ≤ 3.3 GB 留余量），选在**张量少、字节小**处，
**尽量切在段间 encoding 一致处**（不一致 ⇒ 每步 CPU 重量化，`prep` 14 ms → 约 690 ms）；
**切完必须 DCE**（§十三）。详见 §3.3。

---

### 卡 C｜阶段 4：转换 + 量化

```bash
qairt-converter  --input_network <seg>_fixed.onnx --output_path <seg>_fp32.dlc \
                 --target_backend HTP --onnx_skip_simplification -d <input> <dims> ...
qairt-quantizer  --input_dlc <seg>_fp32.dlc --output_dlc <seg>_q.dlc --input_list calib.txt \
                 --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
                 --act_quantizer_calibration min-max --param_quantizer_calibration min-max \
                 --act_quantizer_schema asymmetric --param_quantizer_schema asymmetric \
                 --use_per_row_quantization --target_backend HTP
```

| 门 | 判据 |
|---|---|
| **C1 opset** ✅ | `onnx.load(f, load_external_data=False).opset_import` 一行读出。<br>🔴 **不是"一律降到 17"** —— 2026-09-20 实测：本项目 `text_encoder_part1~4` 与 `vae_decoder` **都是 opset 18 且已交付跑通**。需要降级的是**特定算子**（transformer 那条单元素 `Split`），不是整个 opset。先转，报错了再按算子降（§2.4 订正） |
| **C2 overrides 闸门** | 用 overrides 时 `Processed N quantization encodings` **N ≠ 0**。N=0 = **静默失效**（多半是用了 DLC 内部名；必须用 **ONNX 张量名**，§10.4） |
| **C3 float_bitwidth** | 用 overrides 时**必须显式** `--float_bitwidth 32`，否则整张图被降成 FP16（§10.6） |
| **C4 dtype 落对** ✅ | 读回 DLC 确认白名单为 `Float_16`、权重仍 `sFxp_8`/per-row（§27.8、§29.4 G1）。<br>⊕ 实跑：`snpe-dlc-info -i <q.dlc>` 后 `grep -E "Float_16\|sFxp_8\|axis-quant"`。**输出表格极宽，不 grep 读不了** |
| **C5 健康检查** ✅ | `python scripts/check_quant_health.py <stats.json> [--fp16-candidates]`：算 `levels = median(\|a\|)/scale`，排序，`< 10` 的就是手术名单，并按 §29.2 筛 FP16 候选。<br>🔴 **量程必须用实测值**：自检实测 **26/53** 个张量的真实 amax 超过标定 calib_max，最大低估 **1.9512 倍**（§28.10）。按标定值筛会把超范围张量放进白名单 ⇒ 设备出 inf。没有实测时用「标定 × 4.0」保守代理。<br>⚠️ stats 怎么来：ONNX 加 `graph.output` 跑 FP32 前向，**只算两个标量、不落盘**，且 **§31.5 必须分批 —— 一次声明几百个图输出会把机器压死** |

**决策点·量化配方**（按顺序走，每步都是单变量）：
1. **A16W8 + `--use_per_row_quantization`** ← DiT/Transformer 起手式（**不是** per-channel，后者零 Conv 时空转）
2. 跑健康检查；出现「级/元素 < 10」⇒ 上**手术式选择性 FP16**（§二十九，选择规则三条缺一不可，最优配置可照抄 §29.8）
3. 🔴 **转更多张量到 FP16 不是单调更好的，存在最优点**（§29.7）

🔴 **两条容易漏的前置**：
- 开 per-row **之前**先把需要的 **SNPE CPU 参考跑完落盘** —— 开了之后这条离线校验路径就没了（§11.3）
- **第一次量化就留下 overrides JSON** ⇒ 后续重量化 3 小时 → **0.1 分钟**，且**不需要校准数据**（§35.1）

**代价门 G0-cost（强制）**：凡改 encoding / 量化参数，先用**已有 FP32 张量**做 quantize-dequantize 模拟，
报**新旧两版的纯表示误差（全量 + 主体两个口径）**。
**否决条件**：新 encoding 的纯表示误差 > 当前端到端实测误差 ⇒ **直接否决，不许上机**（CLAUDE.md 约束 4·补，本项目为跳过它付了 80 分钟 + 占用设备）。

---

### 卡 D｜阶段 5：建 context binary

```bash
qnn-context-binary-generator --backend <SDK>/lib/x86_64-windows-msvc/QnnHtp.dll \
  --dlc_path <seg>_q.dlc --binary_file <seg>_ctx --output_dir <dir> \
  --htp_socs sm8750 --config_file backend_ext.json      # 🔴 两个都不能少
# 门 D2（✅ 实跑验证，2026-09-20）：注意 --expect-arch 收的是 int「79」，不是「v79」
python scripts/check_ctx_identity.py <产物>.SM8750.bin --expect-arch 79 --expect-vtcm 8
python scripts/check_ctx_identity.py <产物> <部署版> --diff    # 多产物逐项对照（对照实验必做）
```

| 门 | 判据 |
|---|---|
| **D1 产物名** ✅ | 带 `--htp_socs` 才会产出 `foo_ctx.**SM8750**.bin`；少了它 `rc=0`、**无任何警告**，产出的**不是**设备用的离线缓存 |
| **D2 装置画像** 🔴🔴 ✅ | **读回元数据**核 `dspArch` / `vtcmSize` / **图名**三项。**不看文件名** —— 本项目传了 `--htp_socs`、文件名也带 `.SM8750`，仍因漏 `--config_file` 被静默编成 **v68 / 4 MB VTCM**，**一整轮结论作废，数值差 15 倍** |
| **D3 graphs 段** | 配置文件必须有 `graphs` 段，且 `graph_names` = **converter 产出的 fp32 DLC 文件名主干**。填错 ⇒ 整段**静默落空**（VTCM 退回 4 MB），不报错 |
| **D4 容量** | 建得出、装得上，无 `Failed to find available PD`（要开 FARF 才看得到真实错误，§12.3） |

**坑**：`backend_ext.json` 是**两层结构**（外层指向内层，§二④）；`cores`/`perf_profile`/`rpc_control_latency`
只对 `qnn-net-run` 有效，写进来报 Unknown Key。`snpe-dlc-graph-prepare` 的开关**带不进 `.bin`**（§24.3）。

---

### 卡 E｜阶段 6–8：宿主接口 + 装载 + 数值验证

| 门 | 判据 |
|---|---|
| **E1 I/O 解析模式** 🔴🔴 | `.raw` 的 dtype 与工具标志必须一致：`snpe-net-run` ⇒ **一律 float32**；`qnn-net-run` ⇒ **标志随实际 dtype 走**。两边都会**静默缩张量且 rc=0**，`qnn-net-run` 那条**连字节守卫都骗得过**。用 `Bool_8`/`Int_32` 当模式指示器（§六、§45.8） |
| **E2 字节守卫** ⚠️ | 四项强制检查：输入键集合 / 输入字节数 / 输出名 / **输出字节数**（`scripts/snpe_runner.py`，最后一项是抓静默缩批的唯一防线）。<br>🔴 **换模型前先改文件头常量**：`snpe_runner.py` 写死 `SDK` 与 `DLC_DIR`（2 处），`dlc_contracts.py` 写死 4 个路径（含 `OUT_JSON`）。§47.5 曾把这两个标成"无关，直接用"，**不准确** |
| **E3 级别 2** | 混合流水线出图；**人打开图片看**（不许转述形容词） |
| **E4 级别 2.5** 🔴 **不可省** | 设备上 `qnn-net-run` 跑 `.bin`，**隔离 + 级联两组都做**（只做级联无法归因）。本项目：级别 1/2 全绿、`EXIT=0`，**设备上照样出色块** |
| **E5 对拍参照** | 必须与**参考实现**对拍，不能只和自己比 —— "和自己比"验不出系统性错误（§44.4） |
| **E6 内存口径** | 看 `/proc/meminfo` 的 **`IonTotalUsed`**（1 Hz 采样，热分区路径要缓存），**不是**进程 `VmRSS`。设备实际占用 ≈ 文件字节 **× 1.42** |

🔴 **口径纪律（约束 7）**：用相对 L2 必须同报"前 1% 元素占 ‖a‖² 比例"，>50% 就改用主体口径；
**余弦 < 0.3 的张量退出定量比较**（已与真值去相关，比大小没有意义）。
🔴 **措辞纪律**：离线 `snpe-net-run` 跑 `.dlc` 的结论只能写"**在 SNPE CPU 参考实现上**"，**不得写"设备实测"**。

---

### 卡 F｜阶段 9–10：交付 + 提速

> 🔴 **卡 E 走完只代表"设备上用 `qnn-net-run` 能跑出正确数值"，不代表能交付。**
> 中间还隔着 app 集成（挂载点清单、契约同步、SELinux、存档指纹）—— 见 **§四十八**。

**交付四条铁律**（本项目一夜踩全，代价约 2 小时 + 给出过一个错误结论）：

| # | 铁律 | 本项目怎么违反的 |
|---|---|---|
| 1 | **先证明现状可用，再改一个变量** | 把"用户说之前能出图"当事实 —— 那是 3 天前、另一个 APK |
| 2 | **覆盖任何东西前先备份原件** | APK 直接 `install -r` 覆盖，丢掉唯一能回答"是不是我弄坏的"的证据 |
| 3 | **交付后自己跑一次真实链路** | 做了 sha256/签名全套静态校验 —— 那验的是"文件对不对"，**不是"跑不跑得起来"** |
| 4 | **验证必须在真实运行环境** | 用 `run-as` 做对照，其 SELinux 域是 `runas_app`、真实 app 是 `untrusted_app`，DSP 权限不同 ⇒ **整套对照无效** |

| 门 | 判据 |
|---|---|
| **F1 生效证据** 🔴 | **当预期结果就是"没有变化"时**（逐字节不变的优化），主指标无法区分"生效且无害"与"根本没生效"。**必须有独立于主指标的生效证据**（装载日志里的文件字节数 / sha256 / 图名）。**先问：如果这步完全没生效，我的判据会不会照样通过？** |
| **F2 指纹层级** | 判据必须是 **sha256 且用对应层的**：改 native 比 `.so`、改上层比**整包**。实测两个内容不同的 APK **字节数完全相同**；只改 Kotlin 时 `.so` 指纹**完全相同** |
| **F3 单变量对照** | 对照臂必须与实验臂**只差一个变量**；多个优化叠加后仍用最早的基线 ⇒ 差了**两个**变量 |
| **F4 埋点包住** | 埋点写完先问：**这对标记之间，是不是正好夹着我要改的那段？** 夹不住就挪标记，不是挪结论 |
| **F5 名字一致** | 设备路径**一律以契约里的 `context_binary` 为准**，不以产物文件名为准；换文件必须同改契约的 `sha256`/`size_bytes` 并**断言命中处数** |

**提速顺序（按本项目实测的性价比，不是按难度）**：
1. 🔴 **第一张表从"用户点下去"量到"图出现"** —— 本项目三轮都在优化内部指标，最后才发现生成开始前还有 **148~210 s**，预算表里连一行都没有
2. **装载改 mmap**（§47.8）：只改宿主侧几十行，**速度 −10.9% + MemAvail +616 MiB**，不碰数值 ⇒ **性价比最高，第一个做**
3. **完整性校验换硬件 SHA-256**（§47.9）：本项目**单条收益最大**（129 s → 12 s）
4. 把 `loadGraph` 提出步循环 / 四段常驻（−26.9% / −21.2%）
5. 再拆推理内部、按算量归一找离群段（§45）
6. 🔴 **一个瓶颈解除后，下一个瓶颈往往换人** —— 改完必须在真实路径重新量（本项目 benchmark 15.5× ⇒ 端到端只有 6.6×，因为从计算受限变成 I/O 受限）

**已实测无空间、别再试**（本项目 SM8750）：HVX 线程（已满）、VTCM 调大（上限 8）、功耗档（已最高）、
宿主侧重量化+拷贝（只占段墙钟 **1~5%**）、并行装载（官方文档：同一 device handle 非线程安全）、
`O:3`（段级 −9% 但撞 PD）。

---

## 47.2 坑清单（按代价排序；每条都真实发生过）

> **怎么用这张表**：它按**代价**排序，便于"先防最贵的"；但执行时你需要的是"我这一步该防什么"。
> 下面这张索引把它挂到 §47.1.1 的卡片上。**执行每张卡片前，先扫一眼对应格子。**

| 卡片 | 该阶段要防的（本表关键词） |
|---|---|
| **A** 准备/导出 | 导出残留算子（part1a 29.1% cycles）、烘焙常量、烘焙序列长度 |
| **B** 手术/校准 | 图级手术"语义等价"就重建、校准未覆盖分支 |
| **C** 转换/量化 | overrides 静默失效（N=0）、漏 `--float_bitwidth 32`、改 encoding 前没算代价、**`perform_layout_transformation` 等转换器行为没看** |
| **D** 建 context | 漏 `--config_file` ⇒ v68/4MB、`graph_names` 填错整段落空 |
| **E** 接口/验证 | `qnn-net-run` 静默砍半且骗过守卫、相对 L2 被离群值主导、**拿 `.bin` 的 md5 当可复现性判据**（内嵌构建时间戳） |
| **F** 交付/提速 | 预期"没有变化"时无生效证据、用字节数/错层指纹判断产物、叠加归因、埋点没包住、写死的清单没改、共享 spill-fill 组大小写死、**终态判据把本次交付物判成"不干净"**、**A/B 期间改开关而被测段未跑完**、**拿不同层的指标互证**（cycles vs 墙钟）、**为验一行 C++ 跑完整 NDK 构建** |
| **全程** | 手搓方案替代成熟工具、用分页结果论证"不存在"、根因没查清就换路 ⇒ 见 **§47.10** |

| 代价 | 坑 | 一句话防法 | 出处 |
|---|---|---|---|
| **一整轮结论作废** | 建 context 漏 `--config_file` ⇒ 静默编成 v68/4MB，数值差 15 倍 | 建完**读回**元数据，不看文件名 | §十二、#95 |
| **一整轮结论作废** | 用相对 L2 描述被离群值主导的张量（低估 48 倍） | 同时报"前 1% 元素占 ‖a‖² 比例"，>50% 就改用主体口径 | 约束 7 |
| **数小时 + 误判方向** | `qnn-net-run` 默认按 **float32** 解析输入；喂原生字节会被**静默砍半**，而输出字节数恰好等于契约的原生字节数 ⇒ 字节守卫被骗过 | 显式 `--use_native_input_files --use_native_output_files`；**逐个**输出核字节数；用 `Bool_8`/`Int_32` 张量当模式指示器 | §45.8、`EXECUTION_MODEL` 规则 2·补 2 |
| **数小时** | 图级手术"语义等价"就直接重建 2 GB context | 先查后端算子文档 + 先用最小探针复现，再重建（G-backend 门） | §47.3 |
| **数小时** | 配置里 `graph_names` 填错 ⇒ graphs 段**整段静默落空**（VTCM 退回 4MB） | 图名 = converter 产出的 fp32 DLC 文件名主干；**读回元数据确认** | §15.28.5、§45.5 |
| **数小时** | 把 `.dlc` / `.bin` 的 **md5** 当可复现性判据（里面嵌构建时间戳、图名字符串） | 比**可执行载荷区**，看差异落在哪个区段 | §45.8 |
| **80 分钟 + 占用设备** | 改 encoding 前没算"这个改动本身的表示误差" | 代价门 G0-cost：先在宿主做 quantize-dequantize 模拟 | 约束 4·补 |
| **一次交付翻车** | 共享 spill-fill 的**组大小写死** ⇒ 换配置后装载失败 | 组大小从交付产物**现读**（所有成员所有图取最大） | §38.3、§15.45.3 |
| **一次交付翻车** | 改图名/段数后，某处**写死的清单**没跟着改 | 改名前 `grep -rn "<旧名>"` 全库逐个确认；同一清单出现多处就改成从契约派生 | 约束 11·补 |
| **返工** | `perform_layout_transformation` 等转换器行为没看 | 转换/量化命令**内嵌在量化 DLC 里**，`snpe-dlc-info` 一读就有，动手前先读 | §47.3 |
| **数十分钟** | Windows 上 `du` 统计大盘 >15 分钟；Gradle 树有 >260 字符路径 | 用 `os.scandir` 递归（2 秒）；长路径加 `\\?\` 前缀 | §45.9 |
| **方向错判** | 用带 `head`/分页的搜索结果去论证"**不存在**"——本项目据此断言"app 没有导入模型的入口"，实际有，只是被 `head -8` 截掉了 | **证否必须用全量**：去掉 `head`，或改用 `-c` 计数。分页结果只能证明"存在"，永远不能证明"不存在" | §47.10 模式 3 |
| **一轮重来 + 数据全废** | 把「从设备取」和「打包」耦合成一条流式管道，为省一次磁盘写 ⇒ 写 zip 的参数错一个（单成员 >2 GiB 需 `force_zip64=True`，`allowZip64` 只管整包），**已传好的 4.34 GB 一起废掉** | **拷贝 / 压缩 / 推送分成三步**，各自可单独重跑，本地副本留着下次复用；用 `ZipFile.write`（知道大小、自动开 ZIP64）而不是手搓流 | §47.10 模式 2 |
| **反复失败且改错地方** | app 私有目录导不出来：`run-as` 写 `/data/local/tmp` 报 `Permission denied`，先判成"路不通"换方案、再判成"权限位问题"加 `chmod 777`，**两次都不对** —— 真因是 **SELinux**（`untrusted_app` 不得写 `shell_data_file` 标签） | 导出 app 私有目录用 `adb exec-out run-as <pkg> cat <file>` **逐个文件**拉（单文件 2.9 GB 实测可行，约 26~30 MB/s）；`adb pull` 与任何中转到 `/data/local/tmp` 的做法都会被 SELinux 挡 | §47.10 模式 1 |
| **交付"成功"但其实没生效** 🔴 | **当预期结果就是"没有变化"时（逐字节不变的优化），主指标无法区分「改动生效且无害」与「改动根本没生效」** ——推送失败、文件名写错、app 用了旧文件，都会安静地给出一模一样的图和速度，然后被读成"收益未复现" | **必须有一个独立于主指标的"生效证据"**。本项目用装载日志里的**文件字节数**（新件 2,877,374,464 vs 旧件 3,151,335,424）做断言；没有就用别的能区分新旧的量（sha256、图名、opDataSize）。**先问：如果这步完全没生效，我的判据会不会照样通过？** | §47.1 阶段 9 |
| **误判"改动没进产物"/ 丢存档** | 用**文件字节数**或**错层的指纹**判断产物有没有变。实测：只改上层代码（Kotlin）时 native `.so` 指纹**完全相同**；更巧的是那两个 APK 的**字节数也完全相同**（94,339,337）而内容不同 | **判据必须是 sha256，且要用对应层的**：改 native 比 `.so` sha256，改上层比**整包** sha256；**存档命名也用整包 sha256**——按 `.so` 命名会让只改上层的版本被当成"已有存档"跳过而丢失 | §47.1 阶段 9 |
| **把 A 的功劳记到 B 头上** | 多个优化**叠加**后，速度对照仍用最早那个基线 ⇒ 实验臂与对照臂差了**两个**变量 | 对照臂必须与实验臂**只差一个变量**；按实验臂实际走的配置**自动选同配置的那一臂**，找不到就停（本项目：A 臂开着 mmap 跑，就必须跟 mmap 臂比，不能跟 read 臂比） | §47.4 |
| **交付成功后被指示删掉交付物** | 终态自检把"某 marker 还在"无条件判成「设备不干净」——而那个 marker 正是本次交付的形态 | **"干净"是相对于本次交付意图的状态**：写终态判据前先问"这次交付成功后，设备**应该**是什么样"，再按那个来判 | §47.1 阶段 9 |
| **一轮 A/B 作废** | A/B 期间改了开关状态（marker、配置），而**被测的那一段还没执行完** ⇒ 同一次运行里混用两条路径，数据不是单变量 | 切换开关前**读产物确认上一阶段已完成**（本项目：`[loadpath]` 行数是否已满 8），不要用"按时间推算应该装完了"代替；并在脚本里加一条**路径唯一性自证**（`len(set(paths)) != 1` 即作废） | §47.4、约束 2 |
| **结论互相污染** | 拿**不同层**的两个指标互相"验证"或"推翻"：HTP cycles 层的「搬运算子占 41.2%」与 app 墙钟层的「宿主侧占 0.9%」量的根本不是一回事 | 每个数先问**它量的是哪一层**（cycles / 段墙钟 / 整张墙钟）；不同层只能**并列**，不能互证。写结论时把层写进句子里 | §47.4 |
| **一次插线白费** | A/B 的**埋点位置**没有包住被优化的那段代码：把"开始装载"的日志打在整文件读入**之后**，于是 `开始→结束` 区间恰好把要省的部分排除在外，两臂必然量出一样的数 | 埋点写完先问一句：**这对标记之间，是不是正好夹着我要改的那段？** 夹不住就挪标记，不是挪结论 | §47.4、约束 9 自审第 1 条 |
| **一次插线白费 + 看起来"成功"** | 覆盖设备上的文件时用了**建图产物的名字**，而交付脚本早把它改过名 ⇒ 写到一个没人读的路径，app 继续用旧文件 | 设备上的路径**一律以契约里的 `context_binary` 为准**，不以产物文件名为准；推完 `stat -c %s` 读回核字节数 | §47.1 阶段 9 |
| **app 直接拒绝装载** | 换了某个 `.bin` 却没同步改契约里的 `sha256`/`size_bytes`（契约的哈希是**执行时完整性检查**，不是溯源信息） | 换文件必须同改契约，且**断言命中处数**（本项目多图文件在契约里出现 5 处：基准 + 四比例）；改完读回 | §47.1 阶段 6/9 |
| **可能无法归因** | 磁盘清理把**交付原件**删了，等到要覆盖设备上的文件时才发现没有回滚源 | 覆盖设备上任何文件前，先用**设备实测 sha256** 认领宿主那份原件；认不上就先把设备那份 `adb exec-out run-as … cat` 拉回来再动手 | 约束 11 第 2 条、§47.1 阶段 9 |
| **几十分钟 / 压死宿主** | 为验证一行 C++ 改动去跑完整 NDK 构建，而宿主正被量化/建图占着（本项目建图峰值 **17.8 GB / 23.7 GB**） | 用 `.cxx/**/compile_commands.json` 里该 TU 的原命令起**一个** clang 做 `-fsyntax-only`（30 秒、几百 MB）；⚠️ 那里的 `command` 是**已转义**的，必须还原 `\"` 再拆词，否则报一堆假的 SDK 头文件语法错 | §47.4 |

## 47.3 🔴 图级手术的硬规则（本项目最贵的一课）

**结论（① 实测）**：在 ONNX/FP32 语义下**严格等价**的手术，在 HTP 上**可能毁掉结果**；
更进一步：**同一张图里两处结构完全相同的实例，结论可以相反**。

本项目实例：part1a 里有两个 batch=1 `pad_sequence` 导出的全尺寸 `ScatterElements`（索引全 0、沿大小为 1 的轴 ⇒ 数学恒等）：

> ⊕ **【2026-09-20 实测补正】"两个"指的是**两个大的**（各约 16.2 M 元素，即下表两列）。
> 用 `scripts/scan_export_residue.py` 全图扫描实得：part1a 共 **6 个**恒等 `ScatterElements`
> + **6 个** `pads` 全 0 的 `Pad`。另外 4 个 Scatter 较小（540 K / 530 K / 307 K / 10 K 元素），
> **从未被评估过**（台账 #184）。数量差异不影响下面的结论，但换模型时**别只扫最大的那两个**。

| | 删图像流那个 | 删 unified 流那个 |
|---|---|---|
| ONNX FP32 | 与原图**逐位相同** | 与原图**逐位相同** |
| 同形状小图探针（HTP） | 删掉**无影响** | —— |
| 完整图（HTP） | **9 个输出逐字节相同**，该段 **−14%** | **主体误差 125%，毁图** |
| 建图元数据 | `spillFill` 不变、`opData` −0.3 MiB | `spillFill` +9.9 MB、`opData` **+58 MiB** |

⇒ **规则**：
1. **任何"删除/替换算子后数值不变"的断言，动手前必须过两道门**：
   ① 查后端算子定义（`docs/QAIRT-Docs/QNN/OpDef/HtpOpDefSupplement.html` + `MasterOpDef.html`）；
   ② **最小探针**先复现现象，再筛变体。两门都过才允许写成②推论、才允许重建完整 context。
2. **每一处手术单独上设备验数值**，不得由"同型结构"外推。
3. ②**建图元数据可作预警**：手术后 `spillFillBufferSize` / `opDataSize` 出现扰动的那个变体，
   就是后端会崩的那个（本项目 n=3 全中，样本仍少，**只能当信号不能当判据**）。
4. 验数值用**逐字节**，且至少换**两组差异很大的输入**（本项目用真实 step-0 + 随机各一组）。

## 47.4 提速的标准动作（§45 的执行版）

0. 🔴 **第一张表要从"用户点下去"量到"图出现"**，不是从后端自报的 `generation_time` 拆
   —— 本项目就栽在这：三轮提速都在优化那个内部指标，最后才发现**生成开始之前还有 148~210 s**
   （完整性校验），**比生图本身还长且预算表里连一行都没有**。详见 **§47.9**。
1. **再拆推理内部**（零设备）：用后端单调时钟把单张图拆成「装载 / 文本编码 / 步循环 / VAE / 其余」。
   ⚠️ 装载耗时**不能只看 `Initializing → Initialized`**，标记之前还有整文件读入（本项目 12.2 s vs 8.0 s）。
2. **按算量归一找离群段**：从 ONNX 结构数 MatMul/Gemm 乘法次数，算「段墙钟 ÷ 算量」；
   用**多个输入尺寸**当天然样本（本项目 5 比例 × 4 段 = 20 点）。离群段一眼可见。
   🔴 只有一个高值点时**不要做回归**（共线变量会给出无物理意义的系数）。
3. **逐算子 profiling**（设备）：`--profiling_level detailed` + `--perf_profile burst`；
   代表性门：离线的段间耗时比必须落在 app 实测区间，不过就不下归因结论（本项目离线绝对值比 app 慢 5 倍）。
4. **把 cycles 分成"真算力"与"搬运/转换"**：本项目 part1b 有 **14.2%** 是布局搬运与类型转换，part1a 是 **41.2%**。
   这部分是**导出/量化配置的产物**，不是模型本身的算力需求 —— 提速的空间主要在这里。
   🔴 **这是 cycles 层的数，说的是 HTP 内部**。它**不能**回答"宿主侧的重量化和拷贝值不值得优化"——
   那是墙钟层的另一个问题，必须另外测（**两层不能互相验证，也不能互相推翻**，本项目在这上面绊过一次）。
5. **宿主侧到底占多少：app 内埋点拆墙钟**（① 实测，2026-09-19 本项目结果）。
   把每段墙钟拆成 `宿主准备 / 拷入 QNN 缓冲 / graphExecute / 拷出 / 落库` 五段
   （5 次 `steady_clock`，亚微秒开销，不碰数值；实现见 `QnnModel::ExecSplit` 与 `PipelineZImage::logSegTiming`）。

   | 段 | prep | 拷入 | **graphExecute** | 拷出 | 落库 | 宿主侧占比 |
   |---|---|---|---|---|---|---|
   | part1a | 14 | 3 | **45495** | 185 | 226 | **0.9%** |
   | part1b / 2a / 2b | ~690 | 35~128 | **24352~24624** | 4~206 | 5~218 | 3.0~4.8% |

   ⇒ ①**宿主侧（重量化 + 拷贝 + 落库）只有 1~5%**，即使优化到零，单张最多省 1.9%。
   **这条杠杆在本项目上是关闭的**——换模型时先花一次运行测出这张表，再决定要不要投入，
   不要凭"每步搬几十 MB，肯定很贵"的直觉去优化它（本项目正是这么假设的，实测否定）。
   ⊕ 有意思的是 `prep` 的差别：part1a 只有 14 ms，其余三段各约 690 ms ——
   差在**段间张量的 encoding 不同、需要 CPU 逐元素重量化**。这是分段切口的隐性成本，
   切口选在 encoding 一致处可以省掉它。
   🔴 **埋点也要先自证**：五段合计必须与既有的 `[segtime]` 段级墙钟相差 < 5%，否则是口径错（漏计/重复计），表作废（约束 7）。
   🔴 **必须在真实 app 里测**，不能用 `run-as` 或 `qnn-net-run` 替代（约束 11 第 4 条：换了执行环境就不是对照）。
6. **杠杆清单**（本项目实测结论，换模型要重测）：
   | 杠杆 | 本项目结果 |
   |---|---|
   | 把 `loadGraph` 提出步循环 | −26.9%（已交付） |
   | 四段常驻（不再每步装卸） | −21.2%（已交付） |
   | 多图共享权重 + 共享 spill-fill | 设备占用 35 → 12.3 GiB，速度持平（已交付） |
   | **删导出残留算子**（逐字节安全的那一处） | ✅ **已交付：单张 −12.6 s（−8.8%）**，出图逐字节不变。①实测是按 cycles 占比推算（−6.3 s）的**两倍** ⇒ 该类算子的真实墙钟成本高于线性外推 |
   | **装载改 mmap** | ✅ **已交付：单张 −17.5 s（−10.9%）**，八段装载 29.17 → 14.55 s（−50%），**且 MemAvail 最低点 +616 MiB**（文件页可回收，不占匿名内存）⇒ 速度与内存双收益，**本项目性价比最高的一条** |
   | HVX 线程数 | 🔴 已用满（6 = SoC 上限），无杠杆 |
   | VTCM 调大 | 🔴 无空间（写 64 被钳到 8） |
   | `O:3` | 🔴 段级 −9% 但 part1a 撞 PD、ION 涨 ⇒ 当前配置不可交付 |
   | 功耗档 | 🔴 app 已是最高档 |
   | **完整性校验换硬件 SHA-256** | 🆕 ②**−117 s**（129 → 12 s，①实测 15.5 倍加速）⇒ **单条收益最大，且不碰数值**（§47.9） |
   | **宿主侧重量化 + 拷贝** | 🔴 **只占段墙钟 1~5%**（①实测，见上表）⇒ 关闭，别凭直觉优化 |
   | 并行装载 context | 🔴 **官方文档否决**：依赖同一 device handle 的 API 调用**不是线程安全**（`api_usage_guidelines.html` Multi-Threading）⇒ 未写代码即关闭 |
   | 步数 | 产品决策关闭（本项目） |

## 47.5 已经写好、可直接复用的工具（换模型时先看这张表，别重写）

| 工具 | 作用 | 模型相关性 |
|---|---|---|
| `scripts/qairt_tool.py` | 跑任何 QAIRT python 工具（环境自足） | 无关，直接用 |
| `scripts/lab_dev.py` | 设备操作统一入口（把"掉线"与"进程结束"在类型层面分开） | 无关，直接用 |
| `scripts/check_ctx_identity.py` | 产物装置画像核对（dspArch/vtcm/图名） | 无关 |
| `scripts/code_lint.py` / `doc_audit.py` / `ledger_lint.py` / `ledger_provenance.py` | 强制流程的机器检查 | 无关 |
| `scripts/p2_zero_device_evidence.py` | 时间预算 + ms/G 归一 + 元数据表（零设备） | 改路径即可 |
| `scripts/p2_d1_profile.py` | 逐算子 profiling（preflight/collect/analyze 三段分开） | 改契约路径即可 |
| `scripts/scan_export_residue.py` | **门 A3-b**：扫"全尺寸但语义恒等/填充"的导出残留（`Scatter*`/`Pad`/`Expand`/`Tile`/单输入 `Concat`） | **无关，直接用** |
| `scripts/scan_baked_seqlen.py` | **门 A3-c**：扫烘焙的序列长度常量 + `value_info` 维度，打印消费者 | **无关，直接用** |
| `scripts/scan_requant_edges.py` | **门 B2**：算子内全部张量两两取 scale 比值（含 STATIC），float32 复算 | **无关，直接用** |
| `scripts/scan_where_clip.py` | **门 B3**：`Where`/`Select` 输出是否钳掉掩码分支；并为 B2 提供转储解析器 | **无关，直接用** |
| `scripts/check_quant_health.py` | **门 C5**：量化级/元素 + FP16 候选筛选（§29.2 三规则） | **无关，直接用** |
| `scripts/build_lineage.py` | **交付谱系重建**：设备 sha256 → 宿主 .bin → 建图配置 → DLC → 内嵌命令。**换模型第一天就该建，别等乱了再补** | **无关，改锚点文件即可** |
| ⊕ 上述六个工具**都自带 `--selftest`**（用已知样本验判据本身，约束 8）。**先跑 selftest 再用。** | | |
| `scripts/legacy_export/convert_qairt_zimage.py` | 2026-08-03 原始转换脚本**存档**。唯一价值是 `freeze_dynamic_shapes`（算法已提炼进 §3.1）。🔴 **别直接跑**（量化那段用的是空转的 per-channel） | 仅供对照 |
| `scripts/p2_h1_surgery.py` | 删算子手术 + 六条等价性断言 | 结构相关，但断言可照抄 |
| `scripts/p2_h1_probe.py` | 最小探针（建 5 个变体 + 设备对比） | 模板可复用 |
| `scripts/p2_h1_build.py` | 转换 → 量化 → 建 context 一条龙，**建完自动读回元数据与部署版逐项对比** | 改路径即可 |
| `scripts/p2_h1_deliver_build.py` | 把一个已验证的变体推到交付形态（多比例 + mg 多图 + spillFill vs 设备 marker 核对） | 模板可复用 |
| `scripts/p2_h1_ab.py` / `p2_h1_c2.py` | 设备 A/B（输出 sha256 在设备上算）+ 三方对照（float64 全量/主体/余弦/top1%） | 模板可复用 |
| `scripts/p2_s6_session.py` | **一次插线跑完多个实验 + 交付**：阶段化、判据事前锁定、任一门不过**逆序自动回滚** | 模板可复用 ⭐ |
| `scripts/timing_breakdown.py` | `--seg` 段级墙钟、`--split` 段内五段拆时 + 装载路径（含自证） | 改标记名即可 |
| `scripts/snpe_runner.py` / `dlc_contracts.py` | DLC 执行的唯一入口 + 机器可读契约 | ⚠️ **先改文件头**：各写死 `SDK` / `DLC_DIR` 等 2 / 4 处绝对路径（2026-09-20 实查） |
| `scripts/map_opid.py` | requant 极端边定位（门 B2） | 🔴 **不通用**：无 `--help`，输入路径与研究问题全写死，输出 GBK 乱码。**只能抄算法** |
| `scripts/oneshot_dryrun.py` / `p2_s6_dryrun.py` | **设备会话干跑台**：把 `lab_dev`/`subprocess`/出图全换成桩，整条阶段逻辑（含回滚链）真跑一遍，并写出格式真实的假 logcat 端到端测解析 | 模板可复用 ⭐ |

## 47.6 迁移到新模型的第一天清单

> **本章的读法（三层，别一上来就读细节）**：
> ① **§47.10 五种思维模式** —— 为什么会返工，比坑清单更值得先读
> ② **§47.1 阶段表 + §47.1.1 六张执行卡片** —— 照着跑
> ③ **§47.2 坑清单（按卡片索引）/ §47.3 / §47.7** —— 执行到哪一步查哪一格
> ⊕ 其余四十六章是**证据库**，被卡片指到时才翻，**不要通读**。

0. 🔴 **先读 §47.10（五种反复出现的思维模式）**。§47.2 的坑清单一条只防一件事，
   换模型后新坑会不一样；而那五种模式**每一种本项目都重犯过至少两次**，它们不会变。
1. 读 `CLAUDE.md`（11 条约束）与本章；**不要**通读其余章节。
2. 确认 47.0 的四个维度，尤其**重测 PD 红线与 VTCM 上限**（一条命令级别的探针即可）。
3. 导出后立刻做三件事：查烘焙常量、查 `ScatterElements`/`Pad`/`Expand` 这类导出残留、数 MatMul 算量。
4. 第一次量化**必须**留下 overrides JSON（后续重量化从 3 小时降到 0.1 分钟）。
5. 建第一个 context 后**立刻读回元数据**，确认 `dspArch` / `vtcmSize` / 图名三项，再往下走。
6. 第一次上设备**先做代表性门**：离线单段与 app 内同段的耗时比要对得上，再开始归因。

## 47.7 如果新模型要**自己做导出**：下游会反过来对导出提的硬性要求

> 本项目的 ONNX 是别人给的（§二·补），所以 §47.1 阶段 1 只能写"未亲自做"。
> 但后面九个阶段踩的坑里，**有相当一部分的根在导出**。下面每条都是"我们在下游被它咬过"，
> 换模型时**在导出阶段就满足，代价是分钟级；漏到量化之后再补，代价是小时到天级**。

| # | 下游的硬性要求 | 不满足会在哪一步咬人（本项目实例） |
|---|---|---|
| 1 | **所有形状固定**；每个比例/序列长度导一套图 | 动态轴在转换期就会失败或被固化成错误值；本项目五比例 = 五套图 |
| 2 | **不要留 `pad_sequence`/`Pad`/`Expand`/`ScatterElements` 这类"batch=1 下数学恒等"的残留** | ①它们会变成**全尺寸**算子：本项目两处占 part1a **29.1%** cycles；事后删除是图级手术，**其中一处删了就毁图**（§47.3） |
| 3 | **分段切口按 PD 红线反推**（本项目 3.65 GB/context），切在张量少、字节小的地方 | 单段 3535 MB 的 part2 **装不进** unsigned PD；补救是事后再拆（切口 6 张量 / 65.6 MB） |
| 4 | **导出时烘焙进图的常量必须逐个登记**（缩放因子、mask 极性、归一化常数） | ①VAE 图里已含 `Div(0.3611)`，宿主又除一次 ⇒ 不只解码错，输入还超出校准量程 **2.77 倍被硬钳**（#80/#81） |
| 5 | **IO 张量名稳定、可与契约对上**；图名 = converter 产出的 fp32 DLC **文件名主干** | 名字一变，`quantization_overrides`（按 ONNX 张量名）与建图配置的 `graph_names` 会**静默落空** |
| 6 | **要走 FP16 白名单的算子，导出时就要能按名字选中** | A16W8 下必须对少数算子保精度 + 显式 `Clip(±65504)`（§二十九、#111） |
| 7 | **导出后立刻留一份 FP32 参考**：真实 step-0 的**输入与输出**全张量落盘 | 后面每一次量化/手术/建图的对拍都要它；没有它就只能"和自己比"，而"和自己比"验不出系统性错误（§44.4） |
| 8 | **导出后立刻数算量**（MatMul/Gemm 乘法次数，按段、按比例） | 提速阶段的第一把尺就是「段墙钟 ÷ 算量」；没有它就只能看绝对耗时，看不出哪段异常（§47.4） |

🔴 **第 2 条是本项目最贵的一条**，值得在导出后立刻花 10 分钟做一次结构体检：
按算子类型统计节点数与**输出张量元素数**，把"输出是全尺寸、但语义上是恒等/填充"的算子全部列出来，
在**导出阶段**改掉。等到量化、建图、上设备之后再动它，就要付 §47.3 那套代价（而且可能删不掉）。

## 47.8 【新模型必做】装载改 mmap —— 本项目性价比最高的一条

① 实测（SM8750，2026-09-19，同一个 APK 用 marker 文件切两条路径，严格单变量）：

| | read-into-vector（默认） | **mmap** |
|---|---|---|
| 八段 context 装载合计 | 29.17 s | **14.55 s（−50%）** |
| 单张 generation | 160.5 s | **143.0 s（−10.9%）** |
| ION 峰值 | 8697 | 8707（+10，可忽略） |
| **MemAvailable 最低点** | 511 MiB | **1127 MiB（+616）** |
| 出图 sha256 | — | **逐字节不变** |

**为什么值得第一个做**：不碰任何数值、不碰量化、不碰图结构，**只改宿主侧几十行代码**，
却同时拿到速度与内存两份收益。本项目直到第三轮提速才想起查它，之前一直以为
"`createFromBinary` 已经在用 mmap" —— 那句话只对 `createAndInitModel` 路径成立（见 §45）。

**怎么查自己的模型有没有这个坑**：
1. 看装载路径：把整个 `.bin` 读进 `std::vector` 再 `createFromBuffer`，就是有坑
   （先给 2~3 GB 匿名内存清零，再整份拷贝一遍，然后才交给 QNN）。
2. 改法：`open` + `mmap(PROT_READ, MAP_PRIVATE)` + `madvise(SEQUENTIAL|WILLNEED)`，
   指针直接交给 `createFromBuffer`。QNN 在 `contextCreateFromBinary` 返回后不再引用该缓冲，
   所以映射只需活到初始化结束（用 RAII 守卫 `munmap`+`close`）。
3. **用 marker 文件做开关，同一个 APK 跑两臂** —— 这是保证严格单变量的最省事办法
   （换 APK 做 A/B 会引入编译差异，不是单变量）。

### 47.9.4 ① 改完之后的实测：瓶颈会换人，要重新量一次

本项目改完硬件加速后**又量了一次**，拆解如下（把 `logcat -c` 放在 `am start` **之前**才捞得到；
放在之后的话 `initialize()` 落在录制窗口外，本项目为此白跑两次）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| app 启动 → 开始校验 | 2.2 s | 9% |
| **完整性校验（12.04 GiB）** | **19.4 s** | **79%** |
| 校验完 → 接受请求 | 2.9 s | 12% |

🔴 **实际加速比 6.6 倍，不是 benchmark 的 15.5 倍**：
改进前是**计算受限**（95.7 MiB/s ⇒ 129 s），改进后变成 **I/O 受限**（636 MiB/s 冷读，
低于 benchmark 热读的 1022 MiB/s）。⇒ **继续优化哈希算法已经没有意义**。
要再省只能"不重复算"（缓存 (路径, 字节数, mtime, sha) ，命中即跳过）——
但那等于把信任从**内容**降级到**字节数 + mtime**，动之前先想清楚威胁模型。

⇒ 通则：**一个瓶颈被解除后，下一个瓶颈往往换了角色**。
拿 benchmark 的加速比去推端到端收益会高估；改完必须在真实路径上重新量一次。

🔴 **两个必须同时量的**：
- **装载耗时的口径要包住"读文件"本身**：把"开始装载"的日志打在读文件**之前**，
  否则区间恰好把要省的那段排除在外（本项目第一版就是这么写的，见 §47.2）。
- **内存要看 `MemAvailable` 最低点，不只看 ION**：mmap 的收益主要体现在匿名内存减少上，
  只盯 ION 会看不见（本项目 ION 只差 +10 MiB，MemAvail 却差 616 MiB）。

## 47.9 【新模型必做】别只优化"推理耗时"——先量一遍**用户实际等待的时间**

本项目最迟被发现的一块，也是最大的一块之一。

### 47.9.1 现象

我们三轮提速都在优化 `generation_time`（后端自报的生成耗时），把它从 160.5 s 压到 130.4 s。
直到 2026-09-19 才发现：**用户从点开到看见图要等约 280 s**，其中
**148~210 s 花在生成开始之前**——模型文件的**完整性校验**（sha256 过 12.04 GiB）。

⇒ ①**校验比生图本身还长**，而它**从来没进过我们的时间预算表**，
因为预算表是按后端 SSE 里的 `generation_time_ms` 拆的，那个字段根本不包含启动。

### 47.9.2 根因与修法（① 设备实测）

| | 吞吐 | 12.04 GiB 耗时 |
|---|---|---|
| 纯软件 SHA-256，且 `update()` **逐字节循环** | 95.7 MiB/s | **129 s** |
| **ARMv8 `sha2` 指令**（Crypto Extension） | **1486.3 MiB/s（15.5 倍）** | 8.3 s |
| 端到端（含读文件，4 MiB 缓冲） | 1022 MiB/s | **约 12 s** |

**做法**：
1. 先查 `/proc/cpuinfo` 的 `Features` 有没有 `sha2`（本项目设备还有 `sha512 sha3 aes`）。
2. 硬件函数用 `__attribute__((target("+crypto")))` **单独标记**，其余代码仍按默认 `-march` 编译
   ⇒ 不会在不支持的设备上因指令集而崩。
3. 运行时 `getauxval(AT_HWCAP) & HWCAP_SHA2` 检测。
4. 🔴 **启动自检 + 自动回落**：用已知向量让硬件版与软件版对拍（含**分片边界**），
   **逐位相同才启用硬件版**。哈希算错 = 完整性校验全失败 = app 起不来，这条不许赌。
5. 读缓冲别用 64 KiB —— 实测要 **4 MiB** 才吃得满 I/O。

**验证路径（重要）**：别直接改 app 再测。先把实现编成一个**几十 KB 的独立可执行文件**
push 到设备（本项目 `tools/sha_bench.cpp` + `scripts/build_sha_bench.py`），
单独确认「算得对不对、快多少」，**通过了再集成**。改 app 要走构建+装包+起后端，
一轮十几分钟，而且写错了 app 直接起不来。

**正确性要三重闭合**：硬件版 == 自家软件版（多组向量含分片边界）== **系统 `sha256sum`**
（随机数据逐位一致）。第三重最容易被跳过，但它是唯一能排除"两个实现错得一样"的。

### 47.9.3 通则：**优化目标必须对齐用户实际等待的时间**

这条比上面的技术细节更值钱：

> 我们优化的是一个**内部指标**（`generation_time_ms`），而用户等的是**墙钟**。
> 两者差了 148~210 s，而这段差值在预算表里**连一行都没有**——
> 因为那张表本身就是从内部指标拆出来的。

⇒ 新模型开工时，**第一张时间表要从"用户点下去"开始量到"图出现"**，
把启动、校验、装载、推理、后处理全列进去，再决定优化谁。
别一上来就拆推理内部——那是在一个可能只占一半的盘子里做精细优化。

⊕ 本项目的顺序刚好反了：先做了最难的（图手术，−12.6 s），
最后才发现最大的一块（校验，−117 s）**只需要换个哈希实现**。

## 47.10 五种反复出现的思维模式 —— 比坑清单更值得先读

§47.2 那张表是**具体的坑**，一条只防一件事。但把本项目所有返工按根因归类，
只有五种模式，而且**每一种都重犯过至少两次**。换模型时新坑会不一样，**这五种模式不会变**。

### 模式 1：没弄清根因就行动

| 本项目案例 | 代价 |
|---|---|
| 打包时 `run-as` 报 `Permission denied`，判断成"这条路不通"就换方案 —— 真因是 **SELinux**（`untrusted_app` 不得写 `shell_data_file`），换路后卡死两次；回头再修时又判断成"权限位问题"加 `chmod 777`，**仍然失败** | 4 次返工，约 40 分钟 |
| 由"有效位宽 11.34 bit"推出"HTP 在跑 FP16 relaxed precision"，没查文档 | 假设当场作废（查文档 2 分钟） |

🔴 **识别信号**：你正在说"这条路不通，换一条"，但**说不出它为什么不通**。
⇒ **对策**：报错先问三句——它指向的对象是我验证过的那个吗？根因是什么？
这个根因是能解的还是真的无解？**没回答完不许换路**。
换路会把一个已知的小问题，换成一堆未知的大问题。

### 模式 2：手搓方案替代成熟工具

| 本项目案例 | 代价 |
|---|---|
| 不用 `adb pull`，手搓 `exec-out tar` 一次流 13 GB ⇒ 两次卡死在同一字节数 | 20 分钟 + 两次全量重传 |
| 不用 `ZipFile.write`（自动处理 ZIP64），改手搓流式写 ⇒ `File size unexpectedly exceeded ZIP64 limit`，**已传好的 4.34 GB 一起废掉** | 一轮重来 |

🔴 **识别信号**：为了省一点开销（一次磁盘写、一次传输），把两个独立步骤耦合成一条管道。
⇒ **对策**：**先问这件事有没有现成工具**。耦合的收益通常是几分钟，代价是"任一环节失败全部重来"
且没有中间产物可查。**三步分开做，每步能单独重跑**，几乎总是更快。

### 模式 3：证据没看全就下结论

| 本项目案例 | 代价 |
|---|---|
| `grep ... \| head -8` 的结果里没有 `ModelListScreen.kt`，就断言"app 没有导入入口" —— **实际有**，被 `head` 截掉了 | 方向错判，用户当场指出 |
| 改图名前 grep 过，结果里就有 `main.cpp`，但只看了自己**预期**会改的两处 | `Unknown Z-Image graph`，一次交付翻车 |
| 早已读到 `perform_layout_transformation=True` 却没连起来 | 图手术失败，数小时 |

🔴 **识别信号**：用带 `head`/`tail`/分页的结果，去论证"**不存在**"。
⇒ **对策**：**证否必须用全量**（去掉 `head`，或用 `-c` 计数）。
证据在手却没连起来，与不查文档是同一类失误。

### 模式 4：判据本身没被验证

| 本项目案例 | 会导致什么 |
|---|---|
| G4「touch 一个文件后应全量重算」——实现是**逐文件**失效，按原判据会把正确行为判成失败 | 错误回滚（执行前发现） |
| 用 **APK 字节数**判断"改动进了产物"——实测两个不同内容的 APK 字节数**完全相同** | 反向误判 |
| 终态自检把"marker 还在"判成不干净——那个 marker **正是本次交付物** | 交付成功后被指示删掉交付物 |
| 相对 L2 跨张量可比 / 误差沿 Id 单调 / 对称量化 ⇔ `offset == -128` | 多条结论作废 |

🔴 **识别信号**：判据里出现**具体数值、顺序、粒度、某个编码**，而你没拿已知样本试过它。
⇒ **对策**（约束 8 的延伸）：判据定稿前问一句——
**如果被测对象完全正常，这条判据会不会误报？如果它完全没生效，会不会照样通过？**
两个方向都要问。后者是 §47.2 那条"预期结果是没有变化时"的来源。

### 模式 5：优化内部指标，而不是用户实际等待的时间

| 本项目案例 | 代价 |
|---|---|
| 三轮提速都在压 `generation_time_ms`（160→130 s），而用户等的是墙钟 —— **生成开始之前还有 148~210 s**（完整性校验），**预算表里连一行都没有** | 最大的一块最迟才被发现 |

🔴 **识别信号**：你的时间预算表是从**某个内部字段**拆出来的。
⇒ **对策**：第一张表必须从"用户点下去"量到"结果出现"（§47.9）。
⊕ 推论：**瓶颈解除后要重新量**——本项目 SHA-256 换硬件加速后，
benchmark 显示 15.5 倍，端到端只有 6.6 倍，因为瓶颈从计算换成了 I/O。

---

🔴 **这五条的共同点**：都不是"不会做"，而是**在该停下来确认的地方没停**。
代价分布很不均匀——单次几分钟到几小时，但**每一次都可以用几十秒的确认避免**。

---

# 四十八、【接进 app】从 `.bin` 到手机上能跑

> **为什么单独成章**（2026-09-20 补）：§47 的卡片走到卡 E 就能在设备上用 `qnn-net-run`
> 跑出正确数值 —— 但那**不是**交付。中间还隔着 app 集成，而**本项目踩坑最密的一层正是这里**：
> `Unknown Z-Image graph`、`manifest input byte mismatch`、校验通配符漏掉文本编码器、
> 终态判据把交付物判成"不干净" —— 全在这一层。
> 此前这些知识散在 §33.7 / §35.4 / §42.x / 约束 11 里，**没有按"接新模型"的顺序串过**。
>
> **证据类型**：本章的挂载点清单是 2026-09-20 **实查当前代码**得到的（①），
> 不是从记忆或历史文档转述。行号以当时的工作副本为准，改动后请重新 grep。

## 48.1 先认清三层：谁是模型专属的，谁不是

```
Kotlin UI / 服务        ← 模型列表、尺寸选项、提示词框
    ↓ SSE
C++ 后端 main.cpp       ← 按模型类型分发到某个 Pipeline
    ↓
Pipeline<模型>.hpp      ← 🔴 模型专属：噪声、调度器循环、分段调用顺序、VAE 前后处理
    ↓
QnnModel.hpp            ← 通用：装载 context、执行图、拷贝张量（不必改）
    ↓
<模型>QnnContract.hpp    ← 🔴 模型专属：契约解析 + 必需图清单 + 切口闭合校验
```

⇒ **接新模型 = 新写 `Pipeline<X>.hpp` + `<X>QnnContract.hpp` + 一组常量，其余复用。**

## 48.2 挂载点清单（①实查，2026-09-20）

**C++（`app/src/main/cpp/src/`）**：

| 文件 | 行数 | 是否模型专属 | 接新模型要做什么 |
|---|---|---|---|
| `Config.hpp` | — | 🔴 常量集中地 | **改这里**：`latent_channels`、`canvas_size`、`sizes[]`（支持的比例）、`text_max_length`、`text_template_tokens`、`text_embedding_size`、`turbo_steps`、`vae_scaling_factor`、`vae_shift_factor`。<br>🟢 **这是个好设计，照抄**：模型常量**集中在一处**，不要散到各 `.hpp` |
| `PipelineZImage.hpp` | 675 | 🔴 全专属 | 新写一份。职责：初始 latents 生成 → 调度器循环 → 逐段 `graphExecute` → Euler 步进 → VAE 解码 |
| `ZImageQnnContract.hpp` | 583 | 🔴 全专属 | 新写一份。职责：解析契约 JSON、**必需图清单**、切口闭合校验、多比例变体 |
| `ZImageFlowMatchScheduler.hpp` | — | 🔴 专属 | 按你的调度器写（本项目 FlowMatch Euler） |
| `ZImagePrompt.hpp` | — | 🔴 专属 | chat 模板 / 特殊 token |
| `TextEncoder.hpp` | 629 | 🟡 半专属 | 本项目文本编码器分 4 段；段数变了要改 |
| `main.cpp` | 854 | 🟡 分发 | 加一个分支；**必需文件清单已从契约派生**（见 48.3 规则 1） |
| `QnnModel.hpp` | 1246 | 🟢 通用 | 不改。含 `ExecSplit`（段内五段拆时埋点，§47.4） |
| `Sha256Hw.hpp` | — | 🟢 通用 | 不改。ARMv8 硬件 SHA-256（§47.9），**直接复用** |

**Kotlin（`app/src/main/java/.../`）**：`data/Model.kt`（模型定义）、`data/ZImageSizes.kt`（比例表）、
`ui/screens/*`（模型列表、运行界面、提示词框）。

**宿主侧（生成契约与模型包）**：`scripts/gen_zimage_manifest.py`、`build_app_bundle.py`、
`build_aspect_contract.py`、`freeze_delivery.py`。

## 48.3 三条硬规则（每条都对应一次真实事故）

### 规则 1：清单一律从契约派生，**不要在代码里另写一份**

part2 拆成 part2a/part2b 后 app 报 `Unknown Z-Image graph: transformer_part2` ——
根因是 `main.cpp` 里一份**写死的图名清单**。🔴 **改动前 grep 过该名字，结果里就有这个文件**，
但只看了预期会改的两处（契约层、流水线层）。**证据在手却没连起来**（§47.10 模式 3）。

现在 `main.cpp` 已改成遍历 `contract.graphNames()`，注释里留着事故记录。

⚠️ **但本项目仍有一份写死的清单**：`ZImageQnnContract.hpp` 的 `final_graphs`（约第 217~226 行）
列着 8 个图名 + part2 拆分分支。它是**有意的必需图白名单**（用于判定交付是否残缺），
不是遗漏 —— 但**接新模型时它必须改**，而且它是"补一处清单只是把下一次事故推迟"的活标本。

⇒ **做法**：改任何图名/文件名/段数前，`grep -rn "<旧名>"` **全库**，**逐个文件确认**
（不是逐个"预期会改的文件"）。同一份清单出现多处 ⇒ 改成从单一数据源派生。

### 规则 2：契约的哈希是**执行时完整性检查**，不是溯源信息

换了任何 `.bin` 却没同步契约里的 `sha256` / `size_bytes` ⇒ **app 直接拒绝装载**。
而且改模型时要同步的**不只是哈希**：

| 也必须同步 | 不同步的后果 |
|---|---|
| `inputs/outputs` 的 `fixed_shape` / `exact_bytes` | 真机报 `manifest input byte mismatch: input_ids` |
| `text_seq_len` 一类的规格字段 | 陈旧值被当成事实 |
| **多比例交付：同一文件在契约里出现多处**（本项目 5 处：基准 + 四比例） | 只改一处 ⇒ 其余比例仍指向旧哈希 |

⇒ **改完必须断言命中处数**，并读回核对。

### 规则 3：校验的文件清单**从契约派生**，别用通配符

2026-09-17 交付第二次回滚的根因：设备上 `sha256sum *_ctx.SM8750.bin`，
而文本编码器的文件名是 `…_ctx_sm8750.SM8750.bin` ⇒ **被通配符漏掉，门判"缺失"**。
旧契约恰好不含它们，所以前两次交付没暴露。

⇒ 用 `check_deploy_sha.py --names <契约>`；**宿主上可预演**（拿脚本的通配符去套契约清单，纯字符串比较，秒级）。

## 48.4 交付前的四件事（§47 卡 F 的 app 版）

1. **先证明现状可用**：查 `files/history/` 确认最后一次成功生成是什么时候、哪个 APK。
   "用户说之前能出图"**不是基线**（本项目那次指的是 3 天前、另一个 APK）。
2. **备份原件**：APK 用 `adb exec-out run-as <pkg> cat <file>` **逐个文件**拉
   （单文件 2.9 GB 实测可行，约 26~30 MB/s）。🔴 `adb pull` 与任何中转到 `/data/local/tmp`
   的做法都会被 **SELinux** 挡（`untrusted_app` 不得写 `shell_data_file`）——
   这不是权限位问题，`chmod 777` 没用。
   ⊕ **适用范围（2026-09-25 补）**：上面说的是 **app 私有数据目录**（`files/` 下的模型、契约）。
   **已安装的 APK 本身**不在其内：`adb shell pm path <pkg>` 给出的 `/data/app/…/base.apk`
   可以直接 `adb pull`（①实测 94 MB，2.3 s，sha256 与 `sha256sum` 一致）。
3. **存档命名用整包 sha256**，不是 `.so` 的：实测只改 Kotlin 时 native `.so` 指纹**完全相同**，
   按 `.so` 命名会让只改上层的版本被当成"已有存档"跳过而**丢失**。
4. **自己跑一次真实链路**。静态校验（sha256、签名、切口闭合）验的是"文件对不对"，
   **不是"跑不跑得起来"**；且**必须在真实 app 里**跑，`run-as` 的 SELinux 域不同，不算对照。

## 48.5 与上游/官方版**并存**：只换包名不够（2026-09-25，①真机实测）

**场景**：想在同一台手机上同时保留自己的改版与官方新版。
Android 判定"是不是同一个 app"**只看 `applicationId`**（与名字、图标、版本号无关），
所以包名不同就不会互相覆盖、数据互不可见。**但以下几处不会跟着包名自动隔离**：

| # | 共享的东西 | 不改的后果 | 本项目做法 |
|---|---|---|---|
| 1 | 🔴 **本机回环端口**（后端 HTTP 监听 `localhost:<port>`） | 两个后端抢同一端口：后启动的起不来；**更糟的是界面的 `/health` 探测会拿到对方后端的 200**，把请求发给另一个 app —— 不报错，只是行为莫名其妙 | 8081/8808 → **8091/8818**；**只在 `RemoteProtocol` 定义一次**，其余全部引用（§48.3 规则 1）。C++ 默认值不动（app 总是显式传 `--port`），换来 `.so` 逐字节不变 |
| 2 | 公共导出目录（`Pictures/<名>`、`Downloads/<名>`） | 两个 app 的出图/日志混在一个相册 | 抽成一个常量 `PUBLIC_EXPORT_DIR` |
| 3 | 桌面名字、图标 | 看起来像同一个 app | 改 `app_name`；自适应图标三层（背景/前景/**单色**） |
| 4 | 🔴 **通知栏小图标**常与自适应图标的单色层共用一个资源 | 只换启动图标 ⇒ 通知栏仍显示上游标志 | 先 `grep -rn ic_launcher_monochrome`：本项目有 3 个服务拿它当 `setSmallIcon` |
| 5 | 🔴 **宿主侧脚本按进程名/子串判断死活**（`pidof libstable_diffusion_core.so`、`grep localdream`） | 对方 app 的后端**同名**、包名也含同一子串 ⇒ 判断被对方骗过 | 只认**本包 uid** 名下的进程（`ps -A -o USER,NAME`，先按包名取 USER，再按 USER 过滤） |

**不需要改的**（已查）：私有目录、`SharedPreferences`、数据库、通知渠道 ID 都按 app 隔离；
Intent 若全是显式组件（本项目无广播）则不会串。

**交付验证**（约束 11 四条全做）：签名证书先比对（不一致就**不能** `install -r`，
也**不许**靠卸载绕过 —— 卸载会删掉私有目录里 13 GB 模型）；装前旧版当场出图 = 金标准；
装后在真实 app 里出图仍 = 金标准，并用 `/proc/net/tcp` 确认新端口 LISTEN、旧端口无人监听
（这是**独立于主指标的生效证据**，§47.1.1 门 F1 —— 出图不变本身区分不了"改了"与"没生效"）。

⊕ 自适应图标的做法：用户给的是整张带背景的方图，而主体（光环）伸出了内切圆 ⇒ 直接用会被圆形遮罩切掉。
量主体离中心的最大半径，缩放到 **66dp 安全圈**内；前景边缘按"径向 × 距边框"双重渐隐到透明，
背景层用原图边缘色 —— 否则方形遮罩下会看到原图直边的接缝（本项目第一版就出现了）。
脚本 `qairt/scripts/make_zit_icons.py`，出圆/圆角/方形/主题/通知五联预览图，**先看图再接资源**。

## 48.6 🧩 本章的边界

- 本章回答"**要改哪些地方**"，**不**回答"怎么写一个新 Pipeline" ——
  那取决于新模型的调度器与前后处理，没有可复用配方。
- ①本项目只接过 Z-Image 一个模型进这个 app（`PipelineAnima` 等是上游 local-dream 自带的）。
  ⇒ 上面的分层**在 n=1 上成立**，换个结构差异很大的模型（例如多模态输入）可能要动 `QnnModel` 层。
- 宿主侧的契约生成脚本是 Z-Image 专属的，换模型要重写；可复用的是**字段结构与校验思路**，不是代码。
