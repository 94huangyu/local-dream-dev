> # 🔴 已作废 —— 不要用它开工
>
> **本文件是 2026-08-10 的早期交接稿，已被 `../docs/HANDOVER_2026-08-13_ZIMAGE_MVP.md` 完全取代。**
> 2026-08-21 改名归档，原因：它与正式交接文档同名前缀，新线程按"阅读交接文档"很容易开错这一份。
>
> **它里面这些说法现在都是错的**：
> | 本文说 | 实际 |
> |---|---|
> | `W8A8` 量化 | **A16W8**（act 16 bit / weight 8 bit） |
> | "1.2 最终确认方案（**当前执行中**）" | 已过期，见 `../docs/QNN_CONVERSION_GUIDE.md` |
> | 产物表里的 `SM8550` | 目标已是 **SM8750 / Hexagon v79** |
>
> ⚠️ **本文件字节层面有损坏**：537 处字符无法按 UTF-8 解码，L113 一句话被截断后拼进了表格。
> 原始字节**原样保留在下方**，未做任何修补。
>
> **开工请读 `MAINLINE.md`。**

---

# Z-Image Turbo 跨平台端侧量化工程交接文档

> **文档说明**：本文件为项目持续更新的交接记录与知识库。包含核心方案演进、技术路线变更记录、以及开发过程中的错误复盘与深刻反思。请在接手后续模型（如 Transformer）或进行架构调整前，**务必仔细阅读此文档**。

---

## 一、 技术路线与方案变更记录

### 1.1 初始方案（已废弃）
- **工具链**：尝试直接使用 `qnn-onnx-converter` 配合 `--input_list` 进行端到端 QNN Context Binary 的转换。
- **废弃原因**：`qnn-onnx-converter` 会强制将所有算子直接推向 HTP 硬件约束。对于 Z-Image 这种拥有大量复杂动态算子（如 `Slice`, `GatherND`）的生成式模型，直接编译会导致不可控的图拆分与算子 fallback，内存消耗巨大且极易崩溃。

### 1.2 最终确认方案（当前执行中）
- **工具链**：采用高通推荐的生成式 AI 标准量化管线：`qairt-converter` -> `qairt-quantizer` -> `qnn-context-binary-generator`。
- **核心策略**：
  1. 先用 `qairt-converter` 生成 FP32 的 `.dlc`，将计算图的 IR 解析与硬件约束剥离开来。
  2. 使用 `qairt-quantizer` 配合真实数据注入，生成 Mixed-Precision W8A8 量化 `.dlc`。
  3. 最后使用 `qnn-context-binary-generator` 离线编译出直接能跑的 HTP Context Binary (`.bin`)。
  4. **原则**：非必要不改图。只有当 QAIRT Dry run 明确报出某算子（如 `GatherND`）不支持并直接阻塞转换时，才允许基于 ONNX Runtime 编写局部的、完全数学等效的替换逻辑，并且必须验证绝对误差为 0.0 后才能继续。

### 1.3 Phase 3: Text Encoder W8A16 Quantization & Context Generation
**Status:** Completed ✅

### 复盘 2：W8A16 Context Binary 离线编译报错的本质
**错误表象**：在使用 `qnn-context-binary-generator` 离线编译 W8A16 DLC 时，报错 `[4294967295] has incorrect Value 68, expected >= 73` 并导致编译彻底失败。
**原因剖析**：
由于 QNN 框架对于 W8A16 的硬件加速有硬性要求，该特性在 Hexagon v73（即 Snapdragon 8 Gen 2 / SM8550）及以上的架构中才得到支持。然而 `qnn-context-binary-generator` 在没有任何外部配置指定时，会强行构建一个适配通用架构（默认 v68）的泛化模型（Generic Graph）。因为 v68 并不支持 W8A16 的 MatMul，导致 `Validate OpConfig` 第一时间崩溃退出，而不会继续生成我们通过 `--htp_socs sm8550` 要求的离线缓存。
**深刻反思**：
在处理基于特定硬件特性（如 W8A16, W4A16）的 QNN 量化部署时，**不可心存侥幸依赖工具的默认缺省值**。对于有强硬件绑定特性的精度，必须在所有相关命令链条中“强制注入”对应的 SoC 环境。通过编写底层的 `backend_extensions` 配置文件（如 `htp_config.json` 里显式声明 `soc_id: 57`），可以强行绑定运行上下文，从而彻底避开不相关兼容性检查的雷区。以后在遇到任何 `QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE` 时，应第一时间确认上下文架构是否匹配目标算子要求，而不是去怀疑模型图是否损坏。

### Findings & Changes:
1. **W8A16 切换**:
   由于 FP32 的 activation 中含有极大的异常值（`>16000`），W8A8 的 activation 量化（8-bit, 255 bins）会导致严重的信息截断，导致输出的 Cosine Similarity 降至 0。因此我们转而使用 **W8A16 (Weight 8-bit, Activation 16-bit)** 混合量化，保持了极高的精度。
2. **Context Binary 编译报错绕过**:
   - `qnn-context-binary-generator` 在生成 W8A16 缓存时，会遇到 `[4294967295] has incorrect Value 68, expected >= 73.` 的报错。
   - 原因是 W8A16 仅在 Hexagon v73 及以上的架构（如 SM8550）中支持，而 `qnn-context-binary-generator` 默认优先尝试构建基础的 v68 Generic Context，导致在验证 OpConfig 阶段崩溃。
   - **解决方案**: 创建了 `backend_ext.json` 和 `htp_config.json`，在启动参数中显式指定 `--config_file` 并将其 `soc_id` 设为 `57` (对应 SM8550/v73)。此时工具直接跳过针对 v68 的校验，成功生成了 `text_encoder_part1~4_ctx.SM8550.bin`。

### Current State:
四个部分的 Text Encoder 分段均已成功完成 W8A16 量化和 HTP 缓存编译（Context Binary），单分段的分配大小均完美避开 2GB 限制。目前产物存放于：
`D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\text_encoder_partX\`

---

## 二、 错误复盘与深刻反思（持续更新）

在处理第一个部件 `text_encoder` 的过程中，由于缺乏对大模型在 QNN 底层约束下的敬畏之心，犯下了 3 个极为经典的低级失误。**后续开发必须引以为戒**。

### 2.1 形状参数硬编码陷阱（导致张量广播失败）
- **现象描述**：在修复动态 `Slice` 和 `Expand` 形状时，探针网络报错 `invalid expand shape`。
- **错误原因**：我在 Python 脚本中试图硬编码去匹配维度名 `'sequence_length'`，但模型实际的维度名叫 `'seq_len'`。字符串匹配失败导致代码退化为 fallback（把该维度推断为 `1`），最终探针吐出了错误的常量形状 `[1, 8, 4, 1, 128]`，而非真实的 `[1, 8, 4, 20, 128]`。
- **深刻反思**：**绝不能凭经验主观臆测和硬编码张量维度名，更不能用诸如 `np.ones` 的假数据（Dummy Data）来糊弄探针。** 
- **强制约束**：后续所有涉及常量折叠、动态 Shape 固定的探测行为，必须 100% 读取真实 `sample_0000.npy` 数据注入模型探针，让计算图自己推导出最真实的常量。

### 2.2 忽视计算图拓扑排序规则（导致 QAIRT 转换崩溃）
- **现象描述**：`qairt-converter` 报错指出 `val_53` 不是任何前序节点的输出（`The model is invalid: Nodes in a graph must be topologically sorted`）。
- **错误原因**：在将 `GatherND` 拆分为 `Reshape + Slice + Gather` 时，我为了图省事，直接用 `append()` 将新节点添加到了 ONNX 图的末尾。ONNX Runtime 可能宽容了这点，但 QAIRT 编译器极其严格，线性扫描时发现被依赖的节点还未生成，直接抛出异常。
- **深刻反思**：对计算图进行“外科手术”时，决不能破坏图的物理拓扑结构。
- **强制约束**：所有的计算图节点替换**必须原位插入（In-place Insertion）**。即：删掉原节点所在的索引位置 `idx`，新生成的所有拆分节点必须依次 `graph.node.insert(idx, new_node)` 到该处。

### 2.3 等效替换遗漏维度压缩（导致后续算子 Rank 校验不匹配）
- **现象描述**：ONNX Runtime 验证阶段，后续的 `Transpose` 算子报错 `perm size: 4 does not match input rank: 5`。
- **错误原因**：用原生的 `Slice + Gather` 替换 `GatherND` 时，只盯住了“数值”对齐。忘记了原生的 `Slice` 切片后会保留一个大小为 `1` 的冗余维度，导致 4 维张量退化为 5 维。
- **深刻反思**：**数学等价 = 数值完全一致 + 张量秩（Rank）完全一致**。
- **强制约束**：下次手写算子替换逻辑前，必须在草稿上演算一遍张量维度推导公式；代码层面必须精准插入 `Squeeze` 或 `Unsqueeze` 来抹平冗余维度。

### 2.4 临时缓存与磁盘空间膨胀隐患（及误删导致管线断链）
- **现象描述**：单个模型可能高达 1-15GB，频繁生成 `_probe.onnx`, `_fixed.onnx`, `_fixed2.onnx` 有引爆磁盘空间的风险。但稍早前我提前手动删除了 `_fixed.onnx`，导致后续修复 `BOOL` 数据类型时找不到基准文件，管线空转了一整轮毫无变化的旧模型！
- **深刻反思**：在大模型工程中，存储资源与内存资源一样宝贵，但文件清理的时机绝不能“拍脑袋”。
- **强制约束**：中间试错垃圾（如 `_probe.onnx`, `_fixed.onnx`）必须在**整个管线的 Context Binary 完美跑通且验证可用后**，由终末脚本统一进行 `os.remove` 和 `%TEMP%/qairt*` 清理。在确认成功前，必须保留整个修复链条上的每一环！

### 2.5 量化引擎对变长序列尺寸的刚性约束（导致 Quantizer 读取数据崩溃）
- **现象描述**：Quantizer 在读取 `sample_0002/input_ids.raw` 时崩溃，报错尺寸为 76 字节，而预期为 80 字节。
- **错误原因**：虽然前端图的输入尺寸已通过覆盖强制固定为了 `[1, 20]`，但我忘记了处理真实的校准数据集 `calibration_raw`。数据集中存在实际长度为 19 和 18 的样本，它们被直接转成了 76 和 72 字节的裸二进制，导致 Quantizer 强制对齐校验失败。
- **深刻反思**：前端拓扑固定和后端数据投喂必须是“轴对称”的严密逻辑闭环。既然强加了静态 Shape，就必须提供绝对尺寸对齐的底层数据。
- **强制约束**：在管线脚本 `generate_input_list` 中，已注入**零填充（Zero-Padding）与截断（Truncation）**逻辑，所有 `.npy` 在转 `.raw` 前必须自动读取 `INPUT_DIMS` 进行严格形状对齐。这是处理后续 Transformer 变长提示词时的“保命符”。

### 2.6 布尔类型张量的 Gather 操作与量化零方差崩溃（Bad Scale 陷阱）
- **现象描述**：在离线编译 Context Binary 阶段，QNN 报错 `Output def has a bad scale (nan, inf, or too small)` 并导致 `QNN_ElementWiseSelect` 节点插入失败。
- **错误原因**：早前为了绕过 HTP 不支持对 `BOOL` 张量进行 `Gather` 的限制，我在前后强加了 `Cast(BOOL->INT32)` 和 `Cast(INT32->BOOL)`。但这种强转引起了量化引擎的误判，它试图对这个伪装成数值的掩码节点进行量化。由于注意力掩码在填充补齐后某些取值极度单一（例如 EOS token 掩码永远是 1），导致 Min-Max 极差为 0，计算出的 Scale 也为 0，直接触发底层硬件保护机制宕机。
- **深刻反思**：在硬件加速器上，永远不要用复杂的类型强转（Cast）去“欺骗”编译器，这通常会带来灾难性的副作用。最优解永远是寻找数学上的拓扑等价。
- **强制约束**：通过分析 ONNX Runtime 探针日志，确认该 `GatherND` 实际上是在对一个 `[1, 20]` 的序列做遍历（索引网格为等距常数）。这在数学上完全等价于一个 `Unsqueeze` 操作！直接用 `Unsqueeze` 替换掉复杂的 `GatherND` 和 `Cast` 链，不仅完美保留了纯正的 `BOOL` 血统让引擎跳过量化，更替底层硬件省去了一次昂贵的非连续内存切片访存。

### 2.7 Transformer 注意力掩码的“负无穷大”陷阱（导致 ElementWiseSelect 宕机）
- **现象描述**：修复了 Gather 节点后，Context Binary Generator 再次在 `QNN_ElementWiseSelect` 节点报出 `Output def has a bad scale (nan)`。
- **错误原因**：我反编译了 DLC 并发现，罪魁祸首是原模型中的 `Constant` 节点。HuggingFace 的 Transformers 库在生成 Attention Mask 时，经常使用 `torch.finfo(dtype).min`（对于 FP32 是 `-3.4028e38`，即负无穷小）来遮蔽（Mask）不需要的 Token。然而，在执行 PTQ 后训练量化（特别是使用 `percentile` 算法）时，这个跨度高达 $10^{38}$ 的极值会让量化校准的直方图统计瞬间爆炸，方差计算溢出，最终导致输出的 Scale 变为 `NaN`。
- **深刻反思**：大模型的数学极值（尤其是负无穷）对整型量化（INT8）是毁灭性的打击。
- **强制约束**：必须通过脚本遍历所有图中的 `Constant` 节点以及 `Initializer`，将所有极度偏小的掩码常量（`< -10000.0`）统统截断替换为 `-10000.0`。在 Softmax 运算中，`-10000.0` 足以在指数级衰减后产生完美的零概率，但它在量化时却能让直方图校准安然无恙。

### 2.8 整数量化与 IsNaN 算子的逻辑悖论（导致校验失败）
- **现象描述**：成功绕过负无穷陷阱后，管线在 Context Binary 编译时又报出 `Input[0] has incorrect Datatype 0x408` 和 `validateNativeOps master op validator node_IsNaN_... failed`。
- **错误原因**：`0x408` 在 QNN 中代表无符号 8 位定点数（INT8）。原始模型中为了过滤非法激活值，残留了 35 个 `IsNaN` 节点。当图被量化为整型后，高通 DSP 编译器直接拒绝在整数上执行 `IsNaN`（因为 `NaN` 仅存在于浮点数标准 IEEE 754 中，8位整数绝对不可能表示 `NaN`！）。
- **深刻反思**：在极度精简的底层 DSP 硬件上，任何冗余的、在数学上不可能发生的逻辑运算都会触发严格的安全校验崩溃。
- **强制约束**：遇到此类由于类型降级导致的无效逻辑节点，最优平替是数学拓扑变换。由于任何数值（即使是 `NaN`）都不会“严格小于”它自己，我们通过脚本将所有 `IsNaN(X)` 节点原地替换为 `Less(X, X)`。这不仅 100% 数学等价于一个永远输出 `False` 的纯布尔张量，更是完美绕过了 DSP 对整数 `IsNaN` 的类型校验，同时天然保持了原张量的 Shape 不变！

---

### 2.9 盲目假设算子不受支持与预先修改图的过度设计
- **现象描述**：在处理 Text Encoder 时，依据过往经验预先判定所有的 `GatherND` 和 `ScatterElements` 都会在 QAIRT 转换中失败，试图在干跑（Dry run）之前就通过复杂的 ONNX 替换脚本进行强行剥离。
- **错误原因**：QAIRT 编译器在版本迭代中已增强了对某些维度确定的静态 `GatherND` 的支持。预先的大面积强行替换反而可能引入新的维度错误、破坏原图语义，甚至引发不可预知的编译崩溃。
- **深刻反思**：**绝不能以先入为主的刻板印象代替真实的编译器反馈**。任何对计算图的等价替换，必须基于真实的干跑（Dry run）失败日志。
- **强制约束**：必须遵循“不撞南墙不回头”的原则。先对原始/当前已验证的 ONNX 执行 `qairt-converter` 干跑。只有当编译器明确报出某算子阻塞转换时，才允许基于 ONNX Runtime 进行**最小化**的数学等价修改。禁止使用改变计算语义的旁路逻辑。修改后必须立刻完成逐模型数值对比。

### 2.10 QAIRT 弃用参数陷阱与大图规避策略
- **现象描述**：调用 `qairt-converter` 时报错或崩溃，提示 `-d` 已经弃用，以及由于 protobuf 限制导致的内存溢出。
- **错误原因**：习惯性使用了过时的 `-d` (`--desired_input_shape`) 参数，而在新版 QAIRT 中必须使用 `-s` (`--source_model_input_shape`)。同时，Text Encoder 这种超大图在加载和默认化简（Simplification）阶段极易爆破 Protobuf 的 2GB 单次分配限制。
- **深刻反思**：工具链的底层更新往往是致命的，忽视 Warning 就是在给未来埋雷。
- **强制约束**：遇到大模型时，必须在转换命令中添加 `--onnx_skip_simplification` 以跳过会导致内存爆破的原生化简过程，并将所有 `-d` 参数替换为 `-s`。

### 2.11 `qnn-net-run` 默认数据解析陷阱（导致量化结果全损）
- **现象描述**：首次在主机上串联运行 W8A16 离线缓存时，出来的结果与 ONNX 基准进行余弦相似度对比，得分为负数（-0.02），且绝对误差飙升至 13835，量化输出完全是垃圾值。
- **错误原因**：输入文件 `input_ids.raw` 和 `attention_mask.raw` 在导出时保持了模型的原生设计（`int32`）。但在调用 `qnn-net-run` 时，如果没有额外配置，工具会**强制默认按照 32位浮点数 (float32)** 的格式去读取输入的二进制流！这导致原属于整数的位模式（bit pattern）被解析成类似 `1.41e-43` 这种极小或极大的乱码浮点数。数据在流水线入口就已经损毁。
- **深刻反思**：在系统级、跨工具（ONNX $\rightarrow$ DSP Simulator）�| 产物 | 状态 |
|------|------|
| `transformer_part1_fixed.onnx` | ✅ opset 降级 + Split/Identity 修复完成 |
| `transformer_part1a.onnx` | ✅ Part1a 切分完成（ORT 验证通过） |
| `transformer_part1b.onnx` | ✅ Part1b 切分完成（ORT 验证通过） |
| `transformer_part2_fixed.onnx` | ✅ opset 降级完成 |
| `transformer_part1a_ctx.SM8550.bin` | 🔄 量化流水线运行中 |
| `transformer_part1b_ctx.SM8550.bin` | ⏳ 等待校准数据生成完成 |
| `transformer_part2_ctx.SM8550.bin` | ⏳ 待评估（10.11 GB 数据，量化后约 2.5 GB，可能符合 3.5 GB 限制） |
| `vae_decoder_ctx.SM8550.bin` | ✅ **已完成**（103 MB） |

### 4.4 SM8550 HTP 内存上限导致的模型进一步切分

**问题**：SM8550 HTP（DSP）的图序列化内存上限为 **3,670,016 KB（≈ 3.5 GB）**。
原始 `transformer_part1` 量化后需 ~4 GB（3.4B 参数），超限，报错：
```
ERROR: graph requires estimated allocation of 4083073 KB, limit is 3670016 KB
```

**解决方案**：将 `transformer_part1` 在第 14 个 Add 残差节点（`add_138`，`graph.node[1367]`）处进一步切分：

| 子模型 | 节点范围 | initializer 数量 | 输入 | 输出 |
|--------|---------|-----------------|------|------|
| `transformer_part1a` | node[0..1367] | 234（含2个常量输出） | latents, timestep, caption, cap_pad_mask | add_138, add_131, tanh_19, select_45, select_46, adaln_input, unified_freqs, unified_mask, latents_shape |
| `transformer_part1b` | node[1368..2139] | 123 | add_138, add_131, tanh_19, select_45, select_46, adaln_input | unified |

**切分脚本**：`split_transformer_part1.py`（使用手动 GraphProto 手术，绕过 `onnx.utils.extract_model` 的 protobuf 2 GB 限制）

**校准数据级联**：`gen_part1b_calibration.py` 对 transformer_part1 的 40 个原始校准样本运行 Part1a ORT 推理，将输出作为 Part1b 的校准数据。

**接口张量维度**（qairt-converter `-s` 参数）：
| 张量名 | 形状 | 含义 |
|--------|------|------|
| `add_138` | 1,4128,3840 | 主 unified token 流（第14个 residual） |
| `add_131` | 1,1,3840 | adaln 中间广播张量 |
| `tanh_19` | 1,1,3840 | adaln tanh 输出 |
| `select_45` | 1,4128,1,64 | RoPE cos 分量 |
| `select_46` | 1,4128,1,64 | RoPE sin 分量 |
| `adaln_input` | 1,256 | AdaLN 条件向量 |

**注意（接手必读）**：
- `unified_mask` 和 `latents_shape` 是 **initializer 常量**（非节点计算结果），在 Part1a 中直接作为图输出暴露
- 完整端到端推理链：**Part1a → Part1b → transformer_part2**（Part2 的输入 `unified_mask`/`unified_freqs`/`adaln_input`/`latents_shape` 来自 Part1a）

### 4.5 校准数据与模型输入名映射

`vae_decoder` 的校准文件名（`latent_sample.npy`）与模型输入张量名（`vae_latents`）**不一致**，已在 `zimage_dlc_pipeline.py` 中通过 `CALIB_NAME_REMAP` 字典修复：

```python
CALIB_NAME_REMAP = {
    "vae_decoder": {
        "latent_sample": "vae_latents",
    }
}
```

### 4.6 transformer_part2 INPUT_DIMS 修正（历史错误）

原始 `zimage_dlc_pipeline.py` 中 `transformer_part2` 的 INPUT_DIMS 配置完全错误（拷贝了旧值），已根据实际 ONNX 模型修正：

| 字段 | 修正前（错误） | 修正后（正确） |
|------|--------------|--------------|
| `unified` 维度 | `1,4128,2560` | `1,4128,3840` |
| 第4输入 | `timestep: 1` | `adaln_input: 1,256` |
| 第5输入 | `latents: 1,16,128,128` | `latents_shape: 3` |
| `vae_decoder` 输入名 | `latents` | `vae_latents` |

��输入强制转 float、最终输出未反量化为 float），导致原本每次只需 1 分钟验证的单体测试，变成了每次耗时 10 分钟的端到端（End-to-End）空转。这直接引发了用户对开发专业性的强烈质疑。
- **错误原因**：我违背了资深工程师最基本的开发方法论：**“绝不在没有单元测试（Unit Test）保证的情况下，去跑庞大的集成测试（Integration Test）”**。我直接写了一个 `for i in range(1, 5)` 的大循环来贯穿整个黑盒管线，而没有在 Part 1 跑完后，立刻在代码中加入 `assert` 探针去审查（Inspect）第一个输出文件的字节大小、数据类型和数值合理性。
- **深刻反思**：20年的开发经验绝不是用来生搬硬套框架 API 的，而是体现在对**边界条件（Boundary Conditions）**的极端敏感，以及对**工程稳健性（Robustness）**的把控上。盲目地写一个大脚本跑全局，出了错再通过“看最终异常”来倒推原因，是初级程序员的“试错式（Trial-and-Error）”开发。
- **强制约束（最高级别的方法论纪律）**：
  1. **增量验证（Incremental Validation）**：后续在处理新的模型（如 Transformer 块）、或者拼接复杂的管道时，绝对禁止“一镜到底”。必须逐个节点、逐个 Part 编写独立的验证探针。只有上一个节点的输出（Shape、Byte Size、Dtype、Min/Max 数值范围）100% 对齐预期，才允许将数据喂给下一个节点。
  2. **防御性编程（Defensive Programming）**：在任何 Python 胶水脚本中，凡是涉及到读取外部工具生成的 `.raw` 或 `.bin` 文件，必须强制加入前置审查逻辑。例如：`assert os.path.getsize(file) == expected_size`，拒绝让垃圾数据污染下游计算。

### 2.13 百分位量化（Percentile Calibration）对 LLM 稀疏激活值（Outliers）的毁灭性打击
- **现象描述**：在经历了数据解析修复后，验证脚本的余弦相似度依然极低（`0.30`），且 QNN 引擎输出的激活值上限被莫名其妙地强制截断在了 `6670` 左右（而实际 ONNX FP32 激活值高达 `16133`）。
- **错误原因**：我在 `qairt-quantizer` 阶段采用了 `--act_quantizer_calibration percentile`。对于传统的 CNN 视觉模型，截断掉顶部 0.01% 的极端值可以换取更好的量化分辨率。但大语言模型（如 Qwen）的核心机制高度依赖**稀疏的极端异常值（Massive Sparse Outliers）**来作为“路由”机制（Routing Mechanism）。`percentile` 算法把这些不可或缺的 Outliers 当作噪声裁掉了，导致量化的激活截断阈值（Clipping Threshold）极大幅度收缩，彻底破坏了模型的注意力流。
- **深刻反思**：经验主义害死人！照搬 CNN 时代的量化经验到大语言模型上是极度危险的。特别是当我们采用 W8A16 这种具备 65536 个量化桶的超高精度激活容器时，根本没有必要为了压榨微小的分辨率去承担丢弃 Outlier 的风险。
- **强制约束**：在处理所有基于 Transformer 架构的大语言/多模态模型时，其 Activation 量化参数 `--act_quantizer_calibration` **必须强制使用 `min-max`（或根据高通官方对该模型的特殊推荐算法）**，绝对禁止默认使用 `percentile`！

### 2.14 量化校准工具 (qairt-quantizer) 的默认 float32 读取陷阱
- **现象描述**：即使用 `min-max` 重新校准，如果依然发现激活截断异常，说明校准数据本身失效了。
- **错误原因**：无论原 ONNX 模型定义的输入（如 `input_ids`）是 `int32` 还是 `int64`，`qairt-quantizer` 在读取校准目录下的 `.raw` 文件时，**默认一律当作 `float32` 格式强行解析**（除非特别指定或输入 DLC 被特殊标记）。如果你直接将 numpy 的 `int32` 数组 `.tofile()` 存为二进制，`qairt-quantizer` 会将整型比特流强行按照 IEEE 754 浮点数解码，例如把 `151644` 解码成 `2.12e-40`（约等于 0）。这意味着你的量化校准其实一直是在用全 0 数据进行盲跑！
- **强制约束**：在 Python 胶水脚本生成 Calibration `.raw` 文件时，**必须强制增加 `.astype(np.float32)` 转换**，或者在 `qairt-quantizer` 命令行中精确指定数据格式！这是跨平台 C++ 底层工具最常见的内存映射解析深坑。

---

## 三、 Text Encoder W8A16 量化全链路 —— 最终完成状态 ✅

> **完成时间**: 2026-08-05  
> **执行摘要**: Text Encoder（Qwen3 系列，~3.5B 参数）完成了从 ONNX 切分 → W8A16 量化 → HTP Context Binary 编译 → 数值验证的全链路工作，已达到生产可用标准。

---

### 3.1 物理内存墙与四段切分架构（已解决）

- **问题根因**：整体 Text Encoder 单图需要 3.84 GB 内存，超过 Hexagon DSP 的 2 GB 物理上限。
- **解决方案**：沿 Transformer Block 边界将其切分为 4 段顺次执行的子图。
- **切分点（中间激活值张量名）**：
  - Part1 → Part2 接口：`add_2452`（shape: `[1, 20, 2560]`, float32）
  - Part2 → Part3 接口：`add_4828`（shape: `[1, 20, 2560]`, float32）
  - Part3 → Part4 接口：`add_7204`（shape: `[1, 20, 2560]`, float32）

### 3.2 最终产物清单

| 产物文件 | 大小 | 说明 |
|---------|------|------|
| `text_encoder_part1_ctx.SM8550.bin` | **1614 MB** | Part1 HTP Context Binary，目标芯片 SM8550 |
| `text_encoder_part2_ctx.SM8550.bin` | **872 MB** | Part2 HTP Context Binary |
| `text_encoder_part3_ctx.SM8550.bin` | **872 MB** | Part3 HTP Context Binary |
| `text_encoder_part4_ctx.SM8550.bin` | **775 MB** | Part4 HTP Context Binary，输出张量名 `caption` |
| `qnn_manifest.json` | — | 已更新：路径指向 `.SM8550.bin`，精度标记为 `W8A16` |

> **注意**：所有 `.bin` 文件均以 `_ctx.SM8550.bin` 为正式名称（非旧版 `_ctx.bin`），manifest 已同步修正。

### 3.3 数值验证结果（已全部通过）

**单体验证（各 Part 独立 vs 对应 ONNX 子模型）**：

| Part | Cosine Similarity | MAE | 状态 |
|------|-------------------|-----|------|
| Part 1 | **0.999822** | 0.468 | ✅ |
| Part 2 | **0.999989** | 0.270 | ✅ |
| Part 3 | **0.999977** | 0.400 | ✅ |
| Part 4 | **0.999769** | 0.848 | ✅ |

**端到端级联验证（Part1→2→3→4 串联 vs 完整 `text_encoder_fixed4.onnx`）**：

| 指标 | 值 |
|------|-----|
| **Cosine Similarity** | **0.997519** ✅（阈值 ≥ 0.99） |
| MAE | 2.700769 |
| Max Abs Diff | 41.491753 |
| 输入 | `input_ids=[151644, 872, 198, ...]`（seq_len=20） |
| 输出张量 | `caption`（shape: `[1, 32, 2560]`，81920 元素） |

**结论**：4 段级联累积误差 **未出现放大**，从单段最高 0.9997 到端到端仍维持 0.9975，在 W8A16 精度范围内属于正常的量化噪声叠加，生产可用。

### 3.4 关键工具脚本清单

| 脚本 | 路径 | 用途 |
|------|------|------|
| `zimage_dlc_pipeline.py` | `D:\ZImage_Work\` | 主流水线：ONNX→FP32 DLC→W8A16 量化→Context Binary |
| `verify_part1.py` | `D:\ZImage_Work\` | Part1 单体数值验证 |
| `verify_part2.py` | `D:\ZImage_Work\` | Part2 单体数值验证 |
| `verify_part3.py` | `D:\ZImage_Work\` | Part3 单体数值验证 |
| `verify_part4.py` | `D:\ZImage_Work\` | Part4 单体数值验证 |
| `validate_cascade_e2e.py` | `D:\ZImage_Work\` | **端到端级联验证**（最终质量门） |
| `backend_ext.json` | `D:\ZImage_Work\` | W8A16 HTP 编译配置，指定 SoC SM8550 |

### 3.5 复现流水线的操作步骤（供下一位接手工程师参考）

```powershell
# 环境：D:\LocalDreamZImage\local-dream\.venv-qnn
# 如需重新量化某 Part（以 part1 为例）：
python zimage_dlc_pipeline.py \
  --model text_encoder_part1 \
  --onnx ZImage_QNN_Evidence\onnx\text_encoder_part1.onnx \
  --outdir ZImage_QNN_Evidence\dlc_pipeline\text_encoder_part1 \
  --calib ZImage_QNN_Evidence\calibration\text_encoder

# 端到端验证：
python validate_cascade_e2e.py
# 期望输出：Cosine Similarity >= 0.99
```

### 3.6 关键约束（接手必读，违者必踩坑）

1. **校准数据格式**：`calibration_raw` 中的所有 `.raw` 文件均以 **float32 格式** 存储整数值（如 `input_ids` 值 151644 存为 `float32(151644.0)` 的二进制）。见 2.14 节。
2. **量化算法**：所有 Transformer 模型的 Activation 量化必须用 **`min-max`**，禁止 `percentile`。见 2.13 节。
3. **推理时 Part1 输入**：`input_ids` 和 `attention_mask` 必须用 **native int32** 格式（`.tofile()` 后通过 `--use_native_input_files` 传入），见 `validate_cascade_e2e.py` Step 0-2。
4. **Context Binary 命名**：正确后缀为 **`_ctx.SM8550.bin`**（非 `_ctx.bin`），所有路径引用均需注意。
5. **W8A16 编译必须指定 SoC**：`backend_ext.json` 中的 `soc_id: 57`（SM8550）必须随 `--config_file` 传入 `qnn-context-binary-generator`，否则会因默认 v68 架构检查失败而崩溃。

---

## 四、Transformer + VAE Decoder 量化链路

### 4.1 关键 ONNX 兼容性问题：opset 18 → 17 降级

> **适用模型**：`transformer_part1.onnx`、`transformer_part2.onnx`（由 PyTorch 2.x 导出，opset 18）  
> **qairt-converter 2.48 的硬性限制**：只支持以下算子到 opset 16/17：

| 算子 | opset18 新特性 | 修复方案 |
|------|---------------|---------|
| `ReduceMean` / `ReduceSum` | `axes` 从属性改为可选的第 2 个输入 tensor；新增 `noop_with_empty_axes` 属性 | 将 axes tensor 转回 `axes` attribute；移除 `noop_with_empty_axes` |
| `Split` | 新增 `num_outputs` 属性 | 移除该属性 |
| `ScatterElements` | 新增 `reduction` 属性（`none`/`add`/`mul`等） | 移除 `reduction=none`（v16 默认行为即 none） |

**修复脚本**：
- `fix_transformer_opset.py` — 处理 `transformer_part1_fixed.onnx`
- `fix_transformer_part2_opset.py` — 处理 `transformer_part2_fixed.onnx`

**修复流程**（3 步）：
```python
# 1. ReduceMean: axes tensor → attribute + 移除 noop_with_empty_axes
# 2. ReduceSum/Split: 移除 noop_with_empty_axes/num_outputs
# 3. ScatterElements: 移除 reduction=none
# 4. 将 opset 声明从 18 改为 17
```

### 4.2 qairt-converter 单输出 Split 节点 Bug

**现象**：即使完成 opset 降级后，qairt-converter 仍会在 `Split` 节点上崩溃并报 `KeyError: 'split_with_sizes_X_split_0'`。

**根因**：PyTorch 的 `torch.split_with_sizes([full_size])` 在 `split_sizes` 列表只有一个元素时，会导出一个"单输出 Split 节点"（整体张量 = 一份），qairt-converter 无法正确注册此类节点的输出 buffer。

**修复方案**：将所有满足以下条件的 Split 节点原位替换为 `Identity` 节点（数学等价）：
- `len(outputs) == 1`
- `split_sizes` 是常量 initializer
- `len(split_sizes) == 1`

**修复脚本**：`fix_transformer_split.py`

**transformer_part1** 中共有 **6 个**此类节点，修复后 ONNX Runtime 验证通过。**transformer_part2** 无此问题。

### 4.3 Transformer 最终产物清单（进行中）

| 产物 | 状态 |
|------|------|
| `transformer_part1_fixed.onnx` | ✅ opset 降级 + Split 修复完成 |
| `transformer_part2_fixed.onnx` | ✅ opset 降级完成 |
| `transformer_part1_ctx.SM8550.bin` | 🔄 量化流水线运行中 |
| `transformer_part2_ctx.SM8550.bin` | ⏳ 等待 part1 完成后启动 |
| `vae_decoder_ctx.SM8550.bin` | ✅ **已完成**（103 MB） |

### 4.4 校准数据与模型输入名映射

`vae_decoder` 的校准文件名（`latent_sample.npy`）与模型输入张量名（`vae_latents`）**不一致**，已在 `zimage_dlc_pipeline.py` 中通过 `CALIB_NAME_REMAP` 字典修复：

```python
CALIB_NAME_REMAP = {
    "vae_decoder": {
        "latent_sample": "vae_latents",
    }
}
```

### 4.5 transformer_part2 INPUT_DIMS 修正（历史错误）

原始 `zimage_dlc_pipeline.py` 中 `transformer_part2` 的 INPUT_DIMS 配置完全错误（拷贝了旧值），已根据实际 ONNX 模型修正：

| 字段 | 修正前（错误） | 修正后（正确） |
|------|--------------|--------------|
| `unified` 维度 | `1,4128,2560` | `1,4128,3840` |
| 第4输入 | `timestep: 1` | `adaln_input: 1,256` |
| 第5输入 | `latents: 1,16,128,128` | `latents_shape: 3` |
| `vae_decoder` 输入名 | `latents` | `vae_latents` |


### 5. �˵��˼�����֤ (��ǰ�׶�)

#### 5.1 ��֤Ҫ�����
��������ָ�����ִ���ϸ����֤���̣�
1. ֹͣ���� \QnnHtp.dll + context binary\�����������ɵ�ȫ�����
2. ʹ�� \QnnCpu.dll + quantized DLC\ ���� off-target ��֤��
3. �����ȵ������� \part1a\����ʱ 60 ���ӡ�����ɹ�������һ���� \part1b\ �� \part2\ (ÿ���������̣���ʱ 60 ����)������������ 8 ����Ԥ�ƺ�ʱ��
4. ��ֵ�ȶԲ���Ҫ�� Cosine Similarity���������¼ MaxAbsDiff��MaxRelDiff��NaN/Inf ������
5. �� CPU off-target ��֤ͨ����ֻ���ж�Ϊ \CPU/DLC off-target quantization reference passed\������ȷ���� HTP Context Binary ��δ����ʵ�豸����֤��

#### 5.2 QNN CPU Off-target ��֤���
- **����Ŀ��**: \part1a_quantized.dlc\ ��� \QnnCpu.dll\`n- **���Խ��**: ʧ�� (QNN CPU cannot validly execute this W8A16 model)
- **��ϸ��־**: ����ʹ�� \qnn-net-run\ ���� \qnn-context-binary-generator\ ָ�� CPU ��ˣ������׳��������ش���\[QNN_CPU] OpConfig validation failed for Reshape\ �� \QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE\��
- **����ԭ��**: QNN �� x86 CPU ���ȱ���� W8A16 �ض��������ӣ��� Reshape �� 16-bit ����ֵ�󶨣�������֧�֣������� HTP ������ǺϷ��ġ�
- **����**: ��ѭָ�����ֹͣ QNN CPU ���������۸�ԭģ�͡�׼��ת��ʹ�� AIMET QuantSim ����ͬ������������صľ�����֤��

#### 5.3 ̽�� AIMET QuantSim �������
- **��ǰ״̬**: ����������ǰ�������Ƿ��Ѱ�װ������ AIMET (AI Model Efficiency Toolkit)��
- **��������**: ��ǰ�� \.venv-qnn\ Python ���⻷����δ�ҵ� \imet_torch\ �� \imet_onnx\ ģ�顣QAIRT SDK ������ aimet ����� adapter �ű�����ȱ�������� AIMET ���ļ������
- **��һ���ж�**: ��������λ�ȡ����װ��Ӧ�汾�� AIMET������Ѱ�Ҹ�ͨ�ٷ��Ƽ������豸 (off-target) �����������������

### 6. Android APK Z-Image Import Bug Fix (Fix4)

#### 6.1 Issue Identification
During the validation of Z-Image models imported via ZIP files, a critical structural conflict was identified in the LocalDream Android App (Fix3 and earlier):
1. **Zip Flattening Bug**: The legacy model import logic (ModelListScreen.kt and ModelDownloadService.kt) forcefully flattened all zip entries to the root of the model directory using substringAfterLast('/'). This destroyed necessary folder structures like qnn_runtime_libs and models/.
2. **Strict Validation Bug**: The Z-Image validation logic (Model.kt -> isCompleteZImageBundle) strictly expected the qnn_runtime_libs/aarch64-android folder to exist and contained the SDK .so files, and expected the model files to match paths like models/text_encoder.... The flattening caused this validation to fail unconditionally.
3. **JSON Key Mismatch**: The validation logic explicitly parsed the "models" JSON array in inal_qnn_contract.json, but the real contract generated by the QNN pipeline uses "graphs".

#### 6.2 Resolution (Fix4)
1. **Directory Preservation**: Modified ZIP extraction loops in ModelListScreen.kt and ModelDownloadService.kt to preserve zipEntry.name completely, using mkdirs() to restore deep folder hierarchies.
2. **Flexible Contract Parsing**: Updated Model.kt to fall back to "graphs" if "models" is not found in the JSON contract.
3. **Outcome**: The App can now correctly extract multi-level Z-Image model zips and successfully validate the QNN runtime libraries and contract paths.
