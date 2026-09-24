# ONNX / DLC / context-binary 执行模型（只写**实测确立**的事实）

> 本文件的存在，是因为本项目多次"用 ONNX 的知识推断 DLC 行为"而喂错数据，而且每次都是撞了才查、
> 只改当次报错的那一个。每条规则后面都必须附证据；**没有证据的一律写进"尚未验证"一节，不得当事实用**。

## 一、三种产物的关系

```
.onnx  --qairt-converter-->  _fp32.dlc  --qairt-quantizer-->  _quantized.dlc
                                                                    |
                                        qnn-context-binary-generator |
                                                                    v
                                                          _ctx.<SOC>.bin  （设备上跑的）
```

- 离线用 `snpe-net-run` 跑的是 **`.dlc`**（SNPE CPU 参考实现）
- 真机跑的是 **`.bin`**（HTP 硬件）
- **两者是否数值等价：尚未验证**（见第四节）

## 二、`snpe-net-run` 的输入/输出契约（已实测确立）

### 规则 1：所有输入 `.raw` 一律写 **float32**，与 DLC 内部声明的数据类型无关

即使张量在 DLC 里声明为 `Bool_8`（1 字节）或 `Int_32`，宿主侧仍然写 float32。
DLC 里的 `uFxp_16` / `Bool_8` 是**内部表示**，不是宿主接口类型。

**证据**：
- 正例：校准文件 `latents.raw` = 1,048,576 字节 = 262,144 元素 × 4，DLC 声明 `uFxp_16`（2 字节），
  运行成功且数值正确。`caption.raw` / `cap_pad_mask.raw` 同理（后者 128 字节 = 32 × 4，
  DLC 声明 `Bool_8`）。
- 反例（本项目 2026-08-14 实测）：把 `cap_pad_mask` 按 `Bool_8` 写成 32 字节（uint8）后，
  **snpe 不报错**，但**所有输出张量都变成 1/4 尺寸**（`add_138` 从 63,406,080 → 15,851,520 字节，
  `adaln_input` 从 1024 → 256）。原因：snpe 用「输入文件字节数 ÷ 期望字节数」推断批次数，
  32/128 = 1/4，于是整张图按 1/4 批次执行。

> **这条最危险**：喂错不报错，而是静默产出尺寸/内容都不对的结果。
> 唯一的防线是**校验输出字节数 = 元素数 × 4**。

### 规则 2：所有输出 `.raw` 一律是 **float32**

即使张量声明为 `Bool_8`（如 `unified_mask`）或 `Int_32`（如 `latents_shape`），
`snpe-net-run` 写出的都是 float32。

**证据**：`unified_mask` (1,4128) 声明 `Bool_8`，导出文件 16,512 字节 = 4128 × 4。

### 规则 2·补【2026-08-14 新增，重要】：规则 1/2 **只适用于 `snpe-net-run` + `.dlc`**，
### 不适用于设备上的 QNN API + `.bin`

**不要把规则 1 的"一律 float32"跨路径套用到设备实现上——两条路径的宿主契约是不同的。**

**证据（读 C++ 实现 + 三处独立守卫的静态推理）**：设备侧 `cap_pad_mask` 写的是
**32 字节 uint8**（`PipelineZImage.hpp:101-104`，`std::array<uint8_t,32>`），
而不是 SNPE 路径要求的 128 字节 float32。这**不是 bug**，因为该值被三处独立守卫校验过：

| # | 位置 | 校验内容 |
|---|---|---|
| 1 | `ZImageQnnContract.hpp:164` | `tensorBytes(tensor) != tensor.bytes` 即抛异常；`dtypeBytes("bool")=1` ⇒ 契约里 `cap_pad_mask` 恒为 32 字节，JSON 若写 128 反而会抛 |
| 2 | `PipelineZImage.hpp:390` | `!value.spec && data->size() != input.bytes` 即抛 `"Z-Image host input byte mismatch"` |
| 3 | `QnnModel.hpp:89-94` | 与**真实 QNN 张量**的 `QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize` 逐字节比对，不等即 `FAILURE` |

第 3 处比对的是 `.bin` 里 QNN 张量自己声明的 client buffer 大小。
设备确实跑出了图（`scratch_runs/sm8750_test2.png`），说明这三处**全部通过**
⇒ `.bin` 里 `cap_pad_mask` 的 client buffer 就是 **32 字节 BOOL_8**。

> ⚠️ 推理链说明：「三处守卫都通过」是由「设备跑出了图而非抛异常」**反推**的（②严格推论），
> 不是直接读取设备内存得到的（①实测）。若要升级为实测，需在设备上打印
> `QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize`。

**结论**：`scripts/dlc_contracts.json` 里的 `host_bytes`（`cap_pad_mask`=128）
是 **SNPE 工具路径**的契约，**不是设备 QNN 运行时的契约**。
两者对同一个张量给出不同的宿主字节数，且**各自都是对的**。

> **给后来者**：看到 C++ 里 `cap_pad_mask` 是 32 字节而契约 JSON 写 128，
> **不要"修"它**——改成 128 会让守卫 3 直接失败。本次差点因此误报一个根因。

转换阶段会把可常量折叠的输入直接消掉。

**证据**：`transformer_part2.onnx` 声明 5 个输入（含 `latents_shape`），
但 `transformer_part2_quantized.dlc` 里 **只有 4 个**，`latents_shape` 完全不存在
（`snpe-dlc-info` 转储里 grep 计数为 0）。多喂会报
`Error: invalid input size provided. Please check all raw sizes.`

> 因此：**FP32(ONNX) 侧和量化(DLC) 侧不能共用同一份输入名单。**

### 规则 2·补 2【2026-09-18 新增，🔴 会静默出错】：`qnn-net-run` + `.bin` 是**第三条路径**，
### 默认按 **float32** 解析输入/写出输出；喂原生字节会被**静默砍半**

三条路径的宿主契约互不相同，**不得互相套用**：

| 路径 | 输入 | 输出 |
|---|---|---|
| `snpe-net-run` + `.dlc`（规则 1/2） | 一律 float32 | 一律 float32 |
| 设备 app 的 QNN API + `.bin`（规则 2·补） | 契约声明的原生字节 | 原生字节 |
| **`qnn-net-run` + `--retrieve_context .bin`（本条）** | **默认 float32**；`--use_native_input_files` 才按原生 | **默认 float32**；`--use_native_output_files` 才按原生（文件名加 `_native` 后缀） |

**实测（2026-09-18，part1b 单段，三臂单变量）**：

| 臂 | 输入文件 | 标志 | 结果 |
|---|---|---|---|
| I | float32 全尺寸（`add_138` = 64,143,360 B） | 无 | rc=0，输出 `unified.raw` **64,143,360 B**（float32 全尺寸）✅ |
| J | 原生 uint16（32,071,680 B） | `--use_native_input_files --use_native_output_files` | rc=0，输出 `unified_native.raw` **32,071,680 B** ✅ |
| C | 原生 uint16（32,071,680 B） | 无 | rc=0，输出 `unified.raw` **32,071,680 B** ⇒ 按 float32 算只有 **8,017,920 个元素 = 真值的一半** 🔴 |

🔴🔴 **这条为什么危险**：对 `uFxp_16` 张量，**原生字节数恰好等于 float32 字节数的一半**。
于是「输入喂原生字节」会让运行器按「字节数 ÷ 期望字节数 = 0.5」**静默缩小张量**，
而写出的 float32 输出字节数**正好等于契约里的原生字节数** ⇒
**约束 3 要求的「输出字节数 == 契约」守卫会被这个巧合骗过，报 PASS。**

**正确的自证方式（本项目已改用）**：
1. 显式加 `--use_native_input_files --use_native_output_files`；
2. **逐个**输出张量核字节数（不能只核一个）；
3. 交叉验证 `unified_mask`（`Bool_8`）与 `latents_shape`（`Int_32`）这类「原生 ≠ float32/2」的张量——
   float32 模式下它们是 ×4 和 ×1，是判别当前处在哪种模式的**廉价指示器**。

⚠️ **波及**：#147（`O:3`/`dlbc` 离线测速）用的就是「原生字节 + 无标志」这套装置，
三臂同缺陷 ⇒ **相对比较可能仍成立，但绝对数字（如 part1b 15.9 s/次）不得引用**。

### 规则 4：DLC 的张量名 **不等于** ONNX 的张量名（部分被融合消除）

**证据**：`add_24` / `add_45` 在 ONNX 里存在，在 `transformer_part1a_quantized.dlc` 里不存在
（`--dump_encoding_json` 导出的 1023 个 activation 名单里没有）。请求不存在的名字会报
`error_code=204; Couldn't find name`，且**整个运行失败**，不是跳过。

> 权威名单来源：`qairt-quantizer --dump_encoding_json` 导出的 JSON。

### 规则 5：`%` 前缀可以导出**任意中间张量**，不限于图的正式输出

**证据**：用 `%add_57`（非图输出、也不在历史测量集里）成功导出，文件 63,406,080 字节
= 4128 × 3840 × 4，尺寸精确吻合。

### 规则 6：`snpe-net-run` 的 CPU 后端**不支持部分算子**，会在模型校验阶段直接失败

**证据**：
- `text_encoder_part1`：`No backend could validate Op=node_embedding Type=Gather`（error_code=1002）
- 用 override 产出的 `transformer_part1a` DLC：`Op=node_select_scatter_1 Type=ScatterElements`

> 这是模拟器限制，**不代表该 DLC 在真机上有问题**（这些 DLC 都是设备上正常加载运行的）。

## 三、`qairt-converter` 的行为（已实测确立）

### 规则 7：只要提供 `--quantization_overrides`，转换器**强制跳过** ONNX 简化

`--onnx_skip_simplification` 这个 flag 在这种情况下无效。

**证据**：日志明确打印
`Can't simplify the model when custom ops, quantization overrides or dynamic input shapes are specified, converting without simplification.`
加不加该 flag 都一样。

### 规则 8：override JSON 的 `version` 字段决定走哪个量化器

`"2.0.0"` → `IrQuantizerV2.apply_encodings()`（本 SDK 版本存在原生崩溃）；
`"0.6.1"` / `"1.0.0"` → 老版 `IrQuantizer.quantize()`（稳定）。

**证据**：源码 `qti/aisw/converters/qnn_backend/ir_to_dlc.py:577-593` 的分支条件，非试错所得。

## 四、【尚未验证】—— 不得当作事实使用

1. **`snpe-net-run`（CPU 参考）与 HTP 上执行 `.bin` 是否数值等价。**
   这条影响面最大：13.9～13.17 的全部误差数字都来自 CPU 参考路径，却在文档里被表述为"设备实测"。
   已知一条反向信号：`transformer_part2` 在 **HTP 编译**阶段报过
   `scale too large for requant qu16->qu16: inf`，这是 HTP 路径特有的，CPU 路径不会出现。
2. **`.bin` 的输入/输出契约是否与 `.dlc` 一致**（未查）。
3. **`--use_native_input_files` / `--use_native_output_files` 的确切语义**（只在文档里见过一句话，未实测）。
4. **另外 5 个 DLC（text_encoder ×4、vae_decoder）的完整输入输出契约**（未导出）。
5. **snpe 对"输入文件大于期望"的确切行为**：本项目只观测到"小于期望 → 按比例缩小批次"，
   大于期望的情况未做受控实验。

## 五、查契约的标准做法

```bash
# 导出某个 DLC 的完整信息
python scripts/dlcinfo.py <path.dlc> > <name>_dlcinfo.txt
# 提取输入/输出契约表
python scripts/extract_dlc_contract.py <name>_dlcinfo.txt
```

**动手喂数据之前先查表，不要从 ONNX 推断。**
