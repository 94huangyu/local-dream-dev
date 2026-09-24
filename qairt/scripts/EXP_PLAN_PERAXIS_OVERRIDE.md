# EXP_PLAN_PERAXIS_OVERRIDE —— P0-B 前置门：`--quantization_overrides` 能否注入 **per-axis** encoding？

定稿时间：**2026-08-21**（执行前定稿，**判据不得事后修改**）
归属：执行顺序 **P0-B 前置**。不过这道门，P0-B「Exact-real-FC standalone replay」不成立。

---

## 一、为什么必须先验这个

P0-B 要"抽真实 FC 的精确输入/权重/**encoding**，**不重新校准**，建单算子 context"。
真实 FC 的权重是 **per-row** 量化的（#47），⇒ **要把 per-row encoding 原样注进单算子模型**。
但**本项目至今只用 overrides 注过 per-tensor**（#45/#7）⇒ **per-axis 能不能注入是未验证前提。**

## 二、官方文档（约束 5，先查后做）

`docs/QAIRT-Docs/QNN/general/applyencodings.html` Appendix C（JSON schema **2.0.0**）
明确给出 **Per channel quantization encoding example**：

```json
{ "name": "conv.weight", "output_dtype": "int8",
  "y_scale": [0.00501, 0.00171, 0.00171], "axis": 0 }
```

`quantization.html` 另给 **AIMET 格式**（`activation_encodings`/`param_encodings`），
本项目 #45 打通的正是它（`version="0.6.1"` + **ONNX 张量名**）。
AIMET 约定里 per-channel 表示为**同一张量对应一个多元素的 encoding 列表**。

⚠️ **两种 schema 都只是"文档说支持"，未在本项目验证过** ⇒ 本实验就是验证它。
⚠️ 历史记录称 `version="2.0.0"` 曾触发 SDK 崩溃（#45），⇒ 两种都要试，不预设哪个能用。

## 三、实现路径（最小已知样本，秒级迭代）

**不用真实模型**——用一个我完全掌握答案的最小图，避免"结果不对但分不清是注入失败还是模型太复杂"。

- ONNX：`x[1,8] → MatMul(x, W[8,4]) → y[1,4]`，`W` 为 initializer（在 DLC 里成为 FullyConnected 权重）
- 注入 **4 个互不相同、且我自己指定**的 scale（对应 4 个输出通道）
- 三条臂：

| 臂 | 格式 | 内容 |
|---|---|---|
| **C（V 门）** | 0.6.1 AIMET | `W` 给**单个** encoding dict（per-tensor）——**已知本项目可用** |
| **A** | 0.6.1 AIMET | `W` 给**4 个** encoding dict（AIMET per-channel 约定） |
| **B** | 2.0.0 | `y_scale` 4 元素 + `axis` |

命令序列照抄 #7 已验证可用的那套：
`qairt-converter --quantization_overrides <json> --float_bitwidth 32`（#46：**必须显式给 32**）
→ `qairt-quantizer`（per-tensor 基线参数）→ `qairt-dlc-to-json` 读回 encoding。

## 四、🔒 判据（事前锁定）

### V 门（先过，不过则整个实验作废）

| 门 | 要求 |
|---|---|
| **V1** | 每次转换的日志里 **`Processed N quantization encodings`，N ≥ 1**。<br>N=0 ⇒ **静默失效**（#45 的坑，用错张量名就会这样），该臂作废重来 |
| **V2** | **臂 C 必须成功**：DLC 里 `W` 的 scale **与我注入的值一致**（相对差 < 1e-6）。<br>不过 ⇒ 整条链路（最小模型/命令/读回方法）有问题，**A/B 的结果一律不可采信** |

### 主判据

读回 DLC 中 `W` 的 encoding，数**互不相同的 scale 个数** `n_scale`：

| 结果 | 判定 | 后续 |
|---|---|---|
| **A 或 B 任一给出 `n_scale == 4` 且数值与注入值逐个吻合** | ✅ **per-axis 可注入** | **P0-B 前置门通过**，记录可用格式与 axis 取值 |
| 两臂都给出 `n_scale == 1`（退化为 per-tensor） | ❌ **不可注入** | P0-B 的 "exact replay" 不成立，**必须改设计**（例如改用真实模型切子图，或放弃精确重放改用等价重建） |
| 两臂都报错 / `Processed 0` | ❌ **格式不被接受** | 同上，并记录报错原文 |

⚠️ **不得事后放宽**：`n_scale` 只要不是 4，就不算通过。数值必须逐个吻合，
"有 4 个 scale 但值是量化器自己算的"**不算注入成功**（那只是 per-channel 量化被打开了）。

## 五、失败归因

- V1 不过 ⇒ 张量名错（#45 的坑），改名重试，不算方向失败
- V2 不过 ⇒ 装置本身有问题（最小模型/命令/读回），修装置
- V1/V2 都过而主判据落在"退化为 per-tensor" ⇒ **方向失败**：该 SDK 不支持经 overrides 注入 per-axis，
  如实记录并改 P0-B 设计


---

## 六、【2026-08-21】结果：✅ **前置门通过，两种格式都可用**

| 臂 | DLC `quant_params` | 结果 |
|---|---|---|
| **C**（V 门，0.6.1 per-tensor） | `encoding=0`, `scale_offset` | `is_overridden=True`，scale = **0.001000000047** = 注入值 ✅ |
| **A**（0.6.1，4 个 encoding dict） | `encoding=1`, `axis_scale_offset` | **axis=0，4 个 scale 逐个吻合注入值** ✅ |
| **B**（2.0.0，`y_scale` + `axis`） | `encoding=1`, `axis_scale_offset` | **同样成功** ✅ |

⇒ ②**`--quantization_overrides` 可以注入 per-axis encoding，AIMET 0.6.1 与 schema 2.0.0 两种写法都行。**
⇒ ②**P0-B「Exact-real-FC standalone replay」的前置条件成立。**

### 六·1 三个必须记住的细节（否则下一个人会踩）

1. 🔴 **`axis` 会被转换器重映射**：我注入 `axis=1`（ONNX MatMul 权重 `[K,N]` 的 N 轴），
   DLC 里落成 **`axis=0`** —— 因为 MatMul 权重转成 FullyConnected 时被转置成 `[N,K]`。
   ⇒ **不得假设注入的 axis 原样保留，必须读回验证。**
2. 🔴 **`Processed N quantization encodings` 的 N 不是张量数**：只 override 一个张量也显示 **2**。
   ⇒ #45 说"必须检查这一行"仍然对（N=0 = 静默失效），
   但**判断某个具体张量是否被覆盖要看 DLC 里的 `is_overridden`**，不能靠这个计数。
3. **读回方法**：`qairt-dlc-to-json -i <dlc> -o <json>`（**是 `-i/-o`，不是 `--input_dlc/--output_path`**），
   然后读 `/graph/tensors/<名字>/quant_params`：
   `encoding=0` ⇒ per-tensor（`scale_offset`）；`encoding=1` ⇒ per-axis（`axis_scale_offset`）。

### 六·2 装置层面修掉的三个缺陷（都不是方向问题）

| 缺陷 | 根治 |
|---|---|
| `ModuleNotFoundError: No module named 'qti'` | `qairt_tool.py` 原版没设 `PYTHONPATH`（`qairt_run.py` 的文档串里写着要设，但靠人记）⇒ **改成自足**：自己装好 `sys.path`/`PATH`/`QNN_SDK_ROOT`/`SNPE_ROOT` |
| SDK 工具往 GBK 控制台打非法字符 ⇒ `UnicodeEncodeError` | `qairt_tool.py` 里统一 `reconfigure(encoding="utf-8")` |
| 我的 `Processed N` 解析器把日志的进程字段 `278` 当成 N | 锚定正则 `Processed\s+(\d+)\s+quantization encodings`，**不许"取行内最大数字"** |
