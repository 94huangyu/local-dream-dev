# EXP_PLAN_HTP_VS_CPU —— 直接检验假设 C：HTP 跑 `.bin` 与 CPU 参考跑 `.dlc` 是否数值等价

定稿时间：2026-08-15（执行前定稿，判据不得事后修改）
主线归属：A（排查设备失败）→ 假设树里**唯一未排除**的节点 C

## 0. 为什么是这个实验

假设树现状：A1（transformer 量化）已否定、A3（VAE 量化）已否定、
B（C++ ≠ Python）算法 8 项对齐且本轮又查了 mask 喂法未发现新 bug、
`node_Where_105` 编译报错已于 14.22/14.23 证伪为红鲱鱼。

**剩下的所有验证都有一个共同前提：它们全跑在 `.dlc` + SNPE CPU 参考上，
而设备跑的是 `.bin` + HTP。这两者是否数值等价，从未验证。**
这是唯一能直接证否 C 的实验，也是当前唯一还没被关掉的门。

## 1. 预期结果（假设）

**H-C**：HTP 执行 `.bin` 与 CPU 参考执行 `.dlc`，在相同输入下输出**不等价**，
且差异大到足以解释"橙色无定形色块"。

零假设 H0：两者等价（逐张量相对 L2 < 1%），⇒ C 被否定，
⇒ 假设树**全部节点被排除**，必须回到顶层重新提假设（这也是有价值的结果）。

## 2. 实现目标

对 `transformer_part1a`，在**完全相同的输入**下拿到两侧的 8 个输出张量，逐张量比对。

选 part1a 的理由：它是流水线的**第一段**，输入完全由宿主给定（不依赖上游图的输出），
因此两侧输入可以做到逐字节相同——这是"同输入"这个前提唯一能被严格保证的一段。
若从 part1b/part2 开始，输入本身就来自上游，差异会混在一起无法归因。

## 3. 实现路径

设备上**已有**全部所需产物，无需推送 2.37 GB 的 context binary：
`files/models/ZIMAGE/models/transformer_part1a_ctx_sm8750.SM8750.bin`（app 实际加载的同一个文件）
+ `qnn_runtime_libs/aarch64-android/` 全套 HTP V79 库。

1. 推送 `qnn-net-run`（aarch64-android，5.3 MB）与 `libQnnHtpV79Skel.so` 到 `/data/local/tmp`。
2. 推送 testB step0 的 4 个输入 `.raw`（float32，与 CPU 参考**逐字节相同**的那一份）。
3. 在设备上执行：
   `qnn-net-run --retrieve_context <part1a .bin> --backend libQnnHtp.so --input_list ... --output_dir ...`
   **不加** `--use_native_input_files` / `--use_native_output_files`
   ⇒ qnn-net-run 按 float32 读输入、内部量化；输出反量化为 float32 写出。
   ⇒ 与 SNPE 侧口径一致（`../docs/EXECUTION_MODEL.md` 规则 1/2：SNPE 侧也是 float32 进、float32 出）。
4. 拉回输出，与 `testB/s0_transformer_part1a/out/Result_0/*.raw` 逐张量比对。
5. 指标：余弦相似度、相对 L2（`||a-b||/||a||`）、最大绝对差、两侧各自的 min/max/mean/std。

> **注意（`../docs/EXECUTION_MODEL.md` 规则 2·补）**：宿主字节数在两条路径上**可以不同**
> （`cap_pad_mask`：SNPE 128 B float32 vs QNN 运行时 32 B BOOL_8）。
> 但本实验走 qnn-net-run 的**非 native 模式**，它按 float32 读文件再自行转换，
> 所以喂 128 B 的 float32 文件是正确的。若 qnn-net-run 报字节数不符，
> 属**执行方法错误**（见第 5 节），按其报出的期望值调整，不改判据。

## 4. 验收标准（判据，事后不得修改）

**判据 0（执行有效性，先过这一关）**
- `qnn-net-run` 返回成功，且产出全部 8 个输出张量；
- 每个输出的字节数 = 元素数 × 4（float32）且与 CPU 参考侧对应文件**字节数完全相同**。
- 任一条不满足 ⇒ 本次运行**无效**，结果不得解读（防的是"静默缩批"那类陷阱）。

**判据 1（等价性判定）**——对 8 个输出张量分别计算相对 L2：
- **全部** < 1%（且余弦 > 0.9999）⇒ **H-C 被否定**，HTP 与 CPU 参考等价。
- **存在**任一 > 15.988% ⇒ **H-C 成立**且差异达到"已知会/可能致命"的量级。
  （15.988% 是 Test B 实测标定点：噪声预测相对 L2 达此值时成图**仍然完好**，
  所以它是"无害"的**上界**，超过它才谈得上可能有害。）
- 全部落在 1%～15.988% 之间 ⇒ **差异存在但落在已知无害区间**，
  **不得**据此宣称找到根因；只能记为"C 部分成立，但不足以解释色块"，
  并把实验推进到 part1b/part2（差异会逐段累积）。

**判据 2（定位）**：若 H-C 成立，给出**第一个**显著偏离的输出张量名及其指标，
作为下一步的抓手。

## 5. 成功 / 失败的判断依据

- **成功**：拿到 8 组可复算的数字，并据判据 1 给出一个明确分支结论。
- **失败·执行方法错误**：qnn-net-run 跑不起来（缺库、权限、字节数不符、HTP 初始化失败）。
  ⇒ 修执行方法后重跑，**判据不变**。此类失败**不构成**对 H-C 的任何证据。
- **失败·方向错误**：跑通了，但两侧输出**逐字节完全相同**（相对 L2 恒为 0）。
  这反而说明 qnn-net-run 可能根本没走 HTP（回退到 CPU）——
  必须先用 `--log_level info` 确认后端确实是 HTP，否则结论无效。

> **反向陷阱（必须防）**：若结果显示"完全一致"，不能立刻高兴地否定 C。
> 要先证明这次运行**真的在 HTP 上跑**（看日志里的 backend/skel 加载、
> 以及 `Unsupported HTP Arch` 之类的告警是否出现）。

## 6. 深度预算与放弃条件

- 预算：part1a 一轮。若判据 0 反复不过（≥3 次执行方法失败），停下来回主线重新评估。
- 若 H-C 被否定（全部 < 1%）：**假设树全部节点已排除**，
  这是一个重大结论，必须回到顶层重新提假设，**不得**在 part1b/part2 上继续机械重复。

## 7. 对主线 B 的价值（`../docs/QNN_CONVERSION_GUIDE.md`）

无论结论如何，"如何在真机上用 `qnn-net-run --retrieve_context` 对 `.bin` 做逐张量验证"
本身就是**级别 3 验证**缺失的那一环（指南 §五目前只有级别 1/2 和一句"需要手机"）。
跑通后把命令模板与坑写进指南。

---

# 执行结果（2026-08-15，判据未做任何修改）

设备：PKX110 / **SM8750**（与 `.bin` 目标一致），USB adb。
比对脚本：`scripts/compare_htp_vs_cpu.py`。产物：`D:\ZImage_Work\p0_experiments\htp_vs_cpu\`。

## 同源性（执行前先钉死）

| 项 | 值 |
|---|---|
| app 实际加载的 `.bin` md5 | `773502d86641c86ade231a0bd16ffe5e` |
| 主机 `dlc_pipeline\transformer_part1a\..._sm8750.SM8750.bin` md5 | **同上，逐字节相同** |
| 推到设备 `/data/local/tmp/htpcmp/part1a.bin` 后 md5 | **同上** |

⇒ ①实测：本实验跑的 `.bin` 与 **app 真机跑的是同一个文件**。
⇒ ②推论（**有保留**）：该 `.bin` 由 testB 所用的 `transformer_part1a_quantized.dlc` 编出——
   目录下只有这一份量化 dlc、testB 的 `DLC_DIR` 也指向该目录，但构建日志**没记命令行**，
   拿不到文书级证据。**本次结果不依赖这条**：见下，多数张量逐位相同，
   不同源的两个模型不可能给出逐位相同的输出。

## 判据 0（执行有效性）：✅ 通过

8 个可比张量全部产出，且**字节数与 CPU 参考侧逐个完全相同**
（`add_138` 63,406,080；`unified_freqs` 2,113,536；`unified_mask` 16,512；…）。
无静默缩批。

## 判据 1（等价性）：落在【1% ~ 15.988%】中间带

| 张量 | 元素数 | 相对 L2 | 余弦 | 最大绝对差 |
|---|---|---|---|---|
| `select_45` | 264,192 | **0.0000%** | 1.000000 | **0**（逐位相同） |
| `select_46` | 264,192 | **0.0000%** | 1.000000 | **0**（逐位相同） |
| `unified_freqs` | 528,384 | **0.0000%** | 1.000000 | **0**（逐位相同） |
| `unified_mask` | 4,128 | **0.0000%** | 1.000000 | **0**（逐位相同） |
| `adaln_input` | 256 | 0.0269% | 1.000000 | 4.0e-4 |
| `add_131` | 3,840 | 0.0297% | 1.000000 | 6.5e-4 |
| `tanh_19` | 3,840 | 0.0301% | 1.000000 | 3.5e-4 |
| **`add_138`** | **15,851,520** | **2.3492%** | **0.999729** | **144.82** |

**按事前判据：不满足"全部 < 1%"⇒ H-C 未被否定；
也不满足"存在 > 15.988%"⇒ 差异未达可能致命量级。
落在中间带 ⇒【差异存在，但在已知无害区间内，不得据此宣称找到根因】。**

## 第 5 节"反向陷阱"检查：本次确实跑在 HTP 上

- 4 个张量逐位相同曾触发"是不是回退到 CPU 了"的怀疑，但 `add_138` 有 2.35% 的实打实差异
  ⇒ 计算路径与 CPU 参考不同。
- `--backend libQnnHtp.so`，且加载的是**为 SM8750 编译的 context binary**，CPU 后端根本无法消费。
- `execution_metadata.yaml` 确认图名 `transformer_part1a_fp32`、1 次推理完成。

⇒ 结论有效。

## 判据 2（定位）：差异高度集中，不是弥散噪声

`add_138` 的 |差| 分布（15,851,520 个元素）：

```
p50 = 0.0159   p99 = 0.0973   p99.9 = 0.293   p99.99 = 0.894   p99.999 = 1.63   max = 144.82
|差|>1 : 1,073 个(0.0068%)   >5 : 22 个   >50 : 14 个   >100 : 2 个
```

⇒ ①实测：**99.99% 的元素差异 < 0.9，但存在约 14~22 个数量级完全失控的离群点**
（最大 144.8，而该张量 CPU 侧 std 仅 3.44，即 **42σ**）。
这**不是**量化噪声的形态；15.988% 那个标定点标定的是**弥散**误差，
**不覆盖这种集中离群**，所以不能直接拿它给这些离群点背书。

**空间分布（②由数据推出）**：按 token 维（4128 = 4096 图像 + 32 caption）看，
每行平均差最大的 10 行全部落在 caption 段：
`4110, 4112, 4111, 4127, 4118, 4117, 4126, 4116, 4122, 4123`（即 caption 槽位 14~31）。
按通道维看，ch 85 的平均差 0.964 是中位数的 53.6 倍。

⇒ **HTP 与 CPU 参考的分歧集中在 caption（文本条件）那 32 个 token 上**，
   而不是 4096 个图像 token 上。这是下一步的抓手。
   **但注意**：每行平均差最大也只有 0.241（中位 0.0204），仅 12 倍，
   所以"集中在 caption"是**趋势**，不是"caption 被毁掉"。**不得夸大。**

## 顺带确证（把一条②推论升级为①实测）

`execution_metadata.yaml` 直接给出设备侧 QNN 张量声明：

```
tensor_name: cap_pad_mask   datatype: QNN_DATATYPE_BOOL_8      dimensions: [1,32]
tensor_name: latents        datatype: QNN_DATATYPE_UFIXED_POINT_16  dimensions: [1,16,128,128]
```

⇒ `../docs/EXECUTION_MODEL.md` 规则 2·补里"`.bin` 的 `cap_pad_mask` client buffer 就是 32 字节 BOOL_8"
   此前是从"设备没抛异常"**反推**的②推论，现在有了**设备侧直接证据**，升级为①实测。
   C++ 那个 32 字节写法**确认正确**。

## 本实验的成败判定（对照第 5 节）

- **成功**：拿到 8 组可复算数字，并按事前判据给出了明确分支结论。
- 主线状态：**H-C 既没被否定，也没被证实为致命**。按第 4 节中间带分支的规定，
  下一步是**把同样的方法推进到 part1b / part2**（差异会逐段累积），
  而不是就此收工或宣布找到根因。

## 可复用命令模板（待同步进 `../docs/QNN_CONVERSION_GUIDE.md` 级别 3）

```bash
# 设备上：/data/local/tmp/htpcmp 放 qnn-net-run + libQnnHtp.so + libQnnHtpV79Stub.so
#          + libQnnSystem.so + libQnnHtpPrepare.so + libQnnHtpNetRunExtensions.so
#          + libQnnHtpV79Skel.so（来自 lib/hexagon-v79/unsigned/）
export LD_LIBRARY_PATH=/data/local/tmp/htpcmp
export ADSP_LIBRARY_PATH=/data/local/tmp/htpcmp
./qnn-net-run --retrieve_context part1a.bin --backend libQnnHtp.so \
              --input_list in/list.txt --output_dir out --log_level info
```

关键点：
1. **不要**加 `--use_native_input_files` / `--use_native_output_files`。
   非 native 模式下 qnn-net-run 按 **float32** 读输入、按 **float32** 写输出，
   与 SNPE 侧口径一致，两边才能直接相减。（实测：喂 128 字节 float32 的
   `cap_pad_mask` 给一个 `BOOL_8[1,32]` 张量，工具自行转换，正常运行。）
2. `--retrieve_context` 用于 `.bin`；`--model` / `--dlc_path` 是另外两条路，互斥。
3. app 私有目录 `/data/data/<pkg>/` 本身是 `drwx------`，只 chmod 子目录**没用**。
   正确做法是把主机上**同 md5** 的那份 `.bin` 推到 `/data/local/tmp`，
   既绕开权限又拿到来源确定性。
