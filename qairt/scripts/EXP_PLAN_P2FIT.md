# EXP_PLAN_P2FIT —— 把 per-row 的 part2 context 压进 unsigned PD 的容量上限

定稿时间：2026-08-16 16:20（**执行前定稿，判据不得事后修改**）
台账归属：**#55 / #57**
前置：#57 已用 FARF 日志拿到精确根因

## 0. 事实（①实测，不是推测）

DSP 侧原文：
```
QnnDsp <E> Failed to find available PD for contextId 1 on deviceId 0 coreId 0
           with context size estimate 3652345600
QnnDsp <E> context create from binary failed on contextId 1, err = 1002
```
- **不是内存不足**：同一时刻 `lowmemorykiller: device has enough memory 8047228Kib`
- 是 **unsigned PD 装不下 3.652 GB 的 context**
- baseline part2（能加载）按元数据推算约 **3.62 GB** ⇒ **上限在这 30 MB 之间**

per-row 相对 baseline 的增量（元数据实测）：

| 字段 | baseline | per-row | 增量 |
|---|---|---|---|
| contextBlobSize | 2928.7 MB | 2934.3 MB | +5.5 |
| **spillFillBufferSize** | **242.3 MB** | **258.7 MB** | **+16.4** |
| opDataSize | 334.9 MB | 342.5 MB | +7.6 |
| 合计 | | | **≈ +30 MB** |

**spill-fill 占增量的一半以上 ⇒ 优先冲它。**

## 0.1 为什么值得解（代价门，已完成）

`perrow_feasibility.py 9999 part2` 实测（121 个 FC 权重张量）：
per-tensor 主体表示误差中位 **4.2002%** → per-row **1.1168%**（**0.2781 倍**），
与 part1b（3.7279% → 1.0244%，0.2785 倍）**几乎完全一致**，
且 part2 的 FC 权重张量数是 part1b 的 **2 倍**。
⇒ 先前"part2 注意力 MatMul 无权重、收益小"的担心**不成立**，per-row 在 part2 上潜力同量级。

## 1. 手段（按成本排序，逐条试，一次只试一条）

| 序 | 手段 | 成本 | 机制 |
|---|---|---|---|
| **A** | `--hvx_threads_override <N>` 重建 per-row part2 | 42 min PC + 1 min 手机 | 线程数 ↓ ⇒ 并发工作集 ↓ ⇒ spill-fill 缓冲可能 ↓。**不动模型** |
| B | 选择性 per-row（只点名最大的一批 FC，用 `--quantization_overrides` 灌 per-axis encoding） | 中 | 增量与 per-axis 张量数大致成比例 |
| C | 拆分 part2 为两段 | 5 h+ | 必定可行，但牺牲基线可比性 |

**已排除（各有实测/文档依据，不再重试）**：`vtcm_override`（已是 8 MB 上限）、
`optimization_level`（已是 0，O=3 只会更大）、DLBC（压 inputs 非压 context）、
Sparse Weights Compression（本模型权重稠密）、signed PD（需签名 DSP 镜像，没有）。

## 2. 判据（事后不得修改）

**G1（尺寸门，宿主侧，免费）**：重建后用 `qnn-context-binary-utility` 读元数据，
要求 `spillFillBufferSize + contextBlobSize + opDataSize` 的总和
**≤ baseline 的对应总和（3506.0 MB）**。
- 不满足 ⇒ 该手段无效，**不推设备**，直接转下一条手段。

**G2（加载门，设备侧，10 秒）**：G1 过了才推设备，`qnn-net-run --retrieve_context` 必须
`Finished Executing Graphs`，且 FARF 日志里**不出现** `Failed to find available PD`。

**G3（数值门）**：加载成功后，用与 part1b 同一套口径测 part2 输出 `latents`
对 FP32 原点（`vs_fp32/latents_fp32_s0.raw`）的误差。
比较对象 = **baseline part2 隔离误差 E_all 41.24% / 主体余弦 0.9089**。
- `Δcos ≥ +0.02` ⇒ per-row 对 part2 有效，推进三段在环
- `|Δcos| < 0.005` ⇒ per-row 对 part2 无效（与 part1b 不同），**这本身是重要结论**
- `Δcos ≤ −0.005` ⇒ 变差，记录

⚠️ **手段 A 若改变了 HVX 线程数，它本身可能影响数值** ⇒ G3 若出现变化，
**不能直接归因于 per-row**，必须补一个"baseline + 同样线程数"的对照才能分离。
（先记在这里，避免事后才想起来——这正是 #53 那类归因错误的预防。）

## 3. 设备纪律

- 本实验设备侧只做"加载 + 单次推理"，context 2.93 GB，峰值约 3 GB，
  与既有在环同量级，**不是**把手机压死的那类任务（建图 RSS 5.76~7.7 GB）
- 跑前必须过 `preflight.py --device --device-peak-gb 3.5`

---

# 执行结果（2026-08-16 夜 ~ 08-17 凌晨）

## 手段 A：`--hvx_threads_override 2` —— ❌ G1 尺寸门未过

| 配置 | blob | spill | opData | 合计 |
|---|---|---|---|---|
| baseline（可加载） | 2928.7 | 242.3 | 334.9 | **3505.9** |
| per-row（失败） | 2934.3 | 258.7 | 342.5 | 3535.4 |
| per-row + hvx=2 | 2936.1 | **258.0** | 346.2 | **3540.4** |

线程数对 spill-fill 几乎无作用（−0.7 MB），blob/opData 反而各涨。**按判据不推设备**，转下一手段。

## 手段 external buffer（ChatGPT 建议，独立于原三手段）—— ❌ 已实测关闭

自建最小 probe（`scripts/extbuf_probe/extbuf_probe.cpp`，NDK 交叉编译，仅加载不推理）：

| 档 | createFromBinary | memRegister | contextFinalize |
|---|---|---|---|
| T0 普通 | ❌ 0x3ea | — | — |
| T1 DEFER + spill-fill | ✅ | ✅ 260.0 MB | ❌ 0x3ea |
| T2 DEFER + weights | ✅ | ✅ 2730.5 MB | ❌ 0x3ea |
| T3 DEFER + 两者 | ✅ | ✅ 两个都成功 | ❌ 0x3ea |

**FARF 里 `context size estimate 3652345600` 在外置前后完全相同**
⇒ 外置 weights/spill-fill **不减少 PD 的容量估算**，该路径对本问题无效。

**①首次测得的两个数**（本项目此前从未有过）：
- `QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE` = **2,730,491,904（2730.5 MB）**
- `QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE` = **260,046,848（260.0 MB）**

**顺带确立的事实**：
1. `QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT` **确实能绕过创建时的 PD 分配**
   （同一份 2934 MB bin：普通模式 0x3ea，DEFER 模式成功拿到 handle），
   但真正的 PD 检查发生在 `contextFinalize`，绕不过去。
2. `qnn-net-run` 2.48 **无法使用** `context_configs/spill_fill_buffer|weights_buffer`：
   JSON schema 要 integer、下游解析器要 string，三种取值全部失败。
   设备端五个关键库与 SDK md5 逐一核对**完全一致** ⇒ **排除版本错配，是工具缺陷**。
3. ⚠️ **`rpcmem_alloc` 的 size 参数是 `int`**，2730 MB 会 32 位溢出导致分配失败；
   设备 `libcdsprpc.so` 导出了 64 位的 **`rpcmem_alloc2`**，必须用它。

## 手段 C：拆分 part2 —— 🔄 进行中，前置条件已全部满足

见 `EXP_PLAN_P2SPLIT` 段（下）。
