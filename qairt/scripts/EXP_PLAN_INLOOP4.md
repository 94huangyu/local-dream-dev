# EXP_PLAN_INLOOP4 —— 四段（part1a → part1b → part2a → part2b）全 per-row 在环出图

定稿时间：**2026-08-17 07:40**（**执行前定稿，判据不得事后修改**）
台账归属：**#54**（三段/四段全量 per-row 在环出图，此前⛔受阻于 #55/#57，现已解除）
前置：`EXP_PLAN_P2FIT` 手段 C（拆分 part2）已完成，两个 `.bin` 已验证可加载

---

## 0. 本实验回答的顶层问题

**四段全 per-row 之后，成图能到什么程度？**

这是 `STATUS_FOR_REVIEW` 第八节列出的**唯一未答问题**。
其余候选（#30 单 MatMul 探针等）都排在它之后，因为它们回答的是"若不达标下一步查哪"。

## 1. 事前已核实的事实（①实测，本轮亲自验证，不是转述文档）

### 1.1 契约（从 `.bin` 元数据 dump，不从文件名/ONNX 推断）

`qnn-context-binary-utility --context_binary <bin> --json_file`：

| 图 | graphName | 输入 | 输出 |
|---|---|---|---|
| `part2a_fixed.bin` | `part2a_fixed_fp32` | `adaln_input`[1,256] / `unified`[1,4128,3840] / `unified_freqs`[1,4128,64,2] / `unified_mask`[1,4128] BOOL_8 | `select`[1,4128,1,64] / `select_1`[1,4128,1,64] / `val_105`[1,1,1,4128] / `split_7_split_2`[1,1,3840] / `split_7_split_3`[1,1,3840] / `add_92`[1,4128,3840] |
| `part2b_fixed.bin` | `part2b_fixed_fp32` | 上述 6 个切口张量 + `adaln_input` | `latents`[1,16,128,128] |

🔴 **关键**：part2a 的输入集合与 baseline `part2.bin` 的输入集合**完全相同**
（`adaln_input`/`unified`/`unified_freqs`/`unified_mask`，dtype 逐个一致）
⇒ 在环脚本里 part2 的喂料代码**原样复用**，只需追加 part2b 一段。

### 1.2 设备上的 `.bin` 身份（按字节数与宿主产物逐一对上，不按文件名推断）

| 设备文件 | 字节数 | 宿主对应产物 | 配置 |
|---|---|---|---|
| `part1a.bin` | 2370604072 | `p0_experiments/perrow_p1a/ctx/part1a_perrow_ctx.SM8750.bin` | **per-row** |
| `part1b.bin` | 1472862256 | `p0_experiments/perrow/ctx/part1b_perrow_ctx.SM8750.bin` | **per-row** |
| `part2a_fixed.bin` | 1445077032 | `p0_experiments/p2split/ctx/part2a_fixed_ctx.SM8750.bin` | **per-row** |
| `part2b_fixed.bin` | 1495223120 | `p0_experiments/p2split/ctx/part2b_fixed_ctx.SM8750.bin` | **per-row** |
| （对照，本实验不用）`part2.bin` | 2928735488 | `dlc_pipeline/transformer_part2/..._ctx_sm8750.SM8750.bin` | baseline |

### 1.3 度量的操作化已用已知样本验证（约束 8）

用 `scripts/metrics_unified.py`，参考 = `p0_experiments/vs_fp32/latents_fp32_s0.raw`：

| 已记录值 | 本轮复算 | 一致 |
|---|---|---|
| 三段（1a/1b per-row + part2 基线）**43.41%** | `htp_inloop/s0/latents_dev.raw` → **43.4094%** | ✅ |
| part2 拆分 per-row 单段隔离 **28.5156%** | `p2split/latents_split_perrow_htp.raw` → **28.5156%** | ✅ |

能量集中度：前 1% 元素占 `||a||²` 的 **8.15%** ⇒ **两口径均可**，
按约束 7 允许用 `E_all` 下结论（这正是 `unified` 不允许而本张量允许的原因）。

## 2. 预期结果（事前写死，用于事后判断"是否真的验证了猜想"）

②推论（**未验证的外推**）：级联相对隔离的附加量在基线上是
`43.41 − 41.24 = 2.17 pp`。若该附加量在 per-row 下不变，
则四段级联 `E_all ≈ 28.52 + 2.17 = **30.7%**`。

⚠️ 这个外推有两个已知不确定性，**不得当成预测**：
① 四段比三段多一道切口量化边界（只会更差）；
② 28.52% 是用**基线上游输出**做输入测的，级联时 part2a 吃的是 HTP part1b 的 `unified`。

⇒ **落在 28%~34% 属于"符合预期"，不构成任何新发现**；显著偏离才需要解释。

## 3. 实现路径

1. 改 `scripts/htp_inloop_pipeline.py`：新增环境变量 `INLOOP_P2=split`
   - `split` ⇒ `part2a_fixed` → `part2b_fixed` 两段；缺省 ⇒ 保持原三段（向后兼容）
   - `WORK` 目录随 `INLOOP_TAG` 变化 ⇒ **不覆盖** `htp_inloop/s0/latents_dev.raw`
     （那是 43.41% 的唯一副本；PNG 已在 15.17.6 差点被覆盖过一次，这里是同类风险）
   - 输出 PNG 走既有 `INLOOP_TAG` 机制
2. `preflight.py --plan scripts/EXP_PLAN_INLOOP4.md --peak-gb 8 --device --device-peak-gb 3.0`
3. 前台执行（约束 9·补：不用 `nohup &`，使"通知到达 == 任务真结束"）
4. `metrics_unified.py` 出数；**亲自 Read 生成的 PNG**

**不改动的东西**（保证与 43.41% 那次单变量可比）：分词、FP32 文本编码、prompt、
seed=42、8 步调度器、FP32 VAE、part1a/part1b 的 `.bin`。

## 4. 有效性门 V（先于判据 1/2 判读；任一不过 ⇒ **执行方法错误**，不得据此对 per-row 下任何结论）

- **V1**：四段每段日志出现 `Finished Executing Graphs`，且无 `Failed to find available PD`
- **V2**：`latents` 元素数 == 262144（脚本内既有 assert，防 `snpe/qnn-net-run` 静默缩批）
- **V3**：新 run 的 `s0/latents.raw` 与既有 `htp_inloop/s0/latents.raw` **md5 相同**
  ⇒ 输入逐字节一致，与 43.41% 的对照成立单变量
- **V4**：part2a 的 6 个切口输出文件字节数 == float32 期望值
  （`add_92` 63406080 / `select`,`select_1` 1056768 / `split_7_split_2`,`split_7_split_3` 15360 / `val_105` 16512）
  —— 约束 3：输出字节数是抓静默缩批的唯一防线，切口是新引入的、从未在环上跑过

## 5. 判据（**事后不得修改**）

### 判据 1（数值，主判据）：step-0 噪声预测 `E_all` 对 FP32 原点

参考 = `p0_experiments/vs_fp32/latents_fp32_s0.raw`；口径 = `metrics_unified.py`。
对照刻度：**15.99% 成图完好**／**47.07% 色块**／**当前 43.41%**。

| 分支 | 条件 | 判读 |
|---|---|---|
| **1-A** | `E_all ≤ 15.99%` | 达到成图门槛 |
| **1-B** | `15.99% < E_all ≤ 34.0%` | 显著推进，符合第 2 节预期模型 |
| **1-C** | `34.0% < E_all < 43.41%` | 有推进但**低于**预期 ⇒ 拆分边界或级联放大吃掉了收益 |
| **1-D** | `E_all ≥ 43.41%` | 无推进或倒退 |

同时报告 `E_bulk` / `cos_bulk` / `slope`（能量集中度 8.15%，两口径均可）。
`cos_bulk < 0.3` ⇒ 按约束 7 退出定量比较，只能标记"已毁"。

### 判据 2（成图，必须**亲自打开 PNG 看**，约束 1）

三个档位**事前定义**，且锚点是**本轮亲自看过**的三张图，不凭记忆、不用形容词自由裁量：

| 档 | 定义 | 锚点（已亲自查看） |
|---|---|---|
| **图-A** | 能辨认出**猫**：有头部轮廓 + 耳朵 + 眼/口鼻中至少两项 | `zimage_fp32_full_pipeline.png`（FP32 好图）为上界 |
| **图-B** | 有全局构图与可辨识物体轮廓，但**没有猫** | `htp_inloop_transformer_perrow2of3.png`（43.41%）：上下 1/3 处横贯分界、中上方一块深色带块状伪影的斜向团块、**无任何可辨认物体**、通体马赛克方块纹理 |
| **图-C** | 纯渐变、零结构 | `htp_inloop_transformer_BASELINE_0815.png`（47.07%）：橙/奶油上下渐变，无任何边缘或形体 |

判"图-B"时必须**同时说明相对 43.41% 那张是变好、持平还是变差**，并指出具体依据
（新出现/消失的边界、形体、伪影），不得只写"略好"。

### 判据 3（一致性检查，用于发现"尺子失效"）

若 判据1 与 判据2 方向相反（如 `E_all` 明显下降但图反而更无结构），
⇒ **标定尺在该区间失去分辨力**，这本身是必须记录的重要结论（同 #55 被推翻的那次教训），
此时**不得**用 `E_all` 单独支撑任何"推进/倒退"的说法。

## 6. 决策树（**事前锁定**，不等结果再想；约束 9 第 3 条）

| 结果 | 下一步 | 依据 |
|---|---|---|
| 1-A + 图-A | 固化配方 → 真机级别验证 → 测生图时间 | 目标达成 |
| 1-A/1-B + 图-B | **#30 单 MatMul 探针**（约 30 min，从未做过） | 量化配置类手段已把 part2 从 41.24 打到 28.52 仍不出图 ⇒ 先确认 HTP 是否存在**硬性精度地板**，若有则所有量化配置工作都有天花板，须转 #32（SDK 2.28 vs 2.48） |
| 1-C | 先补做**"拆分 + per-tensor"对照**（#55 划界里明确未做的那次） | 1-C 意味着收益被吃掉，必须先分离"拆分本身的代价"与"级联放大"，否则归因不成立 |
| 1-D | 停止量化配置方向，**先查执行正确性**（V 门虽过但仍需查切口张量数值），再转 #30 | 与隔离实测 28.52% 矛盾 ⇒ 优先怀疑级联链路而非配方 |
| 判据 3 触发 | 暂停一切基于 `E_all` 的决策，先重新标定尺子 | 约束 7 |

## 7. 成本与设备纪律

- 设备：四段各自独立 `qnn-net-run` 进程，峰值 = 单段最大 context ≈ **2.4 GB**（part1a），
  低于三段那次（part2 3.5 GB）⇒ **不是**压死手机的那类任务
- 预计 8 步 × 4 段，约 **25~35 分钟**，全程占用设备
- 宿主：onnxruntime 跑 text_encoder ×4 + vae_decoder，峰值约 **8 GB**
- 跑前必过 `preflight.py`（exit 1 则禁止执行）
