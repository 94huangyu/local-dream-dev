> # 🔴 已冻结的旧快照 —— **不是开工必读**
>
> 本文件是 **2026-08-17** 为一轮外部审阅生成的状态快照，**2026-08-21 冻结**。
> 冻结原因：它与 `MAINLINE.md` 内容重复，靠人工同步维护 ⇒ **必然过期**，事实上也确实过期了。
>
> **下面这些说法现在都是错的**：
> - "CPU 参考出清晰图、**设备 HTP 出橙色色块**" —— 设备早已能出正常图（#54/#71）
> - "**画质缺陷来自 VAE 导出**" —— 归因错误，真因是**反缩放做了两遍**（#80）
> - 一切"当前进度"数字 —— 见 `MAINLINE.md` §2
>
> **开工请只读 `MAINLINE.md`。** 需要给外部审阅者材料时，请**当场从 MAINLINE 生成新快照**，
> 不要复活本文件（重复维护正是它过期的原因）。

---

# Z-Image Turbo → 骁龙 HTP：主线状态（供外部 AI 审阅）

更新时间：**2026-08-17 08:40**。所有数字均为实测。

> 📖 **新线程读法**：本文件 + `MAINLINE.md` 即可开工（合计 ≈15k token）。
> **不要用 Read 读 `CLAUDE.md`**——它作为 project instructions 已自动注入系统提示，再读是纯浪费。
> 已关闭线索的完整依据在 `../docs/LEDGER_CLOSED.md`（按需回查，**不是开工必读**）。

---

## 一、顶层目标

Z-Image Turbo（DiT：MatMul + RmsNorm + 残差，**零 Convolution**）W8A16 量化后
部署到 **SM8750 / Hexagon v79** 的 HTP。
CPU 参考跑同一份量化模型出清晰图，**设备 HTP 出橙色色块**。目标：让 HTP 出正确的图。

工具链：QAIRT 2.48.0.260626。模型分段：part1a / part1b / part2（**现 part2 已再拆为 part2a/part2b**）。

## 二、唯一有刻度的度量

step-0 噪声预测相对 FP32 的相对 L2（FP32 原点已保存为 `vs_fp32/latents_fp32_s0.raw`，
用 Test B 的 CPU 参考复算验证为 **15.9882%**，与历史记录吻合）：

- **15.99% ⇒ 成图完好**
- **47.07% ⇒ 橙色色块**

⚠️ 中间张量 `unified` 的能量 **99.83% 集中在最大 1% 元素**，描述它必须用主体口径（|a| ≤ p99）。

## 三、当前进度

| 配置 | step-0 噪声预测 E_all |
|---|---|
| 目标 | **15.99%** |
| 三段全基线 | 47.07% |
| part1a+part1b per-row、part2 基线 | **43.41%** |

**part2 单段隔离误差**（对 FP32 原点，输入逐字节相同）：

| 配置 | E_all | 主体余弦 |
|---|---|---|
| 基线（per-tensor） | 41.24% | 0.9089 |
| **per-row（拆分后，2026-08-17 实测）** | **28.52%** | **0.9557** |
| CPU 参考（可达上界） | 15.99% | 0.9865 |

## 四、已确立的机制结论

1. **`--use_per_row_quantization` 是目前唯一有效的手段。**
   part1b：E_all 19.24% → 7.87%，主体余弦 0.6033 → 0.7986，slope 0.8622 → 1.0435。
   注意：`--use_per_channel_quantization` 帮助原文限定 *"convolution-based op weights"*，
   本模型零 Conv ⇒ **一直空转**。
2. **"细量化步长有收益"只在动态范围完整保留时成立。**
   per-row（每行自己的量程，范围不丢）⇒ 有收益；
   activation percentile（砍量程换分辨率）⇒ **净损害**（主体余弦 0.7986 → 0.4793；
   事后算出被钳掉的 0.026% 元素占 ||a||² 的 **98.75%**）。
3. **量化表示误差不是 HTP 误差的预测量**：同一改动 CPU 侧损失 0.0014、HTP 侧 0.0526（37.6 倍）。
   ⚠️ 但它可作**单向否决**用：表示层面都没有空间 ⇒ HTP 上不可能有收益（#66 即据此否决）。
3.5 🔴 **权重不是离群值主导的，激活才是**（#67，2026-08-17 实测）。
   权重裁掉 0.0520% 元素只丢 `‖W‖²` 的 **1.12%**；激活 percentile 钳掉 0.0261% 元素
   却丢掉 **98.75%** 能量。**这解释了为何 per-row 对权重有效、percentile 对激活是灾难。**
   连带：**8-bit 下 min-max 往往就是 MSE 最优**（#68）——高斯极值 ~3.7σ 已在 8-bit 最优裁剪
   ~3.9σ 附近 ⇒ **"收紧量程"类手段在 8-bit 上本来就没剩多少空间**（#9 与 #66 同机制）。
4. **权重对称量化更差**（0.6033 → 0.5507），**与官方 "recommended to always use symmetrical
   quantization of weights" 相反**。注意 per-row 在 HTP 上**强制对称**（OpDef 硬性要求），
   所以 granularity 与 symmetry 两个效应在 HTP 上无法完全拆开。

## 五、已关闭的路径（都有实测依据，勿重试）

| 路径 | 结论 |
|---|---|
| 权重对称量化（FC-only，via overrides） | ❌ 更差（0.6033→0.5507） |
| 激活 percentile p99.99 | ❌ 大幅更差（0.7986→0.4793） |
| `--restrict_quantization_steps` | ❌ 只对 A16W16 有效，本项目 A16W8 |
| 图优化级别 O=3 | ❌ 精度无变化（Δcos −0.0005） |
| `--vtcm_override` | ❌ 已是 v79 上限 8 MB |
| `--hvx_threads_override 2` | ❌ context 反而更大（3535→3540 MB） |
| DLBC / Sparse Weights Compression | ❌ 前者压 inputs 不缩 context；后者只对稀疏权重 |
| **external weights / spill-fill buffer** | ❌ 机制全通（DEFER + memRegister 均成功）但 **PD 估算一字节未降**（3652345600 前后相同） |
| signed PD | ❌ 需推送签名 DSP 镜像，没有 |
| A16W16 / A8W8 / 更多分段 | ❌ 内存或速度出局 |

## 六、容量硬约束（SM8750 / v79 / unsigned PD）

`QnnDsp <E> Failed to find available PD ... with context size estimate 3652345600` / `err = 1002`
（不是手机内存不足：同刻 `lowmemorykiller: device has enough memory 8047228Kib`）

| 配置 | blob | spill | opData | 合计 | 可加载 |
|---|---|---|---|---|---|
| part2 baseline | 2928.7 | 242.3 | 334.9 | **3505.9** | ✅ |
| part2 per-row 未拆 | 2934.3 | 258.7 | 342.5 | **3535.4** | ❌ |
| **part2a per-row** | 1445.1 | 258.1 | 183.9 | **1887.0** | ✅ |
| **part2b per-row** | 1495.2 | 255.2 | 176.3 | **1926.7** | ✅ |

⇒ **红线在合计 3506~3535 MB 之间。单 context 建议控制在 3.3 GB 以内。**

## 七、part2 拆分（已完成）

`transformer_part2_fixed.onnx` →（**DCE：去掉 84 个死节点**）→ 反向可达性切分
→ 数值等价验证 **逐位相同**（相对 L2 = 0.00000000%）→ 两段量化（per-row，40 样本，零 OOM）
→ 两段建图 → **设备加载成功**。

切口 6 个张量共 65.6 MB：`add_92`(1,4128,3840) / `select` / `select_1`(1,4128,1,64) /
`split_7_split_2` / `split_7_split_3`(1,1,3840) / `val_105`(1,1,1,4128)。

**关键陷阱（否则量化必 OOM）**：不做 DCE 直接切，84 个死节点（IsNaN 15/Where 15/…）
会把 30 个 `[1,30,4128,4128]`（单个 1.9044 GiB）张量拽过切口，
切割集从 65.6 MB 膨胀到 16.4 GB，量化内存线性涨到 134 GB 后 OOM
（增长步长实测 1.90~1.91 GB = 单个该张量）。
转换器对**完整图**会自己 DCE（生产 DLC 里 IsNaN=0），**一切分就"复活"**。
已查：part1a / part1b 死节点为 **0**。

## 八·新、🎉 顶层问题已有答案：**设备第一次生成出猫**（2026-08-17 21:56）

四段全 per-row（part1a → part1b → **part2a → part2b**）在环出图，**HTP 生成了清晰的橘猫**
（头/耳/绿眼/粉鼻/胡须/虎斑纹/木桌木纹）。**唯一变量** = part2 基线 → part2a/2b per-row。

**代码改动已排除**：今天的代码重跑三段对照，E_all 43.4094% 与 08-16 完全一致、
PNG **逐位相同**（像素差 0.0000）。四道 V 门全过（V3 输入 md5 逐一相同）。

**仍未解决**：HTP 图背景空白（FP32 有完整房间场景），毛发质感粗、虎斑纹弱
⇒ **"能不能出图"已解决，"画质对齐 FP32"未解决**，主线随之前移。

## 八·补、🔴 **本项目的三把标量尺子在当前区间全部与成图质量【反相关】**

| 指标 | 三段（马赛克色团） | 四段（清晰的猫） | 方向 |
|---|---|---|---|
| step-0 噪声预测 `E_all` | 43.41% | **44.80%** | ❌ 更差 |
| 对 FP32 平均像素差 | 41.37 | **44.52** | ❌ 更差 |
| PSNR | 12.46 dB | **11.60 dB** | ❌ 更差 |

**机制**：HTP 出的是**另一个场景**（背景空白）而 FP32 是完整房间 ⇒
逐像素/L2 惩罚"画对了但内容不同"；而色块的平滑渐变反而在 L2 上更接近照片均值。

🔴 **后果**：**暂停一切基于这三个量的"推进/倒退/放弃"决策**，直到建立结构/感知类判别方式。
所有历史上用 `E_all` 做过的判断都要复查（尤其 #56，它据此判定"per-row 几乎没有推进"）。

⚠️ 原"15.99% ⇒ 成图完好 / 47.07% ⇒ 色块"这个标定**只有两个点，且分别来自不同流水线**
（Test B 全 CPU 参考 vs 全 HTP 基线）——metric 与 configuration 一起变了，
**把共变当成了因果**。第三个点（44.80% ⇒ 清晰的猫）把该单调关系打断。

## 八·旧、（历史记录）四段在环曾因设备离线中断（2026-08-17 08:00）

**四段全 per-row 之后，成图能到什么程度？**

代码与方案**已全部就绪**：`scripts/EXP_PLAN_INLOOP4.md` 定稿，
`htp_inloop_pipeline.py` 已扩到四段（part1a → part1b → **part2a → part2b**），
preflight 通过。07:41 启动，推进到 part1a 时**用户带手机上班拔了线**，设备侧全部停摆。

按方案 V1 有效性门：**执行方法错误，不得据此对 per-row 或四段级联下任何结论。**

**已拿到的有效结果**：V3 门通过 —— step-0 的四个输入
（`latents`/`timestep`/`caption`/`cap_pad_mask`）与 43.41% 那次 **md5 逐一相同**
⇒ 四段 vs 三段的对照**单变量成立**。

设备回来后：
```bash
python scripts/matmul_probe_run.py && python scripts/matmul_probe_analyze.py   # ~1 min
INLOOP_TAG=perrow4seg INLOOP_P2=split python scripts/htp_inloop_pipeline.py    # ~30 min
```

## 九、排队中的候选

1. ✅ **#30 单 MatMul 探针 —— 已完成，两假设均被否定**（2026-08-17）。
   `E_htp/E_fxp` = **1.00 / 1.01 / 1.00**（K=64/1024/3840）⇒ **HTP 在单算子上是正确的**，
   **不存在硬性精度地板** ⇒ 🟢 **量化配置这条路没有天花板**。
   ⚠️ 探针输入是**良态**的，**不得**外推到真实模型（`unified` 99.83% 能量集中在前 1%）。
   对照 #39：真实模型里同为 FC、同为 K=3840，HTP 却差 20~230 倍
   ⇒ 差异来自**输入条件数**或**图层面效应（#33/#34）**，二者尚未分离。
2. ❌ **#66 权重校准 `mse/sqnr` —— 已做，代价门否决**（2026-08-17，纯宿主，未占设备）。
   part2 全部 121 个 FC 权重张量：per-row min-max 主体表示误差中位 **1.1168%**，
   mse 的**可达上界** **1.0990%** ⇒ 改善仅 **1.6%**，判据线 5% ⇒ **不许上机**。
   单向否决逻辑：上界都没收益，真实实现更不可能有。
3. **partial per-row（block 二分）**——现已无必要（拆分后余量 1.6 GB）。
4. 🔻 **SDK 版本 A/B —— 已降级（2026-08-17）**。
   原依据"local-dream 用 2.28 跑通 SD1.5/SDXL"是**相关性不是因果**：
   local-dream 成功的关键可能是**它的模型成熟**（SD1.5/SDXL 被无数人转过），
   而本项目的模型**从来没人转过** —— SDK 版本与模型成熟度**完全混淆**。
   ⇒ 不值得为它下载 SDK。若将来要做，**正确设计是用 #30 的极小探针做廉价 A/B**
   （同一模型、只变 SDK，几分钟），**不要**重做整个模型（数小时）。

## 十、环境

- SDK：QAIRT 2.48.0.260626｜设备：SM8750(soc_model 69)/v79/VTCM 8 MB/15.4 GB RAM
- 宿主：Windows 11，23.7 GB RAM，提交上限 121 GB
- 量化：`--act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
  --act_quantizer_calibration min-max --param_quantizer_calibration min-max
  --use_per_row_quantization`｜校准 40 样本
- ⚠️ 工具坑：`rpcmem_alloc` size 是 int（>2 GB 溢出，用 `rpcmem_alloc2`）；
  `qnn-net-run` 2.48 无法使用 `spill_fill_buffer`/`weights_buffer`（schema 与解析器类型冲突，
  已核对库 md5 排除版本错配）；`onnx.utils.extract_model` 会内联外部权重撞 protobuf 2 GB 上限。
