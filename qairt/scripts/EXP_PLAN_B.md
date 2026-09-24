# 实验 B 方案：C++ 实现 vs 已验证的 Python 参考（执行前定稿）

## 1. 为什么这个比对是有效的

`scripts/zimage_fp32_pipeline.py` 现在是一份**经过验证的参考实现**：
- 它生成完美的猫（`zimage_fp32_full_pipeline.png`）
- 把 transformer 换成量化版后**仍然**生成完美的猫（Test B，PSNR 23.89 dB）
- 把 VAE 换成量化版后**仍然**几乎无损（A3，PSNR 51.11 dB）

而设备（C++ 实现）生成橙色无定形色块。**量化已被排除两项**，所以：

> C++ 实现与这份 Python 参考的任何行为差异，都是 bug 候选。

## 2. 目标

对下列 8 个环节逐项给出**有证据的判定**（不是印象）：
`一致` / `不一致（bug 候选）` / `无法判定（需要更多信息）`

每条判定必须附**双方的 file:line 与实际代码**，不得只写结论。

| # | 环节 | Python 参考侧 |
|---|---|---|
| 1 | 分词 + prompt 模板 | `format_zimage_prompt` / `process_zimage_prompt` |
| 2 | text encoder 四段串联与 padding | `main()` 里的 4 次 `run()` |
| 3 | `cap_pad_mask` 构造 | `cap_pad_mask = ones(32); [:token_count] = False` |
| 4 | latents 初始化与 RNG | `np.random.default_rng(seed).standard_normal` |
| 5 | timestep / sigma 序列 | `zimage_flow_match_sigmas(8, shift=3.0)`；`ts = 1 - t/1000` |
| 6 | 调度器更新公式 | `latents = latents - dt * noise`（dt 为负） |
| 7 | VAE 前后处理系数 | `lat/0.3611 + 0.1159`；`(px+1)*127.5` |
| 8 | 三段间张量传递 | part1a 的 9 个输出 → part1b(6) / part2(4) |

C++ 侧文件：`local-dream/app/src/main/cpp/src/` 下的
`PipelineZImage.hpp`、`ZImageFlowMatchScheduler.hpp`、`ZImagePrompt.hpp`、`ZImageQnnContract.hpp`

## 3. 预期结果（执行前登记）

我预测**至少能找到 1 处不一致**。
理由：这条链路上已经抓到过 2 个同类 bug（`cap_pad_mask` 极性、Euler 符号），
而失败现象（语义完全丢失）属于结构性错误的典型表现，量化已被排除。

**我承认另一种结果同样可能**：8 项全部一致。那将意味着问题出在
① 尚未验证的 A2（量化 text encoder 的计算），或
② HTP 执行与 CPU 参考不等价（`../docs/EXECUTION_MODEL.md` 未知项①），或
③ 我这份清单漏了环节（例如 QNN 张量的内存布局/量化反量化在 C++ 侧的实现）。

## 4. 验收标准

- **每一项都有判定 + 双方代码证据** → 本实验完成
- 找到不一致项 → 逐个评估"能否解释语义完全丢失"，**能解释的才算根因候选**
  （已经有过教训：找到差异就当根因，而不问它是否足以造成观察到的现象）

## 5. 防止误判的规则（针对本项目已犯过的错）

1. **不得只看一侧**。每条都要同时引用 Python 与 C++ 的实际代码行。
2. **差异 ≠ 根因**。找到差异后必须单独论证"这个差异能否造成语义完全丢失"。
3. **不得依赖交接文档的描述**（约束 1）。文档说"某处已修复"时，要自己看当前代码确认。
4. 无法从代码判定的（例如依赖运行时数据的），标为"无法判定"，不要猜。

## 6. 失败分析预案

| 现象 | 更可能是 | 后续 |
|---|---|---|
| 找到不一致，且能解释语义丢失 | 方向正确 | 修复 → 装机验证 |
| 找到不一致，但只能造成轻微劣化 | 方向部分正确 | 记录，继续找 |
| 8 项全部一致 | 清单不完整 或 方向错 | 扩展清单（内存布局、量化/反量化实现、QNN 张量填充顺序），并回头补 A2 |
