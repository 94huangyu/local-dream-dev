# EXP_PLAN_ONDEVICE_PREPARE —— HTP 偏离是"宿主离线 prepare"造成的，还是"HTP 执行语义"本身？

定稿时间：2026-08-15（执行前定稿，判据不得事后修改）
前置：根因已定位为 C（`EXP_PLAN_HTP_INLOOP`）；本轮回答**"为什么"**的第一层。
主线归属：A → C 的下一层

## 0. 要区分的两件事

设备上的数值 = **宿主离线 prepare**（`qnn-context-binary-generator` 在 PC 上把 `.dlc`
编成 `.bin`，含图优化/调度/VTCM 分配）+ **HTP 运行时执行**。
目前只知道"合起来"偏离 CPU 参考 19~47%，**不知道是哪一半**。

区分办法：让**设备自己**做 prepare。
`qnn-net-run --dlc_path` 会在设备上加载 `.dlc` 并用**设备自己的 prepare 库**在线建图，
绕开 PC 上生成的 `.bin`。同输入下与 CPU 参考比：

| 结果 | 结论 |
|---|---|
| 设备在线建图 ≈ CPU 参考 | **宿主离线 prepare 是元凶**，`.bin` 编错了 —— 可修 |
| 设备在线建图 ≈ 宿主 `.bin`（同样偏 19%） | prepare 无罪，**HTP 执行语义**与 SNPE CPU 参考不等价 |

## 1. 选 part1b 做首发的理由

- 差异 **19.47%** —— 远高于噪声，信号明确
- `.dlc` 只有 1.39 GB（part1a 2.26 GB / part2 2.72 GB），推送与建图最快
- 设备 15.4 GB RAM、7.5 GB 可用，1.39 GB 的在线建图有充足余量

## 2. 实现路径

1. 推 `transformer_part1b_quantized.dlc` + `libQnnModelDlc.so` 到 `/data/local/tmp/htpcmp`
2. 喂**与实验 A 完全相同**的输入（`testB/s0_transformer_part1b/*.raw`，即 CPU 参考的上游输出）
3. `qnn-net-run --dlc_path part1b.dlc --backend libQnnHtp.so`（**非 native 模式**，float32 进出）
4. 与三方比：CPU 参考 `unified.raw`、宿主 `.bin` 的 HTP 输出（已有，19.47%）

## 3. 验收标准（判据，事后不得修改）

**判据 0**：输出 `unified` 必须是 15,851,520 个 float32（63,406,080 字节）。不符 ⇒ 本次无效。

**判据 1（主判据）**：设备在线建图输出 vs **CPU 参考**的相对 L2，记为 R_dlc。
已知宿主 `.bin` 侧为 R_bin = **19.47%**。

- **R_dlc < 2%** ⇒ 设备在线建图与 CPU 参考基本一致
  ⇒ **宿主离线 prepare 是元凶**。这是可修的工程问题，下一步转向 prepare 参数/SDK 版本。
- **R_dlc 与 R_bin 同量级（相对差 < 20%，即落在 15.6%~23.4%）**
  ⇒ **prepare 无罪，HTP 执行语义本身与 SNPE CPU 参考不等价**
  ⇒ 下一步转向精度配置（`act_bitwidth` / `use_dynamic_16_bit_weights` / per-channel 等）。
- **其余情况（2% ≤ R_dlc < 15.6% 或 R_dlc > 23.4%）** ⇒ 两者都有贡献，
  按 R_dlc 与 R_bin 的比值报告各自份额，**不得**只归因一边。

**判据 2（交叉检查）**：同时算「设备在线建图 vs 宿主 `.bin`」的相对 L2。
若该值 ≈ 0，说明两条路径产出相同的图，判据 1 里 R_dlc 必然 ≈ R_bin，互相印证。

## 4. 成功 / 失败判断

- **成功**：拿到 R_dlc，按判据 1 给出明确分支。
- **失败·执行方法错误**：`--dlc_path` 缺库、设备 OOM、在线建图超时。
  ⇒ 修方法重跑；若设备根本建不了这么大的图，改用**设备侧
  `qnn-context-binary-generator`** 离线编 `.bin` 再跑（同样绕开宿主 prepare），判据不变。
- **失败·方向错误**：输出与宿主 `.bin` **逐位相同**（md5 一致）⇒ 说明 `--dlc_path` 实际
  加载的仍是缓存的 `.bin`，必须查清楚，不得解读为"prepare 无差异"。

## 5. 预算与放弃条件

- 预算：part1b 一轮。设备在线建图预计数分钟至十几分钟。
- 若判据 1 落在"prepare 是元凶"分支 ⇒ 立即在 part2 上复核一次（43% 那段），再下结论。
- 若设备在线建图 3 次都跑不起来 ⇒ 回主线，改走设备侧 context-binary-generator 路线。

---

# 执行结果（2026-08-15，判据未做任何修改）

设备在线建图耗时 **16m26s**（user 14m33s / sys 1m29s），一次成功。

## 判据 0：✅ 通过
输出 `unified` = 15,851,520 个 float32（63,406,080 字节），与契约一致。

## 实测数据（part1b 的 `unified`，输入与实验 A 完全相同）

| 对照 | 相对 L2 | 余弦 | 最大\|差\| |
|---|---|---|---|
| 宿主 `.bin` vs CPU 参考（R_bin） | **19.4727%** | 0.990125 | 2733 |
| 设备在线建图 vs CPU 参考（R_dlc） | **19.4727%** | 0.990125 | 2733 |
| 设备在线建图 vs 宿主 `.bin`（交叉） | **0.0000%** | 1.000000 | **0** |

## 判据 2 先行处置：警报触发 → 查证后解除

交叉项为 0（逐位相同）触发了第 4 节"方向错误"的警报，
按规定**先查清是否 `--dlc_path` 偷用了缓存 `.bin`**，查证结果（全部①实测）：

1. `D1b/` 目录下**没有生成任何 `.bin` 缓存**，只有 `Result_0/` 与 `execution_metadata.yaml`；
   工作目录里三个 `.bin` 的时间戳仍是 08-14 推送时的，未被读改。
2. 日志打印 `ERROR: Failed in cacheSelection during getRecordsByType`
   —— 缓存查找**失败**，即**没有命中任何缓存**。
3. 日志打印 `Composing Graphs` / `Finalizing Graphs`；
   而 `.bin` 路径打印的是 `Creating context from binary file`。**两条路径特征不同。**
4. 耗时 **16m26s vs 12s**，相差约 80 倍，与"真实在线建图"相符。

⇒ **警报解除**：本次确实是设备从 `.dlc` 在线建图，逐位相同是真实结果。

## 判据 1：🔴 **prepare 无罪，HTP 执行语义本身与 CPU 参考不等价**

R_dlc = R_bin = 19.4727%，落在同量级区间 [15.58%, 23.37%] 内（实为完全相等）。

⇒ ②严格推论：**把 graph prepare 从 PC 换到设备本机，结果一个 bit 都没变。**
   因此 15.6 节里"宿主离线 prepare 库对 v79 支持不完整"这一未验证猜想
   **被否定**——prepare 在哪跑都一样，`Unsupported HTP Arch 1 79` 那条告警确认为虚惊。

⇒ 偏离来自 **HTP 后端本身的执行语义**（算子实现 / 累加精度 / 重量化策略），
   与 SNPE CPU 参考实现不等价。

## 附带确立的可复用事实（应进 `../docs/QNN_CONVERSION_GUIDE.md`）

**QNN 的 graph prepare 是确定性且平台无关的**：同一份 `.dlc`，
在 x86 宿主离线编成 `.bin` 后执行，与在 aarch64 设备上在线建图后执行，
输出**逐位相同**。
⇒ 排查 HTP 数值问题时**不必怀疑 `.bin` 是不是"编坏了"**，可以直接跳过这一层。
⇒ 反过来，离线 `.bin` 是安全的优化手段（16m26s → 12s，**80 倍**加载提速），不影响数值。

## 下一步

按判据 1 的分支：转向**精度配置**。候选（均来自 `maskfix_part2_dlcinfo.txt` 的量化命令）：
`act_bitwidth=16`、`weights_bitwidth=8`、`use_dynamic_16_bit_weights=True`、
`use_per_channel_quantization=True`、`bias_bitwidth=32`。
优先结合 `EXP_PLAN_CH85_CAUSAL` 的归因结果决定攻哪一个——
若极端点是主因，则重点查 HTP 对接近 16-bit 量程上界的值的处理。
