# 实验 A3 方案：VAE 量化是否是设备失败的原因（执行前定稿）

## 1. 假设

**A3**：设备失败源于 VAE decoder 的量化误差。

背景：Test B 已否定 A1（transformer 量化）。`MAINLINE.md` 的假设树上，A3 是**最便宜**的
剩余候选，所以先测它。

## 2. 目标

用**同一份真实 latents**，对比 FP32 VAE 与量化 VAE 的解码结果。

## 3. 路径

- 输入：`ZImage_QNN_Evidence/dlc_pipeline/vae_decoder/calibration_raw/sample_0000/latent_sample.raw`
  （来自真实生成过程的 latents，不是随机噪声）
- FP32 侧：onnxruntime 跑 `vae_decoder.onnx`
- 量化侧：`snpe_runner.py` 跑 `vae_decoder_quantized.dlc`（走强制契约检查）
- 两侧输出都转成 PNG，做视觉与 PSNR 对比

## 4. 执行前必须通过的有效性检查

| 检查 | 判据 | 不通过说明 |
|---|---|---|
| **V1** 契约 | 由 `snpe_runner.py` 强制（输入键集合/字节数/输出字节数） | 喂错数据 |
| **V2** latents 真实性 | **FP32 解码结果必须是一张正常的自然图像** | 若 FP32 解码就是乱码，说明这份 latents 不是真实生成产物，本实验无效 |
| **V3** 输入名对齐 | `vae_decoder.onnx` 的输入名与 DLC 契约的输入名可能不同（校准文件叫 `latent_sample`，FP32 脚本用 `vae_latents`），必须分别按各自声明喂 | 名字对不上会静默失败或报错 |

## 5. 预期结果（执行前登记，不得事后修改）

我预测**量化 VAE 解码正常**，与 FP32 的 PSNR > 20 dB，图像内容一致。

理由：VAE 是卷积网络，通常对 W8A16 量化鲁棒；且交接文档第三节记录过 VAE 解码路径的诊断实验
（但那是结构性检查，非数值验证，所以仍有必要实测）。

**我承认相反结果同样可能**，且那将直接定位根因。

## 6. 验收标准（执行前定稿）

| 结果 | 判读 | 后续 |
|---|---|---|
| 量化解码 ≈ FP32（PSNR > 20 dB，内容一致） | **A3 被否定** | 转 B（C++ 实现），并补 A2 的完整验证 |
| 量化解码 = 色块/乱码 | **A3 就是根因** | 停止其他方向，全力修 VAE 量化 |
| 内容一致但明显劣化（PSNR 10~20 dB） | A3 有贡献但可能不是全部 | 记录贡献度，仍需查 B |

## 7. 失败分析预案

| 现象 | 更可能是 | 排查 |
|---|---|---|
| snpe 报输入名不存在 | 执行错 | 查契约里 VAE 的真实输入名（V3） |
| 输出字节数不符（被 runner 拦截） | 执行错 | 输入尺寸不对，见 EXECUTION_MODEL 规则 1 |
| FP32 解码就是乱码 | 执行错 | 这份 latents 不可用，换 sample 或改用 Test B 产出的 latents |
| 两侧都正常且接近 | **A3 被否定** | 按验收标准转下一候选 |

## 8. 本实验**不能**回答的问题（必须写进结论）

- 不能验证设备 C++ 侧的 VAE **前后处理**是否正确（`vae_latents = latents/scaling + shift`
  这类系数、以及 pixel 反归一化）。那属于候选 B，需要单独对齐。
- 不能验证 HTP 上跑 `.bin` 的行为（本实验是 SNPE CPU 参考实现跑 `.dlc`）。
