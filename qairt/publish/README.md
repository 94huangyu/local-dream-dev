---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
pipeline_tag: text-to-image
tags:
- qnn
- snapdragon
- android
- on-device
---

# Z-Image Turbo for Snapdragon 8 Elite (QNN)

Z-Image Turbo compiled to run fully on the phone's NPU (Hexagon HTP) with the
**本地梦-ZIT** Android app, a fork of [Local Dream](https://github.com/xororz/local-dream).

在手机 NPU 上本地运行 Z-Image Turbo 的模型包，配合 **本地梦-ZIT** 安卓 app 使用。

## Requirements / 使用条件

| | |
|---|---|
| Chip / 芯片 | **Snapdragon 8 Elite (SM8750) only**. The binaries are compiled for this chip and will not run on others. 只支持骁龙 8 Elite |
| RAM / 内存 | About 9 GB is used while generating; a **16 GB** phone is recommended. 生成时约占 9 GB，建议 16 GB 内存的手机 |
| Storage / 存储 | 13 GB for the zip, plus the same again during import. zip 13 GB，导入时还需同样大小的空间 |
| App | 本地梦-ZIT (versionCode ≥ 102): <https://github.com/94huangyu/local-dream-dev/releases> |

Tested on one SM8750 phone only. 目前只在一台 SM8750 手机上验证过。

## How to use / 使用方法

1. Install the 本地梦-ZIT APK. 安装 APK。
2. Download `ZImageTurbo_A16W8_SM8750_qnn2.48.zip` to the phone. 把 zip 下载到手机。
3. In the app, import the zip from the model list. 在 app 的模型列表里导入这个 zip。
4. Open the model and generate. 进入模型页即可生成。

Output sizes: 1024×1024, 1184×896, 896×1184, 1344×768, 768×1344. 8 steps.
Prompt limit: 72 tokens.

## What was changed / 与原模型的区别

Derived from [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(Apache-2.0). The model was split into segments, quantized to A16W8 with some
tensors kept in FP16, and compiled with Qualcomm AI Runtime SDK 2.48 for HTP v79.
Images are close to, but not identical with, the original model's output.
Details are in `NOTICE`. The full conversion notes and scripts are in the
[`qairt/` folder of the app repository](https://github.com/94huangyu/local-dream-dev/tree/qairt-dev/qairt)
(archived; see the notice at the top of `docs/QNN_CONVERSION_GUIDE.md` for what it does and does not cover).

This bundle contains no Qualcomm software; the QNN runtime ships inside the app.

## License

Apache License 2.0, see `LICENSE`. Base model © Tongyi-MAI.
