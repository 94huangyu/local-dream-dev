# 交付态冻结快照

> 由 `python scripts/freeze_delivery.py` **现读**生成，不抄任何文档里的旧值（约束 1）。
> 冻结时间：**2026-09-20 10:38**

## 一、宿主侧产物

| 项 | 值 |
|---|---|
| APK | `LocalDreamZImage_armv8a_2.8.1-zimage-mvp.apk` |
| APK 字节数 | 94339337 |
| APK sha256 | `d87e9670c06ff817474f3e0630d951db51c7f120c73e7c6ad3f4c2c1f426beb6` |
| `libstable_diffusion_core.so` sha256 | **`538c36c58263b0745fd571f34a73bf6bd34a48e6777e33e4c1fa3050b70fcb72`** |
| 构建时间 | 2026-09-20 10:23 |
| noscatA part1a（新件） | 2877374464 字节 |
| ↳ sha256 | `67b5583d3af3596b7daa13c5c14d2c9d0069816c706348041022e889371895be` |
| part1a 原件（回滚源） | 3151335424 字节 |
| ↳ sha256 | `eb0ef46adbfc305883f75577919b6d8b542f1ce2fc449e0ab93696e43f92e727` |

## 二、设备侧现态（现读）

| 文件 | 字节数 | sha256 |
|---|---|---|
| `final_qnn_contract.json` | 154981 | `55e8a1e0e1ec2a746e0ae80ead837367bd9ef796b0a863ff9852d29bee54da44` |
| `SHARE_SPILLFILL` | 10 | 内容：`331415552` |
| `LOAD_MMAP` | 0 | 内容：`(空文件=开关打开)` |
| `models/transformer_part1a_mg_ctx.SM8750.bin` | 2877374464 | `67b5583d3af3596b7daa13c5c14d2c9d0069816c706348041022e889371895be` |
| `models/transformer_part1b_mg_ctx.SM8750.bin` | 1762275328 | `d02259aaf16ee85d54e0303b61dfab5d93d477560bd40e82d3932c7087b0ec4c` |
| `models/transformer_part2a_mg_ctx.SM8750.bin` | 1751965696 | `ab01d74f7ccfe1807e4f3c9e67d575fbb8dc2819bf31e673514da8cac150ea23` |
| `models/transformer_part2b_mg_ctx.SM8750.bin` | 1761329152 | `479b378b2dda48fce975ea7750badc5ae56833489b6f5d6fe1989f6d8c84ae7f` |
| `models/vae_mg_ctx.SM8750.bin` | 443772928 | `453c608a2ac098e2ae99e5e5c6ebf93bfdb8dce182bcaf38f331bebfeae1e4f6` |
| `models/text_encoder_part1_L80_ctx_sm8750.SM8750.bin` | 1693160160 | `9ca612b1fb271cc28bad2f7e876e1368992cc2fff83e2cf8f5c15a36b7558f6c` |
| `models/text_encoder_part2_L80_ctx_sm8750.SM8750.bin` | 915248072 | `4ea46c8ec12396c30ace9bc924927bd9f81f242b5c29ded2341a4b5f33c10642` |
| `models/text_encoder_part3_L80_ctx_sm8750.SM8750.bin` | 915248248 | `98243d9565c81eeb170f45c2d5072f4355150016d5283f8128e6d07f8e182f38` |
| `models/text_encoder_part4_L80_ctx_sm8750.SM8750.bin` | 813627936 | `285d365e1d2e3ec9cef314b478ed74a760ef9eb7e8abb5ba45904dd2b116dad9` |

设备上安装的 APK 路径：`/data/app/~~tGiPxkuVnrno5I2IrBO2eA==/io.github.xororz.localdream.zimage-RImydx7gmrXB8zg_gAGGgw==/base.apk`

## 三、验收基线（任何改动后都要能复现）

- **出图 sha256（金标准）**：`49be8e9a4f95b85e320a9c3caf466c7078f315e5242f7c35981f929529f83ad6`
  - 请求：`scripts/app_generate.sh`，seed 42，默认 prompt，1024×1024
  - 这一条是**逐字节**判据：不相同就说明改动影响了数值，不得交付。

