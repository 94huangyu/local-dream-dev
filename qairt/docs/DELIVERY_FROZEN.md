# 交付态冻结快照

## 〇·二、【2026-09-25 晚】现网 = versionCode **102** —— 以本节为准

v102 = v101 + QNN v79 运行库打进 APK（`assets/qnnlibs`，5 个），模型包可以不带 `qnn_runtime_libs/`。
其余（名字、图标、端口、导出目录）同下面 〇 节。

| 项 | 值 |
|---|---|
| APK | `LocalDreamZImage_armv8a_2.8.1-zit2.apk`，sha256 `287a4bbec1152f5526afab54426adc4773cda795a12d3b29e449779260bbc166`（① 设备与宿主一致） |
| `.so` | 仍为 `0357a043…`（① 逐字节不变） |
| 验证 | 自带库的模型包 / 去掉库的模型包，出图 sha256 **都** == 金标准；加载路径由 `/proc/<pid>/maps` 确认（`scripts/EXP_PLAN_APK_QNNLIBS.md`）。**发布包经 app 导入后**出图同样 == 金标准（同文件"追加 D"） |
| 设备上多出的测试物 | `files/models/ZImageTurbo_A16W8_SM8750`（导入测试，12 G）、`/sdcard/Download/ZImageTurbo_A16W8_SM8750_qnn2.48.zip`（13 G）—— 是否保留由用户定 |
| 设备模型目录 | 测试后已还原，两个目录仍各带 101 个库（走老路径） |
| 回滚源 | `logs/apk_backup_20260925/base_v101_installed.apk`（`444ac10f…`） |
| 发布用模型包 | `D:\ZImage_Work\publish\ZImageTurbo_A16W8_SM8750_qnn2.48.zip`，12,945,674,253 B，sha256 `87e3de95b47f4bb7e225c9d1e86f0f19309877d00b4c5cf35ea40ce575472f93`；= 现网模型包去掉 `qnn_runtime_libs/`（其余 22 个条目逐字节相同）+ `LICENSE` + `NOTICE` |

## 〇、【2026-09-25】本地梦-ZIT（versionCode 101）—— 已被 〇·二 取代

> 目的：与官方 Local Dream 3.x（包名 `io.github.xororz.localdream`）在同一台手机上并存。
> **只改了上层**：应用名、图标、本地端口、导出目录。模型、契约、设备配置均未动。

| 项 | 值 | 依据 |
|---|---|---|
| APK | `LocalDreamZImage_armv8a_2.8.1-zit.apk`（versionCode **101**） | 宿主构建 2026-09-25 19:49 |
| APK sha256 | `444ac10f33f5e487ccfa48f973ac271b7267895b4eafebbc2a8b063d5fe61f94` | ① 设备 `sha256sum base.apk` 与宿主逐字节一致 |
| `libstable_diffusion_core.so` sha256 | **`0357a043606d4970ba75120ccd71f20ee63cf15bad2bc71576458e365ca2ad38`** | ① 与被替换的 v100 **逐字节相同** ⇒ native / 数值路径未动 |
| 包名 | `io.github.xororz.localdream.zimage`（官方为 `io.github.xororz.localdream`） | ① `aapt2 dump badging` |
| 应用名 | `本地梦-ZIT`（所有语言） | ① `aapt2 dump badging` |
| 后端端口 | 生成 **8091** / 受控 **8818**（官方 8081 / 8808） | ① 装后设备 `/proc/net/tcp`：`:1F9B` LISTEN，`:1F91` 无 |
| 导出目录 | `Pictures/LocalDreamZImage`、`Downloads/LocalDreamZImage` | ① 用户 2026-09-25 在手机上保存一张图，确认进入 LocalDreamZImage 相册（代码见 `ImageUtils.kt` `PUBLIC_EXPORT_DIR`）。`Downloads/` 那条（日志导出）未单独验 |
| 桌面名字 / 图标 / 通知栏小图标 | 本地梦-ZIT / 星环图标 / 星形 | ① 用户 2026-09-25 手机上目视确认 |
| 签名证书 | Android Debug，SHA-256 `22c13d4b…3ce46d` | ① `apksigner` 新旧一致 ⇒ `install -r` 原地升级 |
| 回滚源 | `logs/apk_backup_20260925/base_v100_installed.apk`（sha256 `09c4a20a…acbb23`） | 装前从设备 `adb pull` |

**交付验证（约束 11 四条铁律）**：

| 步骤 | 结果 |
|---|---|
| ① 基线：装前用 **v100** 当场出图（`ZIT_PORT=8081 app_generate.sh`，seed 42，1024²） | sha256 `49be8e9a…83ad6` == 金标准 ✅，136.0 s |
| ② 备份原件 | ✅ 见上表"回滚源" |
| ③④ 装后在真实 app 里出图（`app_generate.sh`，端口 8091） | sha256 `49be8e9a…83ad6` == 金标准 ✅，133.9 s |
| 装后模型仍在 | ✅ `files/models/ZIMAGE`、`ZIMAGE_sm8750_v2` 各 13 GB |

🔴 **下面第一~三节是 2026-09-20 10:38 的快照，其"宿主侧产物"一节在当天就已过期**（本节 2026-09-25 查出）：
10:38 冻结后，11:03 又装了一次 APK（`scratch_runs/p2s9/installed_d87e9670c06ff817.apk` 是装前备份），
装上的才是 HANDOVER §15.50.1 记录的最终交付版（`.so` `0357a043…`，APK `09c4a20a…`，94,342,619 B）。
⇒ 第一节的 APK / `.so` sha256（`d87e9670…` / `538c36c5…`）**不是**现网版本，保留原文仅供追溯。
第二节的设备侧模型 sha256 与第三节的金标准出图 sha256 **仍有效**（今天两次出图均复现）。

---

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

