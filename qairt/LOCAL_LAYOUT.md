# 本地梦-ZIT（Z-Image Turbo on Snapdragon HTP）—— 项目总索引

> **状态：2026-09-25 项目结束，冻结存档。** 不再开发。
> 本文件说明：东西都在哪、哪些**不能删**、以后要重新编译 APK 需要什么。

## 一、线上（已发布，是"正本"）

| 内容 | 地址 |
|---|---|
| app 源码 + 转换指南/脚本（`qairt/`） | GitHub `94huangyu/local-dream-dev` 分支 **`qairt-dev`** |
| APK v102（`2.8.1-zit2`） | <https://github.com/94huangyu/local-dream-dev/releases/tag/v2.8.1-zit2> |
| 模型包（不含高通库） | <https://huggingface.co/huangyu94/ZImageTurbo-QNN-SM8750>（公开） |
| tokenizers-cpp 构建修复 | GitHub `94huangyu/tokenizers-cpp` 分支 `qairt-dev`（fork 自 mlc-ai） |

⚠️ 用户名：GitHub 是 **`94huangyu`**，HuggingFace 是 **`huangyu94`**。

## 二、本文件夹

```
D:\LocalDreamZImage\
├─ README.md            ← 本文件
├─ CLAUDE.md            项目约束（给 AI 会话用）
├─ MAINLINE.md          线索台账 + 目录说明（§0.1）
├─ docs\                主文档：转换指南、排查全记录、交付冻结快照（DELIVERY_FROZEN.md）、磁盘清单
├─ scripts\             全部脚本与实验方案（EXP_PLAN_*.md）
├─ logs\                过程日志；logs\apk_backup_20260925\ 有 v100/v101 APK 回滚源
├─ scratch_runs\        设备运行记录、出图（金标准图等）
├─ evidence\ reference\ tools\   文档引用的证据图片、SDK 文档摘录、小工具源码
├─ local-dream\         Git 仓库（app 源码；qairt\ 是 docs/scripts 的同步副本）
├─ host-protoc\         编译 APK 要用的 protoc 35.1（local.properties 指向它）
├─ sdk\qairt\2.48.0.260626\   高通 QAIRT SDK（编译 APK 要用；原在 D:\qairt）
├─ release\             发布物：APK、HF 模型卡 README/LICENSE/NOTICE、重打包脚本 repack.py
├─ keys\debug.keystore  🔴 APK 签名文件的**备份**（原件见下）
└─ archive\             历史存档，不是现状：
   ├─ ZImage_Work\          原 D:\ZImage_Work 剩下的小文件（早期契约、证据、配方存档）
   ├─ local-dream-ZIT0813\  08-13 MacBook 独立诊断
   ├─ MAINLINE_bak\         MAINLINE 旧备份
   └─ PATH_MOVES_20260925.tsv   🔎 本次整理的"原路径 → 新路径"对照表（38 项）
```

**文档里写的旧路径找不到时**：先查 `archive\PATH_MOVES_20260925.tsv`。
大文件（实验 DLC/context、源 ONNX、校准数据、原始权重、模型 zip）已于 2026-09-25 删除，
清单见 `logs\cleanup_20260925\deleted_manifest.txt` 与 `docs\DISK_INVENTORY.md` 开头。

## 三、不在本文件夹、但与本项目有关（**别误删**）

| 位置 | 是什么 | 为什么没挪进来 |
|---|---|---|
| `C:\Users\sinai\.android\debug.keystore` | 🔴🔴 **APK 签名原件** | Gradle 编译时从这里读。丢了就无法给已装用户发升级（只能卸载重装 ⇒ 模型一起被删）。备份在 `keys\` |
| `C:\Users\sinai\AppData\Local\Android\Sdk\` | Android SDK/NDK 28.2/CMake 3.22.1（8 GB） | 由 Android Studio 管理；编译 APK 要用。2026-09-25 查过：目前只有本项目用它 |
| `D:\WSL\`（发行版 `Ubuntu2404`） | 8 GB，曾用来编译 C++ 契约解析器 | WSL 发行版不能直接挪；用户说以后可能用到，暂留 |
| `C:\Users\sinai\.claude\projects\D--LocalDreamZImage\` | AI 会话记忆 | 由 Claude Code 管理 |
| 手机 | app 里的模型 `ZImageTurbo_A16W8_SM8750`（发布版） | 另两个旧模型目录和下载目录里的 zip 由用户手动清理 |

## 四、以后要重新编译 APK

1. 需要：Android Studio（自带 JDK）、上面的 Android SDK、`sdk\qairt\2.48.0.260626`、`host-protoc\`、签名原件。
2. `local-dream\local.properties`（不入库）里已写好 `qnn.sdk.dir` 与 `spm.protoc.exe` 两个路径。
3. 在 `local-dream\` 下：`gradlew.bat :app:assembleBasicDebug`（需先设 `JAVA_HOME=C:\Program Files\Android\Android Studio\jbr`）。
4. 编完运行 `gradlew.bat --stop`，否则后台进程会锁住 `app\build`。

## 五、可选的最后清理

`scripts\cleanup_end_of_project.ps1 -Execute`：删 `local-dream\app\build`、`app\.cxx`（编译缓存）并注销 WSL。
**默认只预览**；用户 2026-09-25 决定暂不执行。
