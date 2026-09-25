# EXP_PLAN_APK_QNNLIBS —— QNN 运行库改由 APK 提供（versionCode 102）

**日期**：2026-09-25　**执行前定稿，判据不得事后修改**（约束 4）

## 目的

发布的模型包不能带高通运行库（QAIRT `LICENSE.pdf` §1(iv)：只许 "as incorporated in Your
software application" 分发，不许 "standalone"）。v102 把 5 个 v79 库打进 APK 的 `assets/qnnlibs`，
Z-Image 在模型包没有 `qnn_runtime_libs/` 时改用 APK 解压出的库。要验证：

1. 老模型包（自带库）行为不变 —— 用户手机上的现状；
2. 不带库的模型包能跑，且出图与金标准逐字节相同。

## 宿主侧已完成（① 实测）

- APK `assets/qnnlibs` 5 个文件，与现网模型包 `qnn_runtime_libs/` 同名文件 **sha256 逐字节相同**；
  `libstable_diffusion_core.so` 仍为 `0357a043…`（`apk_qnn_check.py`）。
- ③ **未验证**：这 5 个库对 Z-Image 是否**足够**（模型包里原有 101 个，代码只检查其中 3~4 个）。本实验的 B 臂就是验它。

## 步骤与判据

| 步 | 动作 | 通过判据 | 不通过 ⇒ |
|---|---|---|---|
| 0 | `adb pull` 现装 v101 APK 备份 | sha256 == `444ac10f…` | 停，先查设备上是什么 |
| 1 | `install -r` v102 | versionCode=102；设备 `sha256sum base.apk` == 宿主 | 回滚 v101 |
| A | 模型包不动，`app_generate.sh` 金标准 | 出图 sha256 == `49be8e9a…83ad6`；**且**后端进程 maps 里的 `libQnnHtp.so` 路径在模型目录 `qnn_runtime_libs/` 下 | 回滚 v101 |
| B | 两个模型目录的 `qnn_runtime_libs` 改名为 `.off`，force-stop 后再跑金标准 | 出图 sha256 == 金标准；**且** maps 里 `libQnnHtp.so` 路径在 app 的 runtime 目录（非模型目录） | 改名还原 + 回滚 v101；结论：5 个库不够，需查缺哪个 |
| C | 还原改名 | 两个目录的 `qnn_runtime_libs` 恢复 | — |

🔴 **生效证据（门 F1）**：A/B 两臂出图预期相同，出图 sha 本身区分不了"走了新路径"与"没生效"，
所以**必须**同时读后端进程的 `/proc/<pid>/maps`（`run-as` 同 uid 可读），看实际加载的是哪份 `libQnnHtp.so`。
不用 logcat 里的 `QNN DIR` 行：`app_generate.sh` 在后端就绪**之后**才 `logcat -c` 开录，那一行录不到。

⊕ 跳过"装前基线"的理由：v101 今天 19:5x 已在同一设备、同一模型上出过金标准图，此后未改动设备状态。

## 设备占用

插线约 30 分钟，全程不可拔；结束后明说可以拔。

---

# 执行结果（2026-09-25，判据未改）—— 🟢 全部通过

| 步 | 实测 | 判定 |
|---|---|---|
| 0 | v101 备份 `logs/apk_backup_20260925/base_v101_installed.apk`，sha256 `444ac10f…` | ✅ |
| 1 | v102 `install -r` 成功；versionCode 102；设备 `base.apk` sha256 `287a4bbe…bbc166` == 宿主 | ✅ |
| A | 出图 sha256 `49be8e9a…83ad6` == 金标准（129.7 s）；maps：`…/files/models/ZIMAGE/qnn_runtime_libs/aarch64-android/libQnnHtp.so` | ✅ 老模型包走自带库，行为不变 |
| B | 两目录改名 `.off` 后：模型仍被识别为完整、出图 sha256 == 金标准（132.9 s）；maps：`…/files/runtime_libs/libQnnHtp.so`（APK 解压的） | ✅ **5 个 v79 库足够**，数值逐字节不变 |
| C | 改名还原，两目录各 101 个文件 | ✅ |

⇒ ③ 那条未验证项已关闭：对 Z-Image（SM8750、上下文二进制、非在线建图），
APK 里的 `libQnnHtp.so` / `libQnnSystem.so` / `libQnnHtpV79Stub.so` / `libQnnHtpV79.so` / `libQnnHtpV79Skel.so` 就够了，
模型包里另外 96 个 `.so` 用不上。
⚠️ 适用范围：只验了 1024×1024 金标准这一种请求；其余四个比例走同一组库、同一后端，未单独跑。
工具：`scripts/qnn_lib_origin.sh`（读后端 `/proc/<pid>/maps`）。
