# ZIMAGE 后端引擎启动失败事件分析报告

## 1. 核心结论与当前状态
经过一系列追踪与调试，底层 C++ 引擎（`libstable_diffusion_core.so`）已经**成功被唤醒**，并且成功解析了模型配置文件。但在加载第一个 QNN 模型二进制文件（`text_encoder_part1`）时，QNN 底层抛出了初始化失败的致命异常，导致进程崩溃。

*   **当前最终报错：** `Failed init QNN model: text_encoder_part1`
*   **App 源码路径：** `D:\LDZ_FIX3_20260810\local-dream\`

---

## 2. 时间线与证据链重构

自我们在 Android Studio 尝试绕过 CMake 编译起，系统经历了一连串的连锁故障，每一次修复都暴露出底层潜伏的下一个“地雷”。

### 事件一：C++ 引擎文件丢失（已修复）
*   **现象：** 点击生成后瞬间报错“后端启动失败”。
*   **根因分析：** 在早些时候（Fix4 阶段），由于 Windows 电脑缺失高通 QNN SDK，编译代码时我禁用了 `build.gradle.kts` 中的 `externalNativeBuild` 节点。这使得 Android Studio 成功打出了纯 Java 的 APK 包，但却导致 **C++ 引擎核心二进制文件 (`libstable_diffusion_core.so`) 未被打包进新 APK 中**。
*   **证据：** 通过 `adb shell run-as` 检查手机内部的 `lib/arm64` 目录，发现该文件缺失。
*   **修复动作：** 从历史可运行的 V3 版 APK 中提取了 `libstable_diffusion_core.so`（19MB），强行放置到了源码的 `app/src/main/jniLibs/arm64-v8a/` 目录下，并为 `BackendService.kt` 增加了标准错误流（`stderr`）捕获代码，成功让 C++ 引擎随 App 启动并在 Java 层输出报错。

### 事件二：JSON 根节点版本不匹配（已修复）
*   **现象：** C++ 引擎启动瞬间退出，报错 `Unsupported Z-Image QNN contract version`。
*   **根因分析：** 引擎源码 `ZImageQnnContract.hpp` (第 51 行) 严格要求 Z-Image 最终模型配置文件的根 JSON 键必须是 `"models"`。由于版本脱节，转换工具生成的 JSON 根键为 `"graphs"`。
*   **修复动作：** 通过 ADB 强行修改手机内的 `final_qnn_contract.json`，将 `"graphs"` 替换为 `"models"`。

### 事件三：JSON 张量字段不匹配（已修复）
*   **现象：** C++ 引擎再次退出，报错 `[json.exception.out_of_range.403] key 'name' not found`。
*   **根因分析：** 同样是版本脱节，引擎源码 `ZImageQnnContract.hpp` (第 148 行) 严格要求张量的名字字段为 `"name"`，但文件里写的是 `"tensor_name"`。
*   **修复动作：** 全局替换 `"tensor_name"` 为 `"name"`。

### 事件四（当前阻碍）：高通 QNN 模型初始化崩溃（未修复）
在解决了上述所有的周边问题（二进制丢失、JSON 格式脱节）后，C++ 引擎终于开始实质性地加载模型数据。但在加载第一个模型切片时，底层发生了崩溃。

*   **现场日志捕获：**
    ```text
    08-11 21:20:00.949 I/BackendService(19861): Backend: libc++abi: terminating due to uncaught exception of type std::runtime_error: Failed init QNN model: text_encoder_part1
    ```
*   **根因推测：** 该报错来源于源码 `D:\LDZ_FIX3_20260810\local-dream\app\src\main\cpp\src\QnnRuntime.hpp` 的第 129 行。这说明 C++ 引擎已经成功读取到了 `text_encoder_part1_ctx.SM8550.bin` 文件并交给了高通 QNN 框架，但在初始化上下文时失败。可能的原因包括：
    1.  **QNN SDK 版本不兼容：** 编译出 `.bin` 模型文件所使用的高通 SDK 版本，与设备当前的 DSP/NPU 固件版本，或与 `libQnnHtp.so` 运行库的版本发生了严重冲突。
    2.  **设备内存耗尽：** NPU/DSP 拒绝了分配如此巨大内存上下文的请求。

---

## 3. 手机模型配置文件的修改说明

为了配合旧版 C++ 引擎的苛刻要求，我对模型配置 JSON 进行了硬编码式的字段修正。修改后的文件已经备份至当前项目目录中供您查阅比对。

*   **修改后的文件备份路径：** `D:\LDZ_FIX3_20260810\local-dream\final_qnn_contract_modified.json`
*   **修改内容详单：**
    1.  **根节点替换：** 将 `"graphs": [` 替换为了 `"models": [`。
    2.  **张量键名替换：** 将全篇所有的 `"tensor_name"` 替换为了 `"name"`。

这些修改已经通过后台推送到手机的 `/data/data/io.github.xororz.localdream.zimage/files/models/ZIMAGE/final_qnn_contract.json` 中生效，这使得 C++ 引擎成功跨过了校验阶段，直面了 QNN 初始化失败的最终真因。
