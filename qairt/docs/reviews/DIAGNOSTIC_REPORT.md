# ZImage 后端启动失败 — 完整诊断报告

> **设备**：OnePlus PKX110 · SM8750（骁龙 8 至尊版）· HTP v79 · Android 16 (SDK 36)  
> **App 包名**：`io.github.xororz.localdream.zimage`  
> **模型**：`final_qnn_contract_modified`（位于 `/data/user/0/.../files/models/ZIMAGE/`）  
> **诊断时间**：2026-08-11 ~ 2026-08-12  
> **诊断工程师**：AI Antigravity（通过 ADB 远程分析）

---

## 一、问题现象

| 现象 | 详情 |
|------|------|
| App 启动后点击模型 | 弹出"后端启动失败，您的设备可能不受支持" |
| 点击生图 | `fail to connect to localhost/127.0.0.1:8081` |
| App 已正常安装 | ✅ |
| 模型已正常导入 | ✅ |

---

## 二、排查思路与确认过程

### 阶段 1：定位根因 —— C++ 引擎崩溃（已确认）

**方法**：ADB logcat 全量抓取，过滤 `BackendService` / `libc++abi` / `fastrpc`。

**发现**：
```
I BackendService: Backend: libc++abi: terminating due to uncaught exception of type
  std::runtime_error: Failed init QNN model: text_encoder_part1
I BackendService: Backend process exited with code: 134
```
进程以 SIGABRT (exit 134) 崩溃，原因是 QNN 模型初始化失败。

---

### 阶段 2：定位 QNN 失败原因 —— Skel 文件缺失（已确认）

**方法**：提取 fastrpc 详细日志。

**发现**：
```
Error 0x80000406: remote_handle_open_domain: dynamic loading failed for
  file:///libQnnHtpV79Skel.so?qnn_2_48_0_skel_handle_invoke&_modver=1.0&_dom=cdsp
  (dlerror: cannot open libQnnHtpV79Skel.so, errno 2 (no such file or directory))
```

**分析**：QNN HTP 采用 FastRPC 架构，需要成对文件：
- `libQnnHtpV79Stub.so`（CPU 侧，应用调用）—— 模型包里有 ✅
- `libQnnHtpV79Skel.so`（DSP 侧，Hexagon 固件层）—— **完全缺失** ❌

FastRPC 在以下所有标准路径均未找到 Skel 文件：
- `/data/user/0/.../qnn_runtime_libs/aarch64-android/` → ENOENT
- `/vendor/lib/rfsa/adsp/` → ENOENT
- `/vendor/dsp/cdsp/` → EACCES（无权限但可能存在）

---

### 阶段 3：确认设备上存在 Skel 文件（已确认）

**方法**：`find / -name '*Skel*.so'`

**发现**：设备上有多个 Skel 文件，但**不在 FastRPC 搜索路径**内：
```
/odm/lib/rfsa/adsp/libQnnHtpV79Skel.so          (8.3 MB — 通用版)
/odm/lib/rfsa/adsp/aiboost/signed/libQnnHtpV79Skel.so
/odm/lib64/aiframe/cdsp/signed/libQnnHtpV79Skel.so
/odm/lib64/aiframe/cdsp/unsigned/libQnnHtpV79Skel.so  (9.5 MB — unsigned 版)
/vendor/lib64/hw/audio/libQnnHtpV79Skel.so
```

---

### 阶段 4：尝试修复 —— 推送错误版本 Skel（失败，已识别原因）

**操作**：将 `/odm/lib/rfsa/adsp/libQnnHtpV79Skel.so`（8.3MB）复制到模型运行库目录。

**结果**：Skel 文件被找到并打开，但加载失败：
```
Error 0x80000442: dynamic loading failed
dlerror: RX VA 0xFFF00000 outside ELF segment
```

**原因**：该版本是 **signed** Skel，但进程以 `Unsigned PD` 模式运行（日志显示 `Unsigned:Y, Signed:N`），signed/unsigned 不匹配。

---

### 阶段 5：推送正确的 unsigned Skel（部分成功）

**操作**：
```powershell
adb pull /odm/lib64/aiframe/cdsp/unsigned/libQnnHtpV79Skel.so D:\LocalDreamZImage\
adb push D:\LocalDreamZImage\libQnnHtpV79Skel.so /data/local/tmp/
adb shell run-as io.github.xororz.localdream.zimage cp /data/local/tmp/libQnnHtpV79Skel.so \
    files/models/ZIMAGE/qnn_runtime_libs/aarch64-android/libQnnHtpV79Skel.so
```

**结果**：
```
Successfully opened file .../libQnnHtpV79Skel.so
fopen: 22us, read: 6985us  ← 真正读取了文件（之前是 0us）
error_code 0x0  ← 成功！
```

FastRPC Skel 加载成功，**不再出现 `Failed init QNN model` 崩溃**。

---

### 阶段 6：发现新问题 —— 初始化时间过长导致连接超时

**现象**：
- 后端进程启动后不崩溃，但 8081 端口始终未监听
- 生图时仍报 `ECONNREFUSED`
- 60 秒时进程仍存活，端口未开

**根因分析（通过 `main.cpp` 源码确认）**：

```cpp
// main.cpp:810 — HTTP 服务器在 pipeline->initialize() 全部完成后才启动
if (!pipeline->initialize()) {
    std::cerr << "ERROR: Pipeline initialization failed!\n";
    return EXIT_FAILURE;
}
// 只有走到这里才开始监听
svr.listen(opts.listen_address.c_str(), opts.port);  // Line 837
```

`pipeline->initialize()` 对 ZImage 的执行流程：
1. `ZImageQnnContract::validateGraphFiles()` — 读取全部 8 个 `.bin` 文件计算 SHA-256 校验
2. `loadGraph()` × 8 — 逐一将 context binary 加载到 HTP DSP

**估算耗时**（SM8750 实测）：
- SHA-256 校验：8 × 1.37GB = ~11GB，以 ~100MB/s 读速计算 ≈ **110 秒**
- DSP context binary 加载：每个估计 10-30 秒 × 8 个 ≈ **80-240 秒**
- **合计预计：3-6 分钟**

**补充说明**：
- `--lowram` 标志对 ZImage 无效（代码注释写的只对 sdxl/anima 有效，`PipelineZImage` 构造函数不接受 lowram 参数）
- 热管理日志在 1:45 后出现 `tempLevel=5, coolDown=5`，证明 DSP 在长时间高负荷运行，由于初始化过程过长，用户在等待期间点击生图会遇到 ECONNREFUSED，进而手动停止任务，导致进程被主动 kill。

---

## 三、已做改动（临时修复）

| 改动 | 状态 | 持久性 |
|------|------|--------|
| 将 `/odm/lib64/aiframe/cdsp/unsigned/libQnnHtpV79Skel.so` 复制到 `qnn_runtime_libs/aarch64-android/` | ✅ 已完成 | **临时**（模型重新导入后失效） |

---

## 四、当前状态

| 层级 | 状态 |
|------|------|
| FastRPC Skel 加载 | ✅ 已修复（unsigned Skel 正确加载） |
| 初始化阶段不崩溃 | ✅ 确认（进程存活 60s+ 无 SIGABRT） |
| HTTP 8081 监听 | ❓ **未确认**（需等待 3-6 分钟后验证） |
| 图像生成 | ❓ **未确认**（8081 未就绪前无法测试） |

---

## 五、后续检查方向

### 方向 A：确认初始化最终是否成功（优先）

> **操作**：触发后端后，**等待 5-6 分钟不做任何操作**，然后用 ADB 检查：
> ```bash
> adb shell "cat /proc/net/tcp6 | grep '1F91'"   # 0x1F91 = 8081
> adb shell "ps -A | grep libstable"
> ```
> 如果 8081 在监听，再点击生图测试是否能成功生成图像。

**可能的结果**：
- 8081 开始监听 → 说明初始化成功，只需修复 UI 超时逻辑
- 进程死亡且 8081 未开 → 存在 v73 context binary 在 v79 HTP 上不兼容的问题

---

### 方向 B：SM8550 模型文件在 SM8750（v79 HTP）上的兼容性

当前模型文件名均带有 `SM8550` 后缀（HTP v73 架构），例如：
```
text_encoder_part1_ctx.SM8550.bin
```

若方向 A 证实初始化失败，需确认：
- QNN SDK 的 v79 Skel 是否能向下兼容加载 v73 context binaries
- 如不兼容，需用 QNN SDK（针对 SM8750 / HTP v79）**重新编译**所有 8 个 context binary

---

### 方向 C：源码层面的永久修复（无论方向 A 结果如何都需要做）

#### C1：永久固化 Skel 文件（必须）
当前 Skel 是手动推入的临时文件，模型重新导入后会丢失。

**选项 1**（推荐）：修改 `BackendService.kt` 的 `DSP_LIBRARY_PATH`，追加设备的 ODM Skel 路径：
```kotlin
// BackendService.kt ~L571
// 当前：
env["DSP_LIBRARY_PATH"] = qnnRuntimeDir.absolutePath
// 修改为（SM8750/OnePlus 设备）：
env["DSP_LIBRARY_PATH"] = qnnRuntimeDir.absolutePath + 
    ":/odm/lib64/aiframe/cdsp/unsigned"
```
> ⚠️ 缺点：路径是设备特定的，不够通用

**选项 2**（更健壮）：在模型导入时，自动从 `/odm/lib64/aiframe/cdsp/unsigned/` 复制 Skel 到模型运行库目录，写入 `ModelImportService` 或类似逻辑。

**选项 3**：将 `libQnnHtpV79Skel.so` 打包进 APK 的 `assets/qnnlibs/`（需 Qualcomm 授权）。

#### C2：修复超慢初始化（必须）
核心问题是 `validateGraphFiles()` 在每次启动时都读取全部 ~11GB 数据做 SHA-256，耗时 2+ 分钟。

**选项 1**：在 ZImageQnnContract 中增加校验缓存（将 SHA-256 结果写入 `.sha256cache` 文件，文件 mtime 未变则跳过）。

**选项 2**：在 BackendService 启动时改变状态机，支持"正在初始化"状态，UI 显示进度条而不是立即允许生图。

**选项 3**：实现 ZImage 的真正 lowram 模式 —— 先启动 HTTP 服务器，收到生图请求时才逐步加载模型（类似 Anima 的分阶段加载）。

#### C3：BackendService 运行时校验修正
`BackendService.kt:432` 检查 `libQnnHtpV73Stub.so`，但实际运行时需要的是 **V79Stub + V79Skel**。这个检查虽然能通过（因为包里有 V73Stub），但逻辑上是针对 SM8750 不正确的。
