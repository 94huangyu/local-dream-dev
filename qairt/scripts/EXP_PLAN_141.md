# EXP_PLAN_141 —— 后端进程中途死亡：**先定死因，不修**

> 状态：**执行前定稿（2026-08-29）**。判据锁定，事后不得修改。
> 用户明确要求：**必须明确是问题的前提下再修**。本方案只做诊断。

## 一、已经坐实的（不要重复查）

- **分类**：`BackendService.kt:583` 用 `ProcessBuilder` 执行原生可执行文件
  （`libstable_diffusion_core.so`）⇒ 确实是 app 的**子进程**，Android 12+ 归为 phantom process。
  app 有前台服务（`foregroundServiceType="dataSync"`），但**前台服务只保护主进程，不保护 phantom 子进程**。
- **现象**：画质评测批次连跑到第 7 张时 curl 报 rc=18（生成到 11.8 秒中断），第 8 张 rc=52；
  前 6 张全部 HTTP 200（197~203 s/张）。**两条 prompt 单独重跑均成功** ⇒ 与 prompt 无关。
- **设置**（2026-08-29 实查，Android 16）：`settings_enable_monitor_phantom_procs`、
  `max_phantom_processes`、`max_cached_processes` **三者均为 `null`（用平台默认）**
  ⇒ **可排除「进程数超上限（默认 32）」**——本 app 只派生 1 个子进程。

## 二、🔴 尚未区分的三个死因（本实验要分开它们）

| # | 假设 | 若成立，logcat 里应出现 |
|---|---|---|
| H1 | **LMK 内存压力** | `lmkd` 缓冲区里有 kill 记录，且 MemFree/MemAvailable 在死亡前显著下滑 |
| H2 | **phantom 进程监控主动杀** | `ActivityManager` 侧有 `Kill.*phantom` / `killPhantomProcess` 类记录 |
| H3 | **后端自身崩溃**（段错误/OOM/断言） | `libc`/`DEBUG` 的 signal 记录，或 tombstone |

⚠️ **`I/ActivityManager: Process ... died` 只是「观察到死亡」的通知，三种情况都会打**
——2026-08-29 我据此写过「Android 主动杀除」，那是过度声称，已订正。

## 三、实现路径

| 步 | 动作 | 关键点 |
|---|---|---|
| 1 | `logcat -c` 后开 **`logcat -b all -v threadtime`** | 🔴 **必须 `-b all`**：lmkd 与 crash 不在默认 buffer 里。上一次只抓了默认 buffer，这是死因查不出来的直接原因 |
| 2 | 每轮生成前后记录 `MemFree`/`MemAvailable`、后端 PID 与其 `oom_score_adj` | 给 H1 提供量化曲线，而不是只有一个末态快照 |
| 3 | 连续生成，**每张用不同 basename**（`d141_01`…）直到复现或跑满 10 张 | 🔴 上次补跑用同名把崩溃现场覆盖了，本轮杜绝 |
| 4 | 复现后立刻 `adb pull` logcat 与 tombstone 目录 | 现场只有一次 |

## 四、判据（事前锁定）

| 判定 | 条件 | 后续 |
|---|---|---|
| **H1 成立** | lmkd 缓冲区有对应 PID 的 kill 记录 | 修法方向 = 降内存驻留（与 #132/#16 同族），**且需先确认收益值不值** |
| **H2 成立** | ActivityManager 有 phantom kill 记录 | 修法方向 = 让后端不再是 phantom（前台服务内进程 / 独立 `android:process`），**属 app 架构改动，须单独评估** |
| **H3 成立** | 有 signal/tombstone | 是我们自己的 bug，**优先级最高** |
| **不复现** | 10 张跑完都正常 | ⇒ 现象**不稳定**，不满足「明确是问题」⇒ **降级为观察项，不修**，并记录复现条件未知 |

🔴 **无论哪一种，本轮都只出诊断结论，不动代码。**

## 五、成本与风险

约 40 分钟设备时间（10 张 × ~3.5 分钟，可能提前复现）。
⚠️ 会占满手机；⚠️ 连续生成会把设备内存压低，**这正是要观测的量**，不是副作用。
