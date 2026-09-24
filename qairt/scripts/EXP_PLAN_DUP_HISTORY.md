# EXP_PLAN_DUP_HISTORY —— #126「一次生图写多条历史」根因确认

**定稿时间**：2026-08-26 20:0x（**判据在用户点击生成之前锁定，不得事后修改**）
**状态**：根因已由静态证据定位，本实验只做**事前预测式确认**。

---

## 一、已拿到的事实（全部为实测，非推断）

### 1.1 设备 DB（`generation_history`，2026-08-26 09:16~09:45 五次生图）

| 时间 | 行 | prompt | 磁盘 PNG 数 |
|---|---|---|---|
| 09:16:20.257/.258 | id=34/35 | `一个红色的立方体…` / **`刘亦菲`** | **2**（…257.png、…258.png，同为 863906 B） |
| 09:21:43.935/.935 | id=36/37 | `男人戴着一枚小的银色胸针` / **`刘亦菲`** | 1 |
| 09:32:46.085/.086 | id=38/39 | `一位年轻的东方女性…` / **`刘亦菲`** | **2** |
| 09:39:23.008/.008 | id=40/41 | **`刘亦菲`** / `一位年轻的东方女性…` | 1 |
| 09:45:14.310/.310 | id=42/43 | `一位年轻的东方女性…红色露肩毛衣` / **`刘亦菲`** | 1 |

⇒ ①**今天 5 次生图，5 次全部写 2 条**（不是「间歇性」）。
⇒ ①**每对里除 `prompt` 外，19 个字段逐个相同**（seed / width / height / mode / steps / cfg /
scheduler / generationTime / runOnCpu / useOpenCL / denoiseStrength / upscalerId / negativePrompt…）。
⇒ ①**时间戳相差 0~1 ms**，且当相差 1 ms 时**磁盘上真的有两个 PNG**（文件名 = 时间戳）。

### 1.2 代码（`local-dream/app/src/main/java/io/github/xororz/localdream/`）

- `data/HistoryManager.kt:90` —— `saveGeneratedImage()` 内部自己取
  `val timestamp = System.currentTimeMillis()`，文件名 = 该时间戳
  ⇒ **两次并发调用落在同一毫秒 ⇒ 后者覆盖前者 ⇒ 1 个 PNG + 2 条记录；
  跨毫秒 ⇒ 2 个 PNG + 2 条记录。** 与 1.1 的磁盘现象逐条吻合。
- **生图路径下 `saveGeneratedImage` 只有一个调用点**：
  `ui/screens/ModelRunScreen.kt:1430`，位于 `LaunchedEffect(serviceState)` 的
  `is GenerationState.Complete ->` 分支内（另一个调用点在 2960~3040 的放大器流程，
  本次无 `upscalerId` ⇒ 已排除）。
- 该分支写入的 prompt 取自 `generationParamsTmp`（`ModelRunScreen.kt:309` 的 `remember`），
  **每个 composable 实例各有一份**。
- `serviceState` 来自 `BackgroundGenerationService.generationState`
  （`service/BackgroundGenerationService.kt:64`），是 **companion object 里的全局 StateFlow**
  ⇒ **进程内每一个活着的 ModelRunScreen 都会收到同一个 `Complete`。**
- `AndroidManifest.xml:38` 的 `MainActivity` **没有 `android:launchMode`** ⇒ 默认 `standard`
  ⇒ **每次 start 都新建一个实例**。

### 1.3 设备实测：任务里确实叠了多个 Activity

`dumpsys activity activities`：`Task #7888 … sz=3`，三条 `Hist #0/#1/#2` 全是
`MainActivity`，同一进程 `24678`，各自都有 `windows=[…]`。启动来源：

| 实例 | 来源 |
|---|---|
| #0（root） | `mCallingUid=2000` = **adb shell**（`scripts/app_generate.sh:81` 的 `am start`） |
| #1 | `mCallingUid=10225` = 桌面启动器图标 |
| #2 | 同上 |

⚠️ #0 由 `am start -n` 拉起且**不带 MAIN/LAUNCHER**，导致任务 root 的 intent 与桌面图标
不匹配 ⇒ 之后每次点图标都**再叠一个**，而不是回到已有实例。
⇒ **本项目的脚本放大了这个缺陷，但缺陷本身是上游的（缺 `launchMode`）。**

---

## 二、机制（②由上述数据严格推出）

> **进程里有 N 个活着的 `MainActivity` ⇒ N 份 `ModelRunScreen` composition ⇒
> N 份独立的 `generationParamsTmp` ⇒ 一个全局 `Complete` 触发 N 次
> `saveGeneratedImage()` ⇒ N 条历史记录、1~N 个 PNG。**

**为什么 `prompt` 是唯一不同的字段**：其余字段在各实例里都是同一套默认值
（1024/1024、dpm、20、7.0…），只有 prompt 是用户逐次改的；
**旧实例的 `generationParamsTmp.prompt` 冻结在它自己最后一次生图时的值**
⇒ 今天 5 对里那条恒为 `刘亦菲`（= 08-26 00:18 id=33 那次的 prompt）。

**这同时推翻了两条旧记载**（约束 2，原文保留在 MAINLINE #126/#130）：
- #126「不是生成两次，是同一张图入库两次」 ⇒ ❌ **是 save 例程被调用了两次**，
  跨毫秒时磁盘上确实有两个 PNG。
- #130 由「两条 prompt 不同」推出的「原机制描述需重估」⇒ ✅ 方向对，本条给出正确机制。

---

## 三、事前锁定的判据（用户点「生成」之前定稿，**不得事后修改**）

**执行**：用户正常打开 app（记录打开方式），点一次生成，用一个**从未用过的 prompt**。
我在设备上同步抓 `logcat`，生成结束后重新拉 DB。

| 编号 | 预测 | 判为通过 | 判为失败（机制错） |
|---|---|---|---|
| P1 | 新增历史行数 **N == 生成时进程内活着的 MainActivity 实例数** | 相等 | 只有 1 行，或与实例数不等 |
| P2 | `logcat` 里 `ModelRunScreen: update bitmap` 出现 **N 次** | 出现 N 次 | 只出现 1 次却写了多行 |
| P3 | `logcat` 里 `ModelRunScreen: start generation` 只出现 **1 次** | 1 次 | ≥2 次（则是重复发起，另一种机制） |
| P4 | N 行中**恰好 1 行**是本次新 prompt，其余为各旧实例冻结的 prompt | 符合 | 全部相同 prompt |
| P5 | N 行除 prompt 外字段全同，时间戳相差 ≤ 数毫秒 | 符合 | 差异大 |

### 3.1 🔴 点击前实测到的装置状态（**2026-08-26 20:0x，用户点「生成」之前锁定**）

`dumpsys activity activities`：`Task #7888 … sz=4`（用户点桌面图标后从 3 变 4，**与 §1.3 的
「每点一次图标叠一个」预测一致**）。四个实例的生命周期状态：

| 实例 | state |
|---|---|
| #3（最上层，用户正在看的） | **RESUMED** |
| #2 / #1 / #0 | **STOPPED**（`finishing=false`，进程同一个，未销毁） |

⚠️ **由此必须把 P1 拆成分支判据**（原因：Compose 的 Recomposer 在 lifecycle < STARTED 时
会暂停重组，**停止态的 composition 是否仍会响应全局 StateFlow 的 `Complete`，
本项目从未验证过**，不得当成已知——约束 5/8）：

| 分支 | 观测 | 判读 |
|---|---|---|
| **P1a** | 新增 **4** 行 | ✅ 机制全对：**所有活着的实例都写**，与 state 无关 |
| **P1b** | 新增 **1** 行 | ❌ 机制部分错：停止态不参与 ⇒ 今天那 5 次的 2 条**另有触发路径**，需重查 |
| **P1c** | 新增 **2~3** 行 | ⚠️ 只有部分实例参与 ⇒ 需按 state 细分，补测「事后把某个 STOPPED 实例切回前台是否再多一行」 |

**P6（新增，指纹判据）**：**从未生成过的实例**其 `generationParamsTmp` 仍是初值
⇒ 它写出的行应为 `prompt=""`、**`steps=0`、`cfg=0.0`**。
若出现这种行 ⇒ 直接坐实「该行来自另一个 composable 实例」，是比 prompt 更硬的指纹。

**P1b 的后续动作（事前写好，不许临时想）**：generation 结束后把 #2 切回前台，
再查一次 DB —— 若此时才多出一行，则机制是「**停止态延迟到恢复时才写**」，
仍是同一个根因，只是触发时机不同。

**失败时的区分（约束 4）**：
- P3 失败 ⇒ **方向错**：不是多实例，是生成被发起了多次。
- P1 失败但 P2 通过（`update bitmap` N 次却只写 1 行）⇒ **执行方法错**：
  多消费者成立，但写库处另有去重，需改查 `dao.insert` 的冲突策略。

---

## 三·补、🔴 **G2 空白对照：实例数 1 vs 4**（**修法定稿前必须先做**，2026-08-26 定稿）

**为什么必须做**（用户质疑「你确定这是根因吗」后补的自审，约束 9·触发条件 3）：
已证明的是「**N 个 composition**」，**没有**证明「**N 个 Activity**」。
两者对修法是**决定性分岔**：

| 若第二个 composition 在 | 则 `launchMode="singleTask"` |
|---|---|
| **另一个 Activity 实例** | ✅ 对症 |
| **同一个 Activity 内**（重复 nav 条目 / 转场期同时 composed） | ❌ **完全无效**，等于改错东西 |

⇒ **不做这个对照就改代码，是在未验证的前提上交付**（约束 11·①、ledger_lint R1）。

**设计（单变量：只改实例数，不改任何代码）**：

| 步 | 动作 | 记录 |
|---|---|---|
| G2-0 | 逐次 Back 退出栈顶实例，**每退一层截一张图** | 记录每个实例停在哪个页面 —— 直接检验「只有进过模型页的实例才写」这条③假设 |
| G2-1 | 收敛到 **`sz=1`**，用 `dumpsys` 确认 | 自变量确立 |
| G2-2 | 同 prompt 同 seed 跑一次生成 | 与本轮 4 实例那次构成对照 |

**事前锁定的判据**：

| 编号 | 观测 | 判读 |
|---|---|---|
| **Q1** | 新增 **1 行** 且 `update bitmap` **1 次** | ✅ **多 Activity 实例 = 真因**，`launchMode` 修法对症 |
| **Q2** | 新增 **≥2 行** | ❌ **根因判错**：第二个 composition 在同一 Activity 内 ⇒ `launchMode` 无用，必须重新定位（查 nav 重复条目 / 转场期双 composed） |
| **Q3** | 翻页动画**完整跑完不卡** | ✅ 佐证卡顿机制（第二个 consumer 抢先 `markBitmapConsumed()`） |
| **Q4** | 翻页仍卡但只有 1 行 | ⚠️ 两个症状**不同根因**，卡顿需单独定位 |

**成本**：设备约 6 分钟（含一次 4.5 分钟生成），**零建图、零编译、不改任何文件**。
**对比**：若跳过它直接改代码重编 APK，一轮约 40~60 分钟且要装机，
而**前提错的概率不为零** ⇒ 代价门通过，先做 G2。

## 四、修复方案（**待实验确认后再动手**，不在本轮交付）

两层，缺一不可：

1. **堵住多实例**：`AndroidManifest.xml` 的 `MainActivity` 加
   `android:launchMode="singleTask"`（或 `singleTop` + `onNewIntent`）。
   同时把 `scripts/app_generate.sh:81` 的 `am start` 改成带
   `-a android.intent.action.MAIN -c android.intent.category.LAUNCHER`，
   使脚本拉起的实例与桌面图标同源，不再叠栈。
2. **让消费者幂等**（即使只剩一个实例也该有）：用 `Complete` 的**实例身份**做门，
   已处理过的 `Complete` 不再重复保存 —— 这条对应 #126 早先记的
   「`Complete` 是锁存状态、`clearCompleteState()` 只在退出/dispose 调用」。

⚠️ 交付走**约束 11 四条铁律**（先证现状可用 / 先备份 APK / 交付后自己跑真实链路 /
不得用 `run-as` 验证）。**建议与 #132 提速改动一起打包重编，只承担一次装机风险。**

---

## 五、与主线的关系（约束 6 三问）

① 在不在主线：**不在**主线 A/B，属 app 交付质量缺陷（#126）。
② 对顶层问题的贡献：**间接但真实** —— 重复行会污染「同 prompt 同 seed 两版对照」这类
判断（#130 正是被它带偏过一次）。
③ 前置条件：满足（设备在线、证据齐全、零建图成本）。

⇒ **限定范围**：本轮只做「确认 + 写方案」，修复代码与 #132 一起交付，
不因此挤占提速主线。
