# -*- coding: utf-8 -*-
"""主线 B：把 EXP_PLAN_SPEEDB 的可复用流程性结论追加进 QNN_CONVERSION_GUIDE。

只做「在文件末尾追加一个新小节」，不改任何既有正文（追加只允许切片，禁止 replace）。
"""
import io, os

GUIDE = os.path.join(r"D:\LocalDreamZImage", "docs", "QNN_CONVERSION_GUIDE.md")

with io.open(GUIDE, encoding="utf-8") as f:
    old = f.read()

assert "## 三十七、" not in old, "§三十七 已存在，避免重复追加"
assert old.rstrip().endswith("让被观测的命令自己决定返回码。"), "末尾不符预期，先人工核对"

SEC = u"""

---

## 三十七、🔴🔴 **量 HTP 模型的内存，进程 RSS 是错的口径 —— 要看 `IonTotalUsed`**（2026-08-29 实测）

> 转新模型做内存预算时**第一件要用对的东西**。本项目此前所有内存判断都建立在
> 「context 文件大小」或「PD 分配估算」上，**从未量过设备上真实占用**，
> 三种量还被混在同一个算式里用（见 `EXP_PLAN_SPEEDB.md` §1.2）。

### 37.1 三种量必须分清，它们不相等

| 量 | 怎么取 | 本项目实测（Z-Image，L=80） |
|---|---|---|
| ① context **文件字节** | 宿主 `ls -l *.SM8750.bin` | part1a 2263 MiB／part1b 1398／part2a 1354／part2b 1398 |
| ② **PD 分配估算** | 建 context 失败时报错里的 `context size estimate` | part2a 1887 MB／part2b 1927 MB |
| ③ **设备实际占用** | `/proc/meminfo` 的 **`IonTotalUsed`** | 每个 part2 半段 **1936~1990 MiB** |

⇒ ①**实测倍率：③ ≈ ① 的 1.42×**，而 ③ 与 ② 吻合。
**用文件大小做内存预算会低估约 40%。**

### 37.2 🔴 进程 `VmRSS` / `VmHWM` **看不见 context**

①**实测**：后端进程生成全程 `VmRSS` 只有 **~200 MiB**；唯一的峰值 **2356 MiB**
出现在「把 `.bin` 读进 CPU 缓冲区」那一瞬（≈ part1a 文件大小 2263 MiB），
交给 QNN/DSP 之后 CPU 侧立即释放。

⇒ ②**context 常驻在 DSP/ION 侧，不计入宿主进程的 RSS。**
拿 `ps -o RSS` 或 `dumpsys meminfo <pid>` 做 HTP 模型的内存预算，**量到的是加载缓冲区，不是模型**。

⚠️ 附带一个会误导人的细节：`VmHWM` 可能**小于**同一时刻读到的 `VmRSS`
（实测 2,412,344 < 2,412,992 kB）。内核 `update_hiwater_rss()` 是惰性更新 ⇒
**`VmHWM` 会滞后，不能当唯一的峰值来源**，要配轮询峰值一起看。

### 37.3 正确做法：设备侧 1 Hz 采样 `IonTotalUsed`

```sh
# 关键：启动时把 nsp* 热分区路径缓存一次。每 tick 重扫 85 个分区会把采样率压到 0.5 Hz，
# 而 context 加载是秒级事件，采样率不足会漏掉峰值。
awk '/IonTotalUsed/{print $2}' /proc/meminfo     # 单位 kB
awk '/^MemAvailable/{print $2}' /proc/meminfo    # 判「还塞不塞得下」用这个
```

**不要在宿主上用 `adb shell` 逐 tick 轮询**——每次往返 100~200 ms，实测把采样率压到 0.5 Hz。
把循环推到设备上跑（`scripts/speedb_sampler.sh`），一条长连接把结果流回宿主。

**读法**：分段常驻/释放的流水线，ION 曲线是干净的**方波**，
两个电平之差**就是那一段的真实占用**——这是最可靠的分段内存归因手段，比任何估算都硬。

### 37.4 判「还能不能再常驻一段」的正确判据

**不要拿峰值去比某条历史「被杀线」**——那条线多半是别的口径（本项目的 7300 MB 就是，
拿它当 ION 阈值会把**已经稳定运行的现网配置**判死）。

正确判据只用同一次测量里的两个数：

```
额外需求（下一段的 ION 占用）  vs  峰值时刻的 MemAvailable
```

①**本项目实测**：额外需求 **1936 MiB** vs 峰值可用 **855~980 MiB** ⇒ 缺口约 1000 MiB ⇒ 否决。
**这个论证自足，不依赖任何历史阈值。**

### 37.5 配套：可复现的**速度**测量协议（实测复现性 0.152%）

改任何东西之前先把尺子标定好，否则收益会被淹没：

1. `am force-stop` —— 让后端成为全新进程
2. **静置到冷机**：所有 `nsph*` 分区 ≤ `T_idle + 3 °C`（本机实测 58.8 → 46.1 °C 约需 **6 分钟**）
3. 重启 app 等端口监听，**此段不计时**
4. 固定 prompt + 固定 seed，测**冷机第一张**；主时钟取 `curl time_total`

①**实测 n=3：202.486 / 204.094 / 204.405 s，`\\|B2−B3\\|/mean = 0.152%`**，
三张图 sha256 完全相同，ION 峰值逐字节相同。

⚠️ **划界**：这个口径测的是**冷机第一张**，**不是用户连续使用时的稳态**。引用时必须注明。
①实测热行为：**单张图内 NPU 就冲到 103.4 °C**（起点 48~50 °C），步循环内在 60~103 °C 剧烈摆动
⇒ ②「连续多张耗时爬升」累积的不是 NPU 结温（单张已打满），③但累积在哪未实测。
"""

with io.open(GUIDE, "w", encoding="utf-8", newline="") as f:
    f.write(old + SEC)

with io.open(GUIDE, encoding="utf-8") as f:
    new = f.read()
assert new.startswith(old), "🔴 追加破坏了原文"
print("追加 %d 字符；原文 %d 字符逐字节保留" % (len(new) - len(old), len(old)))
