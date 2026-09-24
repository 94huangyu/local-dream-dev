# 插线手册 —— mg + 共享 spill-fill（2026-09-07 定稿）

> 🟢 **2026-09-17 已执行并交付**。本手册以下内容保留为当时的定稿；**与之不同的实际执行**：
> ① G4 为三档（>=600 / 375~600 / <375，见 EXP_PLAN「判据修订」），实测 734；
> ② **H2 已由用户决定修订**为「ION 平台期 MemAvail 中位数 >= 同日现网」（原「单秒最低 > 800」无分辨力，现网自己也不过）；
> ③ 共享 spill-fill 的**组大小不再写死**，由阶段 0 从交付 .bin 元数据现读、写进 `SHARE_SPILLFILL` 内容（写死曾致 mg 装载失败）；
> ④ 新增 `--smoke`（交付后只出 1:1）；§0 里的 `.so` 字节数已过时（现为 19,116,448 B）。
> 全过程与五次运行的证据见 `scripts/EXP_PLAN_SPILLFILL.md` 各「执行记录」。
> ✅ **§7 清理已于 2026-09-17 执行**（用户指示），清理后复验逐字节同现网。🔴 **此后 `--rollback` 会拒绝执行**（single 文件已不在设备上），恢复路径见 HANDOVER §15.45.6 / 台账 #171。

> **一次插线跑完：验证 → 交付 → 验收。** 判据事前锁定在 `scripts/EXP_PLAN_SPILLFILL.md` §4，
> 任何一门不过 **自动回滚**，不留半截状态。
> **预计 60~80 分钟**（含每个计时臂前的 NPU 冷却等待），**全程不能拔线**。

---

## 0. 插线前（纯宿主，已全部做完，可反复复核）

```bash
python scripts/oneshot_mg_session.py --check
```

现在的状态（2026-09-07）：

| 项 | 状态 |
|---|---|
| `aspect_preflight.py`（含用 app 真正的 C++ 解析器解契约） | ✅ 全过 |
| 交付物 staged 到 `D:\ZImage_Work\deliver_mg`，**sha256 逐个与契约相符** | ✅ 8.26 GiB |
| APK 已编，`.so` 与现网不同（改动确认进产物） | ✅ 19,115,664 B |
| 回滚用的 single 契约 | ✅ `logs/contracts_20260906/final_qnn_contract.single.json` |

---

## 1. 插上线，跑一条命令

```bash
python scripts/oneshot_mg_session.py --run
```

**全程不需要你操作**（2026-09-07 更新：非 1:1 比例已改为经 HTTP 传 `width`/`height` 自动跑，
不再需要在 UI 里选尺寸）。

| 阶段 | 内容 | 时长 |
|---|---|---|
| 1 | 现网 APK 出一张图（铁律 1：当天证明现状可用）＋ 存档 APK 与契约（铁律 2） | ~8 min |
| 2 | **实验 1**：装新 APK，marker 关/开各出一张图 | ~20 min |
| 3 | **交付**：推 8.26 GiB + 换契约 + 开 marker | ~10 min |
| 4 | **五个比例逐个出图 + 自动校验 PNG 实际宽高** | ~35 min |

**合计约 75~90 分钟。**

---

## 2. 判据（事前锁定，脚本只读不改）

**实验 1（决定要不要交付 mg）**

| 门 | 判据 | 不过 |
|---|---|---|
| G2 | 关臂出图 sha256 与现网**逐字节相同** | 回滚，停 |
| G3 | 开臂出图 sha256 与现网**逐字节相同** | 回滚，停 |
| **G4** | 开臂 ION 峰值比关臂低 **>= 600 MiB** | 回滚，停（**#146 就此判负**）|
| G5 | 耗时增量 **<= +5 s** | 回滚，停 |

**交付验收**

| 门 | 判据 |
|---|---|
| H1 | ION 峰值 **<=** 现网同日实测值 |
| H2 | 峰值 MemAvail **> 800 MiB**（#161 被杀时是 208）|
| H3 | 1:1 出图 sha256 与现网**逐字节相同** |
| H4 | 设备占用 **<= 13 GiB**（现网 35 GiB）|

---

## 3. 预期结果（③预测，用来对照，不用来放行）

| 量 | 现网 single | mg + spill-fill（预测）|
|---|---|---|
| 设备占用 | 35 GiB | **12.3 GiB** |
| 后端启动校验 | 35 GiB 的 sha256 | **12.30 GiB → 约 203 秒** |
| 首次用到某个新尺寸 | 每个 **+100 秒** | **0**（五比例共用同一批文件）|
| ION 峰值 | 9111 | **~8649（−462）** |

预测的来历：宿主公式已用两个设备实测标定
（mg−single：公式 +361 vs 实测 +375，误差 3.7%），见 `EXP_PLAN_SPILLFILL.md` 执行记录。

---

## 4. 出问题怎么办

- **脚本自己会回滚**（换回 single 契约 + 装回存档的 APK + 删 marker），并打印
  「设备已回到交付前状态，可以拔线」。
- 手工回滚：`python scripts/oneshot_mg_session.py --rollback`
- 从中途续跑：`python scripts/oneshot_mg_session.py --run --phase 3`
- 🔴 **看到报错先问一句**：它指向的对象，是我实际验证过的那个吗？
  本项目栽过两次（`qnn-net-run failed` 实为 USB 掉线；「设备不受支持」实为过期清单）。

---

## 5. 五比例验收（已自动化）

后端的 HTTP 接口收 `width`/`height`（`RequestParser.hpp:42-43`），
所以五个比例全部由脚本发请求完成，**不需要在 UI 里手动选**。

🔴 **但必须校验出图的实际宽高**：对**未交付**的尺寸，后端是
**静默回落到 1024×1024**（`RequestParser.hpp:82-92`），**不报错**——
不校验的话，一个根本没生效的比例会以「1024 的图」冒充通过。
⇒ 脚本直接读 PNG 头的宽高字段（该解析已用三张已知图验证，含非正方的 1184×896）。

判据 **H5**：五个比例都出图，且 **PNG 实际宽高 == 请求的宽高**。

提示词与种子由脚本固定为 `a cute orange cat sitting on a wooden table, masterpiece, best quality` / **42**
——与宿主所有 FP32 参考所用的**逐字一致**（#160，换词则任何 PSNR 比较都失去意义）。

## 6. 设备占用告知（MAINLINE §7.0）

- 脚本启动即占用设备，**全程不能拔线**
- 结束时会明确打印「设备空闲，可以拔线」
- 中途想停：Ctrl+C 后立刻跑 `--rollback`，不要留半截状态

---

## 7. 🔴 最后一步：清理（**五个比例都验完之后再做**）

`deploy_aspect.sh` 是**只增不删**。交付之后设备上会同时存在：

| | 大小 |
|---|---|
| 旧 single 形态的 .bin（20 个，不再被契约引用） | ~30 GiB |
| 新 mg 的 5 个 + 4 个 text_encoder（契约引用的） | **12.30 GiB** |
| 部署脚本留在 `/data/local/tmp/aspect` 的暂存区 | 8.26 GiB |

⇒ **不清理的话设备占用不降反升，P1 的目标等于没达成。**

```bash
python scripts/oneshot_mg_session.py --cleanup          # 只列不删（先看一眼）
python scripts/oneshot_mg_session.py --cleanup --yes    # 真删
```

它按**设备上当前那份契约**算引用集合，删掉不被引用的 `.bin` 与暂存区。

🔴 **删完之后，回滚 single 需要重新推 25.5 GiB（约 11 分钟）**：
宿主产物仍在 `p0_experiments/aspect/` 与 `p2attr/`，契约在 `logs/contracts_20260906/`，
所以只是慢，不是不可逆。**这也是为什么清理不放在 `--run` 里自动做。**

---

## 8. 已知的坑（都是本轮实测查出来并已修的，别再踩）

| 坑 | 后果 | 已修 |
|---|---|---|
| `.bin` 在 `files/models/ZIMAGE/**models**/`，不在 `ZIMAGE/` | 按错路径 stat ⇒ 报「文件全缺」⇒ **把成功的交付判成失败并回滚** | ✅ |
| 部署暂存区是 `/data/local/tmp/**aspect**`（不是 `aspect_stage`） | 清理时静默漏删 8.26 GiB | ✅ |
| H4 若量整个目录 | 交付后目录含旧 single ⇒ 必然超阈值 ⇒ 误判 | ✅ 改为只量契约引用的文件 |
| 换契约后不 force-stop | `app_generate.sh` 见端口在就复用**内存里的旧契约**，**不报错**，只给出没意义的数字 | ✅ |
| 给 `sh` 传 Windows 路径 | `for f in "$SRC"/*.bin` 通配符不展开、`stat` 找不到文件 | ✅ 统一转 `/d/...` |
| `--phase N` 缺前置 state | 设备已占用时 KeyError 崩掉 | ✅ 加了续跑守卫 |

---

## 9. 顺带采集：P2 的逐段耗时（零额外设备时间）

本轮的 APK 在 `runSeg` 里加了 `[segtime]` 埋点（台账 #169）。
`--run` 会生 7 张图，logcat 已经被 `app_generate.sh` 抓下来，**跑完直接解析即可**：

```bash
python scripts/timing_breakdown.py --seg scratch_runs/os_g0_live_logcat.txt
python scripts/timing_breakdown.py --seg scratch_runs/os_mg_1024x1024_logcat.txt
```

会得到 part1a / part1b / part2a / part2b **各占每步多少毫秒**——
这是 P2「生图速度」的第一块地基（此前只知道「步循环占 82.9%、每步 16.22 s」，再往下没有分辨力）。

⚠️ 段级墙钟**含输入搬运与输出拷贝**，不是纯计算时间。
⚠️ 拿 2026-09-08 之前的 logcat 跑会明确提示「没有 [segtime]」，不会假装有数据。
