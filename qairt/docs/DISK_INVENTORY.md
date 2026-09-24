# 磁盘占用清点与清理判定（2026-08-22）

> **做法**：先量 → 建引用图 → 哈希查重 → 按**再生成本**分类。**不按文件名猜。**
> 全部判定都附了证据来源；执行删除前请对照本表。

## 0. 全局

| 路径 | 大小 | 说明 |
|---|---|---|
| **`D:\ZImage_Work`** | **197.98 GB** | 主要占用来源 |
| `D:\Z-Image-Turbo` | 30.64 GB | 官方模型（对照基准，**必须保留**） |
| `D:\ZIMAGE` | 10.88 GB | 分词器等 |
| `D:\qairt` | 5.24 GB | SDK（**必须保留**） |
| `D:\LocalDreamZImage` | 4.69 GB | 代码与文档 |

D 盘：已用 416.8 GB / **可用 83.2 GB**。

## 1. 关键前置事实（决定了什么能删）

### 1.1 多个 `.onnx` **共享**同一份 `.data` —— 按名字删会毁掉整条流水线

| `.data` | 大小 | 被几个 onnx 引用 | 判定 |
|---|---|---|---|
| `text_encoder_fixed4.onnx.data` | 15.69 GB | **5**（含 `text_encoder_part1~4`，**活流水线**） | 🔴 必须保留 |
| `transformer_part1.onnx.data` | 13.76 GB | **9**（含 `part1a`/`part1b`，**活**） | 🔴 必须保留 |
| `transformer_part2.onnx.data` | 10.86 GB | **8**（含 `part2a`/`2b`/`fixed_dce`，**活**） | 🔴 必须保留 |
| `vae_decoder.onnx.data` | 0.20 GB | 1 | 🔴 必须保留 |
| **`text_encoder.onnx.data`** | **15.69 GB** | **1**（只有已被取代的单体 `text_encoder.onnx`） | ✅ **可删** |

⚠️ **`transformer_part1.onnx` 这个名字看着像废弃的单体模型，但它的 `.data` 是 `part1a`/`part1b` 的权重来源。**

### 1.2 再生成本（指南 §七 实测）

| 阶段 | 耗时 | ⇒ 结论 |
|---|---|---|
| ONNX → **fp32 DLC** | **≈2 分钟** | **fp32 DLC 是廉价可再生品** |
| fp32 DLC → 量化 DLC | ≈3 小时 | 昂贵，保留 |
| 量化 DLC → context `.bin` | ≈3 小时 | 昂贵，保留 |

**已验证**：9 个待删 fp32 DLC **全部**有存活的量化产物，而量化 DLC 里**同时保存 Converter 与 Quantizer 两条命令**
⇒ 删掉 fp32 DLC 不会丢失复现所需的信息。

### 1.3 哈希确认的逐位重复

| 重复项 | 大小 | 证据 |
|---|---|---|
| `part1b_nofuse_ctx` / `part1b_hps_ctx` vs 基线 context | 2.94 GB | md5 三者同为 `83F16066E57A5ED6` ⇒ **同时实测确认 #10/#27 的开关确实被静默忽略** |
| `text_encoder_part1~4` 的 `_ctx.bin`（无 SoC 后缀） vs `_ctx.SM8550.bin` | 4.32 GB | 逐对 md5 相同 |

## 2. 分类结果（>400 MB 的 57 个文件，合计 179 GB）

| 类别 | 大小 | 个数 |
|---|---|---|
| **A 必须保留** | **47.09 GB** | 7 |
| **B 后续可能需要** | **46.81 GB** | 30 |
| **C 可安全删除** | **85.54 GB** | 20 |

### A 必须保留（47.09 GB）
- 三份活 `.data`（`text_encoder_fixed4` / `transformer_part1` / `transformer_part2`）= 40.31 GB
- **app 正在用的 4 个 transformer context**（`part1a_perrow` / `part1b_perrow` / `part2a_fixed` / `part2b_fixed`）= 6.79 GB

### B 后续可能需要（46.81 GB）
- 各配置的**量化 DLC**（perrow / symw / rqs / p2split 等）——再生各约 3 小时
- 现役 text_encoder / vae 的量化 DLC 与 context
- **SM8550/v73 版 context**（约 10 GB）：#21「v79 成熟度」⛔受阻（缺 v73 设备），
  按约束 6「当前做不了 ≠ 已排除」，**不删**
- `opt3` context（1.49 GB）：#93 证明 `O` 生效但精度无影响，保留作对照

### C 可安全删除（85.54 GB）

| 项 | 大小 | 依据 |
|---|---|---|
| **9 个 `*_fp32.dlc`** | **51.44 GB** | 2 分钟可再生；量化产物已保存转换命令（已逐个验证） |
| `text_encoder.onnx.data` + `text_encoder.onnx` | 15.69 GB | 被 `text_encoder_fixed4` 取代；活流水线走 `part1~4` → 指向 fixed4 |
| `p0_4/p0_5/p0_6/p0_10` 四个实验 DLC | 11.15 GB | add_138 float-fallback 实验；**#43 已关闭**，结论已入档 |
| `part1b_nofuse_ctx` + `part1b_hps_ctx` | 2.94 GB | 与基线**逐位相同**（纯重复） |
| text_encoder `_ctx.bin` ×4（无 SoC 后缀） | 4.32 GB | 与 `_ctx.SM8550.bin` **逐位相同**（纯重复） |

⇒ **清掉 C 后 `ZImage_Work` 从 198 GB 降到约 112 GB，D 盘可用从 83 GB 升到约 169 GB。**

## 3. 未细查的部分（约 18 GB 小文件）

`calibration/`（7.45 GB，867 文件）—— **必须保留**：重新量化时要用（#80 的教训：
校准数据分布必须与推理输入一致，是可复查的证据）。
其余为各实验目录下的中间张量 `.raw`，单个 <400 MB，可按需单独清理。

## 4. 复用纪律（转下一个模型时照做）

1. **先量、再建引用图、再哈希查重，最后按再生成本分类**——不按文件名猜
2. **`.onnx` 与 `.data` 是多对一**，删 `.data` 前必须查有几个 `.onnx` 指向它
3. **fp32 DLC 是廉价中间品**（2 分钟），量化 DLC 与 context 是昂贵产物（各 3 小时）
   ⇒ 流水线跑完后**可以只留量化产物**，转换命令在量化 DLC 里有备份
4. **⛔受阻 ≠ 已排除**：为受阻线索保留的产物（如 SM8550 版）不要当垃圾清掉

---

## 5. ✅ 执行记录（2026-08-22）

**C 类 21 个文件已全部删除，释放 79.67 GiB（85.55 GB），0 失败。**
D 盘可用：**83.2 GB → 162.9 GB**。

### 5.1 删除后的强制验证（删完必须验，同约束 11 第 3 条的精神）

删除 `text_encoder.onnx.data` 有一个风险：**若有活模型偷偷指向它，整条流水线就废了**。
逐个加载 11 个活模型（含外部权重，并抽查真实张量）：

| 模型 | 结果 |
|---|---|
| `text_encoder_part1~4` | ✅ 加载正常（抽查 `text_encoder.embed_tok (151936,2560)` 等） |
| `transformer_part1a` / `part1b` | ✅ |
| `transformer_part2` / `_fixed_dce` / `part2a_fixed` / `part2b_fixed` | ✅ |
| `vae_decoder` | ✅ |

⇒ **删除未破坏流水线。**

### 5.2 为什么"51 GB 可再生"仍然删

近期**确实要重跑量化**（#98 的两条杠杆 #53 激活校准、#48+#51 选择性 FP16 都需要），
而重跑需要 fp32 DLC 作输入。但：

- 再生成本 = **每次约 2 分钟**（指南 §七 实测，最大段 part2）
- 保留成本 = **51 GB**，而 D 盘当时仅剩 83 GB，每个新实验又产出数 GB

⇒ 「未来每次多花 2 分钟」远小于「现在占 51 GB」。**这是可算的取舍，不需要问人。**

### 5.3 重新生成 fp32 DLC 的方法

转换命令**完整保存在对应的量化 DLC 里**（已验证 `Converter command` 与 `Quantizer command` 两行都在）：

```bash
python scripts/qairt_tool.py snpe-dlc-info -i <xxx_quantized.dlc> | grep "^Converter command"
```
照抄该命令即可重建 fp32 DLC。⚠️ 建 context 时**别忘 `--config_file`**（#95/#96）。

---

## 2026-08-27 清理记录（用户批准）

共回收 **42 GB**（73 GB → 117 GB）。**删除前逐项核对过不是部署产物**，
四段 TE context 的 sha256 与设备在用的逐一对上后才动手。

### A. 本会话产生的可再生中间品（26 GB）

| 删除项 | 大小 | 可再生性 |
|---|---|---|
| `TIER1_S32/text_encoder_part*/​*_fp32.dlc` ×4 | 14.6 GB | §7.5：fp32 DLC 是 2 分钟可再生的中间品 |
| `TIER1_S32/v1_backup/` | 8.1 GB | 配方在 `scripts/te_s32_calib.py`，重建约 1 分钟；**真正的回滚路径是从未动过的 `dlc_pipeline/`** |
| `TE_BACKUP_20260827/` | 4.1 GB | 与 `dlc_pipeline` 原件 sha256 **已核对完全相同**，纯冗余 |
| `onnx/_qdq_*.onnx`、`_selfchk*.onnx`、`_tmp_qdq*.onnx` | 17 MB | QDQ 模拟的临时产物，`scripts/te_qdq_sim.py` 每次重建 |

### B. 🔴 已删除的历史产物（16 GB）—— **引用处如下，勿再当它们存在**

**1) 全部 `*.SM8550.bin`（8 个，10.44 GB）**

设备是 **SM8750/v79**，这批 SM8550 产物**从未被部署使用**。已删清单：

```
transformer_part2_ctx.SM8550.bin      2.72 GB
transformer_part1a_ctx.SM8550.bin     2.20 GB
text_encoder_part1_ctx.SM8550.bin     1.58 GB
transformer_part1b_ctx.SM8550.bin     1.37 GB
text_encoder_part2/3_ctx.SM8550.bin   0.85 GB ×2
text_encoder_part4_ctx.SM8550.bin     0.76 GB
vae_decoder_ctx.SM8550.bin            0.10 GB
```

**引用它们的文档（正文未改，仅在此登记）**：
`archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md`、`archive/MAINLINE_ARCHIVE_2026-08-17.md`、
`archive/MAINLINE_ARCHIVE_2026-08-21.md`、`docs/DISK_INVENTORY.md`、
`docs/HANDOVER_2026-08-13_ZIMAGE_MVP.md`、`docs/QNN_CONVERSION_GUIDE.md`、
`docs/reviews/{backend_crash_report,DIAGNOSIS_HTP_ZIMAGE,DIAGNOSTIC_REPORT,HTP_DIAGNOSIS_AUDIT_FOR_CLAUDE}.md`、
`local-dream-ZIT0813/`（两处）。
⇒ 这些文档里提到的 SM8550 产物**已不在盘上**；需要时按 `gen_zimage_manifest.py` 的
`--htp_socs sm8550` 重建（约 30 分钟/段）。

**2) dtype 机制的 part1a 产物（3 项，6.5 GB）**

`p2attr/part1a_dt_fp32.dlc`、`part1a_dt_quantized.dlc`、`ctx_part1a_dt/`。
依据：**台账 #123 已判定**「dtype 机制的产物在设备上不兑现收益 ⇒ 不再用它做画质实验」。
**引用处**：`scripts/seg_dtype_build.py`（该脚本可重新产出它们）。

## 🔴 2026-09-02 订正 + 清理记录（本节最新，与下文冲突时以本节为准）

### C 节已过期（#148 同型：文档落后于交付）

`C` 节写于 **Tier 2（L=80，08-29）之前**，其中两条**已不成立**，原文保留于下但**不得再据以决策**：

| C 节原话 | 实情（2026-09-02 查活契约 `logs/tier2_20260829/final_qnn_contract.L80.json`） |
|---|---|
| 「`p2attr/ctx_part1a_fp16/…` —— 契约里 `device_source` 指向它，是当前部署配置的源」 | ❌ 现网是 **`transformer_part1a_clip_L80_ctx_sm8750.SM8750.bin`**，源在 `p2attr/ctx_part1a_fp16_L80/`。⚠️ `device_source` 字段在 L32/L80 两版里**同名**（都是 `/data/local/tmp/htpcmp/part1a_fp16.bin`），**不能用它判别代次，要看 `actual_filename`** |
| 「`TIER1_S32/…` —— 设备上正在跑的 32 槽 TE」 | ❌ 现网 TE 是 **`text_encoder_part*_L80_ctx_sm8750.SM8750.bin`**，在 `TIER2_L80/` |

### 现网（L80）产物清单 —— **不得清理**

- `p2attr/ctx_part{1a,1b,2a,2b}_fp16_L80/*.bin`（2.21/1.36/1.32/1.37 GiB）
- `p2attr/part{1a,1b,2a,2b}_fp16_L80_quantized.dlc`（2.12/1.30/1.25/1.31 GiB）
- `TIER2_L80/text_encoder_part{1..4}/{*_quantized.dlc,*_ctx_sm8750.SM8750.bin}`
- `ZImage_QNN_Evidence/dlc_pipeline/vae_decoder/vae_decoder_ctx_sm8750.SM8750.bin`
- `onnx/*.onnx.data` 四个（**实测引用数 13 / 44 / 36 / 1**，§7.5 的多对一陷阱）

### 2026-09-02 已删（回收 38 GiB，76 → 115 GiB 可用）

| 项 | 大小 | 删除依据（删前逐条验证过） |
|---|---|---|
| `pd_multigraph{,_part1a}/part*_fpG{1..4}.dlc` 8 个 | 13.72 GiB | ①逐字节比对：与母本**只差 6 字节**（3 处图名 × 2 字符）⇒ 纯改名拷贝，2~3 秒/份可由 `pd_multigraph_probe.py` 再生 |
| `p2attr/` 的 08-23 一代：`ctx_part1a_fp16_w11.bin`、`ctx_part1b_fp16_w5.bin`、`part2a_ctrl_quantized.dlc`、`ctx_ctrl/`、`ctx_fp16/`、`fp32/`、`htp/` | 10.16 GiB | ①grep `MAINLINE.md` + `docs/` + `scripts/*.md` **全部 0 引用**；w11/w5 白名单实验已被 #111 Clip+FP16 取代 |
| `TIER2_L80/text_encoder_part{1..4}/*_fp32.dlc`、`actfp16/part1b_actfp16_fp32.dlc` | 15.90 GiB | ①**删前实测确认** `Converter command` 可从对应 `*_quantized.dlc` 取回，且源 ONNX + `.data` 均保留 ⇒ 可再生 |

🔴 **纪律**：删 `*_fp32.dlc` 前**必须**确认同目录的 `*_quantized.dlc` 还在 —— **配方（Converter/Quantizer command）内嵌在量化 DLC 里**，量化 DLC 一旦也删掉，fp32 就不再可再生。

### 保留但可在空间吃紧时释放的（L=32 上一代，20.36 GiB）

> ✅ **2026-09-17 已执行（用户逐项勾选批准）**：三项全部删除，配方与校准输入先存档。见文末「2026-09-17 清理记录」。

**2026-09-02 判断：不删。** 当前可用 115 GiB，而 #86 全量需求实算约 40 GiB（4 个新比例 × 四段量化 DLC ≈23.9 + fp32 中间品瞬时 ≈8 + 四段五图 context ≈7.5）⇒ **删它买不到需要的东西**；而重建代价**未被精确刻画**（§7.5 记「量化约 3 小时/段」，#134 记「重量化 0.1 分钟/段」，两者差 1800 倍，无实测值可判断适用哪个）。

**触发条件（可用空间跌破 40 GiB 时，按此顺序）**：
1. `TIER1_S32/`（8.06 GiB）—— TE 是纯形状改写、#139 验过 0 新增溢出、不在画质关键路径
2. `p2attr/ctx_part*_fp16/`（6.32 GiB）—— 可从保留的 L32 量化 DLC 重建（实测 11~22 分钟/段）
3. `p2attr/part*_fp16_quantized.dlc`（5.98 GiB）—— **删前必须先把内嵌的两行命令抽成文本**，否则配方随文件消失

---

### C. ⚠️ 明确保留（不得清理）

> 🔴 **2026-09-17 标注**：本节已于 09-02 判为过期；其中 `p2attr/ctx_part1a_fp16/…` 与 `TIER1_S32/…` 两条**已于 09-17 经用户批准删除**（L=32 上一代，现网是 L=80 + mg）。原文保留如下，**勿据此认为它们还在盘上**。

- `onnx/*.onnx.data` 三个共 **37.5 GB** —— 源权重，任何重转都要
  （⚠️ §7.5：**多个 `.onnx` 共用一个 `.data`**，删前必须查有几个指向它）
- `dlc_pipeline/*/*_sm8750.SM8750.bin` —— 部署参照（8 个）
- `p2attr/ctx_part1a_fp16/part1a_fp16.SM8750.bin` —— **契约里 `device_source` 指向它**，是当前部署配置的源
- `TIER1_S32/text_encoder_part*/[*_quantized.dlc + *_ctx_sm8750.SM8750.bin]` —— 设备上正在跑的 32 槽 TE

---

## 2026-09-17 清理记录（用户逐项勾选批准；本节最新）

**回收 47.83 GiB，D 盘可用 63.2 → 110.9 GiB。** 脚本 `scripts/cleanup_20260917.py`（显式路径清单、保护名单命中即退出、默认预演）；
逐文件清单与大文件 sha256：`logs/cleanup_20260917/manifest.json`。删后核对：现网 mg 源 + single 回滚 + 重建 mg 所需 DLC + 源权重共 **63 个文件逐个仍在**。

### 删前先做的（删了就无法补做）

- **配方存档**：8 个待删量化 DLC（L32 transformer ×4、S32 文本编码器 ×4）的 `Converter command` / `Quantizer command` 全部抽出（每个 2 行、`snpe-dlc-info` rc=0）⇒ `logs/cleanup_20260917/dlc_recipes_L32_S32.txt`（副本在 `D:\ZImage_Work\recipes_archive_20260917\`）。
- **配方引用的小文件**：S32 量化命令引用 `TIER1_S32\text_encoder_partN\input_list_raw.txt` 与其 `calibration_raw\`；L32 context 目录的建图配置 `d.json/e.json` ⇒ 共 62 个文件拷到 `D:\ZImage_Work\recipes_archive_20260917\`（逐个核字节数）。L32 transformer 的 overrides（`p2attr\part*_fp16_ovr.json`）本就不在删除范围。
- **`deliver_mg` 查重**：5 个文件 sha256 与 `aspect_mg` 源文件、mg 契约三方一致（纯重复）。

### 已删

| 项 | GiB | 依据 | 需要时怎么恢复 |
|---|---|---|---|
| `ZImage_Work\deliver_mg\` | 8.26 | 纯重复（上） | `python scripts/stage_mg_delivery.py`（约 30 s，带契约 sha256 校验）。🔴 **`oneshot_mg_session.py` 以它为源，重跑交付前必须先 staged** |
| `p0_experiments\symw\` | 1.29 | §15.16 选择性对称权重，已否定 | 不需要 |
| `p0_experiments\memspeed_probe\` | 2.81 | #162 图切换 E/F 已出局的探针 | `scripts/memspeed_probe_build.py` |
| `D:\LDZ_FIX3_20260810\` | 4.67 | 08-10 旧构建包（用户手工改过的参考代码），**用户确认不要**。⚠️ 含 1100 个超 260 字符路径，删除需 `\\?\` 前缀 | 无 |
| `ZImage_Work\TIER1_S32\` | 8.06 | L=32 文本编码器，现网为 `TIER2_L80` | 配方 + 校准输入已存档；`scripts/te_s32_calib.py` |
| `p2attr\ctx_part{1a,1b,2a,2b}_fp16\` | 6.32 | L=32 transformer context；**`ctx_*_fp16_L80\`（single 1:1 回滚）未动** | 需先重建 L32 DLC |
| `p2attr\part{1a,1b,2a,2b}_fp16_quantized.dlc` | 5.98 | L=32 量化 DLC | 配方已存档，重建代价见 09-02 节（未精确刻画） |
| `D:\ZIMAGE\models\*.SM8550.bin` ×8 | 10.43 | 设备是 SM8750，从未部署；同类副本 08-27 已删 | `--htp_socs sm8550` 重建（约 30 分钟/段）。⚠️ #21 ⛔ 若拿到 v73 设备需重建 |

**用户选择保留**：`dlc_pipeline` 最初代 per-tensor 模型（18.3 GiB）、per-row 基线 `p2split\` + `perrow\`（10.6 GiB）。
**未列入候选（现网/回滚/未关闭线索依赖）**：`aspect_mg`、`TIER2_L80`、`p2attr` 的 L80 套件、`aspect\` 的 single 形态 context（#171 唯一回滚路径）与量化 DLC（重建 mg 需要）、`onnx\*.data`、`calibration`、`device_only_backup_20260917`、`actfp16`（#48 未关闭）。

**引用已删产物的文档（正文未改，仅在此登记）**：`TIER1_S32` 9 个文件、`memspeed_probe` 3 个、`symw` 2 个、`LDZ_FIX3` 2 个（HANDOVER 第 177 行、`docs/reviews/backend_crash_report.md`）、`deliver_mg` 6 个（`oneshot_mg_session.py`、`stage_mg_delivery.py` 等）。

### 本次踩到的坑（可复用）

1. **`du` 在 Windows 上统计 437 GB 超过 15 分钟没出结果**；`os.scandir` 递归（复用目录项里的大小）**2 秒**得到同样的总量（436.3 GiB，与系统已用量吻合）。
2. **Windows 长路径**：Gradle 构建树里的文件路径超 260 字符，`os.stat` / `shutil.rmtree` 直接失败 ⇒ 路径加 `\\?\` 前缀；而且 `os.path.relpath` 两个参数必须同为带前缀或同为不带前缀。
3. **经 Bash 工具传递的命令文本里，`\\` 会被吃掉一层**（本次把 `\\?\` 写成了 `\?\`；连写进 Markdown 的 heredoc 也同样被吃，本节第 2、3 条初稿就是这样坏的）——靠「打印出来看一眼」抓到，与 MAINLINE §7.4 同类。⇒ 含反斜杠的文本改用编辑工具按字面写入。

---

## 2026-09-21 清理记录（用户批准原则后自动执行；本节最新）

**原则（用户定）**：判据不是「它是不是证据」，而是**「这个结论还需要被重新验证吗」**。
严谨验证过且已关闭的结论，其**大尺寸材料不必保留**（当前交付产物与回滚源除外）；
**小尺寸文件（文档/清单/manifest）一律留**，哪怕过时或已被推翻，但要**统一管理做好标记**。
方法论已固化进指南 **§21.4**。

**结果**：删 34 个文件、释放 **49.23 GB**，D 盘 **69 GB → 115 GB**。
交付白名单删后逐个核对 **✅ 全部仍在**。

| 实验组 | 体积 | 结论出处（删的理由） |
|---|---|---|
| #173 删 ScatterElements 的三个实验臂 + 对照臂 + 最小探针 | 15.83 GB | ✅已交付：noscatA 逐字节安全（−12.6 s）、noscatB 毁图；机制 §47.3 |
| L=20/32 时代的 TE（量化 DLC + context） | 8.65 GB | 已被 TIER2_L80 取代并交付（§三十五） |
| part1 **未拆分**时代 | 7.48 GB | 已被 part1a/part1b 取代 |
| part2 拆分实验产物 | 5.69 GB | ✅已交付（当前 part2a/2b 即其结果） |
| part2 **未拆分**时代 | 5.65 GB | 已被 part2a/2b 取代；§15.18 |
| #48 wFxp_actFP | 2.95 GB | ✅已关闭：§15.33 通路成立、收益不显著 |
| per-row 实验 | 2.87 GB | ✅已关闭并采用：§十一 / §15.17 |
| dtype 探针 | 0.12 GB | ✅已关闭：§15.42 |

🔴 **完整标记**：`logs/cleanup_20260921/deleted_manifest.json`（34 条：路径、字节数、所属实验、
结论出处，以及 8 个实验组代表 DLC 的**内嵌 Converter/Quantizer 命令**）。
**大产物没了，但「它证明过什么」留下了。**

**本次新增/修正的纪律**（详见 §21.4）：
1. **删除走白名单**（只删明确列出的），不走黑名单（"不在保护名单就删"）——
   本次按黑名单跑时**漏过两次**：#171 的 single 回滚源、TE 四段的来源 DLC。
2. 🔴🔴 **谱系断链 = 清理盲区**：TE 目录里没有建图配置 json ⇒ 自动谱系认不出它的来源 DLC
   ⇒ 被判成可删（台账 #186）。**转新模型时每建一个 context 就把配置与日志留在产物旁。**
3. manifest **必须落仓库**，不能落 scratch/临时目录 —— 否则标记跟着被清掉。

**未处理**：`.raw` 中间张量约 19.65 GB（本次脚本不动，需另行逐类判定）。
