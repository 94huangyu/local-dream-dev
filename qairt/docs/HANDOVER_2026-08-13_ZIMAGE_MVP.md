> 🔴🔴 **不要从头读这个文件。** 4913 行 / 46 节，其中 **1344 行（27%）是一条已被否定的方向**，
> 另有约 500 行是第一天的临时笔记（描述的问题都已解决）。
> **先读 [`HANDOVER_INDEX.md`](HANDOVER_INDEX.md)** —— 它把 46 节分成
> 「必读 8 节（约 700 行）／按需查／🔴勿据此推论（已被推翻）／已过期」四类。
> 真正的项目入口是 `MAINLINE.md`（台账 + 执行顺序），本文档是它的证据库。

# Z-Image Turbo MVP App 交接文档 (2026-08-13 早晨 → 持续更新)

> ## ⚠️ 本项目有两条并行主线，本文件只覆盖其中一条
>
> | 主线 | 归属文件 | 说明 |
> |---|---|---|
> | **A. 排查设备生成失败** | **本文件** | "这次为什么坏"。最新状态见**第十四节** |
> | **B. 沉淀可复用的 QNN 转换量化流程** | **`QNN_CONVERSION_GUIDE.md`** | "怎么正确地做"。为后续转换 Z-Image Turbo 微调版、**Flux.2 Klein** 等模型准备 |
>
> **主线 B 是独立交付物。** 排查过程中凡是得到可复用的流程性结论（命令、陷阱、校准要求、
> 验证方法、成本预算），都要同步进 `QNN_CONVERSION_GUIDE.md`，不要只留在本文件的排查叙事里
> ——那样下次转新模型时没人找得到。
>
> ### 全部配套文件（新会话请先看这张表）
>
> | 文件 | 内容 | 何时看 |
> |---|---|---|
> | `CLAUDE.md` | 项目系统约束（读文档纪律、写文档纪律、DLC 调用纪律、实验规划纪律） | **最先看**，它约束你怎么工作 |
> | `MAINLINE.md` | 主线追踪：顶层问题、假设树及每个节点的验证状态、防漂移规则 | 每次开新实验前 |
> | `QNN_CONVERSION_GUIDE.md` | **主线 B**：可复用的转换量化配方 + 换新模型的检查清单 | 要转换新模型时 |
> | `EXECUTION_MODEL.md` | ONNX/DLC/bin 的执行契约，**只收录有实测证据的规则**；第四节是未验证清单 | 动手喂数据前 |
> | `scripts/MASTERY_PLAN.md` | 如何用机制（而非"更小心"）避免重复踩契约的坑 | 参考 |
> | `scripts/EXP_PLAN_*.md` | 各实验的执行前定稿方案（判据不得事后修改） | 做实验前 |
>
> ---
>
> 本文档承接 `../archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md`(模型量化那部分),记录 2026-08-13 起在 App 侧(local-dream fork)做的全部工作。
>
> ## 📑 小节索引（**本文件 66k token，任何情况下都不要通读**，按下表定点读）
>
> 状态标记：✅仍有效 ／ ⚠️部分被订正 ／ 🔴已被推翻（保留是为了知道"这条路为什么被排除"）／ 📜纯历史
>
> | 节 | 内容 | 状态 |
> |---|---|---|
> | 一~七 | 08-13 早期：三类修复、首次诊断、离线排查 | 📜 |
> | 八 | `add_138` 量化误差异常（当时以为是根因） | 🔴 后续否定 |
> | 九 | AIMET / per-channel 激活 / float16 fallback 方向 | 🔴 方向已废 |
> | 十 | 分词器 bug + **建立单变量测试纪律** | ⚠️ bug 已修；纪律部分仍有效 |
> | 十一 | Phase A 纯 FP32 + 一个独立符号 bug | 📜（结论已被十五取代） |
> | 十二 | 🔴 **SM8750 芯片型号发现 + `backend_extensions` 正确格式** | ✅ **配置仍在用** |
> | 十三 | caption 单通道定位（整轮方向） | 🔴 **被十四节明确否定** |
> | 十四 / 十四·续 | transformer 量化不是根因 | ⚠️ 被十五节的单变量实验取代 |
> | **十五** | 🔴 **根因：HTP 执行 `.bin` ≠ CPU 参考执行 `.dlc`** | ✅ **核心结论** |
> | 15.14 | 三个量化开关的实测结论 | ✅ |
> | 15.15 | 🟢 `--quantization_overrides` 闸门打通（必须用 ONNX 张量名） | ✅ |
> | 15.16 | 选择性对称权重 FC-only ⇒ **更差**，与官方通用建议相反 | ✅ |
> | 15.17 | 🟢 **per-row 量化：迄今唯一有效的 HTP 修复手段** | ✅ |
> | 15.18 | part2 拆分（DCE 陷阱、PD 容量上限） | ✅ |
> | 15.19 | 🎉 设备出猫 + app 四段交付上线 | ✅ |
> | 15.20 | app vs PC 在环对照，C++ 第 3 个 bug 假设关闭 | ✅ |
> | 15.21 | 画质缺陷"定位到 VAE 导出" | 🔴 **归因被 15.22 推翻**（定位对、归因错） |
> | **15.22** | 🔴🔴 **根因：VAE 反缩放做了两遍；一行修复已交付并实测** | ✅ |
> | 15.23 | #84 复查：三把尺子的反相关**不是**钳位假象（我的假设被证否） | ✅ |
> | 15.24 | P0-B 前置门：`--quantization_overrides` **可以**注入 per-axis encoding | ✅ |
> | 15.25 | P0-B 单算子精确重放 | 🔴 **结论已被 15.29 推翻**（装置有误） |
> | 15.26 | 三臂对照：offset 项占比 | ⚠️ **数值已被 15.29 订正**（46.5%→6.6%） |
> | 15.27 | #91 分块量化（BQ 只支持 int4）与 #92 per-row bias，两条关闭 | ✅ |
> | 15.28 | 🔴 #10/#27/#34「静默忽略」根因：配置缺 `graphs` 段；#10/#27 关闭 | ✅ |
> | **15.29** | 🔴🔴 **重大订正：漏 `--config_file` ⇒ 编成 v68/4MB，结论翻转** | ✅ **必读** |
> | **15.30** | 🔴🔴 **图层面误差 = RmsNorm 固有放大，非 HTP 缺陷** | ✅ |
> | **15.31** | 🔴🔴 **#99 gamma 8-bit = 3.75%（🟡）；并划出「宿主消融」天花板：只覆盖 12.7%** | ✅ |
> | **15.32** | 🔴🔴 **查文档推翻「选择性 FP16 不可行」：FC 支持 `FP16 激活 + SFxp8 权重`，权重不涨** | ✅ |
> | **15.33** | **#48 wFxp_actFP 最小确认：通路成立、收益不显著（🟡）；配方须用 `--enable_float_fallback`** | ✅ |
> | **15.34** | **#53 激活校准代价门否决：mse/sqnr/entropy 都不收紧量程 ⇒「收紧量程」族已用尽** | ✅ |
> | **15.35** | 🔴 **#102 选择性 FP16：装置全通但 G4 崩 —— `Pow(x,2)` 在 FP16 溢出，38% inf/nan** | ✅ |
> | **15.36** | 🔴 **#103 BF16 与量化互斥（官方文档）⇒「选择性浮点」整族关闭** | ✅ **最新** |
>
> **主线 A 最新状态：§15.22。** 想知道"接下来做什么"请读 `MAINLINE.md`，不要在本文件里找。
>
> **一句话现状**：之前所有"调量化配置修 `add_138`"的方案（12.7 节的 P0～P3 全部）已被证明**在数学上
> 不可能奏效**（收益上限 0.004%，见 13.8）；改用逐层实测重新定位后，误差已经收敛到一个非常具体的
> 位置——**unified 序列里 32 个 caption token 上的单个通道（通道 85）**，误差在两条支流合并处出现
> 28 倍阶跃（见 13.12）。目前正在做第二轮定位，把范围从 125 个算子缩到具体算子（13.13）。
>
> **前面第一到十二节是排查过程记录，不用重读**，其中这些结论已被第十三节推翻，读到时请以第十三节
> 为准：
> - 8.4 节把 `add_54` 标注为"第1层"→ **错**，它前面还有 4 个完整 block（13.9）。
> - 8.4 节把误差归因于"残差张量 per-tensor scale 不够细"→ **错**，误差不来自这些张量自身的编码
>   （13.8）。8.4 节发现的"离群激活通道"现象本身是**对的**，被推翻的只是归因。
> - 12.7 节的 P0～P3 修复计划 → 前提不成立，整体作废（13.8）。
>
> 另外：`C:\Users\sinai\.claude\plans\binary-imagining-mango.md` 末尾有指路总结，但该文件已不再更新，
> 一切以本文档第十三节为准。

## 一、总体结论

**好消息**:App 现在能完整走通"加载模型 → 收到 prompt → 跑 8 步去噪 → VAE 解码 → 返回 PNG"全流程,不再崩溃、不再 OOM。
**坏消息**:生成的图像内容是错的——不是"生成质量差",而是明显的**张量数据错位**(规律的竖条纹,不是随机噪声),bug 已经被诊断实验缩小到 **Transformer 阶段**(text encoder 和 VAE 都已验证正常)。

## 二、今晚做的三类修复(均已验证)

### 1. 路由 bug(已修复,已验证生效)
`PipelineZImage.hpp` 里原来用文件名包含 `.ctx.` 子串来判断是否走 `createAndInitContext`,但真实交付的文件名是 `..._ctx.SM8550.bin`,子串匹配从未命中,导致所有 QNN context binary 都被错误地当成"模型库 .so"去加载(`createAndInitModel`)。已改为无条件走 `createAndInitContext`(因为 final contract 里的 graph 全部都是 context binary)。
**验证**:修复前报错是 `Failed init QNN model: text_encoder_part1`,修复后报错变成 `Failed init QNN context: text_encoder_part1`(说明路由对了,只是当时还有别的问题)。

### 2. 内存耗尽 OOM(已修复,已验证生效——这是最大的隐藏问题)
`PipelineZImage::initialize()` 原来会把全部 8 个 QNN context(约 11GB)一次性加载并常驻到 pipeline 销毁为止。这台设备(OnePlus 13T,骁龙8至尊版,约15GB可用内存)在系统内存管理器(OnePlus 自己的 Osense/lowmemorykiller)介入下,只要进程内存占用冲到 7GB+ 就会被强制杀掉。

**实测数据**(logcat 直接证据):
- 8个文件一起加载:进程被标记 `abnormal_size: 7331208`(~7.3GB),几秒内系统可用内存从 9GB 暴跌到 <100MB,`lowmemorykiller` 明确记录击杀原因 `reason: device is not responding`。
- 即使只分阶段加载(文本编码器一组4.2GB成功;但 Transformer 三段一起6.8GB仍然失败),同样被 OnePlus 的 `OsenseKillAction` 标记为"内存异常占用"并杀掉。

**修复方案**(已验证可行):彻底重写 `generate()`,不再持有任何模型:
- **阶段1 文本编码器**:4个 part 一起加载,跑完立刻释放。✅ 实测成功,约4.2GB 峰值,耗时约8秒。
- **阶段2/3 Transformer**:三段(part1a+part1b+part2)合计约6.8GB **不能同时常驻**。目前方案是:`part1a`+`part1b` 一组(约3.84GB,在已验证安全的范围内)、`part2` 单独一组(约2.93GB),**每一步去噪都重新加载/释放**(不是只加载一次)。这样峰值内存被压到 ~3.84GB,但代价是速度很慢——完整生成一张图约 4 分钟(8步 × 每步约30秒的加载开销)。
- **阶段3 VAE解码**:最后单独加载、解码、释放。

这个改动直接对标 `PipelineAnima.hpp` 里已经验证过的 `seq_dit`(顺序DiT)模式的思路,不是我们自己发明的新模式。

### 3. 编译环境坑(一次性的,以后应该不会再遇到)
- 子模块(尤其是 `tokenizers-cpp` 嵌套的 `msgpack`/`sentencepiece`)第一次 `git submodule update --init --recursive` 有静默失败,需要 `git reset --hard` 强制补齐。
- Windows 默认关闭"开发者模式"导致 CMake 无法创建符号链接(sentencepiece 需要链接 abseil-cpp)——已让用户手动开启。
- `qairt-converter`/CMake 生成的 `protoc` 是 arm64-v8a(Android目标架构)可执行文件,不能在 Windows host 上跑,需要显式指定 host-native `SPM_PROTOC_EXECUTABLE`。
- vendored 的 `tokenizers-cpp/rust/src/lib.rs` 有两处触发新版 rustc `dangerous_implicit_autorefs` 硬性报错的写法,按编译器建议加了显式 `&`。
- `sentencepiece_processor.h` 用 `third_party/absl/...` 和 `absl/...` 两种不同前缀的 include,`tokenizers-cpp/CMakeLists.txt` 里补了两条 `target_include_directories`。

## 三、当前正在诊断的问题:生成的图像是错的

### 现象
用固定 prompt("a cute orange cat sitting on a wooden table...")跑完整 8 步,拿到合法的 1024×1024 PNG,但内容是**有规律的竖条纹**(见 `../evidence/images/zimage_test_output.png`),不是自然图像,也不是纯随机噪声。

### 诊断实验 1(已完成,结论明确):VAE 解码路径是对的
加了一个调试开关(marker 文件 `DEBUG_SKIP_TRANSFORMER`,放在模型目录下即可触发,见 `PipelineZImage.hpp::generate()` 开头),让代码跳过整个 Transformer 循环,直接把初始高斯随机噪声送进 VAE 解码。
**结果**:输出是干净的"电视雪花"随机噪点(见 `../evidence/images/zimage_vae_only_test.png`),**没有条纹**。

**结论**:VAE 的输入量化(`putQuantized`)、输出反量化(`requireFloat`)、以及最后 NCHW→RGB 的像素展开循环(`PipelineZImage.hpp` 第 175-184 行左右)全部正确。`final_qnn_contract.json` 里 `vae_decoder` 的 `pixels` 输出声明是 `fixed_shape: [1,3,1024,1024]`(NCHW),代码里 `pixels[(c*H+y)*W+x]` 的读取方式和这个声明一致,而且被这次实验实测验证过。

### 诊断实验 2(进行中,未完成):数值统计
在 `PipelineZImage.hpp` 里加了 `logStats()` 静态方法,会在以下几个点打印 min/max/mean/std 到 logcat(tag 里带 `[diagnostic]`):
- 初始高斯噪声 latents
- 第0步和最后一步,Transformer 输出的 predicted velocity(`noise`)
- 第0步和最后一步,scheduler 更新后的 latents

**交手机前抢到了第0步的部分数据**(完整跑完前就要交出手机了,后面几步没跑完):

```
initial latents (N(0,1)):              min=-4.6041 max=5.1186 mean=0.0019 std=1.0007
transformer predicted velocity, step 0: min=-6.2600 max=6.8903 mean=0.0700 std=1.4958
latents after step 0:                   min=-4.2909 max=4.8745 mean=0.0050 std=0.9451
```

**这组数字非常关键,而且是意外的**:全部数值都很"健康"——没有 NaN、没有饱和在量化上下限、均值方差都在合理范围(更新后的 latents 标准差还基本保持在 1 附近,符合 Flow Matching 该有的行为)。**这排除了"量化 scale 搞反/溢出/大面积截断"这类会让数值本身跑飞的低级错误**。

结合这一点,可以把上面"下一步怀疑方向"里的优先级重新排一下:**数值本身是对的,问题更可能出在"数值被放在了错误的空间位置"**,也就是某个环节的张量做了不该做的转置/重排(最典型嫌疑:RoPE 的 `select_45`/`select_46` cos/sin 分量如果被空间维度换位,或者 `128×128` 的 latent 网格在某个环节被按错误的 stride 展开/收起来),这样数值分布统计完全正常,但解码出来的图像却是"看起来有内容但排列错了"的条纹——和实际观察到的现象完全吻合。量化 scale 传递错误(`requantize()`)这个可能性可以往后放一放,除非后续几步的数据显示数值开始跑飞。

### 下一步怀疑方向(未验证,按可能性排序)

Transformer 分成三段执行,`transformer_part1a` 会输出一大堆中间张量喂给 `part1b`(据 `../archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md` 4.4 节记录):
```
add_138, add_131, tanh_19, select_45, select_46,
adaln_input, unified_freqs, unified_mask, latents_shape
```
这些张量的桥接**完全靠 `final_qnn_contract.json` 里每个 graph 的 `inputs[].source`/`name` 字段驱动**(见 `PipelineZImage.hpp::valueFor()` 和 `runGraph()`),没有任何硬编码的形状检查。可疑点:

1. **`select_45`/`select_46`(RoPE 的 cos/sin 分量)**——如果这两个张量在传递中被换位、或者量化 scale 弄反,会产生"看起来有规律但内容错"的输出,而不是纯粹的数值爆炸/NaN,这和我们观察到的"条纹而非纯噪声"高度吻合。
2. **`unified_mask`/`latents_shape`**——`../archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md` 里特别提示这两个是 **initializer 常量**(不是节点计算结果),在 `final_qnn_contract.json` 里可能以某种特殊方式声明(不一定是普通的 graph output)。需要确认 `ZImageQnnContract.hpp` 和 `PipelineZImage::valueFor()` 是否正确处理了"这个 input 的数据来自一个常量而不是上一个 graph 的运行结果"这种情况——如果常量被漏掉、被清零、或者被错误地映射到了别的张量,极可能产生这种"结构还在但内容错"的输出。
3. **量化 scale/offset 是否在多段之间正确传递**——`runGraph()` 里的 `requantize()` 函数处理了"同一个张量在两个 graph 里声明了不同 scale/offset"的情况,这段逻辑本身没有被独立验证过(只有 VAE 那种单一 scale 的简单情况被验证了)。

### 建议的下一步排查方法(不需要手机,可以在电脑上做)

1. **读 `final_qnn_contract.json` 里 `transformer_part1a`/`transformer_part1b`/`transformer_part2` 三个 model 条目的完整 `inputs`/`outputs`**(路径:`D:\ZIMAGE\final_qnn_contract.json`,用 `models[]` 数组,通过 `internal_graph_name` 字段找到对应条目),重点看:
   - 每个 input 的 `source` 字段到底指向哪个上游 tensor 名字。
   - 是否有输入的 `source` 是空的/缺失的(可能对应"这个其实是常量,不该从 `values` map 里找")。
   - 量化 scale/offset 在同一个逻辑张量的不同 graph 声明里是否一致。
2. **对照原始 Python/ONNX pipeline**(`D:\ZImage_Work\ZImage_QNN_Evidence\` 及相关切分脚本,尤其是 `split_transformer_part1.py`)确认 part1a→part1b 的九个中间张量的**真实数学含义和形状**,和 `PipelineZImage.hpp::valueFor()` 的桥接逻辑逐一核对。
3. 如果有机会重新拿到手机,可以把 `logStats()` 埋点扩展到 part1a 和 part1b 之间的每一个中间张量(不只是最终 latents),这样能一步定位到具体是哪个中间张量在哪一段出问题。诊断开关 `DEBUG_SKIP_TRANSFORMER`(在模型目录 `files/models/ZIMAGE/` 下放一个同名空文件即可触发)也可以复用来做类似的二分排查(比如只跑 part1a+part1b,跳过 part2,看输出统计是否已经跑偏)。

## 四、当前代码/构建状态

- 项目路径:`D:\LocalDreamZImage\local-dream`(全新 clone 的 xororz/local-dream 主线 + 手工移植的 Z-Image 相关文件)
- 参考对照的旧代码(user 之前手工改过的版本):`D:\LDZ_FIX3_20260810\local-dream`——仅用作参考,不要在这上面继续改。
- APK 输出:`app\build\outputs\apk\basic\debug\LocalDreamZImage_armv8a_2.8.1-zimage-mvp.apk`,包名 `io.github.xororz.localdream.zimage`,versionCode 100。
- 编译命令(需要先在 Windows 设置里开启"开发者模式"):
  ```powershell
  cd D:\LocalDreamZImage\local-dream
  $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
  .\gradlew.bat :app:assembleBasicDebug
  ```
- 手机上已经装好这个 APK,模型也已经在 `/data/user/0/io.github.xororz.localdream.zimage/files/models/ZIMAGE/` 导入完整(不需要重新导入)。
- 关键改动文件:`app\src\main\cpp\src\PipelineZImage.hpp`(核心逻辑全在这)、`QnnRuntime.hpp`(加了 `createAndInitContext`)、`BackendService.kt`(V79 stub 检查、超时时间)。

## 五、给 MacBook 继续诊断的建议

QAIRT SDK (`D:\qairt`)、模型文件 (`D:\ZIMAGE`)、原始转换工程 (`D:\ZImage_Work`) 目前都只在这台 Windows 机器上。如果要在 MacBook 上继续,建议:
1. **纯代码审查路线**(不需要 SDK/设备):把 `final_qnn_contract.json` 拷贝到 Mac 上,配合 `PipelineZImage.hpp` 源码,人工核对第三节提到的中间张量桥接逻辑。这是当前最高价值、最不依赖环境的路径。
2. 如果 Mac 上能装 QAIRT SDK(高通有部分工具链支持 macOS)并且能访问 `D:\ZImage_Work` 里的 Python 验证脚本/ONNX 参考模型,可以尝试离线复现 part1a/part1b 之间九个中间张量的期望数值,和 App 里 `logStats()` 打出来的实际值做对比——但这需要先把 `D:\ZImage_Work` 同步过去,量比较大。
3. 手机拿回来之后,可以直接用本文档"下一步排查方法"里的第3条(扩展 `logStats` 埋点)继续设备端诊断,这是最直接的路线。

## 六、生成的测试图片(可用于对比)

- `D:\LocalDreamZImage\evidence\images\zimage_test_output.png` —— 完整8步跑出来的(有问题的)输出,竖条纹。
- `D:\LocalDreamZImage\evidence\images\zimage_vae_only_test.png` —— 跳过 Transformer、直接解码纯噪声的对照组,证明 VAE 路径正常。

## 七、2026-08-13 白天:离线(无需手机)排查进展

用户白天在 MacBook 上纯靠 adb(无源码)又做了一轮独立诊断(见 `local-dream-ZIT0813\diagnostics_2026-08-13\README.md`),提出三个风险点。回到 Windows 机器后,结合真实源码 + 原始 ONNX 模型逐条验证:

### 7.1 App 侧代码:排除(有代码依据)
- **use-after-free 假设(风险点1)排除**:`QnnModel::executeNamedGraph()`(`QnnModel.hpp:107-108`)在返回前把输出 `memcpy` 进独立的 `std::vector`,`PipelineZImage::runGraph()` 又拷贝一次存进 `values` map——数据在对应 QNN context 释放前已经两次深拷贝,不存在悬空指针。
- **按位置而非按名字桥接的假设(风险点2)排除**:从 `ZImageQnnContract.hpp` 到 `valueFor()` 到 `QnnModel::executeNamedGraph()`,两层全部是严格按 tensor name 字符串查找,没有任何位置/顺序相关逻辑。**如果 `select_45`/`select_46` 真的错位,只可能是 `final_qnn_contract.json` 本身在导出时就标错了,不会是 App 代码引入的。**
- **`latents_shape` 孤儿输出（风险点3）**:确认了"没有报警"是因为 `validatePartSplit()` 那个检查压根不会在 final-delivery(8段)格式下执行,是校验盲区,不是阴性证据,但结论(低优先级)依然成立。

### 7.2 用真实 ONNX 模型 + 真实校准数据做的端到端验证:transformer 拆分被证明完全正确 ✅

用 `D:\ZImage_Work\ZImage_QNN_Evidence\calibration\transformer_part1\sample_0000\` 下的真实校准数据（`latents.npy`/`timestep.npy`/`caption.npy`/`cap_pad_mask.npy`），跑了两轮独立的 FP32 数值对比（脚本见 `D:\LocalDreamZImage\scripts\compare_split_boundary.py` 和 `compare_part1b.py`）：

1. **part1a vs 未拆分的 `transformer_part1_fixed.onnx`**：在拆分边界的全部 9 个张量（`add_138`、`add_131`、`tanh_19`、`adaln_input`、`select_45`、`select_46`、`unified_freqs`、`unified_mask`、`latents_shape`）上，`max_abs_diff = 0.000000`，**逐位精确匹配**。
2. **part1b vs 未拆分模型的原生 `unified` 输出**：同样 `max_abs_diff = 0.000000`，**逐位精确匹配**。

**结论：transformer_part1 → part1a + part1b 的拆分操作在浮点精度层面被双重证实完全正确，不是 bug 来源。** `select_45`/`select_46` 也在这次验证里被间接证明没有错位（如果错位，两次比对不可能同时归零）。

### 7.3 现在唯一还没验证过的环节：W8A16 量化

排除了 App 代码和拆分操作之后，链路里剩下**从没被端到端数值验证过**的环节就是 **PTQ 量化**（`qairt-quantizer` 那一步）。几个支持这个方向的间接证据：
- 交接文档自己的历史记录（第二节 2.6/2.11/2.13/2.14）显示，这个团队在其他模型的量化阶段反复踩过"数值统计正常但结果全错"类型的坑（percentile 校准裁掉 outlier、类型强转导致 Bad Scale、qnn-net-run 默认按 float32 解析裸整数等）——同一类风险在 transformer 量化时没有理由不存在。
- 昨晚设备上抓到的数值统计（min/max/mean/std）虽然"健康"，但**量化引入的错误不一定表现为数值发散**——如果是某几个特定张量的 scale/calibration 没校准好（比如 RoPE 分量、或者 `unified` 这种拼接了 patch token 和 caption token 的联合序列，混合了两种数值分布，量化校准容易失衡)，完全可能产生"数值范围正常但内容错位"的效果。

**受限之处**：交接文档 5.2 节已经记录过"QNN CPU 模拟器跑不动这个 W8A16 模型"，所以没法在电脑上直接跑量化后的版本做数值对比。真要验证这个方向，大概率需要重新走一遍 `qairt-quantizer` 流程（用 `D:\qairt` 这个 SDK），检查量化校准参数，或者尝试换用更保守的量化设置（min-max 而非 percentile，参照交接文档 2.13 节的强制约束）重新编译。这是一个数小时级别的工作量，需要你决定要不要投入。

### 7.4 建议的下一步（按优先级）

1. **（推荐，工作量最小）**：先检查 `qairt-quantizer` 当时用的校准命令参数，确认 transformer 量化确实用的是 `min-max`（而不是 `percentile`）——这是交接文档 2.13 节明确写过的强制约束，值得先花几分钟确认有没有被违反，而不是直接假设它遵守了。
2. **（中等工作量）**：如果校准参数本身没问题，下一步是重新生成量化校准数据、重跑 `qairt-quantizer`，并在生成 W8A16 DLC 后，用真实校准样本跑一次"量化前 vs 量化后"的余弦相似度对比（参照交接文档第三节 text_encoder 当年做过的 `validate_cascade_e2e.py` 同款验证），这是 transformer 这一段目前唯一缺失、但 text_encoder 当年做过的验证步骤。
3. **（如果1、2都排除了）**：回头怀疑更底层的 patchify/unpatchify 或 RoPE 应用逻辑是不是在最初的手工 ONNX 图手术（交接文档第二节列的一堆 GatherND/IsNaN/Slice 替换）里就已经出错——但这个可能性因为 7.2 节的验证结果（拆分前后完全一致）被间接削弱了，因为如果手术阶段就错了，错误会同时存在于拆分前后两个版本，不会因为拆分而改变，所以本质上还是指向拆分之前那一步，只是本次验证没法直接证明或证伪它。

## 八、2026-08-13 深夜：找到根因了——`add_138` 量化误差异常

### 8.1 关键突破：QAIRT SDK 里其实有另一个能跑通 W8A16 的 CPU 模拟器

`validation_report.md`（`D:\ZImage_Work\ZImage_QNN_Delivery\`）记录过"QNN CPU 无法执行这个 W8A16 模型"，用的是 `qnn-net-run` + `QnnCpu.dll`，0.23 秒内直接报 `OpConfig validation failed for Reshape` 崩溃。**但 QAIRT SDK 还打包了另一套完全独立的旧版 SNPE 执行引擎**（`snpe-net-run.exe`，位于 `D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\`），是纯软件的数值模拟器，不像 QNN CPU 后端那样对算子做硬件相关的严格校验。用它直接跑 `transformer_part1a_quantized.dlc`（这次拆分的量化 DLC 文件本身也在本机：`D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\transformer_part1a\`），**成功了**——第一次能在电脑上直接拿到真实量化数值,不需要手机、不需要 HTP 硬件。

用法（供后续复现）：
```powershell
$env:PATH = "D:\qairt\2.48.0.260626\lib\x86_64-windows-msvc;D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc;" + $env:PATH
# input_list 第一行用 % 前缀列出想要的输出张量名，否则默认只导出最后一个输出
& "D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\snpe-net-run.exe" `
  --container <xxx_quantized.dlc> `
  --input_list <input_list.txt> `
  --output_dir <out_dir>
# 输出默认是反量化后的 float32 .raw 文件，直接按声明的 shape reshape 即可比对
```

### 8.2 数值比对结果：`add_138` 量化误差远超正常范围

用真实校准样本 `sample_0000`（`D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\transformer_part1a\calibration_raw\sample_0000\`），把 SNPE 跑出来的**真实量化后**数值，和 7.2 节已经验证过 bit-exact 的 **FP32 参考值**逐张量对比：

| 张量 | max_abs_diff | 相对误差 | 参考值 std |
|---|---|---|---|
| adaln_input | 0.0069 | 0.53% | 0.1581 |
| add_131 | 0.0223 | 1.22% | 0.3150 |
| **add_138** | **321.80** | **26.93%** | 3.1707 |
| select_45 | 0.000015 | 0.0015% | 0.6969 |
| select_46 | 0.000015 | 0.0015% | 0.6107 |
| tanh_19 | 0.0187 | 1.87% | 0.3606 |
| unified_freqs | 0.000015 | 0.0015% | 0.6559 |
| unified_mask | 完全一致(bool) | — | — |

除 `add_138` 外，所有张量的相对误差都在 2% 以内，是 W8A16 量化正常该有的精度损失。**只有 `add_138` 一个张量的误差高达 27%，比其它张量高一个数量级**，绝对误差最大到 321——这不是普通的量化舍入噪声，规模上更像是**部分数值被截断/裁剪**。

### 8.3（更新，纠正了下面 8.3 原文的猜测）：不是校准范围太窄，是极少数离群元素被算错

**先纠正**：从 `final_qnn_contract.json` 反查 `add_138` 实际用的量化参数（`scale=0.026026, offset=-5056`），算出量化器当时校准到的范围是 `[-131.6, 1574.0]`——sample_0000 的真实取值范围 `[-126.5, 1195.1]` **完全落在这个范围内**，不存在越界截断。所以下面 8.3 原文里"per-channel 校准范围没盖住真实极值"的猜测是错的，已经用实测数据推翻。

**真正的情况**（逐元素误差分布分析）：
```
中位数误差: 0.0157   均值误差: 0.0223     <- 和理论舍入误差量级完全吻合，说明量化本身没问题
误差 > 1.0 的元素占比:   0.0247%
误差 > 10.0 的元素占比:  0.0000189%（1585万个元素里只有 3 个）
误差 > 100.0 的元素占比: 0.0000063%
```
把元素按真值大小分组更能说明问题：
```
|真值| < 5（99.85% 的元素，"正常范围"）：      平均误差 0.022，最大误差 1.49  <- 完全正常
|真值| >= 5（0.15% 的元素，"离群激活值"）：    平均误差 0.299，最大误差 321.8 <- 严重异常
```
**最极端的例子**：某个元素真值 410.97，理论上按 scale=0.026 反量化应该复原到 ≈410.98，实际量化后读出来却是 **732.77**——这已经不是精度不够的舍入误差（理论最大舍入误差只有 0.013），量级差了四个数量级，更像是计算链路里某个环节对这个数值做了错误处理（比如 int8 矩阵乘法的累加器溢出/饱和），而不是"量化分辨率不够"。

**往上追了一层源头**：`add_138 = add_129 + tanh_18 * mul_301`（普通加法），`add_129` 是更早的残差流。这个离群值大概率是从 `add_129`（甚至更早，一路追溯回第一层）就已经存在，只是在某个更早的矩阵乘法环节被错误计算了，不是 `add_138` 这个加法节点自己引入的。**要精确定位到具体是哪一层、哪个算子出的问题，需要继续往前顺着残差流逐层追踪**，这是一个比"调整某个张量的量化配置"大得多的工作量，今晚先追到这里，留给下一轮排查。

### 8.3-原（已被上面纠正，仅保留存档）：为什么恰好是这个张量：极端的数值分布

`add_138` 是交接文档 4.4 节里写的"**主 unified token 流（第14个残差节点）**"——DiT 几乎全部隐藏状态都通过它往下传。它的 FP32 参考分布非常极端：

```
shape=(1, 4128, 3840)  mean=0.0588  std=3.1707  min=-126.51  max=1195.08
```

均值只有 0.06、标准差 3.17，但最大值达到 1195——是均值的 **377 个标准差**开外，典型的"绝大多数数值挤在很小范围内、极少数稀疏离群值撑开整个动态范围"分布，正是交接文档 2.13 节专门警告过的"大模型激活值依赖稀疏异常值做路由机制"那种情况。配合当时用的 `--use_per_channel_quantization`（每个通道独立校准 min/max）：只要有个别通道在 10~几十个校准样本里没有充分覆盖到真实的极值区间，该通道的量化范围就会偏小，真正跑到极值时就会被截断——具体某些通道是否真的截断了、还是别的机制导致的这个量级误差，还没有进一步细查，但"是这个特定张量的量化配置有问题"这个定位已经很确定了。

**这也顺带解释了"数值统计看起来很健康但图像内容错位"这个现象**：`add_138` 携带了几乎全部的 token 表征，如果它在部分通道/部分 token 位置被系统性压缩或截断，下游计算会得到内容错误、但幅度仍然合理（不是 NaN、不是数值爆炸）的结果——统计层面的 mean/std 检测不出来，但空间内容已经被破坏了。这和昨晚设备上抓到的"数值健康但图像是条纹"完全吻合。

### 8.4（再次更新，最终定位）：不是某一层的 bug，是经典的"离群激活通道（outlier activation channel）"问题

沿着残差链（`add_54`「第1层」→ `add_66` → `add_78` → `add_90` → `add_102` → `add_114` → `add_126` → `add_138`「第14层」）逐点做了同样的量化误差分布比对（脚本：`D:\LocalDreamZImage\scripts\trace_residual_corruption.py`，参考值缓存在 `D:\ZImage_Work\ZImage_QNN_Evidence\onnx\residual_chain_reference.npz`），结果：

```
tensor     离群元素最大误差
add_54     31.8   <- 第1层就已经有明显误差了
add_66     34.3
add_78     48.2
add_90     47.3
add_102    60.5
add_114    87.3
add_126    149.3
add_138    321.8  <- 一路逐层滚雪球式放大
```

**关键发现：误差从第 1 层就已经存在，不是某一层突然"坏掉"，而是每一层量化都对离群激活值处理得不够精细，然后逐层累积放大。** 进一步查了 `add_54` 里离群值（|真值|≥5）在 3840 个通道上的分布：

```
通道 1238: 4128 个 token 里有 4120 个都是离群值（几乎每个 token 都在这个通道上异常大）
通道   85: 37 个 token 是离群值，但数值极端（最大 647，是整个张量的最大值）
```

**这是大模型量化文献里非常经典的"离群激活通道 / massive activation channel"现象**（LLM.int8()、SmoothQuant 等论文专门讨论过的问题）：少数几个固定通道天生携带远超其它通道的数值（很可能承担某种"寄存器/路由"作用），如果整个 `[4128, 3840]` 张量共用**同一个** per-tensor 量化 scale，这几个通道会把 scale 拉得很粗，拖累其余 3400+ 个正常通道的精度；反过来看，即使量化范围没有截断（8.3 节已确认没有截断），单一 scale 对这种极端不均匀的分布来说本身就是精度不足的，而且这种不足会随着残差流逐层传递、逐层叠加。

查过 `qairt-quantizer --help` 的完整参数：**`--use_per_channel_quantization` 这个开关只对权重（weights）生效，对激活值（activations）没有 per-channel 选项**——所以即使命令行加了这个参数，`add_138` 这类中间激活张量依然只能用 per-tensor 量化，这正是问题的结构性根源，不是某次配置疏忽。

### 8.5 下一步建议（工作量：数小时到一天级别，需要用到 QAIRT SDK 重新量化）

1. **`qairt-quantizer` 支持一个 `--config CONFIG_FILE`（YAML）参数**，目测这是传递"量化覆盖（quantization overrides）"的入口，但具体的 YAML schema、能不能针对单个激活张量指定 per-channel/更高 bitwidth，还没有查文档确认，需要先啃一下 QAIRT SDK 自带的文档（`D:\qairt\2.48.0.260626\docs\`，如果有的话）或者官方在线文档。
2. 另一个可能更简单的方向：`--act_bitwidth` 目前是 16，理论上限已经很高；但 `--weights_bitwidth 8` 如果能针对"产生离群通道的那几层权重矩阵"单独提到更高 bitwidth（如果 override 机制支持按层/按 op 指定），可能比处理激活量化本身更直接。
3. **每次尝试新的量化配置后，都可以完全离线验证效果**，不需要手机：用今天发现的 `snpe-net-run.exe`（x86 CPU 模拟器）跑新产出的 `_quantized.dlc`，对比 `residual_chain_reference.npz` 里的参考值，重点看 `add_54`（第1层）的离群误差是否显著下降——如果第1层就改善了，说明方向对了，再决定要不要推广到后面所有层。
4. 环境注意：`D:\LocalDreamZImage\local-dream\.venv-qnn` 这个专用虚拟环境在本次会话开头的 `git clone` 时被覆盖掉了（原来的 local-dream 目录被替换），今晚是直接用系统 Python 装了 `onnx`/`onnxruntime`/`pyyaml` 后跑通的 `qairt-quantizer --help`。正式重新量化前建议重建一个干净的专用虚拟环境，避免系统 Python 环境的依赖冲突污染量化结果。
5. 验证通过后，同样需要重新走一遍 `qnn-context-binary-generator` 生成新的 `.SM8550.bin`，替换手机模型目录里的对应文件，再用第七节验证过的设备端流程实测一次。

## 九、2026-08-13 晚间：装好了 AIMET，确认 per-channel 激活量化走不通，方向转为 float16 fallback

### 9.1 在 WSL 里装好了 AIMET（可复用）

这台 Windows 机器原本没有 WSL，为了跑 AIMET（官方文档要求 Linux 环境）现场装了一遍：

1. **WSL 平台本身**：需要管理员权限执行 `wsl --install`（本工具无法自行提权，用户在管理员终端跑的），装完重启。
2. **Ubuntu 发行版下载遇到问题**：`wsl --install -d Ubuntu`（以及 `--web-download`、换成具体版本号 `Ubuntu-24.04`）全部固定报错 `Wsl/InstallDistro/0x80072f78`（这是 Microsoft Store 下载通道的问题，不是网络本身的问题——直接用 `Invoke-WebRequest` 测试过，网络是通的）。
   **绕过方法**：不走 Store，直接从 Canonical 官方源下载 WSL 专用 rootfs 包，再用 `wsl --import` 导入：
   ```powershell
   Invoke-WebRequest -Uri "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz" -OutFile "D:\WSL\ubuntu-24.04-wsl-rootfs.tar.gz"
   wsl --import Ubuntu2404 "D:\WSL\Ubuntu2404" "D:\WSL\ubuntu-24.04-wsl-rootfs.tar.gz"
   ```
   现在这个 `Ubuntu2404` 发行版是好的，以后直接 `wsl -d Ubuntu2404 -- <命令>` 就能用，不用重新装。默认用户是 root（`--import` 方式没有走正常的建号流程，够用，不影响功能）。
3. **AIMET 安装**：`pip install aimet-onnx` 默认会连带装完整 CUDA/GPU 依赖链（torch 主包 526MB + 好几个 nvidia_* 包，合计几个GB），这台机器没有能用的 GPU，完全没必要。**先手动装 CPU 版 torch 再装 aimet-onnx，能跳过这部分**：
   ```bash
   python3 -m venv /root/aimet-venv
   /root/aimet-venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
   /root/aimet-venv/bin/pip install aimet-onnx==2.36.0
   ```
   已验证 `import aimet_onnx` 正常，`aimet_onnx.quantsim.QuantizationSimModel` 可用。

### 9.2 关键结论：per-channel 激活量化，这套工具链根本不支持——是硬件限制，不是软件缺功能

AIMET 自带了针对我们目标硬件的现成配置文件 `htp_quantsim_config_v79_per_channel_linear.json`（路径：`aimet_onnx/common/quantsim_config/`，正好是 v79 + per-channel 组合，直接对应我们的场景）。打开看了内容：`"per_channel_quantization": "True"` 只挂在 `op_type` 下的 `Conv`/`ConvTranspose`/`Gemm`/`MatMul`/`PRelu` 几个算子上，而且这个配置的语义是**只对这些算子的权重（weight/param）生效**，不是对激活值（activation）输出生效。

结合 8.4 节已经确认的 `qairt-quantizer --use_per_channel_quantization` 也只对权重生效——**两套独立工具链（AIMET 和 QAIRT）都只支持权重的 per-channel 量化，都不支持激活值的 per-channel 量化**。这不是巧合，是 Hexagon HTP 硬件本身的乘加单元（MAC）不支持运行时按通道切换 scale（权重是静态的，可以提前把 per-channel scale 融合进去；激活值是运行时产生的，要 per-channel 就需要硬件在计算过程中动态换 scale，大多数定点 NPU 不具备这个能力）。**8.4 节原计划的"给 add_138 这类张量配 per-channel 覆盖"这条路，现在可以确定走不通，不用再往这个方向花时间。**

### 9.3 新方向：把离群通道集中的张量标记为不量化（float16 fallback）

`--quantization_overrides` 的 JSON schema（见第八节 8.1 附近提到的 `applyencodings.html` 文档）里，`output_dtype` 字段明确支持 `"float16"` 这个值——意味着可以直接把 `add_54`……`add_138` 这些残差张量标记为**保持浮点、不量化**，绕开 per-tensor 量化精度不够的问题，不需要复杂的校准。这比强行凑 per-channel 方案简单得多，而且是这套工具链明确支持的标准用法（`qairt-converter` 帮助文档里也提到不支持的精度会自动 fallback 到 float16）。

### 9.4 当前的阻塞点：找不到当初转换 `transformer_part1a` 用的确切命令行

想验证 9.3 的方案，需要重新跑一遍 `qairt-converter`（带上 `--quantization_overrides` 指向这几个张量），但：
- `01_qairt_converter.log`（`D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\transformer_part1a\`）是空文件，没有记录实际执行的命令。
- `qairt_wrapper.py` 只是个透传壳子（把 `sys.argv` 转给真正的 `qairt-converter`），本身不包含参数。
- 交接文档 3.4 节提到的主编排脚本 `zimage_dlc_pipeline.py`（应该在 `D:\ZImage_Work\` 下）**在当前磁盘上已经找不到了**，大概率是流水线跑完后被清理掉了。

**不建议凭猜测重建这个命令**——转换阶段的参数（尤其是 `-s`/`--source_model_input_shape`、`--target_soc_model`、opset 相关处理等）如果和原始的有细微出入，产出的 DLC 行为可能跟现在部署的版本不完全一致，会让"改了 override 之后效果对比"这个验证失去意义。

### 9.5 下一步建议

1. **优先找回 `zimage_dlc_pipeline.py`**：检查有没有被同步到别的地方（云端、git 历史、聊天记录里贴过的代码块等）。这是最靠谱的路径，能保证复现时命令完全一致。
2. 如果确认找不回来，退而求其次：参照 `D:\ZImage_Work\Cloud_Package\convert_qairt_zimage.py`（虽然是旧的 W8A8/4段格式，但转换命令的整体结构、常用参数应该类似）加上 `final_qnn_contract.json` 里记录的输入输出 shape，谨慎重建一个新的转换命令，**重建后第一步必须先跑一次不带 override 的转换+量化，用第八节的方法验证误差分布和现在部署的版本完全一致**，确认重建的命令没有跑偏，再加 override 做实验。
3. AIMET 环境已经装好并验证可用，为将来需要更复杂的量化分析（比如更细致的逐层敏感度分析 `analyze_per_layer_sensitivity`，或者需要真的走 AIMET 的 QuantizationSimModel 流程而不是简单的 override）留好了基础，不用重新搭建。

> **重要更新（见第十节）**：上面 9.5 节的"下一步建议"是在发现分词器 bug **之前**写的，当时的假设链条
> （校准范围→per-channel→float16 fallback）建立在"图像错位主要由量化引起"这个前提上。这个前提本身
> 还没有被验证过（从来没跑过纯 FP32 全流程看图片是否正常），而且晚些时候发现了一个**独立的、已确认
> 的新 bug**（分词器装错），所以 9.5 节的建议**不再是当前最高优先级**，先看第十节。

---

## 十、2026-08-13 深夜二：用户要求重新规划 + 发现分词器 bug + 建立单变量测试纪律（本文档当前最新状态）

### 10.0 背景：为什么要重新规划

用户指出，从第七节到第九节的排查方式是"遇到一个问题解决一个问题，没有规划"——发现一个可疑点就
不断往下深挖一层，从来没有做过一次"最省成本、最能一次性回答大方向问题"的检查。这个批评是对的。

因此重新梳理了一份结构化的排查计划，写在 **`C:\Users\sinai\.claude\plans\binary-imagining-mango.md`**
（这是独立于本文档的另一个文件，新会话一定要去读）。核心内容：
- 一份完整的"证据总账"——哪些已经排除、哪些已经确认、哪些从来没查过。
- 重新识别出的最大缺口：**从来没有验证过纯 FP32（不量化）跑完整个 pipeline，图片是否正确**——这个
  检查能一次性回答"是不是量化的锅"，之前却被跳过，直接扎进了量化细节。
- 分阶段计划 Phase A～F，每个阶段前都先问"这一步的结果会不会让后面的假设整个作废"。

### 10.1 重新规划过程中，发现了一个新的、独立的、已确认的 bug：分词器装错了

在着手搭建 Phase A（纯 FP32 pipeline）验证脚本、需要加载分词器给文本编码器构造真实输入时，发现
**手机上实际部署的 `tokenizer.json` 根本不是 Z-Image 文本编码器（Qwen3）需要的分词器**：

| 文件 | 词表大小 | 结论 |
|---|---|---|
| `D:\ZIMAGE\tokenizer\tokenizer.json`（手机部署的那份，**已修复**） | 原来 49408 | ❌ 是 CLIP 分词器（SD1.5/SDXL 那种），装错了 |
| `D:\models\Z-Image-Turbo\tokenizer\tokenizer.json`（原始模型下载） | 151643 | ✅ 正确的 Qwen3 分词器 |

判断依据：正确文件里 `<\|im_start\|>`=151644、`<\|im_end\|>`=151645、`<\|endoftext\|>`=151643，和
`TextEncoder.hpp` 里硬编码的 `kQwenPadId = 151643`（`Config.hpp`/`TextEncoder.hpp:608`）完全对得上；
错误文件的词表只有 49408，连 151643 这个 id 都不存在。

**这意味着**：不管量化精度多准，只要分词器是错的，喂给文本编码器的 `input_ids` 从源头上就是文不对题、
语义随机的（拿 CLIP 词表里 id=27 对应的 token，去查 Qwen3 词表里 id=27 对应的完全不同的 token）。

**已经修复的地方（不是待办，已经生效）**：
1. `D:\ZIMAGE\tokenizer\tokenizer.json` 已替换为正确文件，原错误文件备份在同目录下的
   `tokenizer.json.WRONG_CLIP_backup`。
2. 手机上 `/data/user/0/io.github.xororz.localdream.zimage/files/models/ZIMAGE/tokenizer.json`
   也已经通过 `adb push` + `run-as cp` 替换为正确文件。

### 10.2 单变量测试纪律 + 第一轮测试结果

用户明确要求"一次只改一个变量"，并且指出**优先用真实设备测试，而不是图省事去搭一个可能有自己 bug
的 Python 平行实现**——手机能连就应该用手机测，不要为了方便绕开它。

**已完成的单变量对照**（都是同一个 prompt："a cute orange cat sitting on a wooden table, masterpiece,
best quality"，都走真实的量化设备端引擎，唯一区别是分词器）：

| 测试 | 分词器 | 量化引擎 | 结果图 | 现象 |
|---|---|---|---|---|
| 基线 | 错误（CLIP） | 是（量化） | `D:\LocalDreamZImage\evidence\images\zimage_test_output.png` | 规律竖条纹 |
| 单变量：只换分词器 | 正确（Qwen3） | 是（量化，不变） | `D:\LocalDreamZImage\evidence\images\zimage_tokenizer_fix_test.png` | 马赛克/色块状，深蓝底色+暖色斑块 |

**结论**：换分词器后输出的"错误形态"发生了质变（条纹→色块），证实分词器确实是真实生效的变量，**不是
误报**。但换完之后图片依然明显不是一张连贯照片，**说明分词器不是唯一问题**——大概率量化那条线（第
八、九节定位到的 `add_138` 离群通道量化误差）依然独立存在，但这只是推测，还没有验证过因果关系。

### 10.3 立即的下一步（已经写好代码，还没运行过）

按"一次只改一个变量"的纪律，**保持分词器为"正确"不变**，把"量化"这一个变量单独去掉：跑纯 FP32
（不量化）版本的完整 pipeline，对比是否变正常。

脚本：**`D:\LocalDreamZImage\scripts\zimage_fp32_pipeline.py`**。分词器 + 调度器（Flow Match 8步
时间步）部分已经跑通验证过（可以直接运行确认，几秒钟出结果）。完整的模型串联代码——
`text_encoder_part1~4`（FP32 ONNX）→ `transformer_part1a/1b/2` 循环 8 步（FP32 ONNX）→ `vae_decoder`
（FP32 ONNX）→ 存 PNG——**已经写完，但还没有实际跑通过一次，第一次运行大概率会暴露一些小 bug（张量
名字、shape 对不上之类），需要调试。**

运行方式：
```powershell
cd D:\LocalDreamZImage\scripts
python zimage_fp32_pipeline.py "a cute orange cat sitting on a wooden table, masterpiece, best quality" 42
```

**已知风险**：脚本里 `transformer_part1a`+`part1b`+`part2` 三个 ONNX session 设计成一次性全部加载、
8 步循环复用（避免每步重新加载的开销），但三者合计外部权重数据约 13.7GB+11GB，写这份文档时机器只有
约 15GB 可用内存，**有 OOM 风险**。如果跑挂了，把"三个 session 一次性常驻"改成"每步用完 `del` 释放、
下一步重新 `load`"（脚本里 `load()`/`run()` 两个小函数已经封装好，改起来不难，参照 `PipelineZImage.hpp`
今晚新写的分阶段加载模式）。

**决策分支**：
- 图片连贯正常 → 剩下的问题就是量化（`add_138` 那条线），回到 `binary-imagining-mango.md` 计划文件
  的 Phase B/C/D/E 继续（但注意：9.5 节写的"直接上 float16 override"这个具体方案，在按 Phase B/C
  重新验证因果关系之前，不要直接采用）。
- 图片依然不对（不管是条纹、色块还是别的样子）→ 说明还有第三个尚未发现的问题，回头考虑 ONNX 图手术
  （交接文档第二节，`../archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md` 里那一堆 GatherND/IsNaN/负无穷截断替换）或者其它环节，**不要
  立刻假设"肯定是量化"**。

### 10.4 环境/工具/连接方式清单（全部可直接复用，不用重新搭建）

- **手机无线调试已打通**，不需要一直插 USB：
  ```bash
  # 首次需要 USB 连一次执行（已经做过，仅供以后 IP 变了时参考）：
  adb -s <USB序列号> tcpip 5555
  adb -s <USB序列号> shell ip addr show wlan0   # 找 WiFi IP
  adb connect <WiFi IP>:5555
  ```
  当前已知 WiFi 地址：`192.168.31.157:5555`（如果手机换了网络/重启，IP 可能变化，重新走一遍上面的
  流程即可，需要用户配合插一次 USB）。
- **`D:\LocalDreamZImage\scripts\zimage_device_test.sh`**：一键跑设备端生图测试的脚本（启动 App →
  等 `Server listening` → `adb forward` → `curl POST /generate` → 解析 SSE 里的 base64 图片），当前
  写死了单设备 USB 逻辑，**用无线调试时需要手动指定 `-s 192.168.31.157:5555`** 而不是让它自动挑设备
  （现在同时有 USB/WiFi 两个连接时，`adb devices` 会返回多条，脚本里 `awk 'NR==2'` 逻辑会挑错设备，
  需要小改一下，或者每次手动 `adb disconnect` 掉不用的那个连接）。
- **WSL + AIMET 环境**（第九节搭建）：`wsl -d Ubuntu2404 -- /root/aimet-venv/bin/python3 <script>`，
  已验证 `import aimet_onnx` 可用。
- **SNPE CPU 模拟器方法**（第八节发现）：能在电脑上直接跑真实量化后的 DLC 数值，不需要设备/HTP：
  ```powershell
  $env:PATH = "D:\qairt\2.48.0.260626\lib\x86_64-windows-msvc;D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc;" + $env:PATH
  & "D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\snpe-net-run.exe" --container <xxx_quantized.dlc> --input_list <list.txt> --output_dir <out>
  ```
- **本机 Python 环境**：系统 Python（`C:\Users\sinai\AppData\Local\Programs\Python\Python310\python.exe`）
  已装好 `onnx`、`onnxruntime`、`tokenizers`、`Pillow`、`pyyaml`，可以直接用，不用建 venv。
- **脚本清单**（都在 `D:\LocalDreamZImage\scripts\`）：
  - `zimage_device_test.sh` —— 设备端一键测试
  - `compare_split_boundary.py` / `compare_part1b.py` —— part1a/part1b 拆分正确性验证（已完成，结论
    是拆分完全正确，bit-exact）
  - `trace_residual_corruption.py` —— 残差链量化误差逐层追踪（已完成，定位到 `add_138` 离群通道）
  - `zimage_fp32_pipeline.py` —— **当前进行中**，纯 FP32 全流程验证（Phase A，见 10.3）

### 10.5 给新会话接手的建议顺序

1. 先读 `C:\Users\sinai\.claude\plans\binary-imagining-mango.md`（结构化计划 + 证据总账）。
2. 再读本节（10.0～10.4）了解最新进展。
3. 直接跑 `zimage_fp32_pipeline.py`（10.3 节），这是当前最高优先级、唯一进行到一半的任务。
4. 根据 10.3 节的决策分支决定下一步。
5. 全程记得"一次只改一个变量"、"优先用真实设备测试而不是图省事"这两条纪律，这是用户在本轮反复强调
   纠正过的方法论。

---

## 十二、2026-08-14 凌晨：DLC 验证 + SM8750 芯片型号重大发现 + 正确的 backend_extensions 配置格式

### 12.1 用户远程接入：Chrome Remote Desktop 已配置好

用户白天在公司要用 MacBook Air 远程操作这台 Windows 机器。已装好 Chrome Remote Desktop（服务名
`chromoting`，启动类型"自动"，开机自启，不用担心重启后失效）。手机会被用户带去公司，**设备端测试
在白天不可用**，只能靠离线的 `snpe-net-run.exe` CPU 模拟器验证，真机测试要等用户晚上带手机回来。

### 12.2 FP32 DLC 本身验证干净——不是 DLC 转换环节的问题

用 `snpe-net-run.exe` 单独跑了 `transformer_part1a_fp32.dlc`（未量化版本），同一个 `sample_0000`
校准样本、同一个 `add_138` 张量，和已确认的 FP32 ONNX 参考值（`residual_chain_reference.npz`）比对：

```
max_abs_diff = 0.089（相对 add_138 最大值 1195，相对误差 0.0074%）
mean_abs_diff = 1.7e-6
```

这个量级是正常的浮点运算顺序噪声，**不是真实转换失真**。结论：`qairt-converter`（ONNX→DLC）这一步
干净，问题 100% 在 `qairt-quantizer`（量化）这一步，第八节定位的 `add_138` 离群通道量化误差依然是
唯一确认的剩余问题源头。

### 12.3 重大发现：手机真实芯片是 SM8750，但整晚一直编译的是 SM8550——差两代

```
adb shell getprop ro.soc.model   →  SM8750（骁龙8至尊版，Hexagon v79）
adb shell getprop ro.board.platform  →  sun
```

但 `../archive/ARCHIVE_2026-08-10_OBSOLETE_handover.md` 记录的整条流水线从一开始就是 `--htp_socs sm8550`（`soc_id: 57`，Hexagon
**v73**）——**差了两代芯片**。此前"手机能正常跑（哪怕图是错的）"说明 QNN 运行时有某种向后兼容路径能
在 v79 硬件上跑 v73 编译的 context binary（手机 `qnn_runtime_libs` 里同时打包了 v68~v81 的 skel 库，
运行时按需 dispatch），但显然不是"跑在正确硬件上"该有的状态。

**用户此前被告知"QAIRT 不认 SM8750"是错的**（可能是用了更旧的 SDK 版本，或者环境配置问题）。当前这
套 `D:\qairt\2.48.0.260626` SDK **明确支持 SM8750**，证据：
- `qairt-converter`/`qairt-quantizer`/`qnn-context-binary-generator` 的 Python 库
  `htp_v2.json` 里 `soc_model_to_arch` 表：`"SM8750": "v79"`（`SM8550` 是 `v73`）。
- `QAIRT_ReleaseNotes.txt` 里有多条**专门针对 SM8750 的性能修复**（"Improved model execution time
  performance on SM8750"、SM8750 HTP W8A16 量化专项优化等），说明是被认真维护的目标，不是冷门芯片。
- 官方文档 `offline_graph_caching.html` 的示例命令直接就用 `--htp_socs sm8750` 做演示。
- QNN 头文件 `include/QNN/QnnTypes.h` 里有命名常量：`QNN_SOC_MODEL_SM8750 = 69`
  （`QNN_SOC_MODEL_SM8550 = 43`——注意这套新枚举和旧版 `soc_id: 57` 的编号体系完全不同，不能混用）。

**注意**：Android 侧 `ro.soc.model`/`android_device_constants.py` 里查到的 SM8750 对应"618"/"639"，
那是 **Android 硬件识别用的编号**，和 QNN 内部 `soc_model` 枚举（69）是完全不同的两套编号体系，**不能
混用**——这是踩过的一个坑，记录下来避免下次重踩。

### 12.4 正确的 backend_extensions 两层配置格式（踩了很久的坑，完整记录）

`qnn-context-binary-generator --config_file <path>` 传入的文件**不是**直接写 `graphs`/`devices`
这些字段——那样会导致每一个 key 都报 `Unknown Key = devices/0/soc_model passed in config`（即使
key 名字、结构完全对，因为顶层结构就是错的）。正确格式是**两层**：

**外层**（`--config_file` 直接指向的文件）：
```json
{
  "backend_extensions": {
    "shared_library_path": "D:\\qairt\\2.48.0.260626\\lib\\x86_64-windows-msvc\\QnnHtpNetRunExtensions.dll",
    "config_file_path": "D:\\ZImage_Work\\backend_config_detail.json"
  }
}
```

**内层**（`config_file_path` 指向的另一个文件，真正的 `graphs`/`devices` 配置放这里）：
```json
{
  "devices": [
    {
      "soc_model": 69,
      "dsp_arch": "v79"
    }
  ]
}
```

`soc_model` 用 `QnnTypes.h` 里的枚举值（SM8750=69，SM8550=43），`dsp_arch` 用字符串（"v79"/"v73"）。
`shared_library_path` 在 Windows 上是 `QnnHtpNetRunExtensions.dll`（`D:\qairt\2.48.0.260626\lib\
x86_64-windows-msvc\` 下，对应 Linux 版的 `libQnnHtpNetRunExtensions.so`）。

**踩坑记录（供以后配置任何 SoC 少走弯路）**：
- `cores`/`perf_profile`/`rpc_control_latency` 这些字段**只对 `qnn-net-run` 有效，对
  `qnn-context-binary-generator`（context-binary 生成）无效**，写了会报 Unknown Key。
- `soc_id` 字段在这个 SDK 版本已经从 schema 里移除了（文档写"will be deprecated"，实际已经不认），
  必须用 `soc_model`。
- 完整 JSON Schema 定义在 `docs\QAIRT-Docs\QNN\general\htp\htp_backend.html`（搜索 "QNN HTP Backend
  Extensions" 章节），比 SNPE 那边的示例文档更准确对应 `--dlc_path` 这条新的 QAIRT 工作流。

### 12.5 SM8750 目标编译验证：VAE Decoder 试点成功

用现成的 `vae_decoder_quantized.dlc`（不用重新量化，只是验证"换目标芯片"这条链路本身通不通）+
上面的正确配置，跑：
```powershell
qnn-context-binary-generator.exe --backend QnnHtp.dll --dlc_path vae_decoder_quantized.dlc \
  --binary_file vae_decoder_ctx_sm8750 --output_dir <dir> --htp_socs sm8750 --config_file backend_ext.json
```
**编译成功**，走完 Graph Optimizations → Graph Sequencing → VTCM Allocation → Completion 全部阶段，
产出 `vae_decoder_ctx_sm8750.SM8750.bin`（111MB，和原 SM8550 版本 103MB 量级相近）。过程中有一条
`<E> Unsupported HTP Arch 1 79` 的日志，但整个流程仍然顺利完成产出了完整文件，看起来是无害的早期
探测信息，不是致命错误（**这一点还没有拿到真机验证，只是离线编译通过**，去年内不能在这台电脑上验证
「编译出的 SM8750 binary 能不能被真机正确加载执行」，需要等手机回来）。

### 12.5.1 量化修复（`add_138` 离群通道）尝试受阻：`--quantization_overrides` 触发原生崩溃

**目标**：给 part1a 内部的 8 个残差张量（`add_54`~`add_138`）加量化覆盖，提高精度，缓解第八节定位的
离群通道误差。

**踩坑过程（完整记录，供以后继续）**：
1. `--quantization_overrides` 的 JSON schema 有版本（"0.6.1" AIMET 风格 / "1.0.0" / "2.0.0"），必须显式
   声明 `"version"` 字段，否则报错。用了 2.0.0（`{"version":"2.0.0","encodings":[{"name":...,
   "output_dtype":...}]}`），这是从 `qti/aisw/converters/common/encoding_handler/encoding_handler_v_2_0_0.py`
   源码里直接读出来的真实 schema，比两份文档（`applyencodings.html` 和 `quantization.html`）互相矛盾的
   描述更可信。
2. `output_dtype: "float16"` 这个官方文档明确写"支持"的选项，**在这个 SDK 版本的转换器代码里实际没实现**
   （`op_graph_optimizations.py` 的 `get_tensor_bw()` 只认定点整数类型，float16 直接抛
   `Unsupported datatype found in overrides`）。9.3 节原计划的"标记 float16 不量化"这条路**在这个 SDK
   版本走不通**，不是配置问题，是功能缺失。
3. 改用 `int32`（有真实 y_scale，用 `residual_chain_reference.npz` 里的真实 abs-max 算的对称量化 scale）
   能通过 schema 校验，但无论是 8 个张量一起改还是只改 `add_138` 一个，**都会在 `qairt-converter` 序列化
   阶段的 `IrQuantizerV2.apply_encodings()` 里触发原生 `Windows fatal exception: access violation`**
   （不是 Python 异常，是 C++ 扩展崩溃，没有更多调试信息）。崩溃前有大量
   `[WARN: FP FALL] xxx missing encoding, assumed type f16` 日志，说明只要提供任何
   `--quantization_overrides`，转换器就会尝试对**整张图**做一次量化（没被 override 的张量全部自动退化
   成 f16），而不是像预期那样只记录 override 元数据、真正的量化留给下一步 `qairt-quantizer` 做。
4. 排除了几个可能的干扰变量：`--target_backend HTP` 有没有都一样崩；`qairt-quantizer --help` 确认
   `--quantization_overrides` 是**只属于 qairt-converter 的参数**，quantizer 阶段没法单独接收 overrides
   （只有 `--ignore_quantization_overrides` 这个"忽略"开关）；`qairt-converter` 不接受 `--input_list`
   参数（校准数据只属于 quantizer 阶段），排除了"缺校准数据"这个猜测。

**结论**：这条路线（`--quantization_overrides` 标记特定张量提高精度）在当前这套 `D:\qairt\2.48.0.260626`
SDK 里被这个原生崩溃卡住，没有更深的调试手段（拿不到 C++ 侧的堆栈/日志）。**暂时搁置**，不建议下一个
会话继续无脑重试同样的参数组合——如果要继续这个方向，大概率需要：换一个 QAIRT SDK 版本（更新或更旧）
试试是否还崩、或者改用 AIMET 的量化路径（9.1 节已经装好的 WSL+AIMET 环境，`--use_aimet_quantizer` +
`--apply_algorithms adaround`，这条路完全没试过，理论上能绕开这个 converter 阶段的崩溃，因为走的是不同
代码路径）、或者联系高通报 bug。复现命令和实验文件都留在 `D:\ZImage_Work\`（
`transformer_part1a_overrides.json`、`qairt_converter_wrapper.py`），不需要重新摸索 schema。

**决策**：鉴于用户不在线、这个坑没有清晰的短期解法，把精力转向确定有收益、确定不会崩的工作——**用现成
的、没有改过的 `_quantized.dlc`（已验证正确的量化，只是有 `add_138` 那个已知但还没修复的精度问题）直接
换编译目标到 SM8750**，这本身就有明确的性能收益（Release Notes 里的 SM8750 专项优化），而且是刚验证过
的、稳定不崩的流程。

### 12.5.2 批量为 SM8750 重编译全部剩余部件（进行中）

用 12.4 节验证过的两层 config（`backend_ext.json` → `backend_config_detail.json`，`soc_model: 69`，
`dsp_arch: "v79"`），对 `text_encoder_part1~4` 和 `transformer_part1a/1b/2` 依次跑
`qnn-context-binary-generator --htp_socs sm8750`，全部复用**现成、未改动**的 `_quantized.dlc`（不涉及
重新量化，不会触发上面的崩溃）。产物命名 `<part>_ctx_sm8750.SM8750.bin`，输出目录和原文件同目录。
批处理日志：`D:\ZImage_Work\sm8750_retarget_all.log`。

### 12.5.3 批量编译结果：6/7 干净成功，transformer_part2 有一条新的、和 add_138 相关的可疑报错

`text_encoder_part1~4`、`transformer_part1a`、`transformer_part1b`：编译干净，只有两条已知无害的
host 端提示（`WindowsFileIO couldn't open libcdsprpc.dll`、`Unsupported HTP Arch 1 79`，VAE 那次
成功编译也有这两条，确认无害）。

**`transformer_part2` 多了一条从没见过的报错**（`D:\ZImage_Work\sm8750_retarget_all.log` 第496行）：
```
ERROR:scale too large for requant qu16->qu16: inf; OpID: 0x6994000000a2(q::Requantize)
```
发生在 "Graph Optimizations" 阶段中途，但流程**继续走完了全部后续阶段**（Post Graph Optimization →
Graph Sequencing 100% → VTCM Allocation → Parallelization Optimization → Finalizing Graph Sequence →
Completion），最终确实产出了 `transformer_part2_ctx_sm8750.SM8750.bin`。

**这个现象和第八节定位的 `add_138` 离群通道量化误差高度吻合**（`add_138` 的极端数值经过后续处理，
在换成 v79 目标编译时触发了"requantize scale 算出无穷大"）——不是巧合，是同一个根因在不同代码路径下
的另一种表现形式，进一步佐证 `add_138` 那条线确实是真实存在、还没解决的问题。

**没能验证这个产物是否可信**：`snpe-net-run.exe`（今晚一直用来离线验证的 CPU 模拟器）只能跑 `.dlc`
文件，**跑不了已经编译好的 HTP context binary**（`.bin`）——这一步的正确性只能靠真机跑一次 HTP 硬件
执行来确认，白天没法验证。**这个 `transformer_part2_ctx_sm8750.SM8750.bin` 在真机验证之前，应该当作
"未知是否可信"，不要默认它是对的。**

### 12.5.4 真机验证 SM8750：跑通了，质量和已知基线持平（没有变差，也没有变好）

**部署过程踩的坑（完整记录，供以后复用）**：
1. `adb push` 不能直接写入 App 私有目录，也不能先放 `/sdcard` 再用 `run-as cp`——**scoped storage
   会让 `run-as`（以 App 自己的 UID 运行）读不到 adb shell 用户放在 `/sdcard` 下的文件**
   （`cat: Permission denied`）。正确做法：先 `adb push` 到 `/data/local/tmp/`（不受 scoped storage
   限制），再 `run-as $PKG cat /data/local/tmp/X > 目标路径`，最后清理 `/data/local/tmp/` 里的临时文件。
2. `final_qnn_contract.json` 里 `context_binary` 字段是**相对于 `ZIMAGE/` 目录的相对路径**
   （如 `"models/xxx.bin"`），而不是相对于随便哪个目录——第一次部署时把新文件直接放进了 `ZIMAGE/`
   根目录（少了一层 `models/` 子目录），导致 App 扫描模型列表时判定"不完整"直接从列表消失（`暂无模型`）。
   Kotlin 侧校验逻辑在 `Model.kt::isCompleteZImageBundle()`，会挨个检查 `context_binary` 相对路径是否
   真实存在，缺一个都不行。
3. **`sha256` 字段真的会被校验**（`ZImageQnnContract.hpp:92`），换了文件必须用真实文件重新算哈希填进
   contract，不能偷懒复用旧值。
4. `final_qnn_contract.json` 编码要注意——本地 `D:\ZImage_Work\final_qnn_contract.json` 这份副本不知
   为何被存成了 UTF-16（可能是某次编辑器操作留下的），但**手机上真实部署的那份是标准 UTF-8**。以后要
   改 contract，应该先从手机上 `run-as cat` 一份下来做基准，不要直接信本地这份副本。
5. 新旧两套模型文件（SM8550 原版 + SM8750 新版）**可以同时共存在手机上不冲突**——App 到底加载哪一套，
   只取决于 `final_qnn_contract.json` 里 `context_binary` 指向哪个文件名，不是靠文件名规律匹配或缓存。
   （用户提出"直接删掉旧的以绝对排除歧义"的建议更彻底，已经照做，旧的 SM8550 二进制文件已从手机删除，
   电脑上 `D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\` 里的原始副本完整无损，随时可以重新推回去。）

**测试结果**：用 ASCII prompt（"a cute orange cat..."）跑通了完整 8 步生成，`HTTP 200`，耗时 210 秒，
和之前 SM8550 版本的耗时（~210-230秒）基本持平，没有观察到明显的速度提升（这次没有专门测速对比，只是
数量级上没有量级差异）。logcat 确认加载的是 `libQnnHtpV79Skel.so`（v79，不是 v73），证明真的跑在正确
目标上，不是意外沿用旧路径。

**图像质量**：`D:\LocalDreamZImage\scratch_runs\sm8750_test2.png`——结构连贯（能看出蓬松的猫状轮廓），
但仍然是那种"发闷/颗粒感"的质感，和之前"两个 bug 修复后"那张 SM8550 结果（`two_bugs_fixed_test.png`）
属于同一个质量级别，**没有变得更差，也没有变得更好**。这是一条重要的验证结论：

> **`transformer_part2` 那条"scale too large for requant...inf"报错，实测没有让结果变得更差**
> （用户此前追问"这不代表没问题"是对的——这条结论只回答了"有没有引入新的、更严重的问题"，不代表
> `add_138` 那个根因被解决了。两者是不同的问题，不要混为一谈。）

**一次编码踩坑（记录以免重复怀疑）**：中途用 curl 命令行直接传"刘亦菲"这个中文 prompt，得到瞬间的
`HTTP 500`（0.01秒，logcat 里完全没有 App 日志）。当时怀疑过是不是删除 SM8550 文件的时机（在模型
"加载中"状态时执行的）把状态搞坏了——用同一份文件状态、换成纯 ASCII prompt 重试，正常跑通并观察到
完整的 QNN 初始化日志（`QnnBackend_create`→`QnnDevice_create`→`QnnContext_createFromBinary`→...），
证明文件状态是好的，问题出在 Git Bash/Windows 环境下 curl 命令行传中文 UTF-8 字节被弄坏，不是模型或
部署问题。**以后测中文 prompt，不要在这个 shell 环境里用 curl 命令行直接传中文**，应该写到文件里用
`--data @file.json`，或者干脆在 App UI 里手动输入。

### 12.6 当前证据总账（截至 2026-08-14 早上，用户出门上班前的状态快照）

**已确认、高置信度：**
- `PipelineZImage.hpp` 的两个 bug（`cap_pad_mask` 极性、Euler step 符号）已修复并装机验证，图像从
  纯噪声变成结构连贯——这是本次排查最大的突破，证据链完整（diffusers 源码 + 直接反查 ONNX 图节点）。
- FP32（无量化）用同样修复后的代码跑出**完美清晰**的图（`zimage_fp32_full_pipeline.png`）——证明
  算法/图导出层没有问题，`unified` 拼接顺序、文本编码器隐藏层选择这些此前"未验证"的点可以不用再查了。
- 量化后（设备端 W8A16）会退化成"结构对但发闷"的质感——**唯一确认的剩余问题源头是 `add_138` 离群
  激活通道的量化误差**（第八节定位，27% 相对误差，逐层滚雪球放大）。
- 手机真实芯片是 SM8750（Hexagon v79），此前一直编译的是 SM8550（v73）目标，差两代——已经确认
  SM8750 被 SDK 官方支持（Release Notes 有专项优化记录），已经用现成量化内容重编译全部 8 段并在真机
  验证跑通，**质量和原 SM8550 版本持平**（没变差也没变好，`transformer_part2` 那条可疑报错未造成
  实际影响）。当前手机上部署的就是这一版（SM8750 目标 + 未修复的 add_138 量化问题）。
- FP32 DLC（`qairt-converter` 转换结果，未量化）本身经过独立验证是干净的（vs FP32 ONNX 参考，相对
  误差 0.0074%，正常浮点噪声量级）——问题不在转换层，在量化层，这个结论很扎实。

**已确认受阻、需要新方法：**
- 用 `--quantization_overrides` 给 `add_138` 等张量提高精度（float16 或 int32）这条路，在当前
  `D:\qairt\2.48.0.260626` SDK 里会触发 `qairt-converter` 序列化阶段的原生崩溃（`IrQuantizerV2.
  apply_encodings()` access violation），且 `float16` 这个官方文档写"支持"的选项在这个版本的转换器
  代码里实际没实现（`get_tensor_bw()` 直接对 float 类型抛异常）。完整复现步骤见 12.5.1 节。

### 12.7 下一步排查计划（按优先级排序，供下一个会话/下班后继续）

**P0（最低成本，值得先试）：换一种施加 override 的方式，而不是死磕同一个崩溃**
1. 检查 `--quantizer_log` 参数（`qairt-quantizer --help` 提到"Valid for use with v2.0.0 JSON schema
   for quantization overrides"）能不能在崩溃前拿到更多诊断信息，缩小崩溃的具体触发条件。
2. 尝试**不用 `--onnx_skip_simplification`**（当前所有转换命令都带了这个 flag，会不会是简化步骤缺失
   导致图结构和量化覆盖逻辑对不上，从而触发崩溃？这个变量还没单独测试过）。
3. 尝试**覆盖更多/全部张量**而不是只覆盖 8 个残差张量——崩溃前的 `[WARN: FP FALL]` 级联显示只要用了
   `--quantization_overrides`，转换器就会把没覆盖到的张量全部退化成 f16，这个"大量隐式 FP FALL"的
   状态本身可能才是崩溃触发条件，而不是 override 的具体张量或数值。

**P1（中等成本，路径完全不同，理论上能绕开这个崩溃）：AIMET 量化路线**
第九节已经装好 WSL + AIMET 环境（`wsl -d Ubuntu2404 -- /root/aimet-venv/bin/python3`），从没真正用过。
`qairt-quantizer --use_aimet_quantizer --apply_algorithms adaround`（或者更进一步用 AIMET 的
`QuantizationSimModel` API 直接分析/处理 `add_138` 这类离群通道）走的是完全不同的代码路径，理论上
不会触发同一个 converter 阶段的崩溃（AIMET 量化发生在 quantizer 阶段而不是 converter 阶段）。

**P2（成本较高，但最直接解决"根因"）：换一个 QAIRT SDK 版本**
当前用的是 `2.48.0.260626`。检查有没有更新的版本已经修了这个 `IrQuantizerV2` 崩溃（既然 Release
Notes 里能看到大量 SM8750/量化相关的持续修复记录，这个 bug 存在于下一个版本被修掉的可能性不低）。
换版本前记得同步确认新版本的 `backend_extensions` 两层 config 格式有没有变化（12.4 节踩过的坑）。

**P3（兜底，如果 P0-P2 都不可行）：接受当前质量，或联系高通报 bug**
把 12.5.1 节的完整崩溃复现步骤整理成一个最小复现（一个尽量小的 ONNX 子图 + override 文件），提交给
高通技术支持/论坛。这条路时间成本最高，只在 P0-P2 都失败后才值得投入。

**辅助工作（不阻塞上面任何一条，随时可以做）：**
- 测试 v79 目标下单图 HTP 内存上限是不是比 v73 的 3.5GB 更高——如果更高，`transformer_part1a`+`part1b`
  有机会合并回一张图，减少设备端每步的重复加载次数，可能有速度收益。这个和 add_138 修复是两条独立的
  优化线，互不阻塞，值得抽空测一下（拿 `transformer_part1_fixed.onnx`，就是被删掉的未拆分版 DLC 对应
  的 ONNX 源文件，仍在 `D:\ZImage_Work\ZImage_QNN_Evidence\onnx\`，重新走一遍 converter+quantizer+
  context-binary-generator 用 sm8750 目标，看会不会撞上类似当初 3.5GB 那样的报错）。
- text_encoder 四段和 VAE 不需要再动（已验证正常）。

### 12.8 給下一个会话/下班后接手的建议顺序

1. 先读本节（12.6～12.7）了解最新状态和排查计划，不用重读第一到十一节的排查细节（除非要深挖某个
   已经排除的假设为什么被排除）。
2. 按 P0→P1→P2 顺序尝试修复 `add_138`，每一步都用 `snpe-net-run.exe` 离线对比 FP32 参考值
   （`D:\ZImage_Work\ZImage_QNN_Evidence\onnx\residual_chain_reference.npz`），不需要手机就能知道
   有没有改善。
3. 有实质性数值改善后，再重复 12.5.2～12.5.3 节的流程（换 SM8750 目标 + 重新编译 context binary），
   最后装机做一次视觉确认收尾。
4. 全程记得："一次只改一个变量"、"不要因为没崩溃就默认没问题，需要真的有证据"——这两条是这一整晚
   反复被验证有效、也反复被用户纠正过的纪律。

---

## 十一、2026-08-13 深夜三：跑完 Phase A（纯 FP32），排除量化；顺手发现并修复一个独立符号 bug

### 11.1 Phase A 结果：纯 FP32 依然是噪声——排除"单纯是量化的锅"

跑完了 `zimage_fp32_pipeline.py`（text_encoder 4段 → transformer_part1a/1b/2 循环8步，全 FP32 ONNX
Runtime，无任何量化），耗时 2510 秒（~42 分钟），没有 OOM（三个 transformer session + 前面的
text_encoder session 依次加载/释放，峰值内存扛住了）。

**结果**：`D:\LocalDreamZImage\scratch_runs\zimage_fp32_full_pipeline.png` 是均匀的"电视雪花"随机噪点，
不是竖条纹，也不是换分词器后的马赛克色块——第三种失败形态。8 步下来 latents 标准差从 1.05 一路涨到
2.25（应该收敛，反而在发散）。

按 `binary-imagining-mango.md` 里预先写好的 Phase A 决策分支：**这排除了"纯粹是量化的锅"这个假设**，
不应该继续假设量化，应该去查 ONNX 图手术/结构性 bug。

### 11.2 核对 Python 脚本时顺手发现的独立 bug：`PipelineZImage.hpp` 的 Euler step 符号写反了

写 `zimage_fp32_pipeline.py` 时为了忠实复刻官方 diffusers 公式，调度器更新用的是**加号**：
`latents = latents + dt * noise`（这也是 `ZImageFlowMatchScheduler::step()` 自己的实现，注释里明确写
"对齐 diffusers 官方公式"）。

但真正跑在设备上的 `PipelineZImage.hpp::generate()` **根本没有调用 `scheduler.step()`**，而是手写了一份
内联复刻，符号是**减号**：`latents[i] -= dt * noise[i]`（修复前的 [PipelineZImage.hpp:158-159](local-dream/app/src/main/cpp/src/PipelineZImage.hpp:158)）。

这是同一个代码库里两处互相矛盾的独立证据（`ZImageFlowMatchScheduler.hpp` 自己的 `step()` 方法 vs
`PipelineZImage.hpp` 里没被调用、手写复刻的版本），确定是一个真实、独立的符号翻转 bug——设备端每一步
都在往错误方向走（等价于往回加噪而不是去噪），不管量化对不对，单这一个 bug 就足以让图完全崩坏。**这个
bug 和 Phase A 测的量化问题是两条独立的线，FP32 脚本用的是正确符号，不受这个 bug 影响。**

**已修复**：把 `-=` 改成 `+=`（见文件当前版本），重新编译了 APK（`gradlew assembleBasicDebug`，
BUILD SUCCESSFUL），`adb install -r` 装到手机（`192.168.31.157:5555`，无线连接依然可用）。

### 11.3 修复后设备端重测：结果和 FP32 一致，都是"电视雪花"——收敛证据

用 `curl POST /generate` 直接触发生成（绕开了 `zimage_device_test.sh` 里那次因为盲点击 UI 时机不巧、
没能自动导航进模型页导致的失败——后来手动截图+点击复现并绕过，细节见下面 11.4），拿到
`D:\LocalDreamZImage\scratch_runs\sign_fix_test2.png`。

**结果**：也是均匀"电视雪花"噪声，和 11.1 节 FP32 脚本跑出来的图**视觉上高度相似**——不再是最早的竖
条纹（`../evidence/images/zimage_test_output.png`），也不是换分词器后的马赛克色块（`../evidence/images/zimage_tokenizer_fix_test.png`）。

**这是一个有价值的收敛信号**：符号 bug 修复后，设备端（量化）和 FP32（不量化）两条独立路径产生了**同
一种**失败形态，进一步支持"剩下的问题是量化和非量化路径共有的结构性 bug"这个方向，而不是量化独有的
问题。和 Phase A 的结论一致，指向 ONNX 导出/图手术阶段（交接文档第二节的 GatherND/IsNaN/Slice 替换）
或者 patchify/unpatchify、RoPE 应用顺序这类结构性问题。

### 11.4 顺带记录：`zimage_device_test.sh` 这次自动化失败的原因（脚本本身没问题，是时机问题）

跑 `zimage_device_test.sh` 时超时失败（`Server listening` 120 秒内没出现）。用 `adb screencap` 截图
确认 APP 停在模型列表页，没有导航进 ZIMAGE 生成页——说明脚本里 `sleep 2` 后盲点两下固定坐标
`(608, 1279)` 这次没点中（很可能是列表还没渲染完）。手动用同样坐标点了一次，立刻成功导航、模型开始
加载——证明坐标本身没错，单纯是这次自动化的两次盲点击时机不巧。**脚本逻辑本身不需要改**，以后如果再
遇到同样情况，重跑一次或者手动截图确认+点击即可，不是环境或代码变了。

### 11.4.1 磁盘清理记录（持续更新）

- 删除未拆分版 `dlc_pipeline/text_encoder`（15GB,被4段拆分方案取代）
- 删除未拆分版 `dlc_pipeline/transformer_part1`（17GB,被 part1a+part1b 拆分取代）
- 删除 `transformer_part1b_ctx.bin`/`transformer_part2_ctx.bin`/`vae_decoder_ctx.bin`（旧命名规范的重复文件,正式名称是 `_ctx.SM8550.bin`）
- 删除 `D:\models\Z-Image-Turbo`（31GB,原始 PyTorch checkpoint,已确认无脚本依赖它的权重本身,只有
  tokenizer 路径引用过,已改指向 `D:\ZIMAGE\tokenizer\tokenizer.json`；HuggingFace 上可重新下载,不是
  本地唯一数据）

D 盘可用空间:117GB → 清理后 152GB → 删完原始模型后 183GB。

### 11.5 下一步

按 Phase A 的决策分支（见 `binary-imagining-mango.md`）：不要再假设是量化，回头审查 ONNX 图手术
（交接文档第二节的手工算子替换）、或者对照原始 PyTorch/diffusers Z-Image Turbo 实现，逐层核对
patchify/unpatchify 和 RoPE 应用顺序这类结构性问题。也可以考虑：先给 FP32 脚本的 part1a/part1b/part2
中间张量加类似 `logStats` 的埋点，和已经缓存好的 `residual_chain_reference.npz`（纯 ONNX 单次前向的参
考值）做对比，看多步累积后中间张量是否在某一步开始明显偏离参考分布——这个可以纯离线做，不需要手机。

---

## 十三、2026-08-14 早上：推翻了整条量化 override 路线，并把误差定位到了 32 个 caption token 的单个通道
> 🔴🔴 **【导航标注 2026-09-06】本节整轮方向已被 §十四 否定**（transformer 量化不是根因）。
> 689 行，是本文档最长的一节，**内容全部是一条死路的探索过程**。
> **新接手者不要读**；只在想知道「为什么某条路被排除」时回查。按约束 2 原文保留。

> 本节是当前最新状态，新会话直接从这里接手。
>
> **本节的叙事弧线（按发生顺序，也是推理链条的顺序）**：
> - **13.1～13.7**：解决了 `--quantization_overrides` 的原生崩溃（根因是 override JSON 的 `version`
>   字段），但绕开后卡在一个真机编译校验失败上（`node_MatMul_333`，根因未查明）。
> - **13.8【转折点】**：用纯计算证明**整条 override 路线的收益上限只有 0.004%**——误差根本不来自
>   残差张量自身的编码。这推翻了 12.7 节整个 P0～P3 计划，也让 13.1～13.7 那些工具链问题变得无关紧要
>   （不用再解决 `node_MatMul_333`）。
> - **13.9～13.11**：纠正了 8.4 节两个错误的结构认知（`add_54` 不是"第1层"；part1a 不是 12 个串行
>   block 而是"图像流 2 块 + caption 流 2 块 + unified 8 块"），并发现了一个关键的方法学能力
>   （`snpe-net-run` 可导出任意中间张量，定位工作完全不依赖卡住的工具链）。
> - **13.12～13.13**：补测从未被测量过的前 4 个 block，发现误差在两条支流**合并处出现 28 倍阶跃**，
>   并把它精确收敛到 **32 个 caption token 上的单个通道（通道 85）**。第二轮定位进行中。
>
> 工作目录：`D:\ZImage_Work\p0_experiments\`（13.1～13.7 的产物，文件名带 `p0_1`~`p0_10` 编号；
> 13.10 之后的测量产物是 `snpe_early_chain\`、`snpe_merge_window\` 等）。
> 分析脚本统一在 `D:\LocalDreamZImage\scripts\`（清单见 13.10 末尾）。

### 13.1 崩溃根因找到了，而且是可以直接读代码确认的，不是猜测

按 12.7 节 P0 优先级复现崩溃（`transformer_part1a.onnx` + 单张量 `add_138` 的 int32 override，
schema `"2.0.0"`），确认崩溃点和 12.5.1 节记录的一致，但这次用 `--quantizer_log` 拿到了 Python 层
堆栈（之前没拿到过）：

```
File "...\qti\aisw\converters\qnn_backend\ir_to_dlc.py", line 650, in quantize_cpp_graph
    quantizer.apply_encodings()
RuntimeError: Windows fatal exception: access violation
```

顺着这个堆栈往上读 `ir_to_dlc.py` 的 `get_quant_version()`（第 577~593 行），发现了**决定性的路由逻辑**：

```python
external_quant_params = graph.user_quantization_overrides
if (external_quant_params and "version" in external_quant_params and
    external_quant_params["version"] == "2.0.0" or args.use_quantize_v2):
    return QuantizerVersion.APPLY_ENCODINGS   # -> IrQuantizerV2.apply_encodings() —— 会崩溃的新版量化器
return QuantizerVersion.V1                     # -> IrQuantizer.quantize() —— 老版量化器，稳定
```

**override JSON 的 `"version"` 字段精确等于字符串 `"2.0.0"`，才会走新版 `IrQuantizerV2`（有原生崩溃
bug）；换成 `"0.6.1"` 或 `"1.0.0"` 这两个老版 schema，会走完全不同、稳定的 `IrQuantizer.quantize()`
路径。** 这不是猜测，是直接读到的分支条件。

`0.6.1` 这个 schema 不用凭空造——`qairt-quantizer --dump_encoding_json` 可以直接从**现有已经部署、
确认工作正常**的 `transformer_part1a_quantized.dlc` 导出全部 1023 个 activation + 265 个 param 的
真实编码，格式正好就是 `0.6.1`：

```powershell
qairt-quantizer -i transformer_part1a_quantized.dlc -o dump_test_out.dlc --dump_encoding_json
# 产出 dump_test_out_encoding.json，version 字段是 "0.6.1"
```

**验证**：把这份原样导出、一个字段没改的 `0.6.1` 编码整个喂回 `qairt-converter -q`，**转换干净成功，
没有崩溃**（`D:\ZImage_Work\p0_experiments\p0_4_full061_unmodified.dlc`，`INFO_CONVERSION_SUCCESS`）。
崩溃问题到这里可以认为**彻底解决**。

### 13.2 用这条路验证 `add_138` 精度修复机制本身：float fallback 可行，int32 不可行

在上面这份完整 `0.6.1` 编码里，把 `add_138` 单独一条改成 `{"bitwidth":16,"dtype":"float"}`（完全跳过
定点量化，只保留浮点——这正是 9.3 节最初想要、但在 `2.0.0` schema 下因为 `get_tensor_bw()` 不认
`float16` 而失败的方案），重新跑 `qairt-converter`：**成功**，没有崩溃
（`p0_5_add138_floatfallback.dlc`）。

**顺手确认了一条边界**：先试的是把 `add_138` 改成 `int32`（复用之前算好的、基于真实参考数据的
scale），这个**没有崩溃，但被 SDK 干净拒绝**：
```
RuntimeError: modeltools::IrQuantizer::modifyScaleOffsetWithNewBw: Activation bitwidth conversion
from 16 to 32 is not supported. Supported conversions are 8->16 and 16->8.
```
这是硬件/SDK 的明确限制，不是配置问题——HTP 激活值定点量化只支持 8/16 bit 互转，不支持 32 bit。
以后不用再试这个方向，`add_138` 的精度修复只能走 float fallback（或者在 16 bit 内收窄/裁剪离群值
的 scale，还没试过）。

### 13.3 离线验证（`snpe-net-run` CPU 模拟器）被一个和 `add_138` 无关的图结构问题卡住

按第八节的方法用 `snpe-net-run.exe` 跑 `p0_5_add138_floatfallback.dlc` 想验证误差是否真的改善，结果：
```
error_code=1002; No backend could validate Op=node_select_scatter_1 Type=ScatterElements
```
**关键排除实验**：同样的失败在**完全没改 `add_138`、原样喂回去**的 `p0_4_full061_unmodified.dlc`
上**一模一样复现**，而**已经部署、确认工作正常**的 `transformer_part1a_quantized.dlc` 用同一个
`input_list` 跑 `snpe-net-run` **完全正常**。三者对比说明：这个失败和 `add_138`、和我们的 override
内容**无关**，是"用 override 转换出来的 DLC"这个**转换路径本身**的问题。

**根因**：`qairt-converter` 只要检测到 `--quantization_overrides`，就会**强制跳过内部 ONNX 简化**
（无论有没有传 `--onnx_skip_simplification`，日志明确打印
`Can't simplify the model when ... quantization overrides ... are specified, converting without
simplification`）。已验证的部署版 DLC 是**没有用 override**、走正常两阶段流程（`qairt-converter`
简化 → `qairt-quantizer` 校准量化）产出的，图结构和我们这条"override 直出量化 DLC"的路径不一样。

### 13.4 尝试用第三方 `onnxsim` 预简化绕开——修好了一个真实 bug，但没解决最终问题

思路：既然 `qairt-converter` 自己不让简化，就用独立的 `onnxsim`（`pip install onnxsim`）**在喂给
`qairt-converter` 之前**先把 `transformer_part1a.onnx` 简化好。

**踩的坑 1**：`onnxsim` 简化后模型 >2GB（权重被内联实体化，原文件用的外部数据引用被摊平），必须用
`onnx.save_model(..., save_as_external_data=True, ...)` 存，不能直接 `onnx.save()`（会报
protobuf 2GB 序列化失败）。

**踩的坑 2（真实 bug，已定位并修好）**：简化后的模型喂给 `qairt-converter` 报
`ValueError: Duplicate buffer name, select_45 already exists`。查清楚是因为 `select_45`/`select_46`/
`unified_freqs`/`unified_mask`/`latents_shape` 这 5 个张量本来是"运行时计算得到、同时又是 graph
output"，`onnxsim` 常量折叠时把它们变成了 initializer（因为它们只依赖固定的位置索引，不依赖真实输入
数据，这个常量折叠本身是对的），但**initializer 名字和 graph output 名字撞了**，`qairt-converter`
的图构建器不接受"一个名字同时是常量又是输出"。**修法**：把 initializer 改名加 `__const` 后缀，插一个
`Identity(initializer__const) -> 原名` 节点补上引用，**且这个 Identity 节点必须插在节点列表最前面**
（`qairt-converter` 按文件顺序处理节点，不会自动重新拓扑排序，插在后面会导致更早的消费节点找不到这个
buffer，报 `KeyError`）。

用这个修好的 onnx（`transformer_part1a_simplified_fixed2.onnx`）重新转换：**转换本身成功**
（`p0_10_add138_fixed_simplified.dlc`），`select_45` 相关的报错彻底消失。

**但真机编译（下一节）证明这还不是根因**——`onnxsim` 的简化结果和 `qairt-converter` 自己内部的简化器
**不是同一个实现**，即使解决了这个具体的命名冲突，产出的图在其它地方仍然和"内部简化器产出的图"不等价。

### 13.5 真机编译（`qnn-context-binary-generator`）证明这是真实问题，不只是 CPU 模拟器较真

一开始的判断失误（**已经被用户当场指出，记录在案，避免以后重复**）：以为"CPU 模拟器验证不了"只是
`snpe-net-run` 这个老旧模拟器比较挑剔，真机用的 HTP 后端应该没这么严格，于是跳过继续修
`onnxsim`，直接拿 `p0_5_add138_floatfallback.dlc` 去跑真机编译（`qnn-context-binary-generator
--htp_socs sm8750`）。**结果 200 毫秒内就报错**，而且是一个信息量很大的干净校验失败（不是崩溃）：
```
<E> [4294967295] has incorrect Value -32341, expected equal to -32768.
QnnBackend_validateOpConfig failed 3110
Failed to validate op node_MatMul_333 with error 0xc26
```
**这说明"CPU 模拟器过不了"和"真机过不了"是同一个根因，不是两件独立的事**——这个判断错误已经被用户
当场纠正，教训记录在这里供以后参考：**一个验证渠道过不去，不能想当然认为换一个权威渠道就能绕开，
应该先怀疑是不是同一个结构性问题的另一种表现形式。**

**定位到具体张量**：反查 `-32341` 这个值，发现它精确对应 override 编码里 `val_332` 的
`offset: -32341.0`（`node_MatMul_333` 的输入之一）。而 HTP 对 `MatMul` 某个输入角色的要求是**必须
是精确对称的 16-bit 编码**（`-32768` 是有符号 int16 对称量化的标准零点），但我们从已部署 DLC 原样
导出的 `val_332` 编码是**非对称**的（`offset=-32341`，接近但不精确等于 `-32768`）。

**关键排除实验（和 13.3 一样的方法论）**：这个失败在**完全没碰 `add_138`** 的 `p0_4` 上**同样复现**，
证明和我们的修复内容无关，是通用问题。而且用 13.4 节修好 `select_45` 冲突后的版本重新走一遍，**还是
同一个 `node_MatMul_333` 报错**——说明修掉命名冲突并不足以解决它。

> **⚠️ 本节原来在这里写过一段根因解释（"外部 onnxsim 和内部简化器化简结果不等价，导致张量角色对不
> 上"），那段是未经验证的推断，已删除。存在一个直接反例：被拒绝的那个值 `offset = -32341`（张量
> `val_332`）**本身就是从已部署、已验证能跑的 `transformer_part1a_quantized.dlc` 导出来的**
> （`dump_test_out_encoding.json`，`is_symmetric: false`, `bitwidth: 16`），而那份 DLC 在 08-14 01:27
> 成功编译出了 `transformer_part1a_ctx_sm8750.SM8750.bin` 并真机跑通（12.5.3 / 12.5.4）。**同一个数值
> 在一条链路里合法、在另一条链路里被拒绝**，所以"这个编码值违反硬件约束"不是完整解释。真实原因至少
> 还有两种可能没有排除：(a) 两条链路里消费该张量的算子不是同一个（结构差异）；(b) `--dump_encoding_json`
> 导出→回灌不是无损的（`0.6.1` schema 不记录 QNN 侧数据类型的符号性，也不记录算子级量化属性）。
>
> **不过这个问题现在已经不重要了**——见 13.8，整条 override 路线被证明即使全部走通也修不好画质问题，
> 所以不建议下一个会话再花时间查 `node_MatMul_333`。这里保留记录只是为了说明"我们当时为什么卡住"，
> 以及避免有人重新捡起这条线时以为根因已经查明。

### 13.6 试过的第三条路：`light_weight_quantizer`（`qti.aisw.tools.light_weight_quantizer`）——不适用，已排除

设想：能不能跳过 `qairt-converter` 的转换阶段，直接对**已经结构正确**的 `transformer_part1a_fp32.dlc`
（已存在、简化过、和部署版同源）做"外科手术式"的单张量编码替换？SDK 里确实有一个专门模块看起来像是
干这个的（`D:\qairt\...\lib\python\qti\aisw\tools\light_weight_quantizer\ir_graph_updater.py`，
`IrGraphUpdater.set_encodings()` + `LightWeightIrQuantizer`）。

**读源码后排除**：这个工具是为另一条完全不同的官方工作流（"model preparer pro"）设计的，
`set_given_encodings()` 内部强依赖一个 `self.prepared_model_info`（从 `weight_file_path` 指向的
safetensors/pickle 文件里加载的元数据，包含 `param_name_mapping`、`additional_transpose_info` 等），
这条链路我们完全没有，也没有生成它的工具。而且它期望的编码格式是**按算子名 + input/output 索引**
组织（`{op_name: {"input": {"0": {...}}, "output": {"0": {...}}}}`），和我们从
`--dump_encoding_json` 拿到的**按张量名**组织的格式（`{tensor_name: [{...}]}`）不兼容，需要一个
目前没有的映射关系才能转换。**这条路当前工作量不可控，已排除，不建议下一个会话重新尝试，除非先
找到"model preparer pro"工作流的入口工具（转了一圈没找到对应的 CLI，可能在 QAIRT 更新的版本里，或者
根本不适用于我们这种 ONNX 直接转换的场景）。**

### 13.7 工具链层面的证据总账

**已解决、高置信度：**
- `--quantization_overrides` 的原生崩溃：根因是 override JSON `version=="2.0.0"` 触发有 bug 的
  `IrQuantizerV2`，换成 `version=="0.6.1"` 走稳定的老版 `IrQuantizer` 即可绕开。这个结论有直接读
  到的源码分支作证据（`ir_to_dlc.py:577-593`），不是试出来的巧合。
- `add_138` 用 `dtype:"float"`（float fallback）可以被 `qairt-converter` 接受并成功转换，不崩溃。
  `int32` 方案被 SDK 明确拒绝（激活值定点量化只支持 8↔16 bit 互转），不用再试。

**已确认存在、未解决、但已经不重要（见 13.8）：**
- 只要用了 `--quantization_overrides`（不管改不改 `add_138`，哪怕原样注入已知正确的编码），产出的
  DLC 在真机 HTP 编译阶段会在 `node_MatMul_333` 报编码校验失败。根因**没有查明**（13.5 节记录了
  一个直接反例，推翻了当初写下的解释）。
- 外部 `onnxsim` 预简化路线：修好了一个真实的命名冲突 bug（见 13.4，那个修法本身是对的、可复用），
  但没有解决 `node_MatMul_333`。
- `light_weight_quantizer` 路线已排查并排除（见 13.6）。

---

### 13.8 【决定性结论】整条 override 路线修不好画质问题——收益上限只有 0.004%

> 这是本次会话最重要的结论，**推翻了 12.7 节 P0/P1/P2/P3 整个计划所依赖的前提**，也推翻了
> 13.1～13.6 这一整轮工作的意义（工具链问题即使全部解决，也不会让画质变好）。
> 复现脚本：`D:\LocalDreamZImage\scripts\quant_error_ceiling_analysis.py`（纯计算，不跑 SDK，
> 几秒出结果）和 `diff_float_fallback_lists.py`。

**推理依据**：一个张量用 scale = s 的定点编码存储，如果喂给它的是**完全正确**的浮点值，那么"存储这
一步"能引入的误差上限就是 s/2（round-to-nearest）。所以只要把"实测误差"和"s/2"对比，就能知道误差
到底是不是这个张量自己的编码造成的——**这个判断不需要跑任何 SDK，用已有的
`residual_chain_reference.npz` + `--dump_encoding_json` 导出的真实编码就能算。**

| 张量 | 自身编码误差上限 (s/2) | 8.4 节实测最大误差 | 倍数 | 可归因于自身编码 |
|---|---|---|---|---|
| add_54 | 0.00564 | 31.8 | 5,641× | 0.0177% |
| add_66 | 0.00590 | 34.3 | 5,811× | 0.0172% |
| add_78 | 0.00717 | 48.2 | 6,724× | 0.0149% |
| add_90 | 0.00774 | 47.3 | 6,107× | 0.0164% |
| add_102 | 0.00865 | 60.5 | 6,991× | 0.0143% |
| add_114 | 0.00938 | 87.3 | 9,306× | 0.0107% |
| add_126 | 0.01118 | 149.3 | 13,355× | 0.0075% |
| **add_138** | **0.01301** | **321.8** | **24,729×** | **0.0040%** |

**逐元素验证 `add_138`**：拿 FP32 参考值，假设输入完全正确，用它现有的编码量化再反量化——最大误差
0.013031、均值 0.006506、**误差超过 1.0 的元素数量为 0**。而实测最大误差 321.8，差 **24,695 倍**。

**排除"scale 记错了"**：把 FP32 参考值的 `(max−min)/65535` 和 dump 出来的 scale 逐个对比，8 个张量
全部吻合（差异 0.9%～22%，且 dump 的 scale 系统性略大——这正是"校准用了多个样本、覆盖范围比单样本
更宽"的预期表现）。scale 是真实可信的。

**排除"float 标记有连带效应"**：对比 `p0_4`（原样回灌、未改 `add_138`）和 `p0_5`（`add_138` 标成
float）两次转换的 float fallback 算子列表——**都是 74 个，逐个比对零差异**。所以标成 float 确实只
改变了 `add_138` 一个张量的存储格式，没有让任何上游算子转入浮点计算，不存在"意外获得额外收益"的可能。

**结论（措辞已按独立审查意见收紧，2026-08-14）**：
1. 每个残差张量 **≥99.98% 的误差，在它被存进去之前就已经存在于数值里了**，不是存储这一步造成的。
   这一条严格成立（前提：无 clipping、round-to-nearest、scale/offset 解释正确，均已核对）。
2. 把 `add_138` 标成 float，**只能改变该张量自身至多 0.013 的值**（而它已带有 321.8 的误差），
   无法消除任何从上游传入的误差。
   > ⚠️ 本节原文写的是"收益上限 0.013/321.8 = 0.004%，数学上不可能有可见效果"。**这个表述超出了
   > 证据范围，已收回**：0.004% 描述的是"该张量局部误差的来源占比"，**不能**直接换算成最终画质
   > 收益的上限——下游还有非线性、归一化、门控、注意力和 8 步去噪，微小扰动既可能被衰减也可能被
   > 放大。成立的说法只是"改这些残差张量自身的编码无法修复已从上游传入的大误差"。
3. **同样的判据适用于全部 8 个残差张量**，所以 12.5.1 节原计划的"覆盖 `add_54`~`add_138` 全部 8 个
   张量"同样无效。
   > ⚠️ 本节原文给的"收益上限约 0.018%"来源不清（那实际是表格里单张量最大占比 0.0177%，被误标成
   > 了 8 张量合计）。8 个 s/2 相加 = 0.06867，除以 321.8 = 0.0213%。但**跨层相加本身没有物理意义**
   > （8 个张量位于不同层，误差会经过不同的后续变换），所以这个聚合数字已删除，改为逐张量陈述
   > （见上表"可归因于自身编码"列）。

这也解释了 8.3 节当初的观察（真值 410.97 读出来 732.77——差 12,365 个量化步长，本来就不可能是舍入
误差）。8.3 节自己写过"量级差了四个数量级，更像是计算链路里某个环节出的问题"，但后续 8.4/9.x/12.7
还是走回了"调这个张量的量化配置"，这个方向从那时起就是错的。

**需要保留的 8.4 节结论**：离群激活通道现象本身是**实测的、真实的**（通道 1238 上 4120/4128 个 token
都是离群值）。被推翻的是**归因**——离群通道造成伤害的位置不在残差张量的存储上。

---

### 13.9 【重要纠正】`add_54` 不是"第 1 层"，前 5 个 attention block 从来没有被测量过

8.4 节的表格把 `add_54` 标注为"第1层"，并据此得出"误差从第 1 层就已经存在"。**这个标注是错的**，
我在本次会话中也一度沿用了它（说过"第一个 transformer block 内部就在产出错误数值"，同样是错的）。

**静态图核对结果**（脚本逻辑：从 `add_138` 沿主残差流反向遍历，并统计每个节点之前完成了几个 Softmax）：

```
part1a 共有 12 个 Softmax（= 12 个 attention block）
主残差流共 15 个 add_N 节点：add_54, add_57, add_66, add_69, add_78, add_81, add_90,
                             add_93, add_102, add_105, add_114, add_117, add_126, add_129, add_138

张量        拓扑位置    它之前已完成的 attention 数
add_54        618              5     <-- 8.4 节以为这是"第1层"，实际它前面已经有 5 个 attention
add_66        725              6
add_78        832              7
add_90        939              8
add_102      1046              9
add_114      1153             10
add_126      1260             11
add_138      1367             12
```

`add_54` 之所以是残差链上第一个 `add_N`，只是因为**在它之前主数据流用的是别的命名**
（`add_54 = Add(select_scatter_4, mul_127)`，输入里没有任何 `add_N`），不代表它是第一层。从图输入
走到 `add_54` 一共要经过 **624 个算子**，其中包含 5 个完整的 attention block。

**所以真正的情况是**：误差在"完成 5 个 attention block 之后"已经达到 31.8，而**前 5 个 block 和
embedding/patchify 阶段从来没有任何人测量过**。误差的真正起点仍然未知，它可能在这 624 个算子里的
任何位置。

### 13.10 定位误差起点的方法：已验证可行，且完全不依赖卡住的 override 工具链

**关键能力发现（已实测确认，这是本次会话在方法学上最有用的一条）**：
`snpe-net-run` 的 `input_list` 第一行用 `%` 前缀列出的张量名，**不限于图的正式输出，任意中间张量
都可以按名字导出**。

验证方式：用 `%add_57`（这个张量既不在 8.4 的测量集里，也不是 `transformer_part1a` 的图输出）跑
已部署的 `transformer_part1a_quantized.dlc`，成功产出 `Result_0/add_57.raw`，大小
**63,406,080 字节 = 4128 × 3840 × 4**，尺寸精确吻合，确认导出的就是那个张量本身。

这条能力的意义：**任何"量化后中间数值是多少"的问题，都可以在已部署的、走正常两阶段流程产出的
DLC 上直接问出来**，不需要重新转换、不需要改图、不需要碰 13.1～13.6 那套卡住的 override 工具链，
单次运行约 4 分钟。`node_MatMul_333` 那个坑因此可以直接绕过去，不必解决。

**注意**：转换器会融合掉一部分张量，被融合的名字在 DLC 里不存在，请求它会报
`error_code=204; Couldn't find name. One or more specified output tensors don't exist!` 而整个运行失败
（不是跳过）。**先用 `qairt-quantizer --dump_encoding_json` 导出的那份 JSON 当权威名单**（里面的
1023 个 activation 名字就是 DLC 里真实存在的张量），核对之后再请求。实测 `add_24`、`add_45` 这两个
残差张量就是被融合掉的，不可导出。

配套脚本（本次会话新写，都在 `D:\LocalDreamZImage\scripts\`）：
- `trace_early_blocks.py`：FP32 参考侧（给 ONNX 加 tap → onnxruntime 跑一遍 → 存 npz）。
  **注意 tap 后的 ONNX 必须存回 `ZImage_QNN_Evidence\onnx\` 同一目录**，否则外部权重数据的相对
  路径解析不到。实测 session 加载 88s + 推理 152s。
- `compare_early_blocks.py`：两侧对比，按"误差 vs 该张量自身 s/2"的判据输出定位结论。
- `quant_error_ceiling_analysis.py`：13.8 那套收益上限分析（纯计算，几秒）。
- `diff_float_fallback_lists.py`：对比两次转换的 float fallback 算子列表。

### 13.11 【重要结构发现】part1a 不是 12 个串行 block，是"图像流 2 块 + caption 流 2 块 + unified 8 块"

13.9 说"`add_54` 前面有 5 个 attention"是对的，但把它们理解成 5 个串行层是**错的**。FP32 参考值的
shape 直接给出了真实结构：

| 残差张量 | shape | 所属数据流 |
|---|---|---|
| `add_9`, `add_12`, `add_21`, (`add_24`) | **(1, 4096, 3840)** | **图像流**（4096 个图像 token），2 个 block |
| `add_32`, `add_35`, `add_42`, (`add_45`) | **(1, 32, 3840)** | **caption 流**（32 个文本 token），2 个 block |
| `add_54` 及之后全部 | **(1, 4128, 3840)** | **unified**（4096 + 32 = 4128），8 个 block |

Softmax 拓扑位置 `[138, 245 | 386, 482 | 605, ...]` 和这个划分完全对应：前 2 个属于图像流，中间 2 个
属于 caption 流，从第 5 个（idx 605，紧接着就是 `add_54`）开始才是 unified 流。两条流通过
`select_scatter_4` 合并——这也解释了 13.9 里"残差链命名在 `add_54` 处断掉"的现象：不是断链，是两条
支流在这里汇合。

`part1a` 完整残差链（23 个残差加法，识别签名 = `Add(残差流, mul_N)`）：

```
图像流:   add_9(151)   add_12(174)  add_21(258)  add_24(281)      <- 从未测量
caption流: add_32(398)  add_35(419)  add_42(494)  add_45(515)      <- 从未测量
unified:  add_54(618)  add_57(641)  add_66(725)  add_69(748)  add_78(832)  add_81(855)
          add_90(939)  add_93(962)  add_102(1046) add_105(1069) add_114(1153) add_117(1176)
          add_126(1260) add_129(1283) add_138(1367)
（括号内是拓扑位置；8.4 节测的是 add_54/66/78/90/102/114/126/138，即 unified 段每隔一个采样）
```

**FP32 参考值的数值范围（本次实测，`early_block_reference.npz`）**：

| 张量 | 流 | min | max | std |
|---|---|---|---|---|
| add_9 | 图像 | −7.73 | 34.39 | 0.654 |
| add_12 | 图像 | −7.73 | 40.13 | 0.745 |
| add_21 | 图像 | −7.73 | 41.48 | 0.769 |
| add_32 | caption | −27.45 | 177.74 | 2.580 |
| add_35 | caption | −37.36 | **496.70** | 3.235 |
| add_42 | caption | −56.15 | **626.80** | 5.143 |
| add_54 | unified | −85.20 | 647.37 | 0.913 |

**这是一条重要线索**：撑开整个动态范围的极端数值**产生在 caption 流**（`add_42` 已经到 626.8），
unified 流的 647.4 几乎全部是从 caption 流继承过来的，图像流自己最大只有 41.5。而 8.4 节诊断的
"离群激活通道"正是这个现象在 unified 流上的表现。

**这个结构事实对修复方案的意义（重要，但要等误差定位数据确认后才能下结论）**：caption 流只有
**32 个 token**，是图像流（4096）的 1/128、unified 流的 0.8%。如果误差确实源自 caption 流，那么
"把整条 caption 分支保持浮点、图像分支照常量化"就是一个**成本极低**的干预手段——32×3840 的 float16
只有 245KB。这比 SmoothQuant 那种需要改模型权重的方案简单得多，也比"给单个张量加 override"更对症。
**但在拿到量化侧实测误差之前，这只是一个由数值范围推出的假设，不能当结论。**

> **【上面这个假设已被下一节的实测数据推翻，保留原文只为记录推理过程】**：caption 流虽然数值最大，
> 但它的**相对误差反而是全图最小的（0.09%~0.27%）**。"数值大 ⇒ 量化误差大"这个直觉在这里不成立。

---

### 13.12 【误差起点定位·第一轮，结论已被 13.13 修正】稀疏采样观察到 `add_42` 与 `add_54` 之间有 28 倍阶跃

> **⚠️ 本节标题原为"误差在两条支流合并处出现 28 倍阶跃"，该归因已撤回。**
> 本轮只测了 `add_21`（图像流）和 `add_42`（caption 流），而**两条支流末端的 `add_24` 和 `add_45`
> 当时都没有测量**，`add_42` 与合并之间还隔着一整个完整的 FFN block。因此当时无法从
> "`add_21`/`add_42` ≈1.1 → `add_54` = 31.84" 推出"误差发生在合并操作中"。
> 13.13 的后续实测证明：误差在**合并之前**就已经存在。
>
> 本节下列表述一并撤回：
> - "两条支流进入合并点时误差都只有 ~1.1"
> - "合并之后立刻变成 31.84"
> - "caption 流把大数值干净地传到了合并点"
> - "问题发生在合并这一步之后"
>
> 准确说法：**第一轮稀疏采样观察到 `add_42`(1.07) 与 `add_54`(31.84) 之间存在约 28 倍阶跃；
> 第二轮测量证明该阶跃在合并前已经存在，范围收敛到 caption 分支最后一个 FFN。**

两侧实测数据（FP32 参考 = `early_block_reference.npz`；量化侧 = `snpe-net-run` 跑已部署
`transformer_part1a_quantized.dlc`，同一份 `sample_0000` 输入字节）：

| 张量 | 流 | ref \|max\| | 自身编码上限 s/2 | **实测最大误差** | 倍数 | 相对误差 |
|---|---|---|---|---|---|---|
| add_9 | 图像 | 34.39 | 0.00034 | 1.13 | 3,292× | 3.3% |
| add_12 | 图像 | 40.13 | 0.00038 | 1.15 | 2,983× | 2.9% |
| add_21 | 图像 | 41.48 | 0.00039 | 1.13 | 2,876× | 2.7% |
| add_32 | caption | 177.74 | 0.00158 | 0.16 | 100× | 0.09% |
| add_35 | caption | 496.70 | 0.00410 | 1.34 | 327× | 0.27% |
| add_42 | caption | 626.80 | 0.00522 | 1.07 | 205× | 0.17% |
| **add_54** | **unified** | 647.37 | 0.00564 | **31.84** | 5,648× | **4.9%** |
| add_66…add_138 | unified | … | … | 34.3 → 321.8 | … | → 26.9% |

**核心观察：两条支流进入合并点时误差都只有 ~1.1，合并之后立刻变成 31.84，跳了 28 倍。**
这不是逐层累积，是一个**阶跃**。在支流内部误差是平的（图像流 1.13→1.15→1.13，caption 流
0.16→1.34→1.07，都没有增长趋势），阶跃之后在 unified 段才开始持续放大（8 个 block 放大约 10 倍）。

**阶跃位置的精确定位**（脚本 `scripts/locate_merge_jump.py`，把 `add_54` 的误差按 token 段和通道拆开）：

```
按 token 段拆分：
  图像 token [0:4096]        ref|max|=41.89   误差max= 1.1837   <- 和合并前 add_21 的 1.1339 一致，干净穿过
  caption token [4096:4128]  ref|max|=647.37  误差max=31.8380   <- 31.84 全部在这 32 个 token 里

按通道拆分（3840 个通道）：
  通道 85    误差 31.8380   （该通道 ref|max| = 647.37，正是整个张量的最大值）
  通道 3780  误差  3.6499
  通道 1892  误差  2.7010
  通道 1331  误差  1.4664
  通道 1238  误差  1.1837
  误差 > 1.0 的通道数:  5 / 3840
  误差 > 10.0 的通道数: 1 / 3840
  通道误差中位数: 0.083
```

**结论：整个"量化画质问题"目前可以收敛到——unified 序列里 32 个 caption token 上的、单个通道
（通道 85）**。这和 8.4 节记录的"通道 85：37 个 token 是离群值，最大 647，是整个张量的最大值"
精确吻合，两次独立测量互相印证。

**需要撤回的假设**：13.11 末尾"离群值来自 caption 流 ⇒ 问题在 caption 流"是错的。caption 流自身的
相对误差（0.09%~0.27%）是全图最低的，它把大数值**干净地**传到了合并点。问题发生在合并这一步之后。

### 13.13 【误差起点定位·第二轮，进行中】阶跃只可能来自两条路径，两条都可直接测量

`add_54 = Add(select_scatter_4, mul_127)`（算子 idx 618），所以 31.84 的误差只可能从这两个输入之一
进来。窗口 `[494, 619)` 共 125 个算子，结构如下：

- **路径 A（残差载体）**：`add_24`[图像流末端] + `add_45`[caption流末端] → `Concat(cat_13)` →
  `Pad(constant_pad_nd_4)` → `ScatterElements` → **`select_scatter_4`**（idx 526）
- **路径 B（注意力输出）**：`select_scatter_4` → RMSNorm → QKV(`linear_35/36/37`) → RoPE →
  attention(`scaled_dot_product_attention_4`) → 出投影(`linear_38`) → `mul_126` → **`mul_127`**（idx 617）

> **⚠️ 测量时的硬性限制**：不要随意 tap attention 的 score 张量（`val_953` = MatMul 输出、
> `val_954` = Softmax 输出）。shape 是 4128×4128×30 heads×4 bytes = **2,044,846,080 字节 ≈ 1.90 GiB
> 每个**（本文档早先写成"TB 量级"，是算错了，差约 500 倍，已更正）。单个还能承受，但一次 tap 多个
> 就是十几 GB 的磁盘和内存，仍应避免。已在 `scripts/trace_taps.py` 的注释里写明。

**实测结果（脚本 `scripts/compare_merge_window.py`）——判定为路径 A**：

| 张量 | ref\|max\| | 误差max | 图像段误差 | **caption段误差** | **通道85误差** |
|---|---|---|---|---|---|
| **`constant_pad_nd_4`**（合并前，Concat+Pad） | 646.04 | 31.9181 | 1.2021 | **31.9181** | **31.9181** |
| `select_scatter_4`（合并后） | 646.04 | 31.9181 | 1.2021 | 31.9181 | 31.9181 |
| `mul_112`（注意力输入，RMSNorm 后） | 45.83 | 2.1307 | 2.1307 | 0.4140 | 0.1563 |
| `scaled_dot_product_attention_4` | 73.70 | 8.6475 | 2.6978 | 8.6475 | 0.4619 |
| `linear_38`（注意力出投影） | 5551.61 | 168.8721 | 105.9379 | 168.8721 | 101.1470 |
| **`mul_127`**（路径B 最终输出） | 3.64 | 0.1815 | 0.0880 | **0.1815** | 0.0801 |

> ⚠️ 原文用"路径A 最大误差 31.9181 + 路径B 最大误差 0.1815 ≈ add_54 最大误差 31.8380"来论证，
> **这个论证方式不严谨**（不同张量的最大误差未必发生在同一元素位置，不能跨位置相加）。
> 已按独立审查意见改为下面的**逐点论证**（脚本 `scripts/pointwise_argmax_audit.py`）。

**逐点严格论证**：`add_54` 的最大误差元素位于 **token 4100（即 caption token 4）、channel 85**。
在**同一个 (token, channel) 位置**上：

| 项 | FP32 | 量化 | 逐点误差 |
|---|---|---|---|
| `select_scatter_4`（路径A） | 291.186584 | 323.104706 | **+31.918121** |
| `mul_127`（路径B） | 1.016405 | 0.936276 | −0.080129 |
| 两项之和 | 292.203003 | 324.040985 | +31.837982 |
| `add_54` 实际 | 292.203003 | 324.040985 | +31.837982 |
| **差额（= `add_54` 自身重量化贡献）** | 0.000000 | 0.000000 | **0.000000** |

逐点相加**精确等于** `add_54` 的误差，且 `add_54` 自身重量化的贡献**恰好为 0**。
路径B 在该位置的误差仅 0.080，其全局最大值也只有 0.1815，取上界也无法抵消路径A 的 31.918。
**⇒ 路径A 主导，此结论不依赖任何跨位置的最大值相加。**

**关键判读**：`constant_pad_nd_4` 是 `Concat` + `Pad` 的结果，**只做数据搬运、不做任何算术**，
而它进来时误差就已经是 31.92。所以误差在**合并之前**就存在了，产生位置是
**caption 流的最后一个 FFN block（`add_42` 误差 1.07 → `add_45`，算子 495~515）**。
`add_45` 恰好是 DLC 里被融合掉、无法直接导出的那个张量，所以要用它前面那串可导出的中间张量二分。

> **一个反直觉但重要的观察，值得记住**：`linear_38` 的误差高达 **168.87**（是全窗口最大的），
> 但它经过 RMSNorm + tanh 门控之后，到 `mul_127` 只剩 **0.18**——大误差被归一化彻底吃掉了。
> **所以不能只看绝对误差大小判断危害，必须看它是否会传播到下游。** 反过来，`constant_pad_nd_4`
> 的误差走的是残差直连（`add_54 = select_scatter_4 + mul_127`），没有任何归一化能衰减它，
> 于是一路累积到 `add_138` 的 321.8。

### 13.14 【误差起点定位·第三轮·已完成】caption 流最后一个 FFN block

窗口：算子 495~515，结构（全部是 caption-only 张量，shape (1,32,3840)，每个仅 491KB）：

```
add_42 (误差 1.07, 已测)
  -> [RMSNorm: pow_30/mean_23/add_43/val_798/rsqrt_23/mul_105  —— 全部被 DLC 融合，不可导出]
  -> mul_106      (可导出)
  -> linear_31 (可导出) , linear_32 (可导出)      <- FFN 的两个并行投影
  -> val_801 = Sigmoid(linear_31) (可导出)
  -> silu_4 = linear_31 * val_801 (可导出)
  -> mul_107 = silu_4 * linear_32 (可导出)         <- SwiGLU 门控相乘
  -> linear_33 (可导出)                            <- FFN 下投影
  -> [RMSNorm: pow_31/mean_24/add_44/val_806/rsqrt_24/mul_108 —— 融合，不可导出]
  -> mul_109      (可导出)
  -> add_45 = add_42 + mul_109  【融合，不可导出 —— 但它就是携带 31.92 误差进入合并的那个张量】
```

本轮测量：`mul_106`、`linear_31`、`linear_32`、`silu_4`、`mul_107`、`linear_33`、`mul_109`。
判读：这条链上误差第一次跳到 ~30 量级的位置，就是根因算子。

**实测结果（脚本 `scripts/compare_caption_ffn.py`）**：

| 张量 | ref 范围 | 误差max | **通道85误差** | 相对误差 |
|---|---|---|---|---|
| `mul_106` | ±2.78 | 0.11 | 0.11 | 4.08% |
| `linear_31` | ±25.2 | 0.66 | 0.16 | 2.61% |
| `linear_32` | ±51.0 | 0.69 | 0.28 | 1.36% |
| `silu_4` | [−0.28, 18.1] | 0.68 | 0.01 | 3.75% |
| **`mul_107`**（SwiGLU 相乘，输入） | **[−726.6, 862.6]** | 8.39 | **0.05** | 0.97% |
| **`linear_33`**（FFN 下投影，输出） | **[−1958.6, 5178.5]** | **350.13** | **350.13** | 6.76% |
| `mul_109`（RMSNorm+门控后） | [−52.5, 191.0] | 31.84 | 31.84 | 16.67% |

**`linear_33` 这一个 MatMul 把误差放大了 41.7 倍**（8.39 → 350.13），而且 100% 落在通道 85。

通道 85 的数值在这一步被"创造"出来——它在输入 `mul_107` 里只占张量最大值的 **0.1%**（毫不起眼），
在输出 `linear_33` 里占 **37.6%**，到 `mul_109` 占 **100%**。**这就是那个 massive activation 通道
的诞生位置。**

---

### 13.15 【根因确认】机制已完全闭合：`mul_107` 的激活量化误差被通道 85 的超大权重列放大

在 numpy 里做误差来源分解（脚本 `scripts/decompose_linear33.py` 和 `decompose_linear33_v2.py`），
`linear_33 = MatMul(mul_107, val_802)`，权重 shape (10240, 3840)：

| 对照组 | 通道85误差 | 距设备实测的差距 |
|---|---|---|
| FP32 `mul_107` @ FP32 权重（基准） | 0.0005 | 350.13 |
| FP32 `mul_107` @ 8bit权重 per-tensor | 9.64 | — |
| FP32 `mul_107` @ 8bit权重 per-channel | 9.64 | — |
| 设备 `mul_107` @ FP32 权重 | 349.99 | 43.78 |
| **设备 `mul_107` @ 8bit权重 per-tensor** | **350.13** | **0.0061** ← 精确复现设备行为 |
| 设备 `mul_107` @ 8bit权重 per-channel | 350.13 | 45.80 |
| 设备实测 | 350.13 | 0 |

**倒数第二行和设备实测吻合到 0.0061**，等于在 numpy 里精确复现了设备的计算。由此确认的完整机制：

1. **误差 100% 是从 `mul_107` 继承进来的**，不是 MatMul 内部产生的（不是累加器饱和、不是溢出）。
2. **放大器是通道 85 的权重列**：`|max| = 6.9375`，在 3840 列里**排名第 1**，rms 0.553 是全体中位数
   0.173 的 3.2 倍。`mul_107` 里 rms 仅 0.0703 的量化误差，经过 10240 项与这根超大权重列的点积，
   被系统性地累加成 350（注意：若误差是随机符号，累加只会得到约 3.9，实测 350 说明**误差与权重
   强相关**，是系统性的而非随机的）。
3. **误差高度集中**：32 个 caption token 里，只有 **token 4（误差 350.13）和 token 5（误差 177.13）**
   出问题，其余 30 个 token 误差都在 1~10 量级。
4. **`mul_107` 为什么会有这个误差**：它是 SwiGLU 的乘积输出，范围 **±862 但 std 只有 2.65**
   （max/std = 325），per-tensor 16-bit scale ≈ 0.0242，普通元素只能落在约 110 个量化档位上。
   这是典型的"激活离群值撑大 scale、拖累主体精度"形态。

**附带结论：per-channel 权重量化在这里完全没用。** 因为通道 85 本身就是全权重矩阵最大值所在的那一列，
per-tensor 的全局 scale 和 per-channel 给通道 85 的 scale **完全相同**，两者误差都是 9.64。
（这不否定 9.2 节的结论，只是说明"打开 per-channel"对这个特定问题没有帮助。）

> ### ⚠️ 13.15 原本在这里提出的修复方案是错的，已推翻（见 13.17）
>
> 原文写的是"把 `mul_107` 标成浮点 ⇒ 通道 85 误差 350.13 → 9.64，36 倍改善"。
> **这个推论不成立**，理由和 13.8 推翻 `add_138` 方案的理由**完全相同**：
>
> ```
> mul_107 编码 scale = 0.02443940  ->  自身存储误差上限 s/2 = 0.0122
> mul_107 实测误差 max = 8.3896     ->  是自身上限的 686.6 倍
> ```
>
> `mul_107` 的误差**也是从上游继承的**，不是它自己存储造成的。把它标成浮点只能消掉 0.0122，
> 消不掉那 8.39。而 `decompose_linear33_v2.py` 里"FP32 `mul_107` @ 8bit 权重 ⇒ 9.64"这一行，
> 前提是 `mul_107` **完全精确**，而"标成 float"做不到这一点。
>
> **这是同一个陷阱在同一次会话里犯了第二遍**：把"某张量误差大"直接等同于"该张量自身量化不够好"。
> 判据必须始终是 **实测误差 vs 该张量自身的 s/2**——只有二者同量级时，提高该张量精度才有意义。

### 13.17 【自我纠错 + 修正后的理解】误差是全链路注入并被放大的，没有单张量修复点

> ⚠️ **13.16 是基于 13.15 那个已被推翻的修复方案写的，因此 13.16 的"反转"结论也随之失效**
> ——见本节末尾。

按独立审查意见（`C:\Users\sinai\Desktop\CLAUDE_SECTION_13_REVIEW.md`）补做逐点测量后，对 caption FFN
链条上**每个张量**做了"实测误差 vs 该张量自身 s/2"的判据检查（脚本
`scripts/check_systematic_bias.py`、`pointwise_argmax_audit.py`）：

| 张量 | 自身 s/2 | 实测误差max | 倍数 | 误差来源 |
|---|---|---|---|---|
| `add_42` | 0.00522 | 1.07 | 205× | 继承 |
| `mul_106` | ~0.00004 | 0.11 | ~2600× | 继承 |
| `linear_31` | ~0.0003 | 0.66 | ~2000× | 继承 |
| `linear_32` | ~0.0008 | 0.69 | ~885× | 继承 |
| `silu_4` | — | 0.68 | — | 继承 |
| **`mul_107`** | **0.01222** | **8.39** | **686×** | **继承** |
| `linear_33` | ~0.054 | 350.13 | ~6500× | 继承+放大 |
| `mul_109` | — | 31.84 | — | 继承 |

**链条上没有任何一个张量的误差来自它自身的存储。** 所以"把某个中间张量标成 float"这条路
**对整条链上任何一个张量都无效**，不只是对 `add_138` 无效。

**误差的真实结构**（`check_systematic_bias.py` 实测）：
- 系统性偏置模型只解释 **0.03%**，纯随机模型只解释 **2.88%** —— 两者都不是主因。
- 误差与权重的相关系数 **0.343**，说明量化噪声与权重结构耦合。
- 按输入幅值分组看对输出误差的贡献：

| \|x\| 区间 | 元素数 | \|误差\|均值 | 对输出误差的贡献 |
|---|---|---|---|
| [0, 0.01) | 492 | 0.0106 | 0.23 |
| [0.01, 0.05) | 1597 | 0.0116 | 1.19 |
| [0.05, 0.2) | 3643 | 0.0165 | 7.22 |
| [0.2, 1) | 4063 | 0.0306 | 43.72 |
| **[1, 10)** | **423** | **0.2189** | **147.59** |
| **[10, 1000)** | **22** | **1.8657** | **150.04** |

**85% 的输出误差由 4.3% 的大幅值元素（\|x\|≥1）贡献**。这**否定**了"小值被粗 scale 吞掉"这个
直觉假设——恰恰相反，是**大值元素携带的大误差**在主导。原因是 `mul_107 = silu_4 × linear_32`
是两个量化张量的乘积，误差 ≈ `silu·Δlinear₃₂ + linear₃₂·Δsilu`，幅值越大误差越大。

**修正后的机制理解**：量化误差在**每个 MatMul 处由 8-bit 权重量化持续注入**，然后在两类结构上被
显著放大——
1. **两个量化张量相乘**（SwiGLU 的 `mul_107`）：误差按幅值放大；
2. **与极端权重列做长点积**（`linear_33` 的通道 85，权重 \|max\|=6.94 在 3840 列中排名第 1，
   10240 项累加）：把 rms 0.18 的输入误差累加成 350。

这是"量化噪声沿链路注入并被放大"，不是"某个算子坏了"。因此**不存在单张量 override 式的修复点**。

**13.16 的结论随之失效**：13.16 说"因为找到了有效的单张量修复目标，所以 `node_MatMul_333`
这个 override 工具链障碍重新变成必须解决的问题"。既然那个修复目标不成立，**这个理由也不成立**，
`node_MatMul_333` 是否需要解决，要看最终选定的修复方案是否依赖 `--quantization_overrides`。

13.8 当时的结论是"整条 override 路线收益上限 0.004%，所以 `node_MatMul_333` 那个真机编译校验失败
不必再查"。**那个结论对"给残差张量加 override"这个具体方案仍然成立**，但 13.15 找到的修复目标
不一样了：

- 13.8 否定的是"改 `add_54`~`add_138` 这些**残差张量**的编码"——它们的误差不来自自身存储，所以无效。
- 13.15 要改的是 **`mul_107` 这个中间激活张量**——它的误差**恰恰来自自身的量化**（rms 0.0703），
  而且这个误差会被下游的超大权重列放大 5000 倍。改它是**有效的**，预期 36 倍改善。

而"把某个张量标成浮点"这个动作，唯一的施加途径就是 `--quantization_overrides`——也就是
13.3～13.5 卡住的那条链路。**所以 `node_MatMul_333` 这个障碍重新变成必须解决的问题。**

好消息是 13.5 已经把它的排查基础打好了（包括那个直接反例，说明当初写的根因解释是错的，两个未排除
的可能是 (a) 结构差异 (b) dump→回灌不无损）。13.5 里建议的"先验证 round-trip 是否无损"那个几分钟
的实验现在值得做了。

---

### 13.18 【优化后的验证方向】停止继续微观定位，改测"误差是否传到了最终输出"

13.9～13.17 的微观定位已经把机制查清楚了，但独立审查指出了一个**方法论上的根本问题，我认同**：

> "使用的是中间张量最大绝对误差，不是最终图像质量指标……尚未通过提高精度或替换中间值，
>  证明最终图像会随之恢复。"

**继续往更细的算子里挖，边际收益已经很低**——因为 13.17 证明了误差是全链路注入的，再挖也找不到
单点修复。真正没被回答、而且决定后续所有投入是否值得的问题是：

> **`add_138` 那 26.9% 的中间相对误差，有多少传到了 pipeline 的最终输出（噪声预测）？**

如果最终噪声预测的相对误差只有百分之几，那么画质发闷可能根本不是量化误差造成的，后面所有修复投入
都会打空；如果是几十个百分点，才值得投入。**这个问题一次测量就能回答，而且成本远低于继续微观定位。**

**推荐的下一步实验（按性价比排序）**：

**实验 1（最高优先级）：单步端到端误差测量。** 取一组输入，分别跑：
- FP32 链路：`part1a → part1b → part2` 全 FP32（onnxruntime）→ 得到 `noise_fp32`
- 量化链路：三段依次用 `snpe-net-run` 跑已部署的 `_quantized.dlc`，**上一段的输出喂给下一段**
  → 得到 `noise_quant`

对比二者的相对误差。成本约 3 次 snpe 运行（~12 分钟）+ 1 次 FP32（~10 分钟）。
**这一个数字决定后面是否值得继续做量化优化。**

**实验 2：重复性验证（审查明确要求，成本低）。** 当前所有结论都基于 `sample_0000` 单个样本。
换 `sample_0001`/`sample_0002` 重测，确认误差是否仍集中在同一算子（`linear_33`）和同一通道（85）。
若不集中，说明当前定位只是单样本偶然。

**实验 3：真实条件验证（审查要点）。** 校准样本的 timestep 未必对应真实生成时的 8 个 timestep，
prompt 也不同。应该用真实生成过程中的中间张量（可以从 FP32 pipeline 脚本里 dump）重做测量。

**实验 4（只在实验 1 显示误差确实大时才做）：分段消融。** 分别只量化 part1a / part1b / part2，
其余保持 FP32，看哪一段对最终画质影响最大。这直接指出该往哪一段投入，比继续在 part1a 里挖更有效。

**明确不建议做的事**：
- 不要再往 `linear_33` 更上游做单算子定位（13.17 已证明没有单点修复）。
- 不要基于当前结论去解决 `node_MatMul_333` / override 工具链（13.17 已使 13.16 的理由失效）。
- 在实验 1、2 完成前，不要宣称画质问题的根因已确认，也不要开始设计修复方案。

**审查提出、我已确认成立并已在本文档修正的问题清单**：
1. 13.8 的"0.004% 是最终画质收益数学上限"——表述超出证据范围，已收回。
2. 13.8 的"8 张量合计 0.018%"——数字来源错误（实为单张量最大占比），且跨层相加无物理意义，已删除。
3. 13.12 把阶跃归因于"合并处"——当时 `add_24`/`add_45` 未测，归因无据，已撤回并重写标题。
4. 13.13 用"两个最大值相加"论证路径 A——方法不严谨，已改为逐点 argmax 论证（结论不变，更强）。
5. attention score 张量"TB 量级"——算错约 500 倍，实为 1.90 GiB/个，已更正。
6. 13.15 的修复方案——本会话自查时发现同样踩了"误差≠自身量化"的陷阱，已推翻（13.17）。

---

## 十四、2026-08-14 下午：【决定性结果】transformer 量化不是根因，第十三节整轮方向被否定

> **新会话请从本节读起。** 本节推翻了第十三节的整个工作方向（那些测量数据本身是对的，
> 但打错了目标）。配套文件：`MAINLINE.md`（主线追踪 + 假设树）、`EXECUTION_MODEL.md`
> （DLC 执行契约，只收录实测规则）、`CLAUDE.md`（项目系统约束）。

### 14.1 起因：一个从未被验证的前提

第十三节（乃至第八～十二节）的全部工作都建立在这个前提上：**"设备输出坏、FP32 输出好，
所以差异来自量化"**。

这个前提**从未被验证**。因为：
- FP32 那张好图由 **Python 脚本** `zimage_fp32_pipeline.py` 产生
- 设备那张坏图由 **C++ 实现** `PipelineZImage.hpp` 产生

**两者是不同的实现。** 这个对比同时变了"量化"和"实现"两个因素。而且已知 C++ 侧出过 2 个 bug
（`cap_pad_mask` 极性、Euler 符号），无证据排除还有第三个。

同时纠正一处**事实性错误**：文档 12.5.4/12.6 把设备输出描述为"结构对但质感发闷""能看出蓬松的
猫状轮廓"。**亲自打开图片核对后确认，这个描述不成立**——`scratch_runs/sm8750_test2.png` 是
橙色无定形色块，**没有猫、没有桌子、没有任何可辨识形状**，另有横向条带伪影。
失败类别是**语义/形状完全没生成出来**，不是"精度不够导致质感差"。这两者指向完全不同的排查方向。

### 14.2 Test B：唯一能分离"量化"与"实现"的实验

**设计**（方案定稿于 `scripts/EXP_PLAN_V2.md`，判据执行前写死）：
用**同一个 Python 脚本、同一 prompt、同一 seed(42)**，只把去噪循环里的三段换成量化 DLC，
其余全部保持 FP32：

```
文本编码(FP32) -> [8步: part1a(量化) -> part1b(量化) -> part2(量化) -> 调度器(FP32)] -> VAE(FP32) -> PNG
```

**执行前定稿的判据**：
| 结果 | 判读 |
|---|---|
| 出图 ≈ 清晰的猫 | **C1 被否定**，transformer 量化不是根因 |
| 出图 ≈ 橙色色块 | C1 被确认 |
| 介于两者之间 | C1 有贡献但不足以解释全部 |

### 14.3 结果：出图是清晰的猫 —— C1 被否定

产物：`scratch_runs/testB_hybrid_quantized_transformer.png`
（橘猫端坐木桌上，虎斑纹理、胡须、瞳孔、木纹、背景虚化的画框与绿植全部正常）

| 对比 | 平均\|像素差\| | PSNR |
|---|---|---|
| FP32 基准 vs **混合(量化 transformer)** | **9.32** | **23.89 dB** |
| FP32 基准 vs 设备真机输出 | 46.40 | 11.56 dB |

**结论：transformer 的 W8A16 量化不是设备失败的原因。**

### 14.4 一个很有价值的标定点

Test B 的 step0 实测：**量化 transformer 与 FP32 transformer 的噪声预测，L2 相对误差
高达 15.988%，成图依然完好。**

⇒ 扩散过程对这个量级的误差是**鲁棒的**。
⇒ 以后再看到"某中间张量误差 xx%"，先对照这个标定点再判断严重性。
⇒ **能把图毁成橙色色块的原因，性质上一定不同于"量化噪声"**——应当是结构性错误、
   条件信息丢失、算子语义错误之类，而不是精度不足。

### 14.5 第十三节工作的处置

13.9～13.17 的测量数据（误差逐层分布、`linear_33` 放大 41.7 倍、通道 85 的 massive activation、
numpy 复现设备计算到 0.0061）**在数值上都是正确的、可复用的**，但它们描述的是一个
**与画质问题无因果关系**的现象。

因此：
- **不要**再基于它们设计修复方案。
- **不要**再花时间解决 `node_MatMul_333` / override 工具链（13.16 的"反转"理由已失效两次）。
- 这些数据可以保留作为"该模型量化特性"的参考，仅此而已。

另需修正一处措辞（违反 `CLAUDE.md` 约束 2）：13.9～13.17 里写的"设备实测"，实际全部是
**在 SNPE CPU 参考实现（`snpe-net-run` 跑 `.dlc`）上**测得。CPU 参考与 HTP 跑 `.bin` 是否
数值等价**尚未验证**（`EXECUTION_MODEL.md` 第四节未知项①）。

### 14.6 下一步：剩余候选与优先级

假设树的当前状态（详见 `MAINLINE.md`）：

| 候选 | 状态 | 优先级 |
|---|---|---|
| A1 transformer 量化 | **已否定**（Test B） | — |
| A2 text encoder 量化 | **部分验证**：caption 张量的量化表示无损（余弦 ≥0.99995、rms 相对误差 0.16%）；但**量化 text encoder 的计算**未验证 | 中 |
| A3 VAE 量化 | **未验证**，只有结构性检查 | 中（测起来最便宜） |
| **B  C++ 实现有未发现的 bug** | **完全未排查**，已知出过 2 个同类 bug | **高** |

**建议的下一步实验**（按性价比）：
1. **扩展 Test B：把 VAE 也换成量化版**（只多 1 次 snpe 调用，VAE 模型小，约 5 分钟）。
   出图仍好 ⇒ A3 排除；出图坏 ⇒ 找到根因。
2. **A2 的完整验证**：`snpe-net-run` 的 CPU 后端跑不了 text_encoder_part1（Gather 不支持），
   需另找途径（真机、或只测 part2~4、或用 numpy 模拟 embedding 查表的量化）。
3. **B 的排查**：需要把 C++ 实现与 Python 脚本逐环节对齐（分词、prompt 模板、latents 初始化
   与 RNG、timestep 序列、调度器公式、VAE 前后处理系数）。这条最可能藏着根因，但也最费时。

**注意**：Test B 用的是 Python 侧的分词器与 prompt 模板；设备用的是 C++ 侧的。
第十节发现过分词器装错（CLIP 而非 Qwen3）的 bug，这一类问题正属于候选 B。

### 14.7 实验 A3：VAE 量化也被否定（2026-08-14）

方案定稿于 `scripts/EXP_PLAN_A3.md`，脚本 `scripts/testA3_vae.py`（量化侧走
`snpe_runner.py` 强制契约检查）。

**配置**：同一份**真实** latents（`dlc_pipeline/vae_decoder/calibration_raw/sample_0000/
latent_sample.raw`，来自真实生成过程）分别用 FP32 `vae_decoder.onnx` 与量化
`vae_decoder_quantized.dlc` 解码。

**结果**：平均像素差 **0.32**，最大 29，**PSNR 51.11 dB** —— 几乎无损。
V2 有效性检查通过：两侧解码出的都是清晰自然图像（虎斑猫端坐木桌上，毛发/胡须/瞳孔/木纹/
背景显示器与门框全部清晰）。
产物：`scratch_runs/testA3_vae_fp32.png`、`scratch_runs/testA3_vae_quant.png`。

**按执行前定稿的判据：A3 被否定。**

### 14.8 由 A1 + A3 得到的更强判断

| 候选 | 状态 |
|---|---|
| A1 transformer 量化 | **已否定**（Test B：PSNR 23.89 dB，出清晰的猫） |
| A3 VAE 量化 | **已否定**（A3：PSNR 51.11 dB，几乎无损） |
| A2 text encoder 量化 | 部分验证：caption 张量的量化**表示**无损（余弦 ≥0.99995、rms 相对误差 0.16%）；量化 text encoder 的**计算**未验证 |
| **B  C++ 实现有未发现的 bug** | **完全未排查 → 首要嫌疑** |

三个量化环节里两个已被排除，且都是**接近无损**而非勉强通过。结合 14.4 的标定点
（噪声预测 L2 相对误差 15.988% 仍出好图），可以下一个更强的判断：

> **量化这条线整体上不足以解释"图像变成橙色无定形色块"这种性质的失败。**
> 量化误差的表现是细节劣化，不会抹除语义结构。真正的原因应当是**结构性/语义性**的：
> 条件信息丢失、算子语义错误、数据布局错位、系数写错之类。

这类错误在本项目的已知发生地就是 **C++ 实现**（已抓到 2 个：`cap_pad_mask` 极性反了、
Euler step 符号反了）。

### 14.9 下一步：把 Python 脚本当作"已验证的参考实现"，逐环节对齐 C++

**关键资产**：`scripts/zimage_fp32_pipeline.py` 现在是一份**经过验证的参考实现**——
它生成完美的猫，而且把 transformer 换成量化版（Test B）、把 VAE 换成量化版（A3）之后
**仍然**生成完美的猫。因此：

> **C++ 实现（`PipelineZImage.hpp` 等）与这份 Python 参考的任何行为差异，都是 bug 候选。**

建议按这个清单逐项对齐（全部可离线做，不需要真机）：

1. **分词与 prompt 模板**：Python 用 `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
   + Qwen3 分词器 + pad 到 20；C++ 侧是否一致？（第十节抓到过分词器装错成 CLIP 的 bug）
2. **text encoder 的四段串联与 padding**：段间张量名、attention_mask 的传递方式
3. **`cap_pad_mask` 构造**：Python 是 `True=padding`，前 token_count 个为 False
4. **latents 初始化与 RNG**：Python 用 `np.random.default_rng(seed).standard_normal`；
   C++ 用什么分布/什么 RNG？同一个 seed 是否产生同一组数？
5. **timestep / sigma 序列**：Python 的 `zimage_flow_match_sigmas(8, shift=3.0)` 与
   `timestep = 1.0 - timesteps[step]/1000.0`
6. **调度器更新公式**：Python 是 `latents = latents - dt * noise`（dt 为负）
7. **VAE 前后处理系数**：`vae_latents = latents/0.3611 + 0.1159`，输出 `(px+1)*127.5`
8. **三段之间的张量传递**：part1a 的 9 个输出里哪些进 part1b、哪些直接进 part2

**注意**：Test B / A3 用的都是 Python 侧的分词器与 prompt 模板，所以上述第 1 项
**尚未被任何实验覆盖**。

### 14.10 实验 B 结果：C++ 算法层面无差异；但发现全流水线唯一的退化量化张量

方案 `scripts/EXP_PLAN_B.md`。**8 项对齐全部完成，每项都有双方代码证据。**

| # | 环节 | 判定 | 证据 |
|---|---|---|---|
| 1 | prompt 模板 | **一致** | `ZImagePrompt.hpp:10-11` vs Python `format_zimage_prompt` |
| 1 | 分词 + padding | **一致** | `TextEncoder.hpp:210-219` vs Python `process_zimage_prompt`（同样 encode→填 pad id→前 token_count 填 ids/mask=1） |
| 2 | text encoder 四段串联 | **一致** | 契约接线 part1→add_2452→part2→add_4828→part3→add_7204→part4→caption |
| 3 | `cap_pad_mask` 构造 | **一致** | `PipelineZImage.hpp:101-103`（全填1、前 token_count 填0）极性与 Python 相同 |
| 4 | latents 初始化 | **不一致但无害** | C++ `std::mt19937`（:109-111） vs Python `np.random.default_rng`。不同 RNG 只会生成不同的猫，**不足以造成语义丢失** |
| 5 | timestep 序列 | **一致** | `:144` `1.0f - timesteps(step)/1000.0f` |
| 6 | 调度器更新 | **一致** | `:174` `latents[i] -= dt*noise[i]`，`dt = next_sigma - sigma` |
| 7 | VAE 前后处理 | **一致** | `:186-187` `/0.3611 + 0.1159`；`:214` `(v+1)*127.5` |
| 8 | 图间张量传递 | **一致** | 契约里全部 `source` 为空 ⇒ 按名字连；与 Python 显式串联一致 |
| — | 常量 | **一致** | `Config.hpp`：max_length=20、steps=8、scaling=0.3611、shift=0.1159、latent_ch=16 |
| — | 量化参数 | **一致** | 30 个量化张量的 scale/offset 与 DLC 声明全部相符（`scripts/verify_contract_vs_dlc.py`） |

**唯一无法离线验证的**：手机上实际部署的 `tokenizer.json` 是哪一份（本机 `D:\ZIMAGE\tokenizer\tokenizer.json`
词表 151643 = Qwen3，正确，且 Python 参考用的就是它）。

### 14.11 【最强线索】`transformer_part2` 的 `val_104`：全流水线唯一的退化量化张量

**已证实的事实**（全部可复现）：

1. 扫描全部 8 个图共 **4,940 个张量的 encoding**，只有 **1 个**退化：
   ```
   transformer_part2 / val_104 :  min=-3.40282e+38  max=0  scale=5.192376e+33
   ```
   `-3.40282e+38` 正是 **float32 的最小值（负 FLT_MAX）**，即 ONNX 导出时用来代替 `-inf` 的掩码常量。

2. 它在图里的角色（`transformer_part2.onnx` 实查）：
   ```
   node_Where_105:  Where(unsqueeze_4, val_103, val_104) -> val_105
   val_105 随后被 Add 进 part2 全部 15 个 block 的注意力分数
   （node_Add_107 / _229 / _351 / ... / _1815）
   ```
   即标准的**加性注意力掩码**：`val_103`≈0（允许），`val_104`=-3.4e38（屏蔽）。

3. `val_105`（Where 的输出）被校准成 **[0, 0.0001]，scale 1.526e-09** —— 校准数据里掩码
   **从未激活**过，所以量化器没见过负值。
   于是 Where 要把 scale 5.19e33 的输入重量化到 scale 1.526e-09 的输出，
   比值 **3.4e42 → 溢出成 inf**。

4. HTP context binary 编译报错，**自 2026-08-05 第一次构建起就存在**（不是 SM8750 引入的）：
   ```
   dlc_pipeline/transformer_part2/03_context_binary_generator.log   (SM8550, 8/5)
   D:\ZImage_Work\sm8750_retarget_all.log                            (SM8750, 8/14)
   ERROR ] ...HTP\src\hexagon\ops\include\linearclip.h:248::ERROR:
           scale too large for requant qu16->qu16: inf; OpID: 0x6994000000a2(q::Requantize)
   ```
   这是整个 8 图构建里**唯一的算子级错误**。

5. `transformer_part2` 的输出**就是噪声预测**（喂给调度器的那个张量）。

6. CPU 参考路径（`snpe-net-run` 跑同一份 `.dlc`）产出**完美的猫**（Test B）——
   CPU 路径不经过 HTP 的这个重量化内核。

**【尚未证实的关键一环，不得当事实用】**：
HTP 的 requantize 内核在 scale 为 inf 时，究竟产出 0 还是产出垃圾值。

- 若产出垃圾：`val_105` 会被 Add 进 15 个 block 的注意力分数 ⇒ 注意力全面崩坏
  ⇒ 正好是观察到的糊状输出。
- 若产出 0：此 bug **潜伏但无实际影响**，需要继续找别的原因。

**一条削弱本假设的实测证据（必须一并记录）**：
实测 `unified_mask` 在真实运行中**全部为 1**（4128 个位置无一被屏蔽，见 Test B 各步转储）。
这是**正确行为**——padding 的 caption 槽位是靠 `cap_pad_mask` 在 part1a 里**替换成 pad 向量**
处理的，不是靠注意力掩码。所以 `Where` 恒选 `val_103` 分支，`val_104` 的值在运行时
**从未被选中**。因此本 bug 是否有实际影响，完全取决于上面那个"尚未证实"的问题。

### 14.12 建议的验证与修复路径

**修复本身很标准**：把 ONNX 里的掩码常量 `-3.4e38` 换成一个温和的负值
（如 `-1e4` 或 fp16 可表示的 `-65504`），再重新量化 `transformer_part2`。
这是量化 attention 掩码的常规做法，能同时消除退化 scale 和 inf 重量化。

**分三步验证，每步都有明确判据**：

| 步 | 动作 | 判据 | 成本 |
|---|---|---|---|
| 1 | 改 ONNX 掩码常量 → 重新量化 part2 | `val_104` 的 scale 回到正常量级；`val_105` 范围合理 | 分钟级 + 量化耗时 |
| 2 | 重新生成 context binary | **`scale too large for requant` 错误消失** | 小时级（part2 是最大的一段） |
| 3 | 装机出图 | 生成可辨识的猫 ⇒ 根因确认 | 需要真机 |

**若第 3 步仍然失败**，说明这个 bug 确实是潜伏的，应回到：
- A2（量化 text encoder 的**计算**，CPU 后端跑不了 part1 的 Gather，需真机或数值模拟）
- `EXECUTION_MODEL.md` 未知项①（HTP 与 CPU 参考的等价性）——届时应设法从设备直接
  dump 中间张量，而不是继续在离线侧推断。

### 14.13 边比值扫描：退化掩码污染了 part2 全部 15 个注意力块

14.11 只扫了【单个张量 scale 的绝对量级】。但 HTP 报的 `scale too large for requant` 关心的是
【一条边两端的 scale 比值】。补做了边扫描（`scripts/scan_requant_ratios.py`，零内存开销：
ONNX 只载图结构，encoding 从 dlc-info 转储解析）：

| 图 | 比值 > 1e4 的边 | 全图 scale 最大/中位 | HTP 编译 |
|---|---|---|---|
| **transformer_part2** | **16 条，全部涉及 `val_105`** | **9.05e+36** | **报错** |
| transformer_part1a | 9 条 | 6.12e+04 | 干净 |
| transformer_part1b | 0 条 | 1.13e+04 | 干净 |

part2 的 16 条边 = **1 个 `Where`（val_104→val_105） + 15 个 `Add`（val_105→各 block 的注意力分数）**，
精确对应 part2 的 15 个 transformer block。最极端的三条：

```
3.403e+42  Where node_Where_105   val_104(scale 5.19e+33) -> val_105(scale 1.526e-09)   <- 报 inf 的就是它
2.486e+07  Add   node_Add_1815    val_105(scale 1.526e-09) -> val_1815(scale 3.79e-02)
1.166e+06  Add   node_Add_1693    val_105(scale 1.526e-09) -> val_1693(scale 1.78e-03)
```

**两个编译干净的图，scale 离散度比 part2 低 32 个数量级。** 这个对照支持
"退化掩码是 part2 独有问题"的判断。

### 14.14 修复的局限（执行前就应写明，不要事后才发现）

已实施的修复：`transformer_part2_fixed.onnx` 的 `val_104` 由 `-3.4028235e+38` 改为 `-100.0`，
存为 `transformer_part2_maskfix.onnx`。选 -100 的理由：`exp(-100)` 在 float32 下即为 0，
语义上仍是合格的掩码常量，同时把 Where 边的比值从 3.4e42 降到约 1e6。

**这个修复只覆盖了 16 条问题边中的 1 条（那条报 inf 的 Where）。**
另外 15 条 `Add` 边的畸形来自 `val_105` **自身**的 scale 1.526e-09，
而那是**校准数据里掩码从未激活**（`unified_mask` 全为 1）导致量化器把它校准成 [0, 0.0001] 的结果。
改 `val_104` 不改变这一点。

**因此**：
- 若判据 2（编译错误消失）通过、装机也通过 ⇒ 根因确认。
- 若判据 2 通过但装机仍失败 ⇒ 下一步应针对 **`val_105` 的校准**，
  例如在校准数据里构造掩码真正激活的样本，让量化器看到真实的负值范围；
  或者直接给 `val_105` 施加量化 override（注意：override 通道有 13.x 记录的坑）。

### 14.15 下次真机运行必做的零成本诊断

`PipelineZImage.hpp:251-263` 的 `logStats()` **已经**会打印每步预测速度场的
min/max/mean/std，不需要任何代码改动。

**下次装机时务必正确抓取 native 日志**（本次检查已有的
`scratch_runs/sign_fix_test_logcat.txt` 发现里面 295 行 App 相关记录中
**没有任何 native QNN_INFO 输出**，抓取时被过滤掉了）。

对照基准（Test B 实测，CPU 参考路径）：
```
step 0: noise_std = 1.6396      step 4: noise_std = 1.3801
step 1: noise_std = 1.4411      step 5: noise_std = 1.3730
step 2: noise_std = 1.3534      step 6: noise_std = 1.3613
step 3: noise_std = 1.3854      step 7: noise_std = 1.3564
```

设备端若显著偏离这组数字 ⇒ **直接证明设备 transformer 输出被破坏**，
无需任何推断。这是目前成本最低、结论最硬的设备侧诊断。

### 14.16 判据 1 结果：通过，且与事前估算精确吻合

重新转换 + 重新量化完成（用的是从 DLC 里取出的**原始命令参数**，同一份校准数据
`input_list_raw.txt`，唯一变量是 `val_104`）：

```
转换: qairt-converter --target_backend HTP --onnx_skip_simplification
      -d unified 1,4128,3840 -d unified_mask 1,4128 -d unified_freqs 1,4128,64,2 -d adaln_input 1,256
量化: qairt-quantizer --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
      --act_quantizer_calibration min-max --param_quantizer_calibration min-max
      --act_quantizer_schema asymmetric --param_quantizer_schema asymmetric
      --use_per_channel_quantization --target_backend HTP
      --input_list <与部署版同一份>
```

| 张量 | 修复前 | 修复后 |
|---|---|---|
| `val_104` scale | 5.192376e+33 | **1.525902e-03** |
| `val_104` range | [-3.40282e+38, 0] | [-100, 0] |
| `val_105` scale | 1.526e-09 | **1.526e-09（不变）** |
| **Where 边 scale 比值** | **3.403e+42** | **9.999e+05** |
| 全图含 ±FLT_MAX 或 scale>1e6 的张量 | `['val_104']` | **`[]`** |

事前在 14.14 写下的两条估算全部应验：比值降到约 1e6、`val_105` 不受影响。
**全图已无任何退化张量。**

耗时记录（供以后估算）：
- 转换 ≈ 2 分钟，产出 10.86 GB FP32 DLC
- 量化 ≈ 3 小时（40 个校准样本 × 约 4.5 分钟/样本，part2 是最大的图），产出 2.72 GB
- 对照：部署版当初的量化日志内部时间戳为 9,570,369 ms ≈ 2.66 小时，**同量级**

### 14.17 判据 2 结果：**未通过**。修复打错了目标

重新生成 context binary（SM8750，同 `backend_ext.json`）后：

```
仍然报: linearclip.h:248::ERROR:scale too large for requant qu16->qu16: inf
        OpID: 0x6994000000a2(q::Requantize)
修复前 OpID: 0x6994000000a2   <- 完全相同
```

**同一个算子。`val_104` 的修复没有触及它。**

按 `EXP_PLAN_B` 第 5 节的失败分类：这是**方法正确、目标错误**。
判据 1 确实通过（`val_104` scale 5.19e33 → 1.53e-03，全图按原判据已无退化张量），
但**我从未验证过"报错的算子就是这个 Where"**——`OpID` 是不透明句柄，一次也没映射到
具体张量，却直接假定了因果。这正是 `EXP_PLAN_B` 自己写的纪律"**差异 ≠ 根因**"，仍然踩中。

### 14.18 【工具缺陷】此前所有基于 encoding 的扫描都不可靠，已修正

发现自己的分析正则有 bug：

```python
# 错误：张量名字符类不含点
ENC = re.compile(r"([A-Za-z_][A-Za-z_0-9]*) encoding : ...")
```

`layers.15.attention_norm1.weight_bias encoding : ...` 会从最后一段开始匹配，
被捕获成裸的 `weight_bias`，再被 `setdefault` 去重折叠。**所有带点的张量名都被压成了同一个键。**

**因此 14.11/14.13 里"全流水线 4,940 个张量中只有 1 个退化"这个说法不成立，已作废。**
（`val_104` 本身的发现仍然有效——它的名字不带点。）

修正后（`[A-Za-z_][A-Za-z_0-9.]*`）重扫：

| 图 | 张量数 | scale==0 | scale>1e6 | 含±FLT_MAX | HTP 编译 |
|---|---|---|---|---|---|
| part2（修复前） | 1713 | **90** | 1 (`val_104`) | 1 | 报错 |
| part2（修复后） | 1713 | **90** | **0** | **0** | **仍报错** |
| part1a | 1357 | **71** | 0 | 0 | 干净 |
| part1b | 809 | **44** | 0 | 0 | 干净 |

**90 个零 scale 张量全是 `*.weight_bias`**（RMSNorm 的 bias 伴随张量，bw=32，range ±1e-4）。
除以 0 会得 inf，一度看起来是新根因——**但动手前先做了对照**：part1a 有 71 个、
part1b 有 44 个，而这两个图**编译干净**。⇒ 零 scale 的 bias 张量是**良性的**，不是 inf 来源。
（这次没有重犯"发现异常就当根因"。）

**当前事实**：修复后的 part2 **已无任何按现有判据可识别的退化张量，却仍报同一个错**。
⇒ 这个 inf **不能由单个张量的 encoding 解释**。

### 14.19 【重要】迭代成本比想象的低 7 倍

从构建日志的阶段耗时读出：

```
Graph Optimizations 阶段内，第 310,647 ms（约 5.2 分钟）打印该错误
完整构建各阶段合计 2,040 秒 ≈ 34 分钟
```

**错误在约 5 分钟就会打印，不必等完整的 34 分钟。** 后续排查可以跑到错误出现即终止，
迭代成本从 34 分钟降到 5 分钟。（本项目此前一直以为 context binary 生成是"数小时"级别，
那是 part1a 的 Graph Sequencing 阶段，与本错误的出现时机无关。）

### 14.20 下一步：先确定 OpID 对应哪个算子，不要再猜

14.14 原本的预案是"判据 2 通过但装机失败 ⇒ 转向 `val_105` 校准"。
**现实与预案不同**：判据 2 本身没通过，且全图已无可识别的退化张量。
所以直接跳到 `val_105` 校准**同样是在猜**，不做。

正确的下一步是**把 `OpID: 0x6994000000a2` 映射到具体算子/张量**：

1. 提高 `qnn-context-binary-generator` 的日志级别（`--log_level verbose`）重跑，
   跑到错误出现（约 5 分钟）即终止，看是否能打印出该算子的名称或张量名。
2. 若日志给不出，用二分法：按层裁剪 part2 的 ONNX（15 个 block 逐步减半），
   看错误在哪一段消失。单次 5 分钟，二分 4 次即可定位到具体 block。
3. 拿到算子身份后，再判断它的 requant 为何是 inf，然后才谈修复。

**在拿到算子身份之前，不提出任何修复方案。**

---

### 14.21 【状态快照】新会话看这一节就够了

#### 顶层问题
设备生成的是**橙色无定形色块**（无猫无桌无任何可辨识形状，另有横向条带），
而同 prompt 同流程的 FP32 生成**完美的橘猫**。为什么？

> 两张图都要**自己打开看**，不要采信文档里的形容词（`scratch_runs/zimage_fp32_full_pipeline.png`
> 与 `scratch_runs/sm8750_test2.png`）。本项目已因转述形容词误判过一整轮方向。

#### 假设树的当前状态

| 候选 | 状态 | 依据 |
|---|---|---|
| A1 transformer 量化 | ❌ **已否定** | Test B：量化 transformer + FP32 其余 → 清晰的猫，PSNR 23.89 dB |
| A3 VAE 量化 | ❌ **已否定** | A3：再叠加量化 VAE → PSNR 51.11 dB |
| A2 text encoder 量化 | ⚠️ **部分验证** | caption 张量的量化**表示**无损（余弦 ≥0.99995）；但量化 text encoder 的**计算**未验证（CPU 后端不支持 embedding 的 Gather） |
| B C++ 实现有 bug | ⚠️ **算法层面已排除** | 8 项逐条对齐（模板/分词/接线/掩码极性/timestep/调度器/VAE系数/常量/30个量化参数）全部一致；唯一差异是 latents 的 RNG，无害 |
| **C HTP 执行 ≠ CPU 参考** | 🔴 **当前唯一未排除的主线** | 所有验证都跑在 `.dlc` + CPU 参考上；设备跑 `.bin` + HTP。已知 part2 编译期报 requant inf 错误 |

#### 唯一在手的硬线索
```
transformer_part2 编译期报错（自 2026-08-05 首次构建起就存在，是全部 8 图中唯一的算子级错误）：
  linearclip.h:248::ERROR:scale too large for requant qu16->qu16: inf
  OpID: 0x6994000000a2(q::Requantize)
```
- 已试修复 `val_104`（掩码常量 -3.4e38 → -100）：判据 1 通过（scale 5.19e33→1.53e-03，
  全图退化张量清零），但**判据 2 未通过，同一个 OpID 仍报错**。
- ⇒ 该 inf **不能由单个张量的 encoding 解释**。

#### 立刻要做的（不要跳步、不要猜）
**把 `OpID: 0x6994000000a2` 映射到具体算子/张量。在拿到身份之前不提任何修复方案。**

1. `qnn-context-binary-generator --log_level verbose` 重跑，**跑到错误出现（约 5 分钟）即终止**
2. 若日志给不出：二分法按 block 裁剪 part2 的 ONNX（15 个 block），单次 5 分钟，4 次定位

#### 三个必须记住的标定/成本事实
1. **噪声预测 L2 相对误差 15.988% 时，成图依然完好**（Test B step0 实测）。
   ⇒ 中间张量误差大 ≠ 成图坏。能把图毁成色块的，性质上不是"精度不足"。
2. **该编译错误在第 310 秒（约 5.2 分钟）打印，完整构建 34 分钟。** 迭代成本是 5 分钟不是 34 分钟。
3. **宿主侧所有 `.raw` 一律 float32**，与 DLC 内部声明无关。喂错**不报错**，只静默按比例缩小
   整张图的输出（实测 32/128 → 输出全部缩到 1/4）。必须走 `scripts/snpe_runner.py`。

#### 已产出的可复用资产
- `scripts/testB_hybrid_pipeline.py` —— 混合流水线（Python 参考 + 任意段换量化 DLC）出图，
  这是**级别 2 验证**的模板，换新模型直接改
- `scripts/snpe_runner.py` + `dlc_contracts.py` —— 带四项硬断言的 DLC 调用入口
- `scripts/scan_requant_ratios.py` —— 扫描图上每条边的 scale 比值
  （注意：正则已修正为允许张量名含点，见 14.18）
  > **14.22 更正**：该说法与文件现状不符，且该脚本有第二个更严重的盲区，
  > **不要再用它做体检**。见 14.22.6，替代品是 `scripts/map_opid.py`。
- `scripts/verify_contract_vs_dlc.py` —— 契约 scale/offset vs DLC 声明的全量比对

---

## 十四·续

### 14.22 OpID 定位完成：`0x6994000000a2` = `node_Where_105`，且它是红鲱鱼

实验方案：`scripts/EXP_PLAN_OPID_MAPPING.md`（执行前定稿，判据未事后修改）。
工具：`scripts/map_opid.py`。**全程离线，未启动 verbose 重跑，也未做二分裁剪**
（14.20 预估的 5 分钟/次构建成本没有发生）。

#### 14.22.1 【①实测数据】三份既有日志里，同一个 OpID 报的值并不相同

这三份日志在本次开工前就已存在于磁盘上，此前没有被横向对比过：

| 日志（行号） | 目标 | DLC | 报告值 |
|---|---|---|---|
| `ZImage_QNN_Evidence\dlc_pipeline\transformer_part2\03_context_binary_generator.log:8` | sm8550 | 原始 | `inf` |
| `sm8750_retarget_all.log:496` | sm8750 | 原始 | `inf` |
| `p0_experiments\maskfix\03_ctxgen.log:8` | sm8750 | **maskfix** | **`1000000.000000`** |

`val_104` 掩码修复把该 OpID 处的值从 `inf` 变成了**有限且极其规整的 1e6**。
这是本次定位的全部杠杆——一个可离线精确复算的具体数值。

#### 14.22.2 【①实测数据】算子身份

`OpID: 0x6994000000a2` = **`node_Where_105`**，DLC 图内 `Id=66`，
类型 `Eltwise_Ternary`（ONNX `Where`），是 part2 **全图唯一**的三元算子。

```
输入  unsqueeze_4  Bool_8   [1,1,1,4128]  NATIVE   （转储中无 encoding）
输入  val_103      uFxp_16  [1]           STATIC   range=[0, 1e-4]   scale=1.5259e-09
输入  val_104      uFxp_16  [1]           STATIC   range=[-100, 0]   scale=1.5259e-03  <- maskfix 改过
输出  val_105      uFxp_16  [1,1,1,4128]  NATIVE   range=[0, 1e-4]   scale=1.5259e-09
```

#### 14.22.3 【②严格推论】身份为什么可信：两点精确复算

报错里的 scale = `scale(val_104) / scale(val_105)`：

| 版本 | 复算比值 | float32 后 | 日志实际打印 |
|---|---|---|---|
| 原始 | `5.192376e33 / 1.5259e-9` = 3.4028234663852886e**42** | **inf**（超 float32 上限 3.4028235e38） | `inf` |
| maskfix | `(100/65535) / (1e-4/65535)` = **1000000.0** | 1000000.0 | `1000000.000000` |

两个独立数据点、零自由参数、逐位吻合。排他性：全图比值在 [1e5,1e7] 的还有 10 个
`FullyConnected`（权重 bw8 vs 激活 bw16，正常），但它们在两份转储里数值**完全相同**，
无法解释一个从 inf 变 1e6 的量；**只有 `node_Where_105` 变了**。

> 复算细节：`snpe-dlc-info` 只打印 12 位小数，直接拿打印值算会得到 999936。
> 用精确值算才是整 1e6。以后凡是要拿转储里的 scale 做等式检验，都要注意这个截断。

#### 14.22.4 【②严格推论】14.18/14.21 的一条结论被推翻

> **原文（14.18 / 14.21，保留）**：
> 「修复后的 part2 已无任何按现有判据可识别的退化张量，却仍报同一个错
>  ⇒ 这个 inf **不能由单个张量的 encoding 解释**。」

**推翻理由**：`val_104` 改了，但 `Where` 的**输出** `val_105` 的 encoding **没跟着改**，
仍是 `range=[0, 1e-4]`、`offset=0` 的无符号 uFxp_16——**根本无法表示任何负数**。

所以该结论**方向对但归因错**：它确实不是**单个**张量的问题，
但它完全是**张量 encoding** 的问题——是 `(val_104, val_105)` 这**一对**，
上次修复只动了其中一个。当时之所以看不见，是因为体检脚本只扫"张量自身量级"和
"输入→输出"的比值，而这条病理边是 **STATIC 输入 ↔ 输出**（见 14.22.6）。

#### 14.22.5 【①实测数据】对照组：唯一有此病理的图 = 唯一报错的图

| 图 | Where 数 | 输出区间能否容纳掩码分支 | 编译 |
|---|---|---|---|
| part1a | 1（`node_where_1`） | ✅ 掩码分支 min=-1.68，输出 min=-18.54 | 干净 |
| part1b | 0 | — | 干净 |
| part2 | 1（`node_Where_105`） | ❌ 掩码分支 min=-100，输出 min=**0** | **报错** |

#### 14.22.6 【①实测数据】`scan_requant_ratios.py` 有两个盲区，别再用它体检

1. 正则 `([A-Za-z_][A-Za-z_0-9]*) encoding` **不允许张量名含点**，
   `layers.15.xxx.weight` 这类会被静默漏掉。（14.18 称"已修正为允许含点"与文件现状不符。）
2. **更严重**：它只算「激活输入 → 输出」，且显式排除 STATIC。
   而本次这条病理边恰恰是 **STATIC 常量 ↔ 输出**；`Eltwise` 两个操作数之间的对齐
   （input↔input）它也从不计算。**这正是上一轮漏掉 `val_105` 的原因。**

替代品：`scripts/map_opid.py`，解析同一份转储，对算子内**全部张量两两取比值**，
并按 float32 复算（溢出即 inf，与 SDK 内部一致）。自检：解析出的算子数
（1342）与转储表格行数（1342）一致。

#### 14.22.7 🔴【本节最重要】差异 ≠ 根因：这条线索指向死路径

`val_105` 经 `Add` 进入 part2 **全部 15 个注意力块**
（`node_Add_107`、`node_Add_229` … `node_Add_1815`）再进 `Softmax`。
表面上完全够格解释"语义结构被抹掉"。**但实测直接否定了这条因果**：

```
testB/s0_transformer_part2/unified_mask.raw   4128 个 float32
  取值集合 = {1.0}        ==0 的个数 = 0
```

**该 prompt 下 `unified_mask` 全为 1，没有任何被掩码的位置。**
⇒ `Where` 在全部 4128 个位置都选 `val_103`(≈0) 分支，
   `val_104` 分支**一次都没被选中** ⇒ 该病理常量位于**死路径**。

而且这份 `unified_mask.raw` 正是 **Test B 出好猫那次**用的输入——
出好图的配置下，这条路径同样是死的。

> **②严格推论**：`linearclip.h:248` 这个编译期报错，
> **很可能是"橙色色块"的红鲱鱼**。它是一个**真实但当前未被触发**的潜在缺陷
> （换一个真正需要 caption padding 掩码的 prompt 就会被触发），
> 但在当前 prompt 下它不影响输出。
>
> 补充证据：该报错**从不阻断构建**——`EXIT=0`，2.9 GB 的 `.bin` 正常产出。

注：caption 侧的 padding 确实存在（`cap_pad_mask` 32 个元素里 20 个为 0），
但它在 **part1a** 由 `node_where_1` 用 `cap_pad_token`（一个真实的学习到的 pad embedding）
**替换**掉了，不依赖 part2 的注意力掩码；part1a 那个 Where 的 encoding 是健康的。

#### 14.22.8 下一步：一个廉价且能直接分叉主线的检查（尚未验证）

Python 参考侧喂给 part2 的 `unified_mask` 全为 1（已实测）。
**设备上的 C++ 实现喂进去的 `unified_mask` 是什么？目前没有任何证据。**

- 若 C++ 侧也是全 1 ⇒ 14.22.7 的证伪对设备成立，
  **这条线索彻底出局**，回主线重新找假设 C 的其他抓手。
- 若 C++ 侧喂进了含 0 的掩码 ⇒ 死路径**在设备上是活的**，
  `val_105` 的病理 encoding 会真实生效，这条线索立刻升级为头号嫌疑。

在做完这个检查之前，**不要**去改 `val_105` 的 encoding——那又会是一次
"未验证前提下的修复"（本项目已因此浪费两轮，见 CLAUDE.md 约束 2）。

### 14.23 14.22.8 的检查已完成：C++ 根本不产生 `unified_mask`

#### 14.23.1 【①实测】`unified_mask` 是 part1a 的**输出**，不是宿主喂的

契约（`scripts/dlc_contracts.json`，由 `snpe-dlc-info` 自动生成）：

```
transformer_part1a  OUT: adaln_input, add_131, add_138, latents_shape,
                         select_45, select_46, tanh_19, unified_freqs, unified_mask
transformer_part2   IN : adaln_input, unified, unified_freqs, unified_mask
```

C++ 侧 `unified_mask` **在全部一方源码里一次都没出现**（grep 全仓：只在 .md/.py/.json 里有）。
`PipelineZImage::runGraph` 把每个图的输出直接存进 `values`
（`PipelineZImage.hpp:406-408`），下一个图按名字取用。
⇒ **设备与 Python 参考的 `unified_mask` 同源，都是 part1a 算出来的。**

⇒ ②严格推论：14.22.7 对 `node_Where_105` 的证伪**同样适用于设备**，
   除非 part1a 在 HTP 上算出的 `unified_mask` 与在 CPU 参考上不同（**尚未验证**，
   但这已经属于假设 C 的一般情形，不是这条线索特有的问题）。

> **`node_Where_105` / `linearclip.h:248` 这条线索到此结案：不是根因。**
> 它是一个真实但未被触发的潜在缺陷，已写入 `QNN_CONVERSION_GUIDE.md` §3.2.1 备查。

#### 14.23.2 【重要·差点误报】`cap_pad_mask` 在 C++ 里是 32 字节，这**不是** bug

查 `unified_mask` 时顺带看到 `PipelineZImage.hpp:101-104`：

```cpp
std::array<uint8_t, 32> cap_pad_mask{};                       // 32 字节 uint8
putBytes(values, "cap_pad_mask", cap_pad_mask.data(), cap_pad_mask.size());
```

而 `scripts/dlc_contracts.json` 写着 `cap_pad_mask: host_bytes = 128`，
且 CLAUDE.md 约束 3 的实测反例用的正是 **32/128 → 输出全缩到 1/4** 这组数字。
**看起来完全就是那个已知的静默缩批 bug 出现在了设备代码里。**

**但它不是。** 三处独立守卫否定了这个结论（详见 `EXECUTION_MODEL.md` 规则 2·补）：
`ZImageQnnContract.hpp:164`、`PipelineZImage.hpp:390`、`QnnModel.hpp:89-94`。
最后一处直接与 `.bin` 里 QNN 张量自己声明的 `client_buf.dataSize` 逐字节比对，
不等就 `FAILURE`。设备跑出了图而不是抛异常 ⇒ 三处全过 ⇒
**`.bin` 里 `cap_pad_mask` 的 client buffer 本来就是 32 字节 BOOL_8。**

> **真正的教训（已写入 `EXECUTION_MODEL.md` 规则 2·补）**：
> CLAUDE.md 约束 3 的"宿主侧一律 float32"是 **`snpe-net-run` + `.dlc` 路径**的规则，
> **不是设备 QNN API + `.bin` 路径**的规则。两条路径对同一张量给出不同的宿主字节数，
> **各自都是对的**。`dlc_contracts.json` 的 `host_bytes` 只对 SNPE 工具路径有效。
>
> ⚠️ **不要把 C++ 的 32 字节"改成" 128** —— 那会让守卫 3 直接失败。

这本身也是**假设 C（HTP/.bin ≠ CPU/.dlc）的一个具体实例**：
两条路径的**宿主接口**已被证明不同。至于图**内部**数值是否等价，仍未验证。

---

## 十五、🔴 根因已确定：HTP 执行 `.bin` ≠ CPU 参考执行 `.dlc`（2026-08-15）

方案：`scripts/EXP_PLAN_HTP_VS_CPU.md`、`EXP_PLAN_HTP_VS_CPU_B1B2.md`、`EXP_PLAN_HTP_INLOOP.md`
（三份均执行前定稿，判据未事后修改）。设备：PKX110 / **SM8750**，USB adb。

### 15.1 【①实测】决定性实验：在 HTP 上重做 Test B

把 Test B 的流水线原封不动跑一遍，**只改一件事**——transformer 三段的执行后端
从「SNPE CPU 参考跑 `.dlc`」换成「设备 HTP 跑 `.bin`」。
分词、FP32 文本编码、prompt、seed(42)、调度器、VAE、步数**全部逐字节相同**。

| 配置 | 平均\|像素差\| | PSNR | 图像（**本人亲自打开 PNG 查看**） |
|---|---|---|---|
| Test B：CPU 参考跑量化 transformer | 9.32 | 23.89 dB | 清晰的橘猫 |
| **本实验：设备 HTP 跑同一模型** | **46.73** | **12.06 dB** | **橙色无定形色块，无猫无桌** |
| 设备 app 真机输出 | 46.40 | 11.56 dB | 橙色色块 |

产物：`scratch_runs/htp_inloop_transformer.png`。脚本：`scripts/htp_inloop_pipeline.py`。
24 次 `qnn-net-run` 全部成功，约 9 分钟。

> **HTP 在环几乎逐点复现了真机失败（46.73 vs 46.40；12.06 vs 11.56 dB）。**

### 15.2 【②严格推论】这一个实验同时排除了其余全部候选

本实验里除 transformer 三段外**全是 Python/onnxruntime 的 FP32 参考实现**，照样出色块：

| 候选 | 结论 | 理由 |
|---|---|---|
| A1 transformer 量化 | ❌ 排除 | **同一份量化模型**在 CPU 上出猫 |
| A2 text encoder 量化 | ❌ 排除 | 本实验用 FP32 文本编码 |
| A3 VAE 量化 | ❌ 排除 | 本实验用 FP32 VAE |
| B C++ 实现 | ❌ 排除 | 本实验**根本没用到 C++** |
| 调度器 / Euler 符号 / RNG | ❌ 排除 | 与 Test B 逐字节相同 |
| **C HTP 执行 `.bin`** | 🔴 **坐实为根因** | 唯一的变量 |

### 15.3 【①实测】逐段归因与形态

隔离条件（两侧喂相同的 CPU 参考上游输出）下各段自身的 HTP vs CPU 相对 L2：

| 段 | 张量 | 相对 L2 | 余弦 | 形态 |
|---|---|---|---|---|
| part1a | `add_138` | **2.35%** | 0.999729 | 集中离群（14~22 点达 42σ） |
| part1b | `unified` | **19.47%** | 0.990125 | 最大 180σ |
| part2 | `latents` | **43.04%** | 0.905864 | 已弥散，无 >20σ |
| 级联最终 | `latents` | **47.07%** | 0.898622 | 弥散 |

三个关键性质（均为①实测）：
1. **差异逐段自生**：级联 47.07% vs 隔离 43.04%，仅放大 1.09 倍
   ⇒ 不是"某个上游坏张量污染全链"，而是 HTP 在每段持续偏离。
2. **逐位确定性**：同输入把隔离 part2 重跑一次，两次输出 md5 完全相同
   （`348f870855ec40bee73df469a1720744`）⇒ 系统性偏差，不是运行时噪声。
3. **随图复杂度增长**：2.35% → 19.47% → 43.04%。

### 15.4 【①实测】给"15.988% 标定点"补上了另一侧

- 噪声预测相对 L2 **15.988%** ⇒ 成图**完好**（Test B）
- 噪声预测相对 L2 **47.07%** ⇒ 成图**变色块**（本轮）
- ⇒ 真正的阈值在 **16%~47%** 之间，**未测定**。
  以后再引用 15.988% 时，要同时记住 47% 这个上界。

### 15.5 同源性如何保证（防"跑错模型"）

三个 `.bin` 均先与设备 app 私有目录里的那份比 md5，逐一相同后才使用：

| 图 | md5 |
|---|---|
| part1a | `773502d86641c86ade231a0bd16ffe5e` |
| part1b | `83f16066e57a5ed6baa040b21b65b9e9` |
| part2 | `5994356c1691517a0230cfd97546edf0` |

⇒ 本轮跑的与 app 真机跑的是**同一批文件**。
（app 私有目录 `/data/data/<pkg>/` 是 `drwx------`，只 chmod 子目录没用；
改用"推主机同 md5 副本到 `/data/local/tmp`"，既绕开权限又拿到来源确定性。）

### 15.6 "为什么 HTP 会偏"——已排掉的线索与下一步（**尚未验证**）

按 MAINLINE 规则 2 的深度预算，本轮在"为什么"这一层查了两轮即回主线，未继续下钻。

**已查、结论为"疑似虚惊"的线索**：全部 8 个 sm8750 构建都打印
`<E> Unsupported HTP Arch 1 79`（sm8550 构建不打印）。查证结果：

- ①实测：宿主 `x86_64-windows-msvc` 下**没有任何 `*HtpV*Stub*`，也没有 `cdsprpc`**
  ⇒ 该告警与 `WindowsFileIO couldn't open libcdsprpc.dll` 同类，都是"PC 上没有真实 DSP"。
- ①实测：`qnn-context-binary-utility` 读出的 `.bin` 元数据**正常**——
  sm8750 版 `dspArch=79, socModel=69`；sm8550 版 `dspArch=73, socModel=0`；
  两者 `optimizationLevel` 都是 0、`vtcmSize` 都是 8（即 0 不是异常值）。
- ⇒ ②推论：架构标记正确，且设备成功加载并确定性执行。
  **但"为什么 sm8550 不报而 sm8750 报"仍未解释**，不能完全排除
  "宿主离线 prepare 库对 v79 支持不完整"这一可能。**标注为未验证。**

**下一轮首选实验（尚未规划，须按约束 4 先写方案）**：
**在设备上做 context binary 准备**（SDK 带 `bin/aarch64-android/qnn-context-binary-generator`），
用设备自己的 prepare 库把 `.dlc` 编成 `.bin`，再与宿主编的那份做同样的逐张量比对。
- 若设备编的 `.bin` 与 CPU 参考吻合 ⇒ **根因在宿主离线 prepare**，且可修。
- 若同样偏离 ⇒ HTP 执行语义本身与 SNPE CPU 参考不等价，需转向精度配置
  （`act_bitwidth` / `use_dynamic_16_bit_weights` / per-channel 等）。

### 15.7 现场状态（供下一轮直接复用）

设备 `/data/local/tmp/htpcmp/` 已就绪，**不必重新推**：
`qnn-net-run` + `libQnnHtp.so` / `libQnnHtpV79Stub.so` / `libQnnSystem.so` /
`libQnnHtpPrepare.so` / `libQnnHtpNetRunExtensions.so` / `libQnnHtpV79Skel.so`
+ `part1a.bin` / `part1b.bin` / `part2.bin`（共约 6.8 GB，/data 尚余 612 GB）。

新增脚本：
- `scripts/htp_inloop_pipeline.py` —— **级别 2.5 在环验证**模板，换新模型直接改
- `scripts/compare_htp_vs_cpu.py`、`scripts/compare_htp_b1b2.py`、`scripts/compare_images.py`
- `scripts/map_opid.py` —— 替代有盲区的 `scan_requant_ratios.py`

流程性结论已同步进 `QNN_CONVERSION_GUIDE.md`（新增**级别 2.5**、§九第 1 条改写）。

### 15.8 【①实测】part1b 的 19.47% 在结构上极不均匀，集中在 ch85 的 caption padding 槽

对 `A 隔离 part1b` 的 `unified` 做差异结构分析（脚本见 scratchpad，结论已复核）：

```
p50=0.373(0.02σ)  p99=2.15(0.14σ)  p99.9=3.05(0.20σ)
p99.99=103.2(6.83σ)  max=2733.04(180.9σ)
```

**最大的 8 个差异点全部位于通道 85、caption 的 padding 槽位（token 4116~4127）**：

| token | ch | CPU 参考 | HTP | HTP/CPU |
|---|---|---|---|---|
| 4127 (caption) | 85 | 6169.08 | 3436.04 | 0.557 |
| 4122 (caption) | 85 | 6178.04 | 3451.15 | 0.559 |
| 4117 (caption) | 85 | 6187.81 | 3461.68 | 0.559 |
| 4125 (caption) | 85 | 6180.80 | 3463.34 | 0.560 |

**HTP 稳定给出 CPU 值的约 55.8%** —— 系统性，不是噪声。

关联事实（①实测，来自 `maskfix_part2_dlcinfo.txt`）：
`unified` 的量化 encoding 是 `min=-145.277, max=6265.993, scale=0.0978, offset=-1485`。
**这些出问题的值处在量程的 98.6% 处**（6187.81 / 6265.99）。

通道维上 ch85 也是绝对首位：平均差 105.59，是中位数（0.448）的 **235.6 倍**；
ch85 在 part1a 的 `add_138` 里同样是首位偏离通道（平均差 0.964，中位数的 53.6 倍）。

⇒ ②推论（**待验证**）：HTP 在处理接近 16-bit 量程上界的值时存在系统性偏差，
   而该量程上界正是被 caption pad token 在 ch85 上的极端幅值撑起来的。

⚠️ **但"占误差能量的大头"≠"是毁图的原因"**。已定稿 `EXP_PLAN_CH85_CAUSAL.md`
   做因果分离：把极端点与弥散本底分别注入 CPU 参考的 part2，看谁真正影响下游。
   **在该实验出结果前，不得把 ch85 当成机制结论。**

> **【15.9 更正】上面这条②推论里的"~81%"是错的，实测为 64.28%。**
> 且因果实验已证明 **ch85 极端点不是毁图的原因**（见 15.9），
> 保留原文供对照，但**不要**再照它的方向去查 ch85。

### 15.9 【①实测】ch85 极端点**不是**毁图原因，弥散本底才是

方案：`scripts/EXP_PLAN_CH85_CAUSAL.md`（执行前定稿）。脚本：`scripts/ch85_causal.py`。
设计要点：part2 **四次全部用 SNPE CPU 参考跑**（走 `snpe_runner.py`，约束 3），
唯一变量是喂进去的 `unified` 里哪些元素被换成 HTP 值 —— 因此**不引入 HTP 执行这个变量**。

取 M = |U_htp − U_cpu| 最大的前 16 个元素（即那批 ch85 的 caption padding 点）。

| 变体 | 喂给 part2 的 unified | part2 输出相对基线的相对 L2 | 占 R_all |
|---|---|---|---|
| 基线 | `U_cpu` | — | — |
| 全 HTP | `U_htp` | **52.87%** | 1.000 |
| 仅极端点 | `U_cpu`，M 处换成 HTP | **12.24%** | **0.232** |
| 仅本底 | `U_htp`，M 处换回 CPU | **52.53%** | **0.994** |

**判据 1 判定：`R_base/R_all = 0.994 > 0.7` 且 `R_out/R_all = 0.232 < 0.3`
⇒ 【弥散本底是主因】，H-outlier 被否定。**

判据 2（自洽性）：`sqrt(R_out² + R_base²) = 53.94%` vs `R_all = 52.87%`，
比值 1.020 ⇒ **近似可加，part2 对该扰动基本线性**，归因可信。

判据 3（能量占比复核）：M 占 `||U_htp − U_cpu||²` 的 **64.28%**
⇒ 15.8 节写的"~81%"**被推翻**（那是估算，不是实测）。

> **这条结论反直觉，也正是必须做因果实验的理由**：
> 那 16 个点占**输入**误差能量的 64%，却只造成 23% 的**下游**损害；
> 弥散本底只占 36% 的输入能量，却造成 **99%** 的损害。
> 若跳过这一步、按"误差能量占比"直接归因，就会一头扎进 ch85 这条错路
> ——**这正是本项目反复栽跟头的模式**（差异≠根因）。

⇒ **下一步方向确定：不要再查 ch85 / 量程上界，转向 HTP 的全局数值精度。**

参考量级（①实测）：`unified` 的量化 scale = 0.0978，而 HTP vs CPU 的差异
p50 = 0.373（≈ **3.8 个量化步**）、p90 = 1.21（≈ **12.4 个量化步**）。
⇒ 弥散差异**远大于舍入量级**（舍入应 ≤ 0.5 步），是实打实的计算差异，不是量化表示差异。

### 15.10 【①实测】HTP 是"算错了"，不是"另一种合法舍入"——且 CPU 参考的精度被首次实测

方案 `scripts/EXP_PLAN_VS_FP32.md`，脚本 `scripts/vs_fp32.py`。
用 `transformer_part1b.onnx` + onnxruntime 跑 FP32 当真值，
输入与"实验 A 隔离 part1b"**逐字节相同**。

| | 与 FP32 的相对 L2 | std |
|---|---|---|
| FP32 真值 | — | 14.9664 |
| **SNPE CPU 参考跑量化模型** | **1.1757%** | 15.1097 |
| **设备 HTP 跑同一模型** | **19.2406%** | **12.9183** |

**判据 1：`E_htp / E_cpu = 16.366`，远超阈值 2.0 ⇒【甲】HTP 算错了。**

两条重要推论：
1. **量化模型本身没问题**：CPU 参考把它跑到距 FP32 仅 **1.18%**，是正常量化噪声量级。
   ⇒ 排除了"模型对量化过于敏感"这一可能（判据里的乙分支）。
2. **"CPU 参考可当真值"这个前提，此前一直是默认、从未验证**，本轮首次补上实测：
   它的实测精度就是 **1.1757%**。以后引用"CPU 参考"时可以有底气。

**判据 2（误差方向）**：`cos∠(CPU误差, HTP误差) = -0.3293`，**部分反相关**，
不是"同向放大"。

### 15.11 机制线索收敛：HTP 系统性把幅值算小

三条互相印证（前两条①实测，第三条②推论）：

1. HTP 输出 std **12.9183** vs FP32 **14.9664** —— 只有 **86.3%**
2. ch85 极端值处 HTP 给出 CPU 值的 **55.8%**（15.8 节）—— **幅值越大压得越狠**
3. 误差方向部分反相关（-0.33），不是舍入噪声的形态

⇒ ②推论（**待验证**）：形态像**重量化 scale 偏小或存在饱和/截断**。

### 15.12 【①实测】量化 encoding 条件数很差，但**不是**致命因素

`unified` 的 min-max 校准结果是 `[-145.277, 6265.993]`、scale 0.0978、bw 16，
而张量主体 std 仅 **15.11**：

| 主体范围 | 占用量化级 | 占总级数 | 有效位宽 |
|---|---|---|---|
| ±1σ | 308.9 | 0.471% | **8.27 bit** |
| ±3σ | 926.7 | 1.414% | **9.86 bit** |
| ±6σ | 1853.4 | 2.828% | 10.86 bit |

**16 bit 的量程，主体只用到约 10 bit**；其余被 0.026%（4129 个）超 6σ 的离群值占掉。

> ⚠️ **但不能据此说"encoding 是根因"**：CPU 参考在**同一 encoding** 下只差 1.18%。
> ⇒ ②推论（**待验证**）：条件数差本身不致命，但可能**吃掉了 HTP 的数值余量**。
> 若成立，`--act_quantizer_calibration percentile`（p99.99 可收紧 scale 3.6 倍）
> 是候选缓解手段。**需重新量化 + 重建，成本数小时，动手前应先找更廉价的验证。**

### 15.13 【①实测 + 官方文档】HTP 误差的定量刻画，与"权重非对称量化"这条线索

#### 15.13.1 【①实测】误差可精确分解为「系统性增益」+「随机噪声」

对 part1b 的 `unified`（FP32 当真值，1585 万元素）：

```
HTP 全局斜率 <h,f>/<f,f> = 0.854017      CPU 全局斜率 = 1.009555
  增益项  |1-k|                    = 14.598%
  剩余噪声 ||h-k·f||/||f||          = 12.534%
  两者平方和开根                    = 19.241%   <-- 与实测总误差 19.241% 完全吻合
```

⇒ **HTP 的误差 = 系统性把输出缩小 14.6% ⊕ 12.5% 的随机噪声**，两者贡献相当。

#### 15.13.2 【①实测】HTP 相当于丢了约 4 bit 精度

以 `unified` 的 encoding（range 6411.27，scale 0.0978，bw 16）折算，
理想 16-bit 舍入误差 std 应为 `s/√12 = 0.0282`：

| | 主体区(占 99.6%) 误差 std | = 几个量化步 | 等效位宽 |
|---|---|---|---|
| SNPE CPU 参考 | 0.0441 | **0.45 步** | **15.36 bit**（丢 0.64 bit，≈理想舍入） |
| 设备 HTP | 0.7148 | **7.31 步** | **11.34 bit**（丢 4.66 bit） |

⇒ **CPU 参考的误差正好落在理想舍入极限上（≤0.5 步）；HTP 比正确的 16-bit 实现多丢 4.02 bit。**

#### 15.13.3 ❌ 被官方文档当场否定的一个假设（记录以免重犯）

由"11.34 bit ≈ FP16 尾数精度"曾推出假设：**HTP 在跑 FP16 relaxed precision**。
查 SDK 文档 `docs/QAIRT-Docs/QNN/general/htp/htp_backend.html` 原文：

> `fp16_relaxed_precision` **is deprecated starting from 2.35.0 release.**
> Moving forward, there is no need to set this parameter **for fp functionality**
> and it will be determined based on SoC support.

本项目 SDK 为 **2.48.0**，且该参数**只作用于 fp（浮点）模型**，与量化图无关。
⇒ **假设作废。查文档 2 分钟，验证它要重新量化 + 重建数小时。**（已写入约束 5）

#### 15.13.4 【官方文档】权重对称量化——**必须区分两种适用范围，我第一次读时说宽了**

同一份文档里有两处相关表述，**适用范围不同，不能混为一谈**：

**(a) 通用最佳实践（适用于本项目）** —— 位于"best practices in graph design ...
from a performance and accuracy perspective"一节：

> It is **recommended** to always use symmetrical quantization of weights when quantizing
> the model to obtain best accuracy on HTP based targets. Activation data is recommended
> to be asymmetric.

**(b) 硬性要求，但范围限定在 A16W16** —— 紧接在
"List of Convolution type of operations supported for **A16W16**: Conv2d, DepthConv2d,
TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm"之后：

> **All weights/filters need to be symmetrically quantized.** For Matmul, Input A must be
> asymmetrically quantized, Input B must be symmetrically quantized.

> ⚠️ **本项目是 A16W8**（`act_bitwidth=16`、`weights_bitwidth=8`），**不是 A16W16**。
> 所以 (b) 那条"need to"**不能**直接当成对本项目的硬性要求——
> 我最初就是这样引用的，属于**过度声称，特此更正**。
> 对本项目成立的是 (a)：**强烈建议**，且理由明确写的是"to obtain best accuracy on HTP"。

**(c) 旁证** —— HTP 图配置项 `QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF`：

> Clients that have graphs where **weights are not symmetric** and have Convolution with
> short depths should set this flag **to guarantee accurate results**.

⇒ 官方明确承认：**权重非对称时，HTP 的 HMX 路径存在正确性问题**（该条针对 short-depth
Convolution，本项目是 MatMul/FullyConnected 为主，**是否同样适用未验证**）。

#### 15.13.5 【①实测】本项目的权重确实是非对称量化的

脚本 `scripts/check_weight_symmetry.py`，直接读三段 DLC 的 encoding（判据：
对称量化的 offset 应等于 `-2^(bw-1)`，8-bit 即 -128）：

| 图 | 权重张量数 | 对称 | **非对称** |
|---|---|---|---|
| transformer_part1a | 85 | 1 | **84** |
| transformer_part1b | 51 | 0 | **51** |
| transformer_part2 | 106 | 0 | **106** |
| **合计** | **242** | **1** | **241（99.6%）** |

实例：`layers.15.adaLN_modulation.0.weight_permute` offset=**-190**（应为 -128）；
`layers.15.attention.norm_q.weight` offset=**0**、range=[0, 2.031]（整段落在正半轴）。

与量化命令 `param_quantizer_schema=asymmetric` 完全一致。

#### 15.13.6 【官方文档】另一条：v79 用不了精度补偿

> `QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION_COMPENSATION`：Improves numerical precision for
> models quantized with **16-bit activations** on Hexagon **v81+** targets.
> **Requires min_arch >= 81.** 默认 false。

⇒ ①事实：官方为"16-bit 激活的数值精度不足"专门提供了补偿开关
   —— 这**从侧面证实 HTP 在 16-bit 激活下确有已知的精度短板**，与 15.13.2 实测吻合。
⇒ 但本项目设备是 **SM8750 / v79**，**够不到 v81**，**用不了这个开关**。

#### 15.13.7 当前假设与它的验证方式（**尚未验证**）

**假设 H-sym**：A16W8 + **权重非对称量化** 在 HTP（v79）上导致系统性精度损失，
表现为 15.13.1 的「增益缩小 14.6% + 噪声」，而 SNPE CPU 参考不受影响。

**支持它的证据**：官方通用建议 (a)、旁证 (c)、v79 无精度补偿 (15.13.6)、
以及 241/242 权重非对称的实测。
**它尚缺的证据**：**没有做过对照实验**。上述全部只说明"违反了建议"，
**不能证明"它就是那 19.24% 的原因"**（差异≠根因）。

**唯一能证否它的实验**：用 `--param_quantizer_schema symmetric` 重新量化 part1b，
重建 `.bin`，用完全相同的输入在 HTP 上重测 `E_htp`。
- 若 `E_htp` 从 19.24% 显著下降（比如降到 3% 以内）⇒ H-sym 成立，找到可修的根因。
- 若基本不变 ⇒ H-sym 被否定，转查其它 HTP 配置。

成本：part1b 量化（`.dlc` 1.39 GB，预估 1~2 小时）+ context binary 生成 + 重测。


---

## 15.14 三个量化开关的实测结论（2026-08-15，①实测）

本轮把「还没试过的量化器开关」逐个查文档 + 实测，结果三条全部关闭。
**共同教训：三条都能从帮助文本或文档的适用范围里事先判断出来，我却都是花了时间实测才发现。**

### 15.14.1 `--restrict_quantization_steps` —— ❌ 不适用（花了 45 分钟）

- 帮助原文写着 *"This argument is **required for 16-bit Matmul operations**."*，
  而 part1b 有 14 个 MatMul、输入输出均 `uFxp_16` ⇒ 看起来完全对症。
- 用 `-0x8000 0x7F7F` 重量化 part1b，**852 个 encoding 全部未变，0 个改变**。
- **原因**：HTP 文档里那句硬件限制的适用范围是
  *"INT16 **weight** ... when using **A16W16**"* —— 本项目是 **A16W8**（权重 8-bit），
  该限制根本不触及。
- ⚠️ **我在实测前就已经读到过那句范围限定，却没有与"我们是 A16W8"连起来。**
  这是约束 5「引用文档必须连同适用范围一起引」的**重复违反**。

### 15.14.2 `--use_per_channel_quantization` —— ❌ 对本模型一直是空转（①实测）

帮助原文：*"enable per-channel quantization for **convolution-based op** weights"*。
本模型是 DiT，**零 Convolution** ⇒ 该开关从开始到现在没起过任何作用，
FC 权重实际是 **per-tensor**（与 `map_opid.py` 转储一致：单一 scale/offset）。

> ⚠️ **同时订正一处此前文档中的错误表述**：`reviews/DIAGNOSIS_HTP_ZIMAGE.md` §2 环境表里
> 写了量化配置含 "per-channel"，**该描述不成立**，应为 per-tensor。

### 15.14.3 `--enable_float_fallback` 单独使用 —— ❌ 不允许（①实测）

```
RuntimeError: modeltools::IrQuantizer::fallbackToFloat: Use float fallback flag when
user specify encodings through --quantization_overrides or input network contains
framework trained QAT encodings.
```

⇒ 浮点回退是**「填补用户未指定 encoding 的空缺」**的机制，**不能单独用**。
注意帮助文本只说了"启用后不得提供 input list"，**没有**提到这个前置条件 ——
属于帮助文本未覆盖、只能由运行时错误暴露的约束。

---

## 15.15 🟢 `--quantization_overrides` 闸门已打通（2026-08-15，①实测，**本轮最关键进展**）

### 15.15.1 为什么它是闸门

15.14.3 之后，剩余两条修复路径**都必须经过 overrides**：

| 路径 | 为什么必须走 overrides |
|---|---|
| **选择性 FP16**（审查者建议的 `FP16 激活 + 8-bit 权重`，权重不翻倍） | `--enable_float_fallback` 拒绝单独使用（15.14.3） |
| **选择性对称权重**（消掉 offset 项，见 15.12 / 台账 #40） | 全局 `--param_quantizer_schema symmetric` 会把 RmsNorm 的 gamma 也变成 `SFxp8`，而官方 OpDef 明确 RmsNorm 在 UFxp16 激活下**不接受 SFxp8** ⇒ 图编不出来。只能逐算子指定 |

而本项目历史记录（13.x）称该路径是坏的：`version="2.0.0"` 触发 SDK 原生崩溃，
`"0.6.1"` 撞 `node_MatMul_333` 校验失败。**所以必须先单独验证闸门本身。**

### 15.15.2 实测：两个坑都可绕过

```bash
qairt-converter   --input_network transformer_part1b.onnx   --output_path   part1b_ovr.dlc   --quantization_overrides min_fp16.json   --target_backend HTP
```

`min_fp16.json`：
```json
{
  "activation_encodings": {
    "linear_99":  [{"bitwidth": 16, "dtype": "float"}],
    "linear_100": [{"bitwidth": 16, "dtype": "float"}]
  },
  "param_encodings": {},
  "version": "0.6.1"
}
```

结果：`Processed 2 quantization encodings` → `INFO_CONVERSION_SUCCESS`，**EXIT=0，约 60 秒**。

### 15.15.3 ⚠️ 关键陷阱：名字错了会**静默失效**

第一次尝试用的是 **DLC 内部名** `linear_99_fc`，结果：

```
INFO - Processed 0 quantization encodings
...
INFO - INFO_CONVERSION_SUCCESS: Conversion completed successfully
```

**转换照样成功、EXIT=0、不报任何错**，只是 overrides 完全没生效。

**必须用 ONNX 张量名**（`linear_99`），DLC 的 `_fc` 后缀是转换器加的。
查法：
```python
onnx.load(path, load_external_data=False)  # 看 node.output
# node_linear_99 (MatMul) -> 输出 'linear_99'
```

⇒ **每次使用 overrides 都必须检查 `Processed N quantization encodings` 这一行，
并确认 N == 点名张量数。** 这是唯一的防线，与约束 3 里 `snpe-net-run` 静默缩批是同一类问题。

### 15.15.4 官方文档依据（含适用范围）

`docs/QAIRT-Docs/QNN/general/converters.html` → *Quantization overrides Usecases*：

- **Float mixed precision conversion**：*"if the source model has all tensors with float32
  precision and user wants to change precision of some tensors to [float16, provide]
  **names of the tensor** with type as float16"* ⇒ **只需点名，不必给完整 encoding**
- **Quant conversion**：*"will generate a fully quantized **or mixed precision** graph
  based on the overrides provided"*
- 适用范围：`--quantization_overrides` 是 **`qairt-converter` 的参数**，不是量化器的
  ⇒ 必须从 **ONNX 重新转换**，不能在已有 `.dlc` 上改

### 15.15.5 尚未验证（③假设，不得当事实用）

- 「overrides 指定的 FP16 在 **HTP 上真的以 FP16 执行**」—— 只验证了**转换成功**，
  **没有**验证 context binary 里该张量的实际数据类型，更没有设备实测。
  下一步必须用 `qnn-context-binary-utility --json_file` 确认，方法与 §4.2.1 相同。
- 「对 FullyConnected 指定 `SFxp8` 对称权重能编过图」—— **未验证**，
  动手前必须先用 `check_opdef_support.py` 查 OpDef 矩阵。

---

## 15.16 🔴 选择性对称权重（FC-only）—— **已执行，被否定，且方向相反**（2026-08-15，①实测）

方案与全部原始数据：`scripts/EXP_PLAN_SYM_FC_OVR.md`（判据事前定稿，事后未改主判据）。
台账 **#7 就此关闭**。

### 15.16.1 做了什么

用 `--quantization_overrides` **只对 part1b 的 59 个 FullyConnected 权重**指定对称
`sFxp_8` encoding（scale 直接取自上一轮全局 symmetric 那版 DLC，**不自己推公式**），
**RmsNorm 的 44 个 gamma 不点名**。其余全部与基线逐项相同（同一份 ONNX、同一份校准
input_list、同样的 `act_bitwidth 16 / weights_bitwidth 8 / bias_bitwidth 32 / min-max`）。

**先填上了 15.13.7 遗留的空缺**：这个组合**编得过 HTP 的图** ——
`qnn-context-binary-generator` EXIT=0、1470.6 MB、12 分钟、无任何 `validateOpConfig` 报错。
（对照：全局 symmetric 在 **200 ms** 就挂在 `Failed to validate op rms_norm_node_`。）

### 15.16.2 门禁（全过，结果可解读）

- 转换后 DLC 里 `encoding : bitwidth 8` **恰好 59 行、offset 全 −128**
- 量化后：**59/59 FC 权重 = `sFxp_8` 且 offset==0**；**44/44 RmsNorm gamma 与基线逐字段相同**；
  7/7 FC bias 仍 `sFxp_32`；6/6 图输入 encoding 与基线逐字段相同
- 新旧量化 DLC md5 不同；输出 63,406,080 字节；设备 `.bin` 宿主/设备 md5 一致；
  **同输入两次运行输出 md5 相同**（逐位确定）

### 15.16.3 主结果（真值 = FP32 onnxruntime；主体口径，能量集中度 99.83%）

| 配置 | E_all | E_bulk | **cos_bulk** | slope_bulk |
|---|---|---|---|---|
| CPU 参考 · 基线 | 1.1757% | 7.0863% | 0.9975 | 1.0057 |
| CPU 参考 · 对称 FC | 1.4260% | 8.9153% | **0.9961** | 1.0007 |
| 设备 HTP · 基线 | 19.2406% | 114.798% | **0.6033** | 0.8622 |
| **设备 HTP · 对称 FC** | 21.0129% | 125.825% | **0.5507** | **0.8217** |

**主判据 `Δ = −0.0526` ≤ −0.005 ⇒ 「反向恶化」分支 ⇒ H-sym-FC 被否定。**
（无效变更本底 ±0.001，由 O=3 实测标定 ⇒ 本次是本底的 105 倍。）
次判据同样朝坏：slope 0.8622 → 0.8217，"HTP 把幅值算小"**更严重**了。
CPU 对照 0.9961 ≥ 0.99 ⇒ **不是模型被改坏**。

### 15.16.4 最有价值的副产品：**HTP 的敏感度是 CPU 的 37.6 倍**

| | Δcos_bulk |
|---|---|
| SNPE CPU 参考 | −0.0014 |
| 设备 HTP | −0.0526 |
| **倍数** | **37.6×** |

宿主侧量化了这次改动在权重表示层面的代价（`scripts/repr_cost_sym.py`，59 个全测）：
权重自身量化表示误差中位 **3.73% → 3.97%**，**scale 变粗中位 1.073×、最大 1.548×**。

⇒ ①事实：一次在 CPU 上几乎无害的权重 encoding 改动，在 HTP 上损失放大 **37.6 倍**。
⇒ ②推论：**HTP 误差对权重量化步长/表示质量高度敏感**（与 15.10 的"HTP 是算错了"同向）。
⇒ ③假设（**未验证**）：既然变粗更差，**变细应有收益** ⇒ 指向
   `--use_per_row_quantization`（台账 #47，官方帮助：*"rowwise quantization of
   **Matmul and FullyConnected** ops"*，正好命中本模型）。

### 15.16.5 本实验**不能**得出的结论（划界）

对称化同时做了两件事：① 消掉 offset 项（`z_w=0`）② scale 粗化 7.3%。**两者无法分离。**

- ✅ 否定的是"**靠对称化 FC 权重修 HTP**"这条**路径**。
- ❌ **没有**否定 #40 的机制本身（offset 项/主项 = 0.891~0.979 仍是实测事实），
  只能说消除它的收益（若有）**小于** scale 粗化的代价。
- ❌ **不得**说 "E_bulk 114.8% → 125.8% 恶化 11pp"：两者都 >100%，
  按约束 7，大误差之间比大小无意义。**结论只由余弦支撑。**
- 误差方向：`cos∠(误差_基线, 误差_对称) = 0.5895` ⇒ 部分同向但不重合，
  是被扰到了另一个同样错的位置，不是"同一误差加大"。

### 15.16.6 ⚠️ 与官方文档冲突，**记录以防后人重试**

`htp_backend.html` 通用最佳实践：*"It is **recommended** to always use symmetrical
quantization of weights ... to obtain best accuracy on HTP based targets."*

**本项目实测与之相反。** 该建议在本配置（**A16W8 + 零 Convolution 的 DiT**）下不成立，
可能是针对 Conv 类网络或 A16W16 的经验。
重试一次的完整代价：转换 1 min + 量化 54 min + 建图 12 min + 设备实测 2 min。

### 15.16.7 途中新抓到的工具陷阱（已进 `QNN_CONVERSION_GUIDE.md` 第十节）

1. **`--quantization_overrides` 会把整张图默认降成 FP16**：实测不加 `--float_bitwidth 32`
   得到 **1607 个 Float_16 / 0 个 Float_32**，加了才是全 Float_32。
   且 DLC 内嵌命令行**仍记录 `float_bitwidth=32`** —— **记录值不是行为**。
2. **`Processed N quantization encodings` 的 N ≠ 点名数**（旧写法只对 activation 成立）：
   实测 `N = 2 × 点名数 + Gemm 权重数`（59×2+7=125，2×2+1=5）。
   可靠检查是数产物：`grep -c "encoding : bitwidth 8"` 必须等于点名数。
3. `snpe-net-run` 在混用 `/` 与 `\` 的路径下报 `Failed to create output directory`（换纯反斜杠）。
4. 13.7 GB 的 `.onnx.data` 会让 `onnx.load_external_data_for_tensor` 报 "not regular file"，
   需按 `offset/length` 直接读字节（见 `scripts/repr_cost_sym.py`）。

### 15.16.8 两次判据操作化写错（都在**结果产生前**抓到，约束 8）

| 原写法 | 为什么错 | 已知样本 | 改成 |
|---|---|---|---|
| G4「全部激活 scale 漂移 >1% 的张量数 = 0」 | 权重变 ⇒ 校准值变 ⇒ 激活 encoding 本来就会变，**那是被测效应本身** | symw DLC：**42 个漂移、最大 6.98%** | 改测**图输入** encoding 逐字段相同（上游不可能被权重影响） |
| G3-2「RmsNorm gamma 的 offset != 0」 | **基线自己**就有 14 个 gamma 是 offset==0（`norm_q/norm_k` 值全为正，值域 `[0, 2.015625]`） | 基线 DLC | 改为**与基线逐字段相同**（严格更强） |

第二条与约束 8 表格第一行（"对称 ⇔ offset == −128"）是同一类错：
**把某个具体数值编码当成语义判据**。

---

## 15.17 🟢 **逐行量化（per-row）：本项目迄今唯一有效的 HTP 修复手段**（2026-08-15，①实测）

方案与原始数据：`scripts/EXP_PLAN_PERROW.md`。台账 **#47**。
**这是 #7 失败之后、由 #52 推出的方向——失败实验直接生出了成功方向，过程见 15.16.4。**

### 15.17.1 开关本身：`--use_per_row_quantization`

官方帮助原文：*"enable **rowwise quantization of Matmul and FullyConnected ops**"*。

⚠️ **与 `--use_per_channel_quantization` 是两回事**，后者帮助原文限定
*"convolution-based op weights"* ⇒ 本模型零 Convolution，**那个开关从头到尾空转**（#44）。
本项目 99.99% 的 8-bit 权重在 FullyConnected 上，**per-row 才是对症的那一个**。

**用法上比 #7 简单得多**：直接加在基线量化命令上，
**不需要 `--quantization_overrides`、不需要重新转换**，因而也绕开了 #46 的 FP16 陷阱。

### 15.17.2 官方 OpDef 的硬约束（动手前查到，省掉一轮试错）

`HtpOpDefSupplement.html` → FullyConnected → INT16 → in[1]：

> Given a 2D weight of dimensions **[m n]**, `AXIS_SCALE_OFFSET` / `BW_AXIS_SCALE_OFFSET`
> is supported **with only axis 'm'** and the values are expected to be
> **signed and symmetrically quantized**. 权重 **must have rank 2**。

⇒ **逐行量化在 HTP 上强制要求对称权重** ⇒ 它必然捆绑 15.16 刚被否定的那个改动。
⇒ **但正因为 15.16 已经把「对称 + per-tensor」单独测过**，它成了完美对照组，
   能把「scale 细化」的净效应干净分离出来。**这是 #7 那次失败最有价值的产出。**

### 15.17.3 门禁（全过）

转储原文：`axis-quant: axis: 0, num_elements: 3840 (above encoding is only for the first
(channel_0) of 3840 channels)`。

- **66 个 axis-quant 张量**（59 个 FC 权重 + 7 个 adaLN bias），**全部 `axis: 0`**
- 59 个 FC 权重全为 `sFxp_8`、**offset 全 0**（符合 OpDef 强制的 signed symmetric）
- 实例 `val_1800` 第 0 行 scale **0.015012 → 0.005844**（细 2.57 倍）
- DLC 体积 1385.5 → **1400.0 MB**（+14.5 MB = 多出的 scale 数组）
- 44/44 RmsNorm gamma、6/6 图输入 encoding 与基线**逐字段相同**
- 设备同输入两次运行 md5 相同（逐位确定）

> ⚠️ 工具口径：per-axis 转储打的是 `encoding for channel_0:` 而**不是** `encoding :`，
> 旧正则匹配不到会把 FC 权重显示成"0 个"并误报失败。`check_sym_fc.py` 已加 per-axis 识别。

### 15.17.4 主结果（真值 = FP32 onnxruntime，主体口径）

| 配置 | E_all | E_bulk | **cos_bulk** | **slope_bulk** |
|---|---|---|---|---|
| CPU 参考 · 基线 | 1.1757% | 7.0863% | 0.9975 | 1.0057 |
| 设备 HTP · 基线 | 19.2406% | 114.798% | 0.6033 | 0.8622 |
| 设备 HTP · per-tensor 对称（#7） | 21.0129% | 125.825% | 0.5507 | 0.8217 |
| **设备 HTP · per-row** | **7.8665%** | **78.771%** | **0.7986** | **1.0435** |

- **判据 1（实用性）**：`Δ = +0.1953`，弥合缺口 **49.5%** ⇒ 落在「**部分成立**」档
  （差 0.0047 没够到事前写的 +0.20 线，按约束 4 **不得**宣称"找到根因"）。
  E_all 降 2.4 倍；E_bulk **首次跌破 100%**，该张量脱离约束 7 所说的"已毁"区间。
- **判据 2（机制）**：`Δ_B = +0.2479` ⇒ **HTP 误差随权重量化步长单调变化**，
  这是可外推到新模型的结论。
- **判据 3（CPU 对照）⛔ 无法执行**：
  `error_code=202; Dequantization of axis-quant tensor is not supported for FullyConnected`
  ⇒ SNPE CPU 参考**不支持 axis-quant 的 FullyConnected**（同 #36 类工具限制）。
  **不计为通过。** 替代证据（不等价）：宿主侧权重表示误差 3.7279% → 1.0244%。

### 15.17.5 顺带解释掉 15.13.1 的「系统性增益缩小 14.6%」

| 配置 | 权重 scale 相对基线 | slope_bulk |
|---|---|---|
| #7 对称 per-tensor | ×1.073（**粗**） | 0.8217 |
| 基线 | ×1.000 | 0.8622 |
| **per-row** | **≈×0.26（细）** | **1.0435** |

三点同向且跨越 1.0 ⇒ ②推论：**该"增益缩小"由权重量化步长驱动，不是 offset 项（#40）驱动**。

### 15.17.6 在环出图（只换 part1b 的 `.bin`）——**顺带关闭 #31**

| 配置 | 平均\|像素差\| | PSNR |
|---|---|---|
| Test B：CPU 参考跑量化 transformer | 9.32 | 23.89 dB |
| HTP 在环 · 三段全基线 | 46.73 | 12.06 dB |
| **HTP 在环 · 只换 part1b** | **43.56** | **12.60 dB** |

**①实测·本人亲自打开两张 PNG**（约束 1）：
- 基线：**纯渐变、零结构**，除颜色过渡外无任何边界/纹理/物体
- per-row：**出现全局构图**——下方 1/3 有横贯的水平分界（似桌面边缘）、
  中央有带纹理的主体块、右中部有一个带硬边的深色物体、全图块状条带伪影
- **但没有猫**，主体无法辨认为任何具体物体

⇒ **#31 关闭**：单段修复**有**效果，但**不足以**出图。
   推翻我先前"任何只修一段的方案都无效"的过度声称。

> ⚠️ 两件事必须同时说（约束 2）：像素指标只动了 −3.2，仍稳稳在废图区间；
> 但图像从零结构变成有全局构图。**只看指标会误判成"没变化"**——这正是约束 1 的价值；
> **反过来也不得说"接近成功"**。
>
> ⚠️ 差点丢失证据：`htp_inloop_pipeline.py` 会**覆盖** `htp_inloop_transformer.png`，
> 而那是基线那张的唯一副本。已备份为 `htp_inloop_transformer_BASELINE_0815.png`。
> **改这个脚本时应让输出文件名带配置标识。**

### 15.17.7 由本实验导出的新线索 #53：**#9 的关闭理由已被推翻**

#9（激活 min-max 校准）当初的关闭依据是**表示误差/MSE 论证**
（"MSE 最优解就是 min-max，16-bit 表示误差仅 0.32%"）。
但 #52 + 本实验证明：**表示误差不是 HTP 误差的正确预测量**
——权重表示误差只动了 +6.5%，设备余弦掉 0.0526；表示误差降到 0.28 倍，余弦涨 0.2479。

⇒ **"MSE 最优" ≠ "HTP 最优"，#9 的关闭论证在 HTP 上不成立，应重开。**
15.12 实测 `unified` 的 16-bit 量程里主体只用到约 10 bit，
`--act_quantizer_calibration percentile`（p99.99）可收紧 scale **3.6 倍**
——与 per-row 对权重做的事同构。

---

## 15.18 【2026-08-16 夜 ~ 08-17 凌晨】part2 拆分：从"卡死"到打通

> 本节记录一段**先大幅返工、再定位到真根因**的过程。
> 返工的原因（用错源文件、用错对照）本身比结论更值得后来者读。

### 15.18.1 出发点：part2 是当前的支配项

用标定过的尺子（step-0 噪声预测相对 FP32；15.99% ⇒ 成图完好、47.07% ⇒ 色块）：

| 配置 | E_all |
|---|---|
| 目标 | **15.99%** |
| 三段全基线 | 47.07% |
| part1a+part1b 用 per-row、part2 基线 | **43.41%** |
| 基线 part2 **单段隔离** | 41.24% |

⇒ 上游两段全部改善只贡献约 2 pp，**剩下约 41 pp 是 part2 自己的**。
而 part2 的 per-row 版 context（3535 MB）**装不进 unsigned PD**（上限实测在 3506~3535 之间）。

### 15.18.2 三次返工，逐个记录

**返工 1：用错源 ONNX。**
生产 part2 DLC 的 `--input_network` 是 `transformer_part2_fixed.onnx`
（`dlc_pipeline/transformer_part2/01_qairt_converter.log` 明确记录），
我却按文件名朴素程度选了 `transformer_part2.onnx`，两者**差 108 个节点**。
**验证材料当时就在手上**——我同一小时读过那个日志，只提取了 `input_dim`，
**没看同一行的 `--input_network`**。
> 附带症状：切分后遇到 `latents_shape` 报错，我手工把它固化成常量——
> 而 `_fixed` **早就做过这件事**。**当一个问题被"重新发现"，应立刻怀疑用了旧版本。**

**返工 2：用了一个没跑过推理的对照。**
我用"完整 part2 内存封顶 52.44 GB vs 子图无界增长"论证子图异常。
事后核查：那次探针的 `QnnGraph execute start` 计数为 **0**（历史成功那次是 39）。
**52.44 GB 是加载/建图内存，134 GB 是校准推理内存——两个不同阶段的数字被我直接对比了。**

**返工 3：并发跑了两个量化器。**
测"无 per-row 对照"时，前一个探针（720 s 超时）还没退出就启动了新的，
采样器取"最大的 python 进程"，读到的是旧进程的平台值。**该次测量作废。**
`preflight.py` 就是拦这个的，**而我那次没跑它**。

### 15.18.3 真根因：切分前没做死代码消除（DCE）

`transformer_part2_fixed.onnx` 有 **84 个死节点**
（IsNaN 15 / Where 15 / Gather 10 / Reshape 14 / Cast 6 / …），
它们消费 **30 个 `[1,30,4128,4128]` 的注意力分数矩阵**（fp32 单个 **1.9044 GiB**，合计 57.1 GiB）。

- **转换器对完整图会自己 DCE** ⇒ 生产 part2 DLC 里 `IsNaN` 计数为 **0**，所以完整图量化正常
- **`onnx.utils.Extractor` 切分时保留死节点** ⇒ 它们"复活"成子图的必需部分

| | 切割集 | 量化行为 |
|---|---|---|
| 不 DCE 直接切 | **14 个张量、16.4 GB**（8 个巨型矩阵被死节点拽过切口） | 线性涨到 **134 GB** 后 OOM |
| **先 DCE 再切** | **6 个张量、65.6 MB** | 1 样本：**峰值 32.36 GB 封顶、5.5 分钟完成** |

**①实测指纹**：量化内存增长步长恰好 **1.90~1.91 GB = 单个该张量 fp32 大小**。
⚠️ 已查：**part1a / part1b 死节点为 0**，此问题目前仅 part2 有。

### 15.18.4 正确流程与结果（全部实测）

```
transformer_part2_fixed.onnx
  → DCE（1724 → 1640 活节点，去掉 84 个）
  → 反向可达性切分（切口 6 个张量 65.6 MB）
  → 数值等价验证：**相对 L2 = 0.00000000%、最大绝对差 = 0、余弦 = 1.0、逐位相同**
  → 转换：part2a 4.95 GB / part2b 5.16 GB
  → 1 样本量化内存探针：峰值 32.36 GB 封顶 ✅
  → 40 样本正式量化（per-row）：part2a 71 min → 1.25 GB；part2b 65 min → 1.31 GB，**零 OOM**
     （part2b 的校准数据用 _fixed 版 part2a 重新生成，23 分钟）
```

自检：切分后 844 + 796 = **1640**，与 DCE 后活节点数精确一致；两子图残留 `IsNaN` 均为 **0**。

### 15.18.5 顺带关闭的一条：external weights / spill-fill buffer

自建最小 probe（`scripts/extbuf_probe/`，NDK 交叉编译，仅加载不推理）四档实测：

| 档 | createFromBinary | memRegister | contextFinalize |
|---|---|---|---|
| 普通 | ❌ 0x3ea | — | — |
| DEFER + spill-fill | ✅ | ✅ 260.0 MB | ❌ 0x3ea |
| DEFER + weights | ✅ | ✅ 2730.5 MB | ❌ 0x3ea |
| DEFER + 两者 | ✅ | ✅ 两个都成功 | ❌ 0x3ea |

**FARF 的 `context size estimate 3652345600` 外置前后完全相同** ⇒ **不减少 PD 估算，该路关闭。**

①首次测得：`WEIGHTS_BUFFER_SIZE` = **2730.5 MB**、`MAX_SPILLFILL_BUFFER_SIZE` = **260.0 MB**。
②推论：真正的 PD 检查在 `contextFinalize`，`DEFER_GRAPH_INIT` **只推迟不豁免**。

三个工具级发现：
1. `rpcmem_alloc` 的 size 是 **`int`**，分配 2730 MB 直接失败（32 位溢出）⇒ 必须用 **`rpcmem_alloc2`**
2. `qnn-net-run` 2.48 **无法使用** `context_configs/spill_fill_buffer|weights_buffer`
   （schema 要 integer、解析器要 string）。**设备端五个关键库与 SDK md5 逐一核对一致，排除版本错配**
3. `qairt-net-run` 的 schema 反而要 string（配置能过），但其 HTP 后端在平台信息解析就失败

### 15.18.6 方法论：本轮固化进 CLAUDE.md 的三条

- **约束 9**：定期自审（触发条件 + 7 条清单），**不许等被质疑才回头看**
- **约束 9·补**：长任务必须挂"真完成"监听；判断死活只看**进程表 + 时间戳**，不看日志内容
- **约束 4·补 G0-cost 代价门**：改 encoding 前先算纯表示误差，
  **超过当前端到端误差就直接否决**（用它复算 #53，10 秒就能拦下那次 80 分钟的无效实验）
- 机械化门禁 `scripts/preflight.py`：过不了返回 **exit 1**，不由口头判断

### 15.18.7 🟢 最终结果：卡点突破，且 per-row 对 part2 的收益首次被实测

**G1 尺寸门（宿主侧）**：

| 配置 | blob | spill | opData | 合计 |
|---|---|---|---|---|
| part2 baseline（红线参照·可加载） | 2928.7 | 242.3 | 334.9 | **3505.9** |
| part2 per-row 未拆（❌ 装不下） | 2934.3 | 258.7 | 342.5 | 3535.4 |
| **part2a per-row** | 1445.1 | 258.1 | 183.9 | **1887.0** |
| **part2b per-row** | 1495.2 | 255.2 | 176.3 | **1926.7** |

每半段约 1.9 GB，**距红线余量 1.6 GB** —— 不但装得下，还为将来的改良留出空间。

**设备加载（①实测）**：两个 `.bin` 用普通 `createFromBinary` **全部成功**，无任何 PD 报错。

**精度（part2 单段隔离，输入与基线隔离实验逐字节相同，真值 = FP32 原点）**：

| 配置 | E_all | 主体余弦 | slope |
|---|---|---|---|
| 基线（per-tensor） | **41.2448%** | 0.9089 | 0.9079 |
| **per-row（拆分后，本轮）** | **28.5156%** | **0.9557** | 0.9287 |
| CPU 参考（可达上界） | 15.9882% | 0.9865 | 0.9984 |

⇒ **per-row 把 part2 的隔离误差从 41.24% 降到 28.52%，弥合到 CPU 参考缺口的 50.4%。**

> ⚠️ **划界**：该测量含**两个变化**——per-row 量化 **+ 拆分本身**（多一道切口量化边界）。
> 切口边界只会**变差**，所以 per-row 的真实收益**至少有这么大**，混淆方向保守。
> 若要精确分离，需再做一次"拆分 + per-tensor"的对照（尚未做）。

### 15.18.8 下一步（新线程从这里开始）

**唯一还没回答的顶层问题：三段（实为四段）全 per-row 之后，成图能到什么程度？**

已知的级联点：
- 三段全基线：噪声预测 **47.07%**
- part1a+part1b per-row、part2 基线：**43.41%**
- 现在 part2 也能上 per-row 了（其隔离误差 41.24% → 28.52%）

**待做**：把 `scripts/htp_inloop_pipeline.py` 从三段扩到**四段**
（part1a → part1b → part2a → part2b），跑在环出图，测：
1. step-0 噪声预测 E_all（对照 15.99% 的成图门槛）
2. 成图（**必须自己打开 PNG 看**，约束 1）

设备上已就位：`part2a_fixed.bin` / `part2b_fixed.bin`，两者均已验证可加载。
切口传递 6 个张量共 65.6 MB（`add_92` / `select` / `select_1` /
`split_7_split_2` / `split_7_split_3` / `val_105`）。

## 15.19 【2026-08-17 晚】设备出猫 + app 四段交付上线

### 15.19.1 🎉 主结果：HTP 第一次生成出清晰的猫

`EXP_PLAN_INLOOP4`，判据事前定稿。**单变量**：part2 基线 → part2a + part2b（per-row 拆分），
其余（分词、FP32 文本编码、prompt、seed 42、8 步调度器、FP32 VAE、part1a/part1b 的 `.bin`）不变。

| 配置 | step-0 `E_all` | 对 FP32 像素差 | PSNR | 图像（本人亲自打开看） |
|---|---|---|---|---|
| 三段（part2 基线） | 43.4094% | 41.37 | 12.46 dB | 通体马赛克方块，**无任何可辨认物体** |
| **四段（part2a/2b per-row）** | **44.7978%** | **44.52** | **11.60 dB** | 🎉 **清晰橘猫**：头/耳/绿眼/粉鼻/胡须/虎斑纹/木桌木纹 |

四道 V 门全过；V3 step-0 输入 md5 与 43.41% 那次逐一相同。
产物：`scratch_runs/htp_inloop_transformer_perrow4seg.png`。

**代码改动已排除**：当日改过 `sh()`/`require_device()`。用同一份代码重跑三段对照
（`htp_inloop_transformer_ctl3seg.png`）：`E_all` = 43.4094% 与 08-16 完全一致、
PNG 与 08-16 那张**逐位相同（平均像素差 0.0000）** ⇒ 差异确实来自 per-row 拆分段。

**仍未解决**：HTP 图背景空白（FP32 有完整房间场景），毛发质感粗、虎斑纹弱
⇒ **"能不能出图"已解决，"画质对齐 FP32"未解决**。

### 15.19.2 🔴 三把标量尺子全部反相关（本轮最重要的方法论结论）

上表三个指标**一致地指向"四段更差"**，而图从认不出任何物体变成清晰的猫。
⇒ 判据 3（事前锁定）触发：**标定尺在当前区间失去分辨力**。

机制：HTP 出的是**另一个合理场景**（背景空白），逐像素/L2 惩罚"画对了但内容不同"；
而色块的平滑渐变反而更接近照片均值。
原标定（15.99% 完好 / 47.07% 色块）**只有两个点且来自不同流水线**
（Test B 全 CPU 参考 vs 全 HTP 基线）——metric 与 configuration 一起变了，**共变被当成因果**。

⇒ 已登记 #69/#70/#56b；已写入 `QNN_CONVERSION_GUIDE.md` 第十七节。
⇒ **暂停一切基于 E_all / 像素差 / PSNR 的取舍决策**，直到建立结构/感知判别方式。

### 15.19.3 #30 单 MatMul 探针：H-floor 与 H-acc 均被否定

`E_htp/E_fxp` = **1.00 / 1.01 / 1.00**（K=64/1024/3840）⇒ **HTP 单算子上是正确的**，
**无硬性精度地板** ⇒ 🟢 量化配置这条路没有天花板。
⚠️ 探针输入是**良态**的，不得外推到真实模型（`unified` 99.83% 能量集中在前 1%）。

### 15.19.4 🔴 `models` vs `graphs` 之争：**确定用 `models`**（困扰本项目已久，本轮读死）

**从 C++ 源码定论**：
1. `loadGraph(name)` → `contract_.graph(name)`，图名不在契约里即抛 `Unknown Z-Image graph`
2. `PipelineZImage` 请求 8 个名字：`text_encoder_part1..4` / `transformer_part1a` / `part1b` / `part2` / `vae_decoder`
3. `ZImageQnnContract::load()` 在**没有 `models` 键**时走 legacy 分支，
   而该分支只登记 `kRequiredGraphs[4] = {text_encoder, transformer_part1, transformer_part2, vae_decoder}`

⇒ **走 `graphs` 的契约，第一次 `loadGraph("text_encoder_part1")` 必然抛异常——它从来跑不起来。**

**设备现场佐证**（app 私有目录 `files/models/ZIMAGE/`）：
- `final_qnn_contract.json` 用 **`models`** ← 正在跑的
- `final_qnn_contract.imported-graphs.json` 是 `graphs` 版遗留（带 BOM，未被使用）

**顺带修掉一个潜伏 bug**：`Model.kt` 原来写
`json.optJSONArray("models") ?: json.optJSONArray("graphs")`
⇒ `graphs` 版 bundle **能通过 Kotlin 校验**、却在原生流水线里崩
（"能进模型列表，一生成就失败"）。已改为**只接受 `models`**，并在拒绝时明确提示 legacy schema。

### 15.19.5 ⚠️ 一个由"看错目录"导致的错误结论（已纠正）

我一度断言"app 装的是 SM8550 基线 bin"，依据是 `/storage/.../Download/ZIMAGE_RECOVERY_STAGING/`。
**错了**：那是 staging 副本（且 schema 是 `graphs`）。
app 私有目录里实际装的是 **SM8750**（字节数与宿主基线产物逐一对上：
part1a 2367265832 / part1b 1471522864 / part2 2928735488）。
⇒ **判断"设备上装的是什么"，必须看 app 私有目录，不能看 Download 里的副本**
（`run-as <pkg> ls files/models/...`，debuggable 构建可用）。

### 15.19.6 app 四段交付（已上线，按正式标准执行）

**代码改动三处**：
1. `ZImageQnnContract.hpp`：`loadFinalDelivery` 按 manifest **自动识别** 3 段/4 段布局
   （检测 `transformer_part2a` 是否存在），新增 `hasSplitPart2()` / `finalTransformerGraph()`，
   新增 `validatePart2Split()`（part2a 每个输出都必须被 part2b 消费）
2. `PipelineZImage.hpp`：4 段布局时依次跑 part2a → part2b。
   **切口 6 个张量无需显式搬运**——`runGraph` 按张量名从 `values` 解析，
   `requantize` 自动处理两侧 encoding 不一致
3. `Model.kt`：`requiredGraphs` 按布局二选一；并只接受 `models` schema（见 15.19.4）

**交付脚本**：`scripts/build_app_bundle.py`
- 基准 = **app 私有目录里正在跑的那份契约**（不是 Download 副本）
- transformer 四段的 I/O 规格从 `.bin` 元数据 dump 取、sha256 从文件算，**禁止手写**
- **只动 transformer**（text_encoder ×4 / vae 条目与文件逐字保留）⇒ 单变量
- 自检：图集合、切口闭合、part2a/part2b 每个输入都有来源

**部署实测**：
- APK：`assembleBasicDebug`（设备上原版本**本身就是 DEBUGGABLE**，非降标准）。
  `adb install -r` **Success** ⇒ 签名一致、就地升级、**11 GB 模型数据完好**。
  ⚠️ 先用 `install -r` 探签名是**非破坏性**的：签名不符只会失败，不会清数据。
- 模型：4 个文件**在设备本地 `cat | run-as cat >` 复制**（6.32 GiB，**不经 USB**），
  源就是 `/data/local/tmp/htpcmp` 里出猫那次用的文件本身
- **完整性**：设备侧 `sha256sum` 与契约中宿主算出的值**四个全部一致**
- **可回滚**：旧契约存为 `final_qnn_contract.3seg-backup.json`，
  旧的 `transformer_part1a/1b/part2_ctx_sm8750.SM8750.bin` **均未删除**

**⚠️ 尚未验证（交付时必须一并说明）**：
出猫实验用的是**宿主 FP32 文本编码 + FP32 VAE**，而 app 用**量化版**；
且 app 是 **C++ 实现**，与 Python 流水线是两套实现（已知出过 2 个 bug）。
⇒ **app 端到端是否同样出猫，只能装机实测，必须自己打开图看**（约束 1）。

### 15.19.7 🔴 app 四段交付事故复盘（2026-08-17 夜，代价约 2 小时）

**结果先说**：四段交付**最终成功**，真实 app 端到端生成出正常图像
（`scratch_runs/app_4seg_first.png`，本人亲自打开看：结构完整的写实人像，
发丝/织物/景深/手部结构均成立）。对比同一台设备 08-14 那版的橙色色块是质变。
实测 `transformer_part1a` 4935 ms、`transformer_part1b` 1918 ms；
契约校验（约 11 GB 全量 sha256）约 119 s，仅后端启动时一次。

#### 事故 1：`main.cpp` 里一份过期的硬编码图名清单

**现象**：app 弹「后端启动失败，您的设备可能不受支持」。
**根因**：`main.cpp:320` 在契约加载成功之后，又拿**写死的 8 个图名**逐个
`contract.graph(name)` 查文件。part2 拆分后契约里已无 `transformer_part2`
⇒ 抛 `Unknown Z-Image graph` ⇒ 后端 exit 1 ⇒ 健康检查失败 ⇒ UI 报"设备不受支持"。

🔴 **我为什么漏掉**：改动前我 grep 过 `transformer_part2`，**结果里就有 `main.cpp:320`**，
但我只看了自己**预期**会改的两个文件（契约层、流水线层），没按 grep 的实际结果逐个追。
与约束 5 的"适用范围我早已读到却没连起来"同类。

**根治**：不补第三处清单，而是**让清单不存在**——新增 `ZImageQnnContract::graphNames()`，
`main.cpp` 遍历契约自身的图集合。以后布局再变（如 part1 拆分）此处不需改也不会错。

#### 事故 2：用 `run-as` 验证 app 行为 —— 整整一小时的无效排查

修好事故 1 后生成仍失败（`Failed init QNN context: text_encoder_part1`）。
我用 `run-as` 手动跑后端做了大量对照：换三段契约、重启设备、
甚至**把我全部 4 处改动还原重编基线 APK** —— 全部同样失败，
据此我告诉用户"**不是我的代码**"。

🔴 **该结论的证据全部无效**：`run-as` 的 SELinux 域是 **`u:r:runas_app`**，
真实 app 是 **`u:r:untrusted_app`**，DSP 访问权限不同。
真实 app 里 `QnnDevice_create` / `QnnContext_createFromBinary` 全部正常。
⇒ **`run-as` 不能用来验证 app 的 HTP 行为，用它做的"app 能不能跑"结论一律不成立。**

#### 事故 3：覆盖 APK 前没备份 —— 摧毁了归因能力

模型和契约都留了回滚路径，唯独 APK 没有。于是在排查中**无法回到已知可用状态**，
也就无法回答"是不是我弄坏的"。这是本轮最严重的操作失误。
（后来靠"精确还原 4 处改动重编"补出了基线，但那已是被动补救。）

#### 事故 4：交付前从未验证基线可用

我把"用户说 app 之前出过橙色色块"当成既定事实——那是 **08-14、且是另一个 APK**。
自查 `files/history/ZIMAGE/` 得知：**app 最后一次成功生成是 2026-08-14 07:59，
此后三天无人跑过**。⇒ "改之前是好的"这个前提**从未被验证**（约束 1）。

#### 两次"报错说谎"

| 表面报错 | 真实原因 |
|---|---|
| `qnn-net-run failed [part1a/s0]` | USB 掉线（`sh()` 吞掉了 adb 的 stderr 与返回码） |
| 「您的设备可能不受支持」 | 一份过期的硬编码图名清单 |

⇒ **报错指向错误的对象，比报错本身贵得多。** 两次都直接导致在错误方向上排查。

#### 完整变更台账

`D:\ZImage_Work\p0_experiments\app_change_log\CHANGELOG.md`
（13 条变更 + 时间 + 产物 + 结果 + 已排除假设 + 证据作废标注）。
两个 APK 均已存档：`BASELINE_pre-my-changes.apk` / `FOURSEG.apk`。

## 15.20 【2026-08-18 凌晨】app vs PC 在环对照：C++ 实现的第 3 个 bug 假设被关闭

### 15.20.1 第一次对照是无效的（我自己犯的错，先记）

拿 app（seed 42）与 PC 在环（seed 42）直接比像素，得到 127.99 / 117.01 / 44.52 三个数，
**全部无效**：

- PC 脚本：`np.random.default_rng(42).standard_normal()`（**PCG64**）
- C++ app：`std::mt19937 generator(42)` + `std::normal_distribution<float>`（**MT19937 + 极坐标法**）

⇒ **同一个 seed，两套 RNG 给出完全不同的初始噪声**，两张图是同一模型的**不同采样**。
我设计"同 seed 对照"时**没有先核对两侧的随机数生成方式**——又一次"判据操作化未验证"（约束 8）。

### 15.20.2 正确做法：让设备生成 latents，而不是在 Python 里复现

先尝试在 Python 里复现 libc++ 的 `normal_distribution`（Marsaglia 极坐标 + 缓存 + float32
`generate_canonical`）。**MT19937 引擎通过官方测试向量**（第 10000 个输出 = 4123659995），
但与真机对拍：**前 4 个值逐位相同，第 5 个起差 ~1e-7**
——libc++ 全程用 float32 算 `sqrt(-2*log(s)/s)`，我用了 double 再收窄。

事后量化：Python 复现 **73.01% 逐位相同**、最大绝对差 **4.873e-6**、相对 L2 **0.000007%**。
**实用上无差别，但不逐位等价** ⇒ 若直接采用，后续所有"逐位相同"的表述都会是假的。

⇒ 改为 **NDK 交叉编译一个 20 行探针**（`p0_experiments/rngprobe/gen_latents.cpp`，
代码与 `PipelineZImage.hpp` 初始化逐字对应），在设备上生成 latents 写成 raw，
再喂给在环脚本（新增 `INLOOP_LATENTS` 环境变量）。
**等价性由构造保证，不靠复现。**

### 15.20.3 结果：两条实现路径收敛到同一个采样

| 对比 | 平均像素差 | PSNR | 有效性 |
|---|---|---|---|
| **app(C++/量化TE+VAE) vs PC(Py/FP32 TE+VAE)** | **34.26** | **14.74 dB** | ✅ 输入逐字节相同 |
| app vs `zimage_fp32_full_pipeline.png` | 117.01 | 5.49 dB | ❌ **无效**：该参考图用 numpy latents 生成，是不同采样 |
| PC 在环 vs 同上 | 126.76 | 4.59 dB | ❌ **无效**，同上 |

**①实测·本人亲自打开两张图**：同一只橘猫、同姿态、同构图、同木桌、同深色背景。

**②结论**：
1. **C++ 实现没有第 3 个 bug**（除已知的 `cap_pad_mask` 极性、Euler 符号）——该怀疑关闭。
2. **量化版 text encoder / VAE 没有破坏条件信息。**
3. ⇒ **剩余画质差距全部来自 transformer 的 HTP 量化执行。**

**③划界（不得超出）**：34.26 的像素差是**真实差异**，集中在背景细节
（app 那张能看出橱柜/摆件层次，PC 那张更暗更平）。
**本实验无法区分该差异来自"C++ vs Python 实现"还是"量化 TE/VAE vs FP32 TE/VAE"**
——要拆开需再加一臂（PC 在环 + 量化 TE/VAE）。**不得归因给任何一方。**

### 15.20.4 遗留缺口

**目前没有可比的 FP32 参考。** 现有 `zimage_fp32_full_pipeline.png` 用的是 numpy latents。
要量出真实画质差距，需用**同一份 C++ latents** 重跑一次 FP32 全流程
（纯宿主，约 10~15 分钟）。这是后续第一步。

## 15.21 【2026-08-19】🔴 画质缺陷定位到 **VAE 导出**，与 HTP/量化无关

### 15.21.1 结论先行

**"毛发蜡质、边缘发硬"这个用户从一开始就指出的画质缺陷，100% 来自我们的 VAE ONNX 导出，
与 HTP、与量化、与 transformer 全都无关。**

在 FP32、零量化的条件下即可复现。此前所有针对该缺陷的量化优化，**靶子从一开始就是错的**。

### 15.21.2 证据链（每步单变量）

**第一步：官方 PyTorch 全流程 vs 我们的 ONNX FP32 全流程**
（同噪声 `latents_cxx_seed42.raw`、同 caption、同调度器；唯一差异 = 导出）

| step | 官方 noise_std | 我们 ONNX | 相对差 |
|---|---|---|---|
| 0 | 1.5485 | 1.5522 | −0.238% |
| 3 | 1.3317 | 1.3318 | −0.008% |
| 7 | 1.2858 | 1.2836 | +0.171% |

⇒ **逐步数值几乎一致（±0.24%）**，但成图：平均像素差 **13.627**、PSNR 23.38 dB、
**高频能量（平均梯度幅值）高出 +30.7%**。
产物：`scratch_runs/official_pytorch_22tok.png` vs `fp32_22tok.png`。
官方耗时 9874 s（每步约 1200 s，CPU）。

**第二步：VAE 双臂对照**（同一份 final latents，唯一变量 = VAE 实现）

| | 全流程对照 | **仅 VAE 对照** |
|---|---|---|
| 平均像素差 | 13.627 | **13.326** |
| PSNR | 23.38 dB | **23.51 dB** |
| 高频能量差 | **+30.7%** | **+30.5%** |

⇒ ②**两组数字几乎重合 ⇒ transformer 8 步累积的差异对成图贡献可忽略，全部差异来自 VAE。**
脚本 `scripts/vae_ab.py`，产物 `scratch_runs/vae_ab_pytorch.png` / `vae_ab_onnx.png`。

### 15.21.3 静态检查：图与权重完全等价，缺陷是**运行时数值**

| 检查项 | 官方 decoder | 我们的 ONNX | 结论 |
|---|---|---|---|
| Conv2d | 35 | Conv 35 | ✅ |
| GroupNorm | 30 | InstanceNormalization 30 | ✅ 标准分解 `Reshape→InstanceNorm→Reshape→Mul→Add` |
| SiLU 调用 | 14 block×2+1 = 29 | Sigmoid 29 | ✅ |
| Upsample2D | 3 | Resize `nearest`/`asymmetric`/`floor` | ✅ |
| Attention | 1 | Softmax 1 + MatMul 6 | ✅ |
| GroupNorm eps | 1e-6 | 1e-6 | ✅ |
| 权重 | — | 73 个实权重逐位吻合；4 个注意力权重**转置后**相对差 0.000000 | ✅ |

⇒ ②**结构与权重完全等价 ⇒ 30.5% 的高频差异只能来自运行时数值行为。**

**③未验证假设（当前最可疑）**：GroupNorm 被分解为 `InstanceNormalization` 后，
在 1024×1024 分辨率下每组空间元素数极大；若 ORT 的 InstanceNormalization 用
`E[x²]−E[x]²` 计算方差，会产生明显数值损失，表现为对比度/锐度的系统性偏移。
**需逐层发散定位证实，不得当结论用。**

### 15.21.4 顺带确立的两条

1. **transformer 导出是忠实的**——逐步 noise_std 与官方差 ±0.24%，此前从未验证过。
2. **文本编码器导出也是忠实的**——官方 vs 我们的 ONNX 前 20 槽相对 L2 **0.8628%**、
   余弦 **0.999990**；`hidden_states[-2]` 口径正确（ONNX 35 层 / config 36 层）；
   chat 模板逐字一致；`Qwen3Model` 与 `Qwen3ForCausalLM` 的 hidden_states **相对 L2 0.0000%**。

### 15.21.5 已降级的一条：20-token 截断

我们的 text_encoder 被硬编码固定在 **20 token**（ONNX 内部 Reshape 常量 `{1,20,-1,128}`），
官方默认 `max_sequence_length=512`；本项目 prompt 完整模板 22 token ⇒ 末尾
`<|im_start|>assistant\n` 被截掉。

**但单变量实测（22 vs 20，同噪声同 transformer）：两张图同级，蜡质缺陷原样存在**
（`fp32_22tok.png` vs `zimage_fp32_cxxlatents.png`，像素差 20.27）。
⇒ ②**截断不是当前画质缺陷的主因**，降级。
⚠️ **但该结论只对这个 22-token 短 prompt 成立**——正文只剩约 14 token 空间，
长 prompt 的代价未测。transformer 的 caption 上限是 32 槽，
**只把 text_encoder 重导到 32 即可吃满现有容量，且 transformer 不用动**。

### 15.21.6 方法论：又一次"数值对、图不对"

若只看逐步数值（±0.24%），会判定"导出忠实"并继续在量化上打转。
**是"打开图看"抓住的**——与 #69/#70（标量指标在此区间不可靠）同源，这次是**正向印证**：
数值几乎相同而画质差异明显。

⚠️ 本轮疏漏：官方那次的 final latents **未保存**（脚本只存 PNG），
用户要求用官方 latents 做 VAE 对照时只能改用我们的（末步 std 差 0.17%，对隔离 VAE 无影响）。
⇒ **中间产物默认落盘**。

### 15.21.7 🔴 关键补充：缺陷**传导到设备**，且被量化放大（2026-08-19 实测）

三臂 VAE 对照，**完全相同的 final latents**（`fp32_steps/lat_8.raw`）：

| VAE 实现 | 高频能量 | vs 官方 | 像素差 vs 官方 |
|---|---|---|---|
| 官方 PyTorch（fp32） | 2.892 | — | 0 |
| 我们的 ONNX（fp32/ORT） | 3.773 | **+30.5%** | 13.33 |
| **设备 QNN（量化/HTP）** | **4.021** | **+39.0%** | 13.28 |

产物：`scratch_runs/vae_ab_qnn.png`；设备侧 `qnn-net-run` 输出字节数 12582912 校验通过（约束 3）。

⇒ ②**两个可分离的来源**：导出引入 **+30.5%**，量化再叠加 **+8.5 pp**。
⇒ ②**修 VAE 导出对最终产品有直接收益** —— 这条关闭了 `reviews/REVIEW_VAE_EXPORT_2026-08-19.md`
§5 第 4 条列为"当前计划最大不确定"的疑问（我原先无法排除"ORT 的问题不传导到 QNN，
修了对产品无收益"）。

⚠️ **划界**：本实验用的是**当前那份量化 VAE**（A16W8、per-tensor）。
VAE 从来不是嫌疑对象，**从未尝试过 per-row 或其它配置** ⇒
那 8.5 pp 中有多少可优化**未知**，但优先级低于导出问题。

---

## 15.22 【2026-08-21】🔴🔴 根因：**VAE 反缩放被做了两遍**（宿主与 app 全线，非导出缺陷）

### 15.22.1 结论先行（①实测，判据事前锁定，一次通过）

**`vae_decoder.onnx` 的第一个节点就是 `Div(vae_latents, 0.3611)` —— 反缩放在图内部已经做过一次。
而所有调用方（6 个宿主脚本 + app 的 C++ 流水线）在喂进去之前又做了一遍
`lat / 0.3611 + 0.1159`。**

⇒ **除以了两遍。** 三臂实测（同一份 `lat_8.raw`，对照官方 PyTorch VAE）：

| 喂法 | 平均\|像素差\| | PSNR | 高频能量比 |
|---|---|---|---|
| `lat/0.3611 + 0.1159`（**现行**，全部脚本 + app） | **13.326** | 23.51 dB | **1.3049×** |
| `lat`（裸，无 shift） | 0.751 | 46.52 dB | 0.9976× |
| **`lat + 0.1159×0.3611`（正确）** | **0.001** | **79.92 dB** | **1.0000×** |

⇒ ②**我们的 VAE ONNX 导出与官方 PyTorch 数值等价（79.92 dB）。导出从来没有缺陷。**

脚本 `scripts/vae_input_convention.py`；产物 `scratch_runs/vae_conv_{current,raw,fixed}.png`。

### 15.22.2 正确喂法的推导

- 官方语义：`decoder( lat/s + shift )`，`s=0.3611, shift=0.1159`
- 我们的 ONNX：`decoder( input/s )`（**图内只有 Div，没有 shift**）
- 令两者相等 ⇒ `input = lat + shift·s = lat + 0.041856`

⚠️ 注意 `input = lat`（裸喂）已经能到 46.52 dB —— **shift 是次要项，双重除法才是全部问题**。

### 15.22.3 第二重损害：设备侧量化量程被打穿

VAE 量化的校准数据是**裸 latents**（与图内 Div 的期望一致）：

| | std | 范围 |
|---|---|---|
| 校准样本 0000~0004 | 0.8567 ~ 1.0810 | ±3.4 ~ ±4.5 |
| 裸 `lat_8` | 1.0996 | −4.435 ~ 3.874 |
| **app 实际喂入 `lat/s+shift`** | **3.0451** | **−12.167 ~ 10.843** |

⇒ ②**app 喂给量化 VAE 的输入超出已标定激活量程 2.77 倍，被量化器硬钳。**
⇒ ②这解释了 §15.21.7 里记在"量化"账上的那 **+8.5 pp**（30.5% → 39.0%）。
⇒ ②**一行修改同时消除双重除法与量程越界，不需要重导出、不需要重量化。**

### 15.22.4 影响面：哪些历史结论要改

| 原结论 | 处置 |
|---|---|
| §15.21.1「画质缺陷 100% 来自我们的 VAE ONNX 导出」 | 🔴 **定位对（在 VAE 这一步），归因错**（是调用方喂法，不是导出） |
| §15.21.3「结构与权重完全等价 ⇒ 缺陷是运行时数值行为」 | 🔴 **前半对，后半错**。静态检查全过本应指向**输入**，我却推向了"运行时" |
| §15.21.3 ③ GroupNorm→InstanceNorm 方差精度假设 | 🔴 **作废**，从来不存在 |
| §15.21.7「缺陷传导到设备且被量化放大 +8.5pp」 | 🔴 **现象对，机制错**：是量程越界被钳，不是量化本身 |
| 台账 #78「增益 ×1.3324 ＋ 结构残差 9.86%」 | 🔴 **现象全部正确，归因错**：两者都是同一个喂法 bug 的表现 |
| §15.21.4 transformer / text_encoder 导出忠实 | ✅ **不受影响**（不经过 VAE 输入） |
| §15.21.5 20 vs 22 token 同级 | ✅ **不受影响**（双臂共模） |
| 所有"我们的流水线 A vs 我们的流水线 B"对照（#54/#69/#75 等） | ✅ **不受影响**：两臂共用同一个错误喂法，共模抵消 |
| 所有"我们 vs 官方"对照 | 🔴 **全部被污染，须用正确喂法重做** |

### 15.22.5 是什么抓住它的（方法论，值得记）

**是一个"已知答案的样本"门（约束 8）**：逐层探针里我加了 V2d ——
`conv_in` 的权重与输入两侧都逐位相同 ⇒ **它的 std 之比必须是 1.0000**。
实测 **2.7595**，而 `1/0.3611 = 2.7693`。**这个数字直接把根因指出来了。**

没有这道门，我会拿着"×1.33 增益"去逐层找"增益在哪累积"，
在一个根本不存在的缺陷上继续挖 —— 正如此前两天在"GroupNorm 方差精度"上做的那样。

⚠️ 同时记一条我自己的误判：v1 探针作废后，我诊断为"Conv 按序匹配错位"。
**这个诊断是错的** —— v2 用权重哈希做双射，得到的置换是**恒等置换**，v1 的顺序本来就是对的。
v1 真正的问题与 v2 相同：**输入喂错**。
⇒ 教训：**发现异常时不要先怀疑最近改过的那一环**，要先跑"已知答案的样本"。

### 15.22.6 🔴 设备门实测：修复在【量化 QNN VAE】上同样成立，且"量化放大"根本不存在

方案 `scripts/EXP_PLAN_VAE_DEVFIX.md`（判据事前锁定），脚本 `scripts/vae_devfix.py`。
单变量：同一台设备（SM8750）、同一份 `vae.bin`、同一份 `lat_8.raw`，**唯一变量 = 喂法**。

| 臂 | 输入 std / 范围 | 对官方平均\|像素差\| | PSNR | 高频比 |
|---|---|---|---|---|
| **CURRENT**（app 现行 `lat/s+shift`） | 3.0451 / [−12.167, 10.843] | **13.284** | 23.68 dB | **1.3904×** |
| **FIXED**（`lat + shift·s`） | 1.0996 / [−4.393, 3.915] | **0.200** | **54.33 dB** | **0.9999×** |

**三道门全过**：V1 `Finished Executing Graphs`；V2 `pixels.raw` = 12582912 字节（约束 3）；
**V3 基线复现精确命中** —— CURRENT 臂高频 **4.0207**（§15.21.7 记录 4.021）、
像素差 **13.284**（记录 13.28）⇒ 确是同一配置，可跨实验比较。

**改善率 R = 0.9850。**

⇒ ②**§15.21.7 的"量化把缺陷从 +30.5% 放大到 +39.0%"这个机制结论作废。**
不存在量化放大；那 +8.5 pp 全部来自**输入超出校准量程被硬钳**。
⇒ ②**我们的 VAE 量化质量一直很好**：喂对之后，A16W8 量化 + HTP 执行相对官方 fp32 PyTorch
仅 **0.200 平均像素差 / 54.33 dB**。
⇒ ②**残余 0.200 就是纯量化损害的上界**（fp32 + 正确喂法对官方已是 0.001 / 79.92 dB，
   ⇒ 残余不可能来自导出或喂法）。

产物：`scratch_runs/devfix_{current,fixed}.png`。

### 15.22.7 待办：app 侧一行修改（交付类操作）

`local-dream/app/src/main/cpp/src/PipelineZImage.hpp:203`
```cpp
// 现行（错）
value = value / zimage_vae_scaling_factor + zimage_vae_shift_factor;
// 正确
value = value + zimage_vae_shift_factor * zimage_vae_scaling_factor;
```
⚠️ 按约束 11 四条铁律执行：**先证明现状可用 → 备份原 APK → 只改这一个变量 →
交付后自己跑真实链路 → 在 `untrusted_app` 真实环境验证（不得用 `run-as`，见 #72）。**

### 15.22.8 ✅ app 交付完成并在真实环境验证（2026-08-21，按约束 11 四条铁律）

**改动**：`local-dream/app/src/main/cpp/src/PipelineZImage.hpp`
```cpp
- value = value / zimage_vae_scaling_factor + zimage_vae_shift_factor;
+ value = value + zimage_vae_shift_factor * zimage_vae_scaling_factor;
```
grep 复核：项目源码里 `zimage_vae_scaling_factor` 只有 3 处（2 处定义 + 这 1 处使用），无第二份清单。

**四条铁律逐条执行**：

| # | 铁律 | 本次怎么做的 |
|---|---|---|
| 1 | 先证明现状可用 | 用**旧 APK** 端到端生成一张图（`vaefix_BASELINE.png`，HTTP 200 / 223.8 s / seed 42），**本人亲自看过**：清晰橘猫，且用户抱怨的"毛发蜡质、边缘发硬"在这张上一目了然 |
| 2 | 覆盖前备份原件 | `adb pull` 存 `scratch_runs/apk_backup/PRE_VAEFIX_base.apk`（94316812 B，sha256 `108df9a8…`），版本信息一并存档 |
| 3 | 交付后自己跑真实链路 | 重新生成 `vaefix_AFTER.png`（HTTP 200 / 241.6 s / **同 seed 42、同 prompt**） |
| 4 | 在真实运行环境验证 | 走 app 自己的 HTTP `/generate`（`untrusted_app` 域），**未使用 `run-as`**（#72） |

**交付核实**：设备端 APK sha256 `2643460a…` 与本地新构建**逐位一致**；旧版 sha256 不同，回滚路径完好。

**结果（真实 app 端到端，唯一变量 = 那一行）**：

| | mean | std | 钳到 0 | 钳到 255 | 梯度能量 |
|---|---|---|---|---|---|
| BEFORE | 77.390 | 67.378 | **2.528%** | 1.476% | 4.2916 |
| **AFTER** | 83.904 | 55.609 | **0.001%** | **0.000%** | 2.9257 |

⇒ ①**死黑像素减少 1692 倍**，高光溢出归零。
⇒ ①**本人亲自看两图**：AFTER 毛发柔和有层次、背景房间细节（砖墙/置物架/罐子）恢复，
   BEFORE 里那些细节被压成死黑；"蜡质发硬"消失。
⇒ ②**用户从项目一开始就指出的画质缺陷，到此解决。**

⚠️ **划界**：两图平均像素差 18.107、梯度比 1.4668，**大于**隔离实验里的 13.284 / 1.3904。
两者用的 latents 不同源（隔离实验用宿主 22-token 那次的 `lat_8.raw`，app 用自己的
C++ RNG + 量化 TE + HTP transformer）⇒ **数值不应精确吻合，不得当成不一致来解读**。

### 15.22.9 两条顺带修掉的工具缺陷

1. **`zimage_device_test.sh` 用 `grep "Server listening"` 当就绪判据会误报。**
   2026-08-21 实测：backend 已在 `127.0.0.1:8081` 监听（`/proc/net/tcp` 里 `0100007F:1F91`），
   日志里却没有那句话 ⇒ 脚本报 "backend never came up"。**又一次"报错说谎"。**
   ⇒ 新脚本 `scripts/app_generate.sh` 改为**直接探测端口**，并固定 seed 使前后对照成为单变量。
2. **Git Bash 会把 `adb shell` 里的 `/sdcard/xxx` 转成 Windows 路径。**
   `uiautomator dump /sdcard/ui.xml` 因此写到了 `/Files/Git/sdcard/ui.xml`，
   pull 回来是空文件，一度让我以为 UI 树抓不到。
   ⇒ 凡在 `adb shell` 里出现设备侧绝对路径，**必须 `export MSYS_NO_PATHCONV=1`**。
3. **全新安装后首启需要 ~119 s 契约校验 + 11 GB 模型加载**，原脚本 120 s 窗口太短，
   会把"正在加载"误判成"没起来"并强杀重来。⇒ 已改为先长轮询（15 min）再考虑重启。

### 15.22.10 ✅ 顺带关闭 P0-C：我们的 FP32 全流程与官方 PyTorch 已基本一致

执行顺序表里的 **P0-C**（"我们的 FP32 vs 官方 Z-Image Turbo 实现"，此前是最大的未验证前提）
用两张已在盘的图即可回答：

| | 平均\|像素差\| vs 官方 | PSNR | 梯度能量 |
|---|---|---|---|
| 官方 PyTorch 全流程（基准） | 0 | — | 2.8869（1.0000×） |
| 我们 ONNX 全流程·**旧喂法** | **13.627** | 23.38 dB | 3.7734（**1.3071×**） |
| 我们 ONNX 全流程·**修正后** | **1.018** | **41.61 dB** | 2.8916（**1.0016×**） |

⚠️ **划界**：两侧 latents 不同源（官方那次 final latents 未保存，末步 std 差 0.17%），
且我们这侧的 8 步 transformer 逐步差 ±0.24% ⇒ **残留的 1.018 里包含这两项，不是纯 VAE 残差**。

⇒ ②**阶段 0（导出保真度）三项全部通过**：text_encoder ✅（余弦 0.999990）、
transformer ✅（逐步 ±0.24%）、VAE ✅（79.92 dB）。
⇒ ②**"我们的 FP32 参考不如官方"这个风险已排除，FP32 参考可以作为靶子使用。**
⇒ 主线可以推进到阶段 A（HTP correctness），且**这次的靶子是可信的**。

---

## 15.23 【2026-08-21】#84 复查：反相关**不是**钳位假象——我的假设被证否

方案 `scripts/EXP_PLAN_METRIC_RECHECK.md`（判据事前锁定），脚本 `scripts/metric_recheck.py`。

### 15.23.1 先收窄了一半范围（零成本）

#69 主张三把尺子全部反相关。但 **`E_all` 算在 latents 上、完全不经过 VAE**
⇒ ②喂法 bug 在原理上影响不到它 ⇒ **"E_all 43.41%→44.80% 而图从认不出任何物体变成清晰的猫"
这一条独立成立，无需复查。** 本实验因此只查经过 VAE 的两把：像素差与 PSNR。

（先想清楚"这个量到底经过哪些环节"，就砍掉一半工作量，不花任何成本。）

### 15.23.2 装置（纯宿主，未占设备）

- **末步 latents 重建**：末步 `dt = 0 − σ₇` ⇒ `final = lat_s7 + σ₇·noise_s7`（σ₇ = 0.3000，
  与 FP32 日志 step 7 `dt=-0.3000` 一致）。输入 `htp_inloop_{ctl3seg,perrow4seg}/s7/`
- **FP32 参考**：用**同一份初始噪声与 caption**（md5 逐位相同）重跑 FP32 全流程，
  修正后喂法，2304.6 s（`FP32_TAG=inloopref_vaefix`）

### 15.23.3 🔴 V2 门以极高精度通过 —— 装置本身被验证

| 臂 | 旧喂法重解码 | 历史记录 |
|---|---|---|
| 3seg | **41.368** | 41.37 |
| 4seg | **44.523** | 44.52 |

⇒ ②同时验证三件事：**末步 latents 重建公式正确**、**新 FP32 参考与历史那份等价**、
**整套装置可信**。（这是"已知答案样本"验证的又一次应用，约束 8。）

### 15.23.4 结果：混淆因素被消除，反相关原样存在

| 喂法 | 臂 | 平均\|像素差\| | PSNR | 钳位像素 |
|---|---|---|---|---|
| 旧 | FP32ref / 3seg / 4seg | — / 41.368 / 44.523 | — / 12.46 / 11.60 | 3.469% / 1.592% / 1.143% |
| **新** | FP32ref / 3seg / 4seg | — / **33.608** / **37.425** | — / **14.35** / **13.34** | **0.021% / 0.004% / 0.002%** |

**本人亲自看两张新解码图**：3seg = 无定形橙色色团，**认不出任何物体**；
4seg = **干净完整的橘猫**（胡须/眼睛/毛发层次/木纹俱全）。

⇒ ②**钳位确实被消除了**（3.469% → 0.021%，降两个数量级）；
⇒ ②**反相关原样存在，差距还从 +3.155 扩大到 +3.816**；PSNR 同向；
⇒ ②**我的「钳位假象」假设被证否。#69/#70 成立，且因排除了混淆因素而【比原来更强】。**

### 15.23.5 对后续的影响

| 项 | 处置 |
|---|---|
| #84 | ✅ **关闭·假设被证否**（方向错误，如实记录） |
| #69 / #70 | ✅ **维持，并升级**：混淆因素已排除，结论更硬 |
| §2.3「暂停基于标量指标的取舍决策」 | **继续有效**，不解除 |
| P1「三层指标面板」 | **确认必要**，不再是可选项 |
| #56b | 仍未查。但注意它依据的是 `E_all`（不经 VAE），**与本实验无关，需单独处理** |

**顺带的产品事实**：修正喂法后 HTP 那两张图本身也变好了（4seg 像素差 44.52 → 37.43，
钳位 1.143% → 0.002%）——**喂法 bug 此前同样在损害 HTP 路径的成图**，不只是 FP32 路径。


---

## 15.24 【2026-08-21】✅ P0-B 前置门：`--quantization_overrides` **可以**注入 per-axis encoding

方案 `scripts/EXP_PLAN_PERAXIS_OVERRIDE.md`（判据事前锁定），脚本 `scripts/peraxis_override_probe.py`。
最小已知样本：`x[1,8] @ W[8,4]`，注入 4 个**由我指定**的 scale `[0.001, 0.002, 0.004, 0.008]`。

| 臂 | 格式 | DLC 读回 | 结论 |
|---|---|---|---|
| C（V 门） | 0.6.1 per-tensor | `encoding=0` / `scale_offset` / scale=**0.001000000047** | ✅ 装置可信 |
| **A** | 0.6.1，4 个 encoding dict | `encoding=1` / `axis_scale_offset` / axis=0 / **4 个 scale 逐个吻合** | ✅ |
| **B** | 2.0.0，`y_scale`+`axis` | 同上 | ✅ |

⇒ ②**per-axis 可注入，两种 schema 都行 ⇒ P0-B「Exact-real-FC standalone replay」前置条件成立。**

### 15.24.1 三个会坑下一个人的细节

1. 🔴 **`axis` 会被转换器重映射**：注入 `axis=1`，DLC 里落成 **`axis=0`**
   （MatMul 权重 `[K,N]` 转 FullyConnected 时被转置成 `[N,K]`）⇒ **必须读回验证，不得假设**。
2. 🔴 **`Processed N quantization encodings` 的 N 不是张量数**：只覆盖 1 个张量也显示 **2**。
   #45「必须检查这一行」仍然对（N=0 = 静默失效），但**判断某张量是否被覆盖要看 `is_overridden`**。
3. **读回方法**：`qairt-dlc-to-json -i <dlc> -o <json>`（**`-i/-o`**），读 `/graph/tensors/<名字>/quant_params`。
   `encoding=0` ⇒ per-tensor，`encoding=1` ⇒ per-axis。

### 15.24.2 顺带根治的工具链缺陷

`scripts/qairt_tool.py` 原版**没设 `PYTHONPATH`** ⇒ `ModuleNotFoundError: No module named 'qti'`。
`qairt_run.py` 的文档串里写着"需要 PYTHONPATH=<SDK>/lib/python"，**但靠人记得去设**——
按 #73 的做法根治：**让遗漏不可能**，helper 自己装好
`sys.path` / `PATH` / `QNN_SDK_ROOT` / `SNPE_ROOT`，并统一把 stdout/stderr 重配置为 UTF-8
（SDK 工具会往 GBK 控制台打非法字符）。


---

## 15.25 【2026-08-22】~~P0-B 完成：#30 与 #39 的矛盾已解决~~ 🔴 **结论已被 §15.29 推翻**
> 🔴 **【导航标注 2026-09-06】本节的装置有误**：建 context 时漏传 `--config_file`，
> 生成器按 **Hexagon v68 + 4 MB VTCM** 编译（尽管传了 `--htp_socs sm8750`、尽管文件名带 `.SM8750`）。
> **本节数值一律不得引用。** 见 §15.29。

方案 `scripts/EXP_PLAN_P0B_FC_REPLAY.md`（判据事前锁定）。
脚本 `scripts/p0b_extract_encodings.py` / `p0b_build.py` / `p0b_run_analyze.py`。

### 15.25.1 装置：真正的"精确重放"

把 `node_linear_99`（K=3840 的注意力投影）单独拎出来建成单算子 context，
**输入 x、权重 W、以及全部 encoding 都取自真实那次运行，不重新校准**。

| 门 | 结果 |
|---|---|
| **V1 encoding 注入** | 读回 DLC：per-tensor scale **0.015012254938** / offset **−126**，与真实基线**逐位一致**；per-channel 条目数 0 ✅ |
| **V2 设备执行** | `Finished Executing Graphs`；输出 **63,406,080 B** 精确（约束 3） ✅ |
| **V3 输入同一性** | md5 `eb9b583a0b41f8f37bfb69cc8077a9b5`，与图内那次同一份文件 ✅ |
| **float64 精确性** | \|acc\| 上界 **2.16e9** ≪ 2⁵³ ⇒ 整数乘加在 float64 下无舍入 ✅ |
| 🔴 **模拟器已知答案门** | 用同一套模拟复算**图内** HTP：**E_ingraph = 2.8714%**，#39 记录 **2.72~2.87%** ⇒ **复现** ✅ |

最后一道门是关键：它证明**模拟器与口径本身是对的**，standalone 的数字才可采信（约束 8）。

### 15.25.2 结果

| 量 | 值 |
|---|---|
| **E_standalone**（单算子重放 vs 正确定点） | **1.5544%**（主体余弦 0.999879） |
| **E_ingraph**（完整图 vs 正确定点） | **2.8714%**（0.999591） |
| **比值** | **0.541** |

⇒ ②**判据判定：`E_standalone ≥ 0.5×E_ingraph` ⇒ 真实数据/encoding 就能触发 ⇒ 问题在算子级。**

### 15.25.3 这解决了什么

**#30（合成 MatMul 探针，`E_htp/E_fxp` = 1.00）与 #39（真实 FC 差 20~230 倍）的矛盾，
自 2026-08-17 悬置至今，现已解决：差别在于输入与 encoding 是不是真实的。**

单算子 + 真实数据/encoding ⇒ 复现 54% 的图内误差；
单算子 + 合成良态数据 ⇒ 误差为零。
⇒ ②**#30 的结论「HTP 单算子是正确的」必须加上限定：仅对良态输入成立。**
台账 #30 原本就写了「探针输入是良态的，不得外推」——**该警告现被实测坐实。**

### 15.25.4 ⚠️ 不得放大：还有 46% 没有被解释

比值 **0.541 只是刚过判据线（0.5）**。诚实读法：

- **算子级效应真实存在且略占多数（54%）**
- **仍有约 46% 的图内误差，单算子重放不出来 ⇒ 图层面效应同样真实存在**

⇒ **#33/#34（算子融合 / VTCM 分块）应当降级，但【不得关闭】。**
那 46% 是一条新的、有量化依据的线索（登记为 #88）。

### 15.25.5 途中避开的一个会静默作废的坑

首次建重放时我用了 **per-row** encoding，但 #39 的 2.72~2.87% 来自
`fxp_sim.py`，而它读的 dump 是 `transformer_part1b_quantized.dlc` = **基线 per-tensor**，
`fc_probe/htp/linear_99_fc.raw`（08-15 15:31）也是同一轮。
**拿 per-row 重放去对基线的锚点，判据会落空。** 已改为基线配置重建。
⇒ **判据锚在哪个配置上，重放就必须用哪个配置。**

### 15.25.6 顺带确立：per-row 的量化轴 = ONNX 的 dim1

DLC 里 `val_1800` 标 `axis=0 / 3840 通道`，但那对应 **ONNX 的 dim1（列）**——
FullyConnected 权重从 `[in,out]` 转置成了 `[out,in]`。
实测：按 dim1 反推 3840 个 scale **100% 吻合**（中位相对差 2.11e-08）；按 dim0 只有 **0.99%**。
⚠️ **W 是 3840×3840 方阵，形状检查完全掩盖了这一点** —— 与 #85 在最小样本上抓到的
「axis 会被转换器重映射」是同一件事，这次在真实模型上再次证实。


---

## 15.26 【2026-08-22】~~三臂算子级对照：offset 项是 HTP 误差的最大单一来源~~ 🔴 **结论已被 §15.29 推翻**
> 🔴 **【导航标注 2026-09-06】同 §15.25：装置漏 `--config_file`，编成了 v68/4MB。**
> 正确装置下 `E_standalone` 是 **0.1034%** 而非本节的 1.5544% —— **差 15 倍**。
> **本节数值一律不得引用。**

在 P0-B 建起的单算子重放台上做三臂对照。**唯一变量 = 权重 encoding**，
输入逐字节相同（md5 `eb9b583a…`），每臂的"HTP 超出误差"都对**各自的**正确定点计算
⇒ 表示误差被约掉，量的是纯粹的"HTP 做得比正确定点差多少"。

| 臂 | 权重 encoding | 表示代价（输出域，vs FP32） | **HTP 超出正确定点** |
|---|---|---|---|
| **A** 基线 | 非对称 per-tensor，offset **−126** | 1.3323% | **1.5544%** |
| **B** | 对称 per-tensor，offset **0** | 1.3550%（+0.0227 pp） | **0.8310%（−46.5%）** |
| **C** | 对称 **per-row**，3840 scale | **0.5050%** | **0.6591%（再 −20.7%）** |

三臂 V 门全过（V1 encoding 注入逐位精确；V2 输出字节数精确；V3 输入同一份文件）。

### 15.26.1 三条 ②级结论

1. **非对称权重的 offset 项是 HTP 算子级超出误差的最大单一来源，占 46.5%。**
   支撑 #29/#40（#40 实测 offset 项/主项 = 0.891~0.979 的灾难性抵消）。
2. **在消除 offset 之上，细化 scale 还能再降 20.7%**；A→C 合计 **−57.6%**，
   且表示代价同时从 1.3323% 降到 **0.5050%**。
3. ⇒ **`--use_per_row_quantization`（#47）为何是本项目唯一有效手段，现在有机制解释**：
   它**同时**做到「消除 offset 项」和「细化而非粗化 scale」。

### 15.26.2 🔴 #7 的"对称更差"与本结果的矛盾已解开（不是矛盾）

#7 端到端测对称权重 ⇒ 更差（余弦 0.6033→0.5507）。#7 自己的划界写明
*"对称化同时粗化了 scale（中位 1.073×），两者无法分离"*。

**本实验分离了**（因为量的是"超出**各自**正确定点"的部分）：

- 对称化**消除 offset 项** ⇒ −46.5%
- 对称化**同时粗化 scale 1.073×**，而 **#52 实测 HTP 对 scale 粗化的敏感度 = CPU 的 37.6 倍**
- ⇒ 端到端两者相抵、净负 ⇒ **这正是 #7 观察到的现象**

⇒ ②**"对称化"本身没错，错的是"用粗化 scale 换对称"。** per-row 两头都占，所以有效。

### 15.26.3 划界（不得越界引用）

- 本节全部数字是**单算子（`node_linear_99`，K=3840 注意力投影）算子级**的，
  **不是端到端**。端到端还叠着 #88 的图层面 46%。
- C 臂的激活 encoding 取自 per-row DLC（输入 scale 0.001154021244 vs 基线 0.001149723656，
  差 0.37%），**不是完美单变量**；该差异量级远小于观察到的效应，但须记在案。
- 判据判定：B 落在「部分贡献」（距「主要因素」线 0.777% 差 7%）；C 落在「部分叠加」。
  **按字面判据都不是最强档，本节措辞已按此收敛，不得写成"证明 offset 项是唯一原因"。**


---

## 15.27 【2026-08-22】#91 分块量化（BQ）与 #92 per-row bias：**两条线索均关闭，全程未占设备**

方案 `scripts/EXP_PLAN_BQ.md`（判据事前锁定），脚本 `scripts/bq_probe.py`。

### 15.27.1 #91 BQ：**QAIRT 2.48 的分块量化对静态权重只支持 int4**

`qairt-quantizer` **没有** BQ 的 CLI 开关（已读完整 `--help`）⇒ 只能经 overrides 注入 schema 2.0.0。

**廉价试探（单变量：只改 `output_dtype`）**：

| `output_dtype` | 分块 encoding block=128 | 结果 |
|---|---|---|
| **int4** | ✅ | **converter rc=0，转换成功** |
| **int8** | ✅ 同样的 JSON 结构 | ❌ `modeltools::StaticTensorQuantizer::quantizeAndPackStaticTensors: Unhandled quantization encoding type` |

⇒ ②**不是格式错、不是后端不支持 BQ 本身，而是静态权重量化器只实现了 int4 的分块 encoding。**
（两次都打了 `Processed N quantization encodings` ⇒ JSON 被正常解析，问题在其后的打包环节。）

⚠️ 途中一个格式坑：给二维 `y_zero_point` 会让转换器在 `contain_decimal_num()` 里对 ndarray 调
`round()` 而 `TypeError`。`y_zero_point` 是可选项（默认 0），对称时直接省略即可。

### 15.27.2 那 int4 + 细分块值得吗？——不值得，差一个数量级

输出域表示误差（与 HTP 超出误差同量纲）：

| 配置 | 表示误差 | 权重体积 |
|---|---|---|
| **int8 per-row（当前部署）** | **0.5051%** | 14 MB |
| int4 BQ block=128 | 6.2933%（+5.79 pp） | 7.8 MB |
| int4 BQ block=32（最细） | **5.0470%（+4.54 pp）** | 9.2 MB |

⇒ ②**位宽起支配作用，分块细化补不回来。** 即使体积砍到约一半，5% 的表示误差会把
HTP 那 0.66% 的超出误差完全淹没 ⇒ **#91 关闭**，不上机。

### 15.27.3 #92 `--enable_per_row_quantized_bias`：本模型是空转

part1b 的 **60 个 FullyConnected 全部只有 1 个 STATIC 输入（= 权重），无 bias**。
⇒ ②该开关对本模型无作用，**#92 关闭**。与 #13（"44 个 norm、0 个 bias"）一致。

### 15.27.4 方法论价值

这两条都是**在宿主上用代价门 + 廉价试探关掉的**，合计约 20 分钟、零设备占用。
其中 BQ 那条的关键是**把"支持性"和"值不值得"拆成两问**，先问便宜的那个。


---

## 15.28 【2026-08-22】🔴🔴 **#10 / #27 / #34 的"静默忽略"根因查明：配置里根本没有 `graphs` 段**

方案 `scripts/EXP_PLAN_VTCM.md`（V0 生效门救了场）。全程宿主，未占设备。

### 15.28.1 起点：V0 门抓住 `--vtcm_override` 被忽略

用 `qnn-context-binary-generator --vtcm_override {0,8,4,2}` 建单算子 context，
**四份产物 md5 完全相同**（`e273d197…`）⇒ 选项被静默忽略，与 #10/#27 同一现象。
按事前判据，**#34 标为 ⛔受阻，不得下"VTCM 无影响"的结论**。

### 15.28.2 换 `snpe-dlc-graph-prepare`：开关全部生效，但**带不进 `.bin`**

台账 #10/#27 早就写了"文档指向 `snpe-dlc-graph-prepare`，待换工具重试"。实测该工具把三条受阻线索
要的开关全部作为**一级 CLI 参数**提供：`--vtcm_override`、**`--htp_high_precision_sigmoid`**、
**`--htp_advanced_activation_fusion`**，另有 `--optimization_level/preset`、`--htp_dlbc(_weights)`、
`--htp_slc_alloc`、`--htp_cached_weights`、`--num_hvx_threads`。

**md5 门：8 个配置产出 8 个不同 md5，且体积明显不同**
（default 31.82 MB / vtcm4 33.48 / vtcm2 37.63 / opt1 30.41）⇒ **真的生效**。

🔴 **但产物是"内嵌 HTP cache 的 DLC"，而不是 `.bin` context。实测**：
把 `default.dlc` / `vtcm2.dlc` / `opt1.dlc` 分别喂给 `qnn-context-binary-generator --dlc_path`，
**产出的 `.bin` 三者 md5 完全相同，且与直接从未 prepare 的量化 DLC 生成的那份也相同**（`e273d197…`）。

⇒ ②**`qnn-context-binary-generator` 丢弃 graph-prepare 内嵌的 cache，按自己的默认重新编译。**
⇒ ②**要用 graph-prepare 的开关，必须走 DLC 执行路径；`.bin` context 路径带不动它们。**
⚠️ **app 部署用的就是 `.bin`** ⇒ 这条对产品化是硬约束。

### 15.28.3 🔴 真正的根因：`backend_config_detail.json` 里**没有 `graphs` 段**

官方 `htp_backend.html` 给出的 schema 是：

```json
{ "graphs": [ { "vtcm_mb": ..., "graph_names": ["..."], "dlbc": 1 } ],
  "devices": [ { ... } ] }
```

而本项目一直在用的 `D:\ZImage_Work\backend_config_detail.json` **只有**：

```json
{ "devices": [ { "soc_model": 69, "dsp_arch": "v79" } ] }
```

⇒ ②**#10/#27 当年根本没把选项写进生效的位置** —— 不是"后端不支持"，是**配置写错**。

### 15.28.4 补上 `graphs` 段后实测（单算子，图名正确）

| 配置 | md5 | 体积 |
|---|---|---|
| 基准（无 `--config_file`） | `e273d197…` | 22.34 MB |
| **图名写错**（`fc99`） | `4cf8b914…` | — |
| `vtcm_mb` = 8 / 4 / 2 | **`9ccc464d…` 三者相同** | 17.05 MB |
| **`O` = 3** | **`39d81fa6…`** | 17.08 MB |
| **`O` = 1** | **`8f850693…`** | 15.64 MB |
| **`dlbc` = 1** | **`45e6abd7…`** | 17.05 MB |

⇒ ②**`O` 与 `dlbc` 经 `.bin` 路径生效**；`vtcm_mb` 被接受但对本图无效果（2/4/8 产物相同）。

### 15.28.5 🔴 `graph_names` 必须填对，否则整段配置落空

图名**不是** ONNX 图名，**也不是**量化 DLC 名，而是 **converter 产出的 fp32 DLC 的文件名主干**：

| DLC | 图名 |
|---|---|
| `fc99_quantized.dlc`（由 `fc99_fp32.dlc` 量化而来） | **`fc99_fp32`** |
| part1b per-row context | **`transformer_part1b_fp32`** |

**读法**：`qnn-context-binary-utility --context_binary <bin> --json_file <out>`
→ `info.graphs[].info.graphName`。

⚠️ 填错时**不报错**，只是整段配置落到空处（本次实测 `fc99` vs `fc99_fp32` 产出不同 md5，
但四个不同选项之间毫无差别 —— **典型的"看起来生效了其实没有"**）。

### 15.28.6 台账处置

| 线索 | 原状态 | 新状态 |
|---|---|---|
| #10 `HIGH_PRECISION_SIGMOID` | ⛔受阻 | 🔶 **未查·阻塞已解除**（配置写法已知；单算子无 sigmoid，需在 part1b 上测） |
| #27 `advanced_activation_fusion` | ⛔受阻 | 🔶 **未查·阻塞已解除**（同上） |
| #34 VTCM 分块 | ⛔受阻 | 🔶 **未查·部分澄清**：`.bin` 路径下 `vtcm_mb` 对单算子无效果；graph-prepare 路径生效但带不进 `.bin` |


### 15.28.7 🔴 #10 与 #27 **关闭为「不适用」**（文档 + 实测 + 有分辨力的对照）

阻塞解除后立刻测了这两条。**官方 schema 的键名与注释**（`htp_backend.html`）：

- `"advanced_activation_fusion": {"type":"boolean"}` —— 注释明写
  ***"Note that this option has no effect on quantized graphs."***
- `"use_high_precision_fp16_sigmoid": {"type":"boolean"}` —— 注意**正确键名不是**
  `HIGH_PRECISION_SIGMOID`，而且语义是 **fp16 sigmoid**；本项目激活是 `uFxp_16` 定点

**实测（量化图，图名正确，`.bin` 路径）**：

| 配置 | md5 |
|---|---|
| `advanced_activation_fusion` = false / true | **`9ccc464d…` 相同** |
| `use_high_precision_fp16_sigmoid` = true / false | **`9ccc464d…` 相同** |
| **对照 `O` = 1** | **`8f850693…` 不同** ✅ |

⇒ ②**两个开关在量化图上确实无效果**；**对照组证明配置通路是活的**，不是又一次静默失效
（这一条是关键——没有对照就只能说"又没反应"，而不能说"确实不适用"）。

⇒ **#10 / #27 关闭为「不适用于量化图」**，与 #4（`fp16_relaxed_precision` 只作用于 fp 模型）同类。

⚠️ **划界**：只否证了"在**量化**图上无效果"。若将来走**选择性 FP16**（#43/#48）路线，
图里出现真正的 fp16 段落时，这两个开关**可能重新变得相关**，届时需重测。


---

## 15.29 【2026-08-22】🔴🔴 **重大订正：P0-B 系列（§15.25/§15.26）的装置有误，结论翻转**

### 15.29.1 装置错在哪

我建 P0-B 单算子 context 时**没有传 `--config_file`**，而设备上部署的那些 context **都传了**
（§12.4 的两层配置，detail 里有 `devices: [{soc_model:69, dsp_arch:"v79"}]`）。

**读回产物元数据（`qnn-context-binary-utility`）**：

| 产物 | `dspArch` | `vtcmSize` | E_standalone |
|---|---|---|---|
| 我的（无 `--config_file`） | **68** 🔴 | **4 MB** | 1.5544% |
| 带 `devices` 配置 | **79** ✅ | **8 MB** | **0.1034%** |
| 带 `devices` + `graphs` | 79 | 8 | 0.1034%（**与上逐位相同**） |

⇒ ②**不带 config 时，生成器按 Hexagon v68 + 4 MB VTCM 编译**，尽管传了 `--htp_socs sm8750`，
**尽管产物文件名就是 `.SM8750.bin`**。
⇒ ②`graphs` 段没有作用；全部效果来自 **`devices`**。

**设备侧核查：部署的 6 个 context 全部是 `dspArch 79 / vtcmSize 8`** ⇒ **产品没有问题**，
错误只存在于我自建的 standalone。

### 15.29.2 🔴 #64 的判别方法失效（这是本次没能提前拦住的直接原因）

#64 原文：*"漏 `--htp_socs` 会静默产出非 SoC 定向的 context，**判别方法：看文件名有没有 `.SM8750` 后缀**"*。

**我的错误产物文件名就是 `c_NOCFG.SM8750.bin`——完全通过该检查，而它是 v68/4MB。**

⇒ **#64 的判据必须替换为**：
读回 `contextMetadata.info.dspArch == 79` **且** `graphs[].info.graphBlobInfo.info.vtcmSize == 8`。
命令：`qnn-context-binary-utility --context_binary <bin> --json_file <out>`。

### 15.29.3 用正确配置重测（同输入、同 encoding、同判据）

| 臂 | 权重 encoding | E（错的 v68） | **E（正确 v79）** | 余弦 |
|---|---|---|---|---|
| A baseline | 非对称 per-tensor | 1.5544% | **0.1034%** | 0.999999 |
| B symtensor | 对称 per-tensor | 0.8310% | **0.0966%** | 1.000000 |
| C perrow | 对称 per-row | 0.6591% | **0.0680%** | 1.000000 |

### 15.29.4 结论的实际变化

| 原结论 | 处置 |
|---|---|
| **§15.25 / #87**「算子级占 54%」 | 🔴 **推翻**。正确值 `E_standalone/E_ingraph = 0.1034/2.8714 = **0.036**` ⇒ 按事前判据（≤0.4%）**判定为「只有完整图才触发」⇒ 问题在图层面** |
| **#88**「图层面占 46%」 | 🔴 **改为约 96%**（图层面几乎全部） |
| **§15.26 / #89**「offset 项占 46.5%」 | 🔴 **推翻**。正确值：0.1034% → 0.0966%，**只降 6.6%** |
| **#90**「per-row 因两机制叠加而有效」 | ⚠️ **方向仍成立但权重变了**：per-row 相对对称 per-tensor 降 **29.6%**（0.0966→0.0680），
即**细化 scale 才是主要项，消除 offset 只是次要项** —— 与原结论的主次**相反** |
| **#30**「HTP 单算子是正确的」 | ✅ **反而被加强**：真实数据、真实 encoding 下单算子误差仅 **0.068~0.103%**，
原先"仅对良态输入成立"的限定**不需要** |

### 15.29.5 为什么一个失误会波及全部结论

**因为它在测量链的共享部分。** 那台单算子重放台是后续每个数字的共同来源，
装置的设备画像错了，挂在上面的结论就一起移动 —— 这是共享装置出错时**必然**的波及范围。

**我把门设在了错误的地方**：给**模拟器**设了很强的已知答案门（复现 #39 的 2.87%，且**通过了**），
却**没给装置本身设门**。该有而没有的那一道是：
🔴 **standalone 与图内那份必须用同一套配方构建**。
我核对了输入逐字节相同、encoding 逐位相同、输出字节数正确 —— **唯独没核对构建配置相同**。

**本该能避免的三条路**：
1. 建完就 diff 产物元数据（`dspArch`）——5 秒
2. **照抄部署用的构建命令**，而不是从台账碎片里拼一条新的（分歧就是在这里进来的）
3. #64 本该拦住，但**它的判据操作化是失效的**（见 15.29.2）

⚠️ **这次能抓到靠的是运气**：O/dlbc 实验的对照臂恰好带了 config，出现 15 倍落差大到无法忽略。
**没做那个实验，错误结论就留下来了。**


---

## 15.30 【2026-08-22】🔴🔴 图层面误差 = **RmsNorm 的固有数学放大**，不是 HTP 缺陷

方案 `scripts/EXP_PLAN_RMSNORM_AMP.md`（判据事前锁定）。全程宿主，**零设备占用**。
⚠️ 本节正文此前**缺失**（目录已列 §15.30 但文件到 §15.29 即结束），2026-08-22 补写。

### 15.30.1 观察（零成本，来自已有 fc_probe 产物）

HTP 图内 vs CPU 参考，同一次运行，**所有高误差张量无一例外都是 RmsNorm 输出**：

| OpID | 类型 | 输入(误差) | 输出(误差) | 放大 A_htp |
|---|---|---|---|---|
| 14 | **RmsNorm** | linear_97 3.52% | **mul_308 22.60%** | **×6.43** |
| 101 | **RmsNorm** | linear_105 10.95% | **mul_333 44.36%** | **×4.05** |
| 84 | RmsNorm | linear_102 2.94% | mul_326 7.33% | ×2.49 |
| 35 | RmsNorm | view_111 2.76% | mul_314 3.08% | ×1.12 |
| 29 | FullyConnected | mul_304 2.75% | linear_99_fc 2.87% | **×1.04** |

⇒ FC 几乎不放大；RmsNorm 放大 ×1.12~×6.43。

### 15.30.2 🔴 第一版 V3 门 3/4 未过，以及真正的原因

用 float64 精确复现 RmsNorm、对 `x_cpu`/`x_htp` 各跑一次求固有放大 `A_float`。
**V3 门**（"我复现的算子必须与真实算子一致"，要求 <1%）第一版结果：
**1.3085% / 1.5508% / 1.4510% / 0.1785%** ⇒ **只有 OpID 35 通过，而它恰恰是放大最小的那个**。

诊断链（**没有猜，逐条实查**）：

1. ONNX 实查：Pow 指数 = **2.0**、eps(`val_240`) = **1e-05**、ReduceMean **`axes=[-1] keepdims=1`**
   ⇒ 公式的三个假设**全部正确**（约束 8：操作化先验证，不靠直觉）
2. 逐通道最小二乘拟合 `y = (x/rms(x)) * g_eff` ⇒ **残差 0.0001%**
   ⇒ **结构精确**，唯一偏差在 gamma 本身（`|g_fit − g_onnx|/|g|` 中位 0.63%）
3. CPU 参考跑的是 `transformer_part1b_quantized.dlc` 且未加 `--enable_cpu_fxp`（**#35**）
   ⇒ **浮点执行量化 DLC** ⇒ 运行时 gamma = **反量化后的量化 gamma**，不是 ONNX 里的 fp32 gamma
4. DLC encoding 实查（`snpe-dlc-info -d -s csv`）：这些 gamma 是 **`uFxp_8`，8-bit，per-tensor**

⚠️ **没有用拟合值回填**——那会让 V3 退化成循环论证。
用**从 DLC 独立取出**的 scale/offset 做 Q/DQ，并与"只用 CPU 臂做的最小二乘拟合"**互验**。

### 15.30.3 修正后重跑：四门全过

脚本 `scripts/rmsnorm_amp_probe_v2.py`。**判据一字未改**（属方案 §五「修正后重跑」分支）。

| OpID | V3（fp32 gamma → 量化 gamma） | 互验 | A_htp | A_float | **A_htp/A_float** |
|---|---|---|---|---|---|
| 14 | 1.3085% → **0.0001%** | 0.0000% | 6.43 | 6.17 | **1.043** |
| 101 | 1.5508% → **0.0001%** | 0.0000% | 4.05 | 3.96 | **1.022** |
| 84 | 1.4510% → **0.0000%** | 0.0000% | 2.49 | 2.49 | **1.001** |
| 35 | 0.1785% → **0.0632%** | 0.0691% | 1.12 | 1.12 | **1.000** |

判据线是 `A_htp ≤ 1.3 × A_float ⇒ 固有`，实测 **1.000~1.043**，V1/V2/V3 全过 ⇒ **4/4 判定「固有」**。

### 15.30.4 ①实测 / ②结论

- ①**实测**：RmsNorm 对输入误差的放大，在精确 float64 下与在 HTP 上**数值相同**（差 0~4.3%）
- ②**结论**：图层面那 ≈96% 是**误差沿链累积 + RmsNorm 的固有数学放大**，
  **不是 HTP 的实现缺陷** ⇒ **没有「HTP 侧可修」的东西**
- ⇒ 导出 **#98**：唯一杠杆是**降低进入 RmsNorm 的输入误差 = 改善上游量化**。
  这也解释了为何 `--use_per_row_quantization`（#47）是迄今唯一有效手段。

### 15.30.5 回查记录（约束 6，三个触发条件全中）

**#15「RmsNorm 是误差放大点」曾被撤回**，理由是*"那些张量主体余弦仅 0.11~0.48，废墟里比大小无意义"*。
本次余弦 **0.9202~0.9998**，全部远超 0.3 阈值 ⇒ **撤回理由针对的是当时那批已毁数据，不是机制本身**，
现象可以重新提出。#12（gamma 量化质量）与 #14（输出 encoding 余量）已被排除 ⇒ 旧机制解释不了，
本节给出的是**新机制**（固有传播）。

### 15.30.6 🔴 划界（不得越界引用）

- 四个样本**全部取自 part1b 的 layers.7/8**。§3.2 实测 **part2 才是支配项（41 of ~44 pp）**，
  本节**未在 part2 上验证**。"RmsNorm 固有放大"外推到 part2 属**未验证假设**。
- `A_float` 的计算依赖"CPU 臂 = 无 HTP 缺陷的参考"这一前提；CPU 是**浮点执行量化 DLC**（#35），
  不是定点参考 ⇒ 本节量的是"HTP 相对**浮点**执行的额外放大"，**不是**相对"正确定点"的。
- 本节**不**支持"量化配置无可为"——恰恰相反，②直接导出 #98（改善上游量化）。

### 15.30.7 副产物：一条新线索（#99）

诊断途中确立的事实：**RmsNorm 的 gamma 是 8-bit、per-tensor，全模型 205 个
（part1a 71 / part1b 44 / part2 90），且部署用的 per-row DLC 里实查确认仍是 `uFxp_8`**
⇒ **#47 从未碰过它**。其单层输出域注入误差 **1.31~1.55%**，
而**已被 per-row 优化过的** FC 权重只注入 **0.5050%**（§15.26 C 臂）。
结构原因：FC 权重误差在 K=3840 的求和中按 ~√K 平均掉，**gamma 逐元素相乘不发生平均**。
⇒ 登记为 **#99**，方案 `scripts/EXP_PLAN_GAMMA_BW.md`。


---

## 15.31 【2026-08-22】#99 RmsNorm gamma 8-bit：**次要但非零（3.75%）**，并划出「宿主消融」这一整类手段的天花板

方案 `scripts/EXP_PLAN_GAMMA_BW.md`（判据 + 止损**执行前定稿**）。纯宿主，**零设备占用**，
两臂合计 **10.6 分钟**（止损线 90 分钟）。

### 15.31.1 装置与三道门

三段 FP32 ONNX 链（`part1a→part1b→part2`），输入与 `fp32_noise_ref.py` 逐字节相同。
**B 臂只改 205 个 RmsNorm gamma**（part1a 71 / part1b 44 / part2 90），
按量化器规则 Q/DQ 回 float32，其余一字节不动。

实现要点：gamma 用**结构检测**定位（`Mul(A,init)` 且 `A=Mul(x,Reciprocal(...))`），不靠张量名；
只把这 205 个小张量**内联**进新 `.onnx`，其余保持 EXTERNAL 引用，
新文件写进同一目录 ⇒ 相对 `location` 仍有效，**不重写 13.7 GB 外部权重**。

| 门 | 要求 | 实测 | |
|---|---|---|---|
| **V2 操作化门** | encoding 规则复现 4 个从 DLC 实取的样本 | 相对差 5e-12~2e-11，offset 逐个精确 | **PASS 4/4** |
| **V1 单变量门** | 除 205 个 gamma 外初始化器逐字段相同 | 三段各 71/44/90，无意外改动 | **PASS** |
| **V0 装置门** | A 臂复现已存原点 `latents_fp32_s0.raw` | **0.000000%** | **PASS** |
| **V3 口径门** | FP32 参考前 1% 能量占比 ≤50% | **8.15%**，与 §3.1 记录值吻合 | **PASS** |

🔴 **V2 第一版 FAIL 4/4**，原因是我用 float64 算 scale，而 **DLC 里 scale 以 float32 存储**。
规则定稿为：`lo=min(min,0)`、`hi=max(max,0)`、`scale=float32((hi-lo)/255)`、`offset=round(lo/scale)`。
（约束 8：判据的操作化必须先用已知样本验证——这次是**验证拦住了错误**，不是事后补救。）

### 15.31.2 结果与落档

①**实测**：`E_gamma = 3.7460%`（step-0 噪声预测全量相对 L2，原点 = FP32）
⇒ `share = (3.746/15.99)² = ` **5.49%** 占权重量化地板能量。

按事前判据（🟢≥7.15% / 🟡2.26~7.15% / 🔶<2.26%）落 **🟡 次要但非零**。

🔴 **不得写「已排除」**：#52 实测 HTP 对权重量化步长敏感度 = CPU 的 **37.6 倍**，
本实验是**宿主浮点**测量 ⇒ **只是 HTP 影响的下界**。#9 正是栽在「用表示误差论证关闭」。

### 15.31.3 🔴 事后修正：「捆绑，边际成本≈0」这句判据后果是错的

判据 🟡 档原写「捆绑进下一次重量化，边际成本≈0」。**捆绑确实便宜，但它摧毁单变量归因**
（本项目已因非单变量栽过 #61/#72/#74/#95 四次）。正确表述分两种模式：

| 模式 | 做法 | 代价 |
|---|---|---|
| **归因模式**（要知道哪个改动起作用） | **必须单变量**，一个改动一次重量化 | 每个 3h |
| **推进模式**（只求尽快拉画质） | 可捆绑 | **结果不可归因，必须在文档里如此标注** |

### 15.31.4 🔴🔴 本节最有价值的产出：宿主浮点消融这一整类手段的天花板

两个**直接实测**点：

| | 权重 | 激活 | 执行 | E vs FP32 |
|---|---|---|---|---|
| CPU 参考 | 8-bit 量化 | **浮点**（#35：漏 `--enable_cpu_fxp`） | 浮点 | **15.99%** |
| 部署 HTP | 8-bit per-row | **uFxp_16** | HTP | **44.80%** |

⇒ **两者之间变了三样，不是一个变量。**
正交叠加假设下（**该假设未验证**，仅作量级估算）：

| 成分 | 能量 | 占比 | 等效 E |
|---|---|---|---|
| CPU 参考（权重量化，激活浮点） | 255.7 | **12.7%** | 15.99% |
| **「激活 16-bit 量化 + HTP 执行」合项（尚未分离）** | 1751.4 | **87.3%** | **41.85%** |
| 其中 gamma 8-bit | 14.0 | **0.70%** | 3.75% |

🔴 **订正记录**：本小节初稿把那 87.3% 写成「**HTP 执行超出部分**」，**是错的**——
CPU 参考是**浮点执行**（#35），连激活量化都没有，
所以这个合项里含着「激活 16-bit 量化」这份 **A16W8 的固有代价，不是 HTP 缺陷**。
当场订正（约束 2：不得写超出证据范围的量化表述）。

🔴 **这两项至今没有被分开过**，根因是 **#36**：本 SDK 给不出 A16W8 真定点 CPU 参考
（`error_code=703`，CPU 定点只支持 INT8）。已知的分离工具只有 **#37 的 numpy 定点模拟器**。

②**对手段选择的实际含义**：**「宿主浮点消融」只能直接量到那 12.7%**
⇒ **再做更多宿主浮点消融的边际价值很低。**

⚠️ **不得读成「权重量化工作只值 12.7%」——那是错的。**
#47 per-row 把 part1b 从 19.24% 降到 7.87%，远超其表示误差改善所能解释；
机制是 #52 的 **37.6 倍敏感度** ⇒ **权重量化改动同时作用于那 87.3%**，
只是其杠杆**在宿主浮点上量不出来**。这同时说明 #99 的 🟡 有可能被低估。


---

## 15.32 【2026-08-22】🔴🔴 查文档推翻了「选择性 FP16 不可行」的前提

**触发**：准备进 P1 的 #48/#51 时，按约束 5 先读文档（此前从未读过 wFxp_actFP 的原文）。
**代价**：约 20 分钟文档检索，**零设备、零重量化**。

### 15.32.1 被推翻的是什么

`EXP_PLAN_FC_VS_REST.md` §0 的否决：*"浮点化 FullyConnected 则必然出局"*
（FC 占 8-bit 权重 99.99% ⇒ +6154 MB ⇒ 撞 unsigned PD 3.3 GB 红线，#57）。

**该否决的隐含前提是「算子转 FP16 = 权重也转 FP16」。查 HTP OpDef supplement：不是。**

| 算子 | `in[0] 激活` \| `in[1] 权重` \| `in[2] bias` \| `out[0]` | |
|---|---|---|
| **FullyConnected** | `FLOAT_16` \| **`SFIXED_POINT_8`** \| `FLOAT_16` \| `FLOAT_16` | ✅ **权重留 8-bit，只有激活变 FP16** |
| RmsNorm | `FLOAT_16` ×4 | ✅ 全 FP16（gamma 合计 ~1.5 MB） |
| ElementWiseMultiply / Add | `FP16` ×3 | ✅ 全 FP16，无权重 |
| Softmax | `FP16` ×2 | ✅ 全 FP16，无权重 |

⇒ ②**体积不再是否决理由。** 剩余风险转为：**HTP 上 FP16 的速度代价 / PD 估算是否仍在红线内 /
量化器实际会不会产出该配置**——三条**全部未验证**。

### 15.32.2 两条文档原文（连同适用范围一起引）

1. `SNPE/general/tools.html`：`--keep_weights_quantized` … *"**Required to enable wFxp_actFP configurations**"*
   —— 🔴 **紧跟其后**：*"Note: These modes are **not supported by all runtimes**.
   Please check corresponding **Backend OpDef supplement** if these are supported"*。
   MAINLINE 原先只引了前半句。**必要条件≠充分条件**，支持性必须逐算子查后端表。
2. `QNN/general/quantization.html`：`--enable_float_fallback` 在 qairt-quantizer **已是 no-op**，
   由 `qairt-converter` 默认处理；机制是「**缺 encoding 的张量落成浮点**」；
   且它与 `--input_list` **互斥**。
   ⇒ 选择性 FP16 的正确做法是**给一份故意漏掉某些张量的 encodings**，不是加开关。

### 15.32.3 途中的一个解析错误（记下来避免重犯）

第一次解析 OpDef HTML 时用 `h.find('FullyConnected')` 定位，**抓到的是 Convert 算子里
提及"FullyConnected weights"的说明文字**，据此得出的"组合数"全错。
🔴 **正确做法是用 HTML 锚点 `id="<小写算子名>"` 定位算子段，并确认列头
`Configuration | in[0] | in[1] | in[2] | out[0]` 之后再读数据类型。**

### 15.32.4 方法论（本轮由用户提出并被证实）

**复杂且陌生的分析，先查官方文档/找同类案例，再推理。** 分工是：
- **机制 / 适用范围类问题**（开关存不存在、对量化图生不生效、后端认不认）⇒ **必须查文档**，
  推理不合法。本项目栽过：`fp16_relaxed_precision`（已废弃且只作用于 fp 模型）、
  `use_high_precision_fp16_sigmoid`（fp16 专用）、`use_per_channel_quantization`（只作用于 conv，**一直空转**）。
- **量级 / 占比类问题**（谁主导误差）⇒ 文档给不出，**只能实测**。

⇒ 实操纪律：**把推理链切短，每一段推理以一次实测或一条文档引用收尾。**
同日反例见 §15.31.4 的订正——连推三步没有落点，把「激活量化」错算到「HTP 执行」头上。


---

## 15.33 【2026-08-22】#48 wFxp_actFP 最小确认：**通路成立，收益不显著（🟡）**

方案 `scripts/EXP_PLAN_WFXP_ACTFP.md`（判据 + 止损**执行前定稿**）。
阶段 1 纯宿主，阶段 2 设备占用 **< 5 分钟**。

### 15.33.1 阶段 1（宿主）：三门全过

| 门 | 内容 | 实测 |
|---|---|---|
| **V1 产出门** | 从量化 DLC 读回数据类型 | `x=Float_16` / `val_1800=sFxp_8` / `out=Float_16` **PASS** |
| **V2 生效性门** | 两个不同取值 ⇒ 两个不同产物（§7.3） | 带/不带 `--keep_weights_quantized` 两对臂 md5 均不同 **PASS** |
| **V0 装置门** | `check_ctx_identity.py --diff` | 两臂 dspArch=**79**、vtcmSize=**8**、graphNames 一致 **PASS** |

🔴 **产物侧验证了文档结论**：wfxp context **14.88 MB** vs perrow **15.10 MB**
⇒ **权重确实留在 8-bit**，`EXP_PLAN_FC_VS_REST.md` 的体积否决（+6154 MB）作废。

### 15.33.2 🔴 配方：`--input_list` 必须换成 `--enable_float_fallback`

第一版 V1 **FAIL**：`x=uFxp_16 / out=uFxp_16`。根因是 **quantizer 拿 `--input_list`
把「故意不给 encoding」的激活又量化回去了**——converter 阶段的浮点回退被 quantizer 覆盖。

正确配方（文档原文：二者**互斥且必居其一**）：

```
qairt-converter  --quantization_overrides <只含 param_encodings 的 json> --float_bitwidth 16
qairt-quantizer  --weights_bitwidth 8 --use_per_row_quantization
                 --keep_weights_quantized --enable_float_fallback     # ← 不给 --input_list
```

⚠️ **执行中的偏离已明写在方案里**：定稿时把"允许的换配方范围"列错了（漏掉这一项），
用掉的仍是**一次**预算，判据与止损一字未改。

### 15.33.3 阶段 2（设备）：判据落 🟡

真值 = 原始 fp32 权重的精确 float64 matmul，两臂共用。

| 臂 | 全量相对 L2 | **主体相对 L2** | 主体余弦 | wall |
|---|---|---|---|---|
| perrow（**同装置重跑**，不引用历史值） | 0.4614% | **0.5006%** | 0.999987 | 0.9 s |
| **wfxp_actFP** | 0.4530% | **0.4911%** | 0.999988 | 1.0 s |

`E_wfxp / E_perrow = 0.981`（判据线 0.5）⇒ **🟡 有改善但不显著**。速度 1.19×（判据线 3×）。

②**结论**：在这个 FC 上，把激活从 `uFxp_16` 换成 `Float_16` 几乎没有收益
⇒ **残余误差基本全是权重 8-bit 的代价**（两臂权重同为 SFxp8 per-row）。

**一致性校验**：#90/§15.26 记的 per-row 表示代价 **0.5050%**，本次独立测得 **0.5006%**，吻合；
HTP 超出正确定点的 0.0680% 叠在其上确实是小项 ⇒ **装置与历史数据自洽**。

### 15.33.4 🔴 划界（不得越界引用）

- **单算子**、**part1b 的 `linear_99`**、#30 已划界「探针输入是良态的」。
- 🔴 **本次真值前 1% 元素只占 ‖a‖² 的 18.13%，而 `unified` 是 99.83%** ——
  输入良态得多。**不得据此推断「激活量化在全模型上也几乎免费」，
  更不得用它去否定 #53（激活校准）。**
- 不回答「全模型 PD 估算是否超红线」。见 #101：单算子上
  `spillFillBufferSize` 从 **1.97 MB 涨到 88.67 MB（45 倍）**，全模型量级未知。


---

## 15.34 【2026-08-22】#53 激活校准：**代价门否决**，且「收紧激活量程」这条杠杆在 SDK 选项内已用尽

方案 `scripts/EXP_PLAN_ACTCAL_G0.md`（判据 + 止损**执行前定稿**）。纯宿主，**零设备占用**，约 20 分钟。

### 15.34.1 装置：让 SDK 自己在目标张量上算，不复现算法

原设计是「复现 mse/sqnr/entropy 再外推到 `unified`」，有 V2 操作化门失败的风险。
执行时**加强**为：造一个输入就是 `unified` 的极小探针（`MatMul` 到 32 列），
用 `unified_fp32.raw` 作校准集，**让 `qairt-quantizer` 自己算**，再读回 encoding。
⇒ **零近似，V2 门不再必要。**（方案里已明写这处加强。）

**V1 参照门**：复算 min-max ⇒ 全量 **0.1864%** / 主体 **4.4741%**，
与 CLAUDE.md 约束 4·补 所载的 0.1884% / 4.5227% 吻合 ⇒ **口径与张量都对**。
**V3 口径门**：`unified` 前 1% 能量占比 **99.83%**，与 §3.1 精确吻合。

### 15.34.2 结果：三种方法与 min-max 几乎完全相同

| 方法 | E_full | **E_bulk** | 量程宽度 | 被钳元素数 |
|---|---|---|---|---|
| **min-max**（当前部署） | 0.1864% | **4.4741%** | 6332.10 | 1 |
| mse | 0.1879% | 4.4827% | 6344.43 | 1 |
| sqnr | 0.1911% | **4.4571%** | 6307.37 | 11 |
| entropy | 0.1864% | 4.4741% | 6332.10 | 1 |

判据（🟢 需 `E_bulk < 2.2371%` 且 `E_full ≤ 0.3727%`）：**三者全部落「不值得单独立项」**。
⚠️ `entropy` 给出的 encoding 与 min-max **逐位相同**。

### 15.34.3 ②机制结论（本节的真正价值）

**这三种方法根本不收紧量程**——被钳元素 1/11/1 个（总量 1585 万），量程宽度差异 <0.6%。
⇒ scale 几乎不变 ⇒ **主体误差纹丝不动**。

⇒ **在 SDK 提供的激活校准方法里，唯一会「拿量程换分辨率」的只有 `percentile`，
而它已被实测证明是灾难**（主体余弦 0.7986 → 0.4793；事后 G0-cost 显示纯表示误差 32.02%
已超当时端到端 7.87%）。

⇒ ②**「收紧激活量程」这一整族手段，在 SDK 选项范围内已经用尽。**
与 #67（激活才是离群值主导的，钳 0.0261% 丢 98.75% 能量）
与 #68（8-bit 下 min-max 往往就是 MSE 最优）方向一致。

### 15.34.4 划界

- 本节只量了 **`unified` 一个张量**（选它是因为它是决策相关的病态张量，集中度 99.83%）。
  未扫描其他激活张量。
- 本门**只能否决，不能证明有收益**（#52：表示误差不是 HTP 误差的预测量）。
  本次结论是**否决**，正落在该门有效的方向上。
- `--act_quantizer_schema`（对称/非对称）**未测**，本节不覆盖。


---

## 15.35 【2026-08-22】#102 选择性 FP16：**装置全通，数值门崩** —— `Pow(x,2)` 在 FP16 里溢出

方案 `scripts/EXP_PLAN_ACTFP16.md`（判据 + 止损**执行前定稿**，含 §三·补 偏离声明）。

### 15.35.1 前三道门全过

| 门 | 结果 |
|---|---|
| **G1 注入门** | **PASS**：620 个激活 `Float_16` / 24 个保持 `uFxp_16`；权重 59 个 `sFxp_8`、409600 条 per-row scale，与部署一致 |
| **G2 容量门** | **PASS**：context **1558.0 MB** vs 参照臂 1472.86 MB（**仅 +5.8%**），**未撞 PD 红线**。宿主编译 36.2 min vs 参照 16 min |
| **G3 装置门** | **PASS**：dspArch **79** / vtcmSize **8**，与参照臂画像一致 |

⚠️ 但 `spillFillBufferSize` **222.8 MB → 1224 MB（5.5×）**。
context 1.56 GB + spillFill 1.22 GB = **2.78 GB，距 #57 的 3.3 GB 红线只剩约 0.5 GB**
⇒ **推到更大的 part2 时这是硬约束。**

### 15.35.2 🔴 G4 数值门 FAIL：38% 输出是 inf/nan

| 臂 | 全量 | 主体 | 主体余弦 | inf/nan |
|---|---|---|---|---|
| perrow（**同装置重跑**） | 7.8665% | 78.7714% | **0.7986** | 0 |
| actfp16 | nan | nan | nan | **6,072,398 / 15,851,520 (38%)** |

✅ **已知答案门通过**：参照臂重跑给出 0.7986 / 7.8665%，与 #47 历史记录逐位吻合
⇒ 装置与流水线可信，失败不是装置问题。

### 15.35.3 根因：**筛选判据错了**（机制解释已于当日订正，见 15.35.7）

🔴 **初稿写的是**：*"图里 RmsNorm 是拆开的算子，`x²` 是独立的 FP16 张量"* —— **该表述已作废**，
它是**拿 ONNX 结构推理 DLC 行为**。实查：**DLC 把 RmsNorm 融合成单算子**
（part1b/part2a 的 `pow_*`/`mean_*`/`rsqrt_*` 张量数**均为 0**，part2a 显示 `RmsNorm: 46` 个）。

- FP16 上限 65504 ⇒ **平方不溢出要求 `|x| ≤ sqrt(65504) ≈ 255.9`**
- 实测：**44 个 RmsNorm 输入里 37 个（84%）超标**，最极端 `linear_153` `|a|max=238418` ⇒ `x² = 5.68e10`

我筛的是「张量自身 `|a|max ≤ 65504`」，**漏掉了下游算子会放大量级这一层**。

⇒ ②**RmsNorm——恰恰是误差放大发生的地方（#97）——在这张图里基本没法走 FP16。**
按约束 4 区分：这是**方向错误**，不是执行方法错误。

### 15.35.4 可复用纪律（已入指南 §27.9）

**选 FP16 张量时，判据不能只看张量自身的动态范围，必须看它在下游算子里被放大到多少。**
最少要检查：`Pow(·,2)` / `Mul(x,x)` / 累加类算子。
一般形式：**对每个候选张量，沿其消费者算子推算「最大中间量级」，再与目标浮点格式的上限比。**

### 15.35.5 处置

- **#102 按事前止损关闭**（§六：落 ❌ ⇒ 关闭，不做 BF16、不做其他段）。
- **#103 登记**：BF16 指数范围同 FP32 ⇒ 无此溢出问题；尾数 8 bit ⇒ 相对精度 ~0.4%，
  仍比 uFxp_16 在 `unified` 上的 4.47% 好约 11 倍；OpDef 实查各相关算子**都有 BF16 行**。
  🔴 **是否追属于方向性取舍 = 用户的成本优先级，不自行决定。**

### 15.35.6 补测：溢出是**全模型性质**（零成本，读现成 encoding）

| 段 | RmsNorm 输入数 | 平方后溢出 FP16（`|x| > 255.9`） | 占比 |
|---|---|---|---|
| part1b | 44 | 37 | **84%** |
| **part2** | 90 | **81** | **90%** |

part2 最极端：`linear_15` `|a|max=115849`（`x²=1.34e10`）、`linear_23` 85837、`linear_39` 82794。

⚠️ 注意对比：若只看「张量自身是否超 65504」，part2 只有 **8/1389 = 0.6%** 超标，
**看起来 99.4% 都能转 FP16** —— 这正是让我踩坑的那个假象。
按正确判据（看 `Pow` 下游）则是 **90% 的 RmsNorm 走不了 FP16**。

⇒ ②**FP16 这条路在全模型都走不通，#102 关得对**；BF16（#103）是这一族里唯一的幸存者。


---

## 15.36 【2026-08-22】#103 BF16：**G0-cost 过了，但工具设计封死** —— 「选择性浮点」整族关闭

方案 `scripts/EXP_PLAN_ACTBF16.md`（§〇 明写本方案**推翻了 #102 止损里的「不做 BF16」**及理由）。
纯宿主，**零设备占用**，从提出到关闭约 15 分钟。

### 15.36.1 G0-cost 代价门：**过了**，而且很有说服力

| | 全量表示误差 | **主体表示误差** | 动态范围 |
|---|---|---|---|
| uFxp_16 min-max（当前部署） | 0.1864% | **4.4741%** | — |
| FP16 | 0.0167% | 0.0209% | 🔴 `x²=5.68e10` 溢出（#102 死因） |
| **BF16** | **0.1315%** | **0.1675%（好 26.7 倍）** | ✅ `BF16(238418²)=5.69e10` 有限 |

且 BF16 范围足够 ⇒ **不需要钉任何激活张量**（#102 钉了 24 个名字），配方反而更干净。

### 15.36.2 🔴 死因：BF16 与量化互斥（官方文档）

converter 报错：

```
ERROR - Encountered Error: Currently, BF16 isn't supported with models having
        quantization overrides or QDQ nodes.
```

官方文档 `QNN/general/converters.html` §"BF16 Graph generation use cases" 逐字：

> *"BF16 graph generation is supported through QNN and QAIRT for conversion of ONNX models
> **without QDQ nodes or overrides**."*
> *"**Quantization isn't supported for BF16 graphs.**"*

⇒ 要 BF16 就得**整张图 BF16、权重也 2 字节**（当前 8-bit 是 1 字节）⇒ 权重翻倍
⇒ 与 **#16** 实测的 A16W16 同一量级死法（part1a+1b **7279 MB vs 实测被杀线 7300 MB**，
且加载主导致生图时间翻倍）。

### 15.36.3 ⇒ 「把激活换成浮点」整族关闭（三条独立证据）

| 路径 | 死因 | 证据类型 |
|---|---|---|
| FP16 选择性（#102） | `Pow(x,2)` 溢出（part1b 84% / part2 90% 的 RmsNorm 输入超标） | **设备实测** 38% inf/nan + 机制 |
| BF16 选择性（#103） | 与量化互斥 | **官方文档** + 报错实证 |
| 全图浮点 | 权重翻倍撞内存被杀线 | #16 已实测 |

⚠️ **不得读成「浮点没用」**：表示误差侧浮点是压倒性的（BF16 好 26.7 倍、FP16 好 214 倍）。
**是工具与内存不给做，不是想法错。**
⇒ 若将来 SDK 放开「BF16 + 量化」，**这条应当第一个重开**。

### 15.36.4 方法论记录：我推翻了自己的止损，以及为什么这次是对的

`EXP_PLAN_ACTFP16.md` §六 写着「落 ❌ ⇒ 不做 BF16」。本轮推翻了它。

**推翻的理由必须可检验，否则止损就形同虚设**。本次的理由是：
那条止损写于**不知道失败模式**时，目的是防"一条条换花样试"；
而失败已被**精确诊断为动态范围不足**，BF16 的指数范围同 FP32 ⇒
**它是针对该诊断结论的直接解，不是另一个花样**。

**做法上没有绕过纪律**：新开方案、重新定稿判据与止损、把推翻动作明写在案（约束 2）。
**结果也证明这次推翻是划算的**：15 分钟、零设备，换来整族的确定性关闭
——如果当初真的"不做 BF16"，这一族会一直挂在台账上当悬案。


### 15.35.7 🔴 当日订正：#102 的**机制解释**降级为未验证假设（实测事实不变）

| 内容 | 类别 |
|---|---|
| FP16 版设备输出 **38% inf/nan**（6,072,398 / 15,851,520） | ①**实测** |
| RmsNorm 输入 `|a|max` 达 238418，平方 5.68e10 **远超 FP16 上限 65504** | ①**实测** |
| part1b **37/44 (84%)**、part2 **81/90 (90%)** 的 RmsNorm 输入平方后超上限 | ①**实测** |
| **DLC 把 RmsNorm 融合成单个算子，图里无独立 `x²` 张量** | ①**实测（当日新查）** |
| ~~"溢出发生在独立的 `Pow` FP16 张量上"~~ | 🔴 ③**未验证假设，措辞作废** |
| "溢出发生在融合算子内部的平方和累加上" | ③**未验证**：融合算子内部是否高精度累加，本项目未查证 |

**⇒ #102 的关闭结论不变**（FP16 实测不可用），**但不得再宣称已知其精确机制。**

**方法论教训（与 #61/#73 同类）**：**要解释后端行为，必须查后端制品（DLC / OpDef），
不能拿源模型（ONNX）的结构代替。** 本次是自查发现的——
起因是在 part2a 的 DLC 里找不到 `mul_22`/`pow_6`，顺手核对了 part1b，才发现融合。


---

## 15.37 【2026-08-22】~~part2 算子级归因：RmsNorm 主导~~ 🔴 **结论已被 §15.39 推翻**

> 🚫 **本节的归因结论作废（2026-08-23）。** 原因：**探针太稀** ——
> 链条从 `linear_5` 直接跳到 `mul_23`，中间的 `mul_21`（SwiGLU 输出）**不在探针集里**，
> 其 **×9.14** 的放大被整个记到了 RmsNorm 头上。
> ⚠️ **V2 可比性门当时 12/12 全过**，仍未能拦住 —— **门过 ≠ 探针够密**。
> **本节的原始测量数据仍然有效**（每个张量的误差/余弦是实测），失效的是**按算子类型的归因**。
> 正确结论见 **§15.39**。

方案 `scripts/EXP_PLAN_P2_ATTR.md`（判据事前锁定；§三 含"改用 FP32 ONNX 作真值"的改道声明）。

### 15.37.1 装置

- **HTP 侧**：`qnn-net-run --dlc_path part2a_fixed_perrow.dlc --set_output_tensors <12 个张量>`
  ⇒ **不必焊 tap、不必重量化**（今日实测订正：`--set_output_tensors` 要求**在线建图**，
  不是"图已 finalize"，见 MAINLINE §7.3 订正）。设备端在线建图约 **25 分钟**。
- **真值侧**：**FP32 ONNX + onnxruntime**（原计划的 SNPE CPU 参考行不通，见 **#105**：
  `error_code=202 Dequantization of axis-quant tensor is not supported for FullyConnected`）。
- 段内隔离口径（喂 FP32 输入），与 §3.2 的 41.24%/28.52% 基线可比。
- 依赖链由 ONNX 反推（**不按 Id 顺序**，约束 8），**V0 门通过**：每个探针都能追到上游探针或图输入。

### 15.37.2 结果（V2 可比性门 12/12 全过，主体余弦均 >0.9）

| tensor | op | 主体相对 L2 | 主体余弦 | Δcos |
|---|---|---|---|---|
| mul_1 | RmsNorm | 5.5150% | 0.998483 | +0.001517 |
| linear_1 | FC | 2.5694% | 0.999680 | −0.001198 |
| mul_4 / mul_6 | RmsNorm(q/k) | 2.76% / 2.80% | 0.99962 / 0.99961 | +0.000062 / −0.001126 |
| scaled_dot_product_attention | MatMul | 3.5906% | 0.999366 | +0.000253 |
| linear_4 | FC | 2.6892% | 0.999644 | −0.000278 |
| mul_16 | RmsNorm | 3.7392% | 0.999301 | +0.000343 |
| add_8 | 残差 Add | 5.8614% | 0.998292 | +0.001009 |
| mul_19 | RmsNorm | 7.5482% | 0.997165 | +0.001127 |
| linear_5 | FC | 3.6112% | 0.999349 | −0.002184 |
| **mul_23** | **RmsNorm** | **45.2519%** | **0.905659** | **+0.093690** |
| linear_9 | FC | 12.7038% | 0.992448 | +0.005844 |

**按算子类型聚合**：

| 类型 | Δcos 总和 | 个数 |
|---|---|---|
| **RmsNorm(Mul)** | **+0.095613** | 6 |
| MatMul/FC | +0.002436 | 5 |
| 残差 Add | +0.001009 | 1 |

判据线是「RmsNorm ≥ 2 × 其余合计」，实测 **28 倍** ⇒ ✅ **与 part1b 同机制**。

### 15.37.3 ②结论

1. **part2 的误差增长同样由 RmsNorm 主导，FC 不放大**（多处 Δcos 为**负**，即 FC 输出的余弦
   反而高于其输入——与 #97 在 part1b 测到的 FC ×1.04 一致）。
   ⇒ **§3.2「part2 是支配项」不意味着 part2 有不同机制**；它只是同一机制作用在更长的链上。
2. 🔴 **误差高度集中在单个算子**：`mul_23` 一个 RmsNorm 贡献了 **+0.093690 / +0.095613 = 98%**
   的误差增长，主体余弦 0.999349 → **0.905659**，主体相对 L2 3.61% → **45.25%**。
3. `mul_23 = RmsNorm(linear_7) * layers.15.ffn_norm2.weight`，
   而 **`linear_7` 是 massive activation 张量**（FP32 真值 `std=176.27`、`|a|max=77704`；
   量化 encoding 记录 76493）——**与 part1b 里放大最猛的 `linear_105 → mul_333`
   （|a|max=175619，A_htp=×4.05）结构完全一致**。

### 15.37.4 待完成（方案 §四 的并行条款）

对 `mul_23` 检验 `A_htp / A_float ≤ 1.3`（float64 精确复现 + **DLC 量化 gamma**，同 §15.30 装置）
⇒ 判定该放大是**固有**还是 **HTP 缺陷**。需要 `linear_7` 的 HTP 值，第二轮设备取数已启动。


---

## 15.38 【2026-08-23】`mul_23` 的放大是固有的、HTP 算术无误差 🔴 **但"收口"的结论已被 §15.39 推翻**

> 🚫 **本节关于 `mul_23` 的测量全部有效**（`A_htp/A_float=1.002`、HTP 超出仅 0.7587%、余弦 0.999971）。
> **作废的是由它推出的"已拿到能拿的全部"** —— 因为 `mul_23` **不是主要矛盾**（见 §15.39）。
> 教训：**在一条未经加密的链上测出"某算子无问题"，不能推出"整条链无问题"。**

方案 `scripts/EXP_PLAN_P2_ATTR.md` §四 并行条款。装置沿用 §15.30（float64 精确复现 + **DLC 量化 gamma**）。

### 15.38.1 判定

| 量 | 值 |
|---|---|
| `A_htp` | **1.449** |
| `A_float`（精确 float64，同输入、同量化 gamma） | **1.446** |
| **`A_htp / A_float`** | **1.002**（事前判据线 ≤1.3 ⇒ 固有） |
| 🔴 **直接判据**：HTP 输出 vs「同输入同 gamma 的精确 float64 计算」 | **超出 0.7587%，余弦 0.999971** |

V2 余弦门 PASS（输入 0.9519 / 输出 0.9057 / 模拟 0.9060）；
V3 复现门 PASS（**0.0000%**）。

⇒ ②**HTP 在该算子上的算术本身是准确的**（超出仅 0.76%）。

### 15.38.2 🔴 口径修正（数据到达**之前**发现并声明）

初版把 `A_float` 写成 `relerr(rmsnorm(x_fp32, g_q), rmsnorm(x_htp, g_q))` —— 两臂都用量化 gamma
⇒ **gamma 量化代价在 A_float 里被约掉，却仍留在 A_htp 的分子里**（真值来自 FP32 ONNX，用 fp32 gamma）
⇒ 比值被系统性抬高，**偏向"HTP 缺陷"的结论**。

修正为：`A_float` 与 `A_htp` **共用同一真值 `y_fp32`**，模拟臂用 HTP 实际使用的量化 gamma。
并增设**更强的直接判据**（不依赖比值）：`relerr(rmsnorm(x_htp, g_q), y_htp)`。
**该修正在看到设备数据之前完成并写入脚本注释**，判据阈值未改。

⚠️ 相关：V3 门的 gamma 选择取决于**参考系** ——
本次真值来自 **FP32 ONNX（fp32 gamma）**，故 V3 必须用 fp32 gamma（实测 0.0000%）；
§15.30 那次参考是**量化 DLC 的 CPU 执行（量化 gamma）**，故必须用量化 gamma。**用错就复现不出来。**

### 15.38.3 整条链的完整解释

| 环节 | 主体相对 L2 | 说明 |
|---|---|---|
| `linear_7` 抵达 | **31.22%**（余弦 0.9519） | 上游累积 |
| ├ 其中**它自身的 uFxp_16 表示代价** | **14.08%** | 🔴 见 15.38.4 |
| └ 其余约 17 pp | | 更上游带入 |
| RmsNorm 放大 ×1.449 ⇒ `mul_23` | **45.25%**（余弦 0.9057） | **固有**（float64 给 ×1.446） |
| HTP 自身算术 | **0.76%** | 可忽略 |

### 15.38.4 🔴 病灶被定位到具体机制：**massive activation 毁掉 16-bit 的分辨率**

`linear_7` 的激活 encoding：`bitwidth 16, min −49337.46, max 76492.65, scale **1.92004**`。
**16 bit 本该给出 0.01% 级的表示误差，实测主体却是 14.0797%。**
根因：`|a|max` 被 massive activation 撑到 76493，而主体值远小于此
⇒ **量程被离群值占据，主体只分到极少的量化级**（#67 机制，首次在 part2 关键张量上量化）。

另：**gamma 量化本身贡献 1.3326%**（V3 对照行：fp32 gamma 复现 0.0000% vs 量化 gamma 1.3326%）。
仅「`linear_7` 量化 + gamma 量化」两项，在**精确 float64** 下就把误差推到 **21.20%**（放大 1.51 倍）——
**完全没有 HTP 参与**。

### 15.38.5 ②⇒ 主线结论

1. **#98 最终确认**：误差 = 上游累积 + RmsNorm 固有数学放大；**没有「HTP 侧可修」的东西**。
2. **病灶明确**：A16 定点在 massive activation 张量上分辨率不足（单张量 14.08%）。
3. **药也明确**：浮点激活。**但整族被工具链挡死**（§3.7）：
   FP16 动态范围不够（实测 38% inf/nan）；BF16 **官方明写与量化互斥**；全图浮点撞内存（#16）。

⇒ ②**在 QAIRT 2.48 + A16W8 + v79 这个组合下，我们已经拿到了能拿的全部。**
**这是一个完整的答案，尽管不是想要的那个。**


---

## 15.39 【2026-08-23】🟢🟢🟢 **#106 手术式 FP16：有效，主体余弦 0.7099 → 0.9455**

方案 `scripts/EXP_PLAN_FP16_SURGICAL.md`（判据 + 止损**执行前定稿**）。

### 15.39.1 装置（两臂同配方，唯一变量 = 8 个张量）

两臂**都由我重建**，配方完全相同，只差"那 8 个张量给不给 encoding"：

- **ctrl**：718 个激活全部给 encoding ⇒ 零浮点
- **fp16**：其中 **8 个不给**（7 个 SwiGLU 输出 `mul_{21,46,71,96,121,146,171}` + `linear_55`）

⚠️ 两臂都必须打「对称修正」补丁（#109），否则 `node_MatMul_106` 校验失败建不出来。

四道门：**G1 PASS**（8 个白名单全为 `Float_16`，权重 61 个仍 `sFxp_8`）、
**G2 PASS**（1443.5 / 1440.3 MB，未撞 PD 红线）、
**G3 PASS**（两臂 dspArch 79 / vtcm 8，画像一致）、**G4 PASS**（两臂 inf/nan = 0）。

### 15.39.2 结果

| 臂 | 主体相对 L2 | **主体余弦（vs FP32 真值）** |
|---|---|---|
| 部署 DLC（在线建图，仅作参照） | 73.7440% | 0.719663 |
| **ctrl**（重建，零浮点） | 74.9263% | **0.709929** |
| **fp16**（8 个张量转浮点） | **33.0326%** | **0.945511** |

**主体余弦提升 `+0.235582`**（事前判据线 **≥0.03** ⇒ **超出 7.8 倍**）；
主体相对 L2 **74.93% → 33.03%，不到一半**。

### 15.39.3 有效性论证（防止把装置噪声读成效应）

- **基线保真**：ctrl 与部署产物"离真值的距离"几乎相同（余弦 0.7099 vs 0.7197，**差 0.0097**）
- **信噪比**：**效应 0.2356 ÷ 基线不确定度 0.0097 = 24 倍**
- 两臂输出有 **12,737,077 / 15,851,520** 个元素不同 ⇒ 是两份真实结果，非复制
- ⚠️ ctrl 与部署**彼此**余弦仅 0.8057 —— 不矛盾：两者都被量化噪声主导（离真值余弦都 ~0.71），
  各自噪声互不相关，故彼此差得远、离真值一样远

### 15.39.4 机制闭环

这与 #106 的病灶诊断完全对上：`mul_21` 等 SwiGLU 输出每个主体元素只分到 **0.06~0.82 个量化级**，
纯表示误差 **53.77%**（与 HTP 实测 54.40% 吻合）。把这 7 个换成 FP16（表示误差 ~0.02%）
⇒ part2a 输出误差腰斩。**诊断 → 手术 → 效果，三者自洽。**

### 15.39.5 🔴 划界（不得越界引用）

- **唯一变量是"8 个张量不给 encoding"，但浮点实际扩散到 153 个激活**（8 + 其输入锥，约占 21%）
  ⇒ 措辞只能是「**这一组改动**」，不得说成「只改了 8 个张量」。
- **只做了 part2a**。不得外推四段 / 端到端 / 成图质量
  （#69/#70/#84：标量尺与成图质量反相关，**最终必须人工看图**）。
- 速度：ctrl 18.4 s / fp16 20.1 s（含加载），**1.09 倍**，远在 #16 的 3 倍红线内。


---

## 15.40 【2026-08-23】🟢🟢🟢 **四段手术式 FP16 端到端：`E_all` 43.41% → 30.68%，差距关掉 46.4%**

### 15.40.1 结果（标定尺，唯一有刻度的量）

标定：**15.99% 成好图 ｜ 47.07% 橙色色块**（§3.1）

| 配置 | **E_all** | 主体余弦 | inf/nan |
|---|---|---|---|
| per-row 四段（基线，同装置重跑） | **43.4094%** | 0.903123 | 0 |
| **四段手术式 FP16** | **30.6819%** | **0.948845** | 0 |

⇒ **改善 12.73 pp**；到 CPU 参考地板（15.99%）的差距 **27.42 pp → 14.69 pp，关掉 46.4%**。
⇒ **自 #47 per-row 以来第一次真正的端到端推进。**

### 15.40.2 做了什么

四段各自把「量化分辨率被摧毁且 FP16 装得下」的 **SwiGLU 输出**转成 FP16，其余保持定点：

| 段 | 白名单 | context | G2 | G3 |
|---|---|---|---|---|
| part1a | 8（11 个 SwiGLU 中） | 2367.8 MB | ✅ | 79/8 ✅ |
| part1b | **2**（8 个中，6 个量程超 FP16） | 1471.8 MB | ✅ | 79/8 ✅ |
| part2a | 8 | 1440.3 MB | ✅ | 79/8 ✅ |
| part2b | 8（8 个全通过） | 1490.0 MB | ✅ | 79/8 ✅ |

**合计 26 个张量**。速度：单步四段合计约 84 s（part1a 33 / part1b 15 / part2a 18 / part2b 18）。

### 15.40.3 机制闭环（诊断 → 手术 → 效果三者自洽）

1. **诊断**（§15.37/#106）：SwiGLU 输出 `silu(x)*y` 量程被离群值撑开，
   `mul_21` 每个主体元素只分到 **0.21 个量化级**，纯表示误差 **53.77%**
   —— 与 HTP 实测 **54.40%** 吻合（两条独立路径）
2. **单段验证**（§15.39）：part2a 主体余弦 **0.7099 → 0.9455**（+0.2356，判据线 0.03）
3. **端到端**（本节）：`E_all` **43.41% → 30.68%**

### 15.40.4 成图（🔴 亲自看图，非转述）

`scratch_runs/htp_inloop_transformer_fp16seg.png`：橘猫坐在木桌上，
**毛发质感自然**（per-row 基线那张是尖刺状、过锐的"crunchy"感）、背景干净、
**多出一把椅子**、桌面木纹与反光完整。

### 15.40.5 🔴 划界

- **不是"只改了 26 个张量"**：浮点沿输入锥扩散（part2a 实测 153 个激活为 `Float_16`，约 21%）
  ⇒ 措辞是「这一组改动」。
- **part1b 只救到 2/8** —— 6 个 SwiGLU 量程超 FP16（`mul_481` 62,259 等）。
  **该段仍是短板**，也是下一步的明确目标。
- 选择规则两套：part2a 用「**实测** `|a|max` × 2.0」，其余三段用「**标定** × 4.0」保守代理
  （校准低估 1.83~1.95 倍 ⇒ ×4 ≈ 实测 ×2.05）。**后者更严，纳入更少。**


---

## 15.41 【2026-08-23】🟢 **三层指标面板建立 + 最优配置确定 + 确定性验证**

### 15.41.1 顶层目标的澄清（本节最重要的一条）

**目标是「与原始模型的一致性」，不是「图好不好」。** 此前长期用 step-0 `E_all` 一个数
笼统评价，且因 #69/#70/#84 的「标量尺与成图质量反相关」而不敢用它做判断。

🔴 **划界订正**：#69/#70/#84 那条警告是在回答「**图好不好**」时成立；
**当前目标是「一致性」，对一致性这些指标是恰当的。** 两个问题不要混。

### 15.41.2 三层指标（工具：`scripts/panel3.py`，自带同噪声同 caption 的 md5 前提核对）

| 配置 | L1 `E_all` | L2 终点 latents | **L3 图像 PSNR** | L3 像素 L2 |
|---|---|---|---|---|
| per-row 基线 | 43.41% | 69.34%（0.7848） | **11.93 dB** | 37.04% |
| **FP16 26 张量（最优）** | **30.68%** | **57.70%（0.8463）** | **15.53 dB** | **24.46%** |
| FP16 29（part1b 2→5） | 39.09% | 65.78%（0.8056） | 14.32 dB | 28.13% |
| FP16 32（再加 part1a 8→11） | 39.11% | 64.59%（0.8188） | 14.87 dB | 26.40% |
| **FP16 26 复现** | **30.68%** | **57.70%（0.8463）** | **15.53 dB** | **24.46%** |

①**三层排序完全一致** ⇒ `E_all` 作为**序数代理**已验证有效（可用于快速筛选），
但**报告结论必须给 L3**。

✅ **确定性验证**：26 张量配置**重建后复现，五个数逐位相同**
⇒ 量化→建图→设备执行→VAE 整条链确定性，**配置间对比可信**。

### 15.41.3 前提核对（实测，非文档转述）

三配置的 `s0/latents.raw` md5 同为 `99ff38fdde23823c`、
`const/caption.raw` md5 同为 `2ae471cca0e8e90d` ⇒ **同噪声同 caption 成立**。

### 15.41.4 🔴 误差在图上的分布不均匀（亲自逐张看图）

| | 猫 | 桌子 | 背景 |
|---|---|---|---|
| **FP32 参考** | 橘猫正面坐姿 | **圆桌** | **丰富**：墙上挂画、绿植、左右木柜、椅背 |
| FP16 26 张量 | **高度相似** | 方桌 | 近乎空白，仅一个椅背 |
| FP16 32 张量 | 相似 | 方桌 | 完全空白，无椅子 |

②**主体受 caption 强约束 ⇒ 收敛；背景是弱约束自由度 ⇒ 轨迹一偏就换成完全不同的场景。**
⇒ 这解释了 PSNR 只有 15.53 dB（大面积背景不同），**指标没有误导**；
⇒ 且**评估一致性时背景比主体灵敏**，看图时应优先看背景。

### 15.41.5 当前位置

| | L3 PSNR |
|---|---|
| 今日起点（per-row） | 11.93 dB |
| **今日最佳（26 张量）** | **15.53 dB** |
| 参照上限（我们 FP32 vs 官方） | **41.61 dB** |

**推进 +3.60 dB，距参照上限仍差约 26 dB。**


---

## 15.42 【2026-08-24】🔴 dtype 机制验证：**工具有效，但我的选择规则被推翻两次**

### 15.42.1 结果

| 配置 | 浮点张量数 | L1 `E_all` | L2 终点 | **L3 PSNR** |
|---|---|---|---|---|
| **FP16 26 张量（旧机制，仍是最优）** | **509** | **30.68%** | **57.70%** | **15.53 dB** |
| dtype 26 张量（新机制） | **102** | 42.12% | 64.37% | **14.56 dB** |

**把浮点张量从 509 压到 102 ⇒ 变差 0.97 dB。**

### 15.42.2 ①工具本身是好的，②推论是错的

- **dtype 机制有效**：四段 G1 全 PASS，白名单精确落地
  （part1a 8→32、part1b 2→8、part2a 8→30、part2b 8→32 个 Float_16，含自动 Convert）。
  它提供了**精确指定任意一组张量**的能力——这是它的真正价值。
- 🔴 **被推翻的是「浮点越少越好」**。该推论源自 #112（FP16 毁良态张量），
  **端到端实测证否**：那多出来的 407 个浮点张量是**净有益**的。

### 15.42.3 🔴 交叉点分析（宿主实测，16 个张量）

对每个张量比较「uFxp_16（用其部署 encoding）」与「FP16」的**主体表示误差**：

| 张量 | `|a|max` | uFxp16 主体 | FP16 主体 | 倍数 |
|---|---|---|---|---|
| `mul_4` | **10.3** | **0.0069%** | 0.0206% | 定点赢 3× |
| `mul_6` | **16.3** | **0.0101%** | 0.0208% | 定点赢 2× |
| `mul_19` | 21.5 | 0.5192% | 0.0204% | FP16 赢 25× |
| `linear_1` | 98.0 | 0.0575% | 0.0209% | FP16 赢 2.8× |
| `add_8` | 6,206 | 4.3243% | 0.0209% | FP16 赢 **207×** |
| `mul_21` | 9,818 | **53.7738%** | 0.0205% | FP16 赢 **2620×** |

⇒ ②**FP16 在 16 个里赢 14 个。** 机制：
**FP16 的主体误差恒定在 ~0.021%（其相对精度地板），而 uFxp_16 从 0.007% 到 53.77% 剧烈波动。**
**只有当张量量程小到定点步长已细于 0.021% 时，定点才赢**（本样本中的分界在 `|a|max` ≈ 16~21）。

⚠️ **溢出必须单独筛，不能靠这个指标**：`linear_7`（量程 77,704）的 FP16 主体误差看似只有 0.0208%，
**是因为主体口径排除了前 1%，溢出成 inf 的元素被排除在外**。

### 15.42.4 这解释了今天全部三个结果

| 观察 | 修正后的解释 |
|---|---|
| 旧机制 509 浮点 > 新机制 102 浮点 | **浮点越多越好**（FP16 几乎处处更优） |
| 手术式 26 张量 > 全定点（+3.60 dB） | 只治了最坏的几个，收益已很大 |
| #112：加 part1b 那 3 个反而变差 | 🔶 **与新规律冲突，需重查**（那 3 个量程 17k~26k，按新规律应受益） |

### 15.42.5 ⇒ 下一步方向：**最大化 FP16**

选择规则从「外科手术式少量张量」改为「**除两类外全转**」：

1. 排除**溢出风险**：实测 `|a|max × 安全余量 > 65504`（配 §EXP_PLAN_CLIP_FP16 的显式 Clip 可救回一部分）
2. 排除**定点已更优**：量程极小、uFxp_16 主体误差已低于 ~0.021%

**dtype 机制（指南 §三十）正是实现它的工具** —— 可精确指定任意一组张量。

## 15.43 【2026-09-06】🟢🟢🟢 **#86 多比例交付完成：五个比例真机出图；mg 被内存否决，single + 惰性校验胜出**

### 结果（实测，图逐张打开看过）

| 尺寸 | 比例 | 耗时 | ION 峰值 | 峰值时 MemAvail | 备注 |
|---|---|---|---|---|---|
| 1024×1024 | 1.000 | 160 s | 9138 MiB | 560 | 回归，与交付前一致 |
| 1184×896 | 1.321 | 280 s | 9158 | 616 | 已知良品（D2 实测 D=0.37 dB） |
| **896×1184** | 0.757 | 290 s | 9157 | 596 | 🎉 首次上机 |
| **1280×720** | 1.778 | 280 s | 8889 | 678 | 🎉 首次上机 |
| **720×1280** | 0.562 | 270 s | 8885 | 739 | 🎉 首次上机 |

耗时含首次切该尺寸的 sha256 校验（约 100 s），第二次起约 170 s。
三张新比例图：构图完整、解剖正确、无黑边无拉伸，竖/横构图自然 ⇒ **真按比例生成，不是裁切**。

### 交付形态：为什么放弃「多图共享权重」

🔴 **mg（5 图共享一份权重）实测被系统杀，两次复现**：ION 峰值 **9486 MiB / MemAvail 208**。
现网同一处是 9111 / 580 —— **本来就只剩 580 MiB 余量**，mg 多出的 375 MiB 直接撞死。

根因（实测）：**共享块在「只启用一个图」时也全部常驻**。part1a 的权重常驻——

| context 内图数 | shared+const | 相对单图 |
|---|---|---|
| 单图 | 2159 MiB | — |
| 2 图 | 2154 | −5（几乎零惩罚） |
| 5 图 | 2499 | **+340** |

⇒ **每多一个图约 +115 MiB 常驻**；`ENABLE_GRAPHS` 只控制解开哪个图，**不控制共享块驻留多少**。
⇒ **共享省存储（5 比例 11315→3005 MiB，−73%）但涨常驻，两者方向相反。**

**改用 single**（每比例独立 context，无共享开销）⇒ 峰值回到现网档位。
体积 35 GiB 的启动开销用**惰性校验**解决：启动只校验基准（10667 MiB / **183 s**，与交付前持平），
首次用到某尺寸时再校验那一份，每文件一进程只算一次。安全性不变：**装载前必已校验**。

⊕ 顺带：**换尺寸不再重启后端**（Z-Image 的图是 `generate()` 按请求现取，进程与尺寸无关）
⇒ 每次切换省 214 s。

### 本轮踩的坑（按代价排序）

1. 🔴 **该单变量却一把梭**：三比例 + 五图共享 + APK 改动一次性交付，失败时无法直接归因。
   用户当场指出。改成「1:1 → 已知良品 4:3 → 三个新比例逐个」后一路顺。
2. 🔴 **跳过了自己写的冒烟第二阶段**。手册里写了判据「MemAvail 峰值 > 800 MiB」，
   mg 实测 208 —— **那一步本可在推 8.3 GiB 之前拦下这次失败**。
3. 🔴 **抽稀采样漏掉真峰值 ⇒ 诊断反复**：2 秒采样但每 5 点打印，读到「失败时 7557」，
   而对照臂（能跑通的现网）是 9111，于是错误地宣布「不是内存」。逐点重采后真峰值 **9486**。
   ⇒ **峰值类指标一律全点取 max；且必须有对照臂。**
4. 🔴 **契约生成器算出了新 base 却没写回 `contract["models"]`** —— 症状是
   **app 照常工作、不报任何错**，只是设备上多占 6.4 GiB、启动多 100 秒。
   是在宿主上用 WSL 编译 app 真正的 `ZImageQnnContract` 解析落盘契约才发现的；
   生成器自己的汇总打印用的是内存变量，**自证不了**。
5. ⚠️ 同一个「拿错参照物」的 bug 在预检里也有两处（「新增文件」算成 0 个、
   「预测 ION」漏掉基准的 3320 —— 漏掉的恰好是风险最大的那个）。
6. ⚠️ **`QNN_CONTEXT_CONFIG_OPTION_CUSTOM == 0`**：用 `option` 值当「配置设过没有」的哨兵，
   而 `{}` 零初始化恰好等于 CUSTOM ⇒ 判断恒为真 ⇒ 把 `customConfig==nullptr` 交给 HTP ⇒
   设备 SIGSEGV，**现网 1:1 也一起中招**。修法：用独立布尔位。
7. ⚠️ **§7.4 的转义展开陷阱本轮触发 7 次**（`
`/`\r`/`\0` 被展开成真控制字节）。
   已加 `scripts/check_stray_ctrl_bytes.py`，它还顺手抓到一处更早留下的。
   **含反斜杠的字符串一律用 Edit 工具写，不走 heredoc。**

### 新增的可复用工具（都用已知样本试过）

| 工具 | 作用 |
|---|---|
| `scripts/aspect_preflight.py` | 插线前总判决，15+ 项，含**用 WSL 编译 app 的 C++ 解析器解落盘契约** |
| `scripts/check_mg_equivalence.py` | 多图 vs 单图元数据等价（M1~M5） |
| `scripts/check_size_lists.py` | C++/Kotlin/契约三处尺寸清单一致 |
| `scripts/check_stray_ctrl_bytes.py` | 裸控制字节（转义被展开） |
| `scripts/aspect_device_smoke.py` | 插线后第一个跑：逐图装载 + ION 实测 |
| `scripts/aspect_range_multistep.py` | 多步激活量程，**差值口径 + 强制 1:1 对照臂** |
| `scripts/aspect_fp32_refs.py` | FP32 参考（VAE 标定 + 画质参考两用），带 md5 互异门 |
| `docs/RUNBOOK_ASPECT_DEVICE.md` | 插线操作手册，每步有继续/中止判据与回退 |

### 下一步：内存/速度工作点是否最优 —— 方案见 `scripts/EXP_PLAN_MEMSPEED.md`

🔴 **现工作点是「擦线通过」**：MemAvail 峰值余量 **560 MiB = 总内存的 3.6%**，
与交付前持平（580）。四段常驻 9.1 GB，而计算任一时刻**只用一段**（约 3.3 GB）⇒ **4 倍过配**。
用户多开几个后台 app 就可能被杀（#141）——**这是产品可用性问题，不是性能问题**。

官方文档里有两条没用上的路（本轮才查到）：
- **Graph Switching (Beta)**：`is_persistent_binary` + `memory_limit_hint`，多图 enable 但未加载，
  按需懒切换；⊕ 我们的 `createFromBinary` **已经用 mmap+madvise**，正是它的前置要求。
- **Multi-Graph Switching (Beta)**：加 `graph_retention_order`，内存紧张时自动从低优先级卸载。

**决定性未知量是 `t_switch`（一次图切换耗时）**：≤2 s ⇒ 常驻可降到约 3.5 GB、MemAvail 涨到约 6000；
≥4 s ⇒ 出局。**先测这一个数，用 `qnn-net-run` 就能测，不用改 app。**


## 15.44 【2026-09-06~07】🔴🔴🔴 **主线 A 根因：Tier 2 的序列手术引入 15.5%，而基座导出是忠实的**

> **起因**：用户质询「部署的推理与官方是否完全一致」。查完发现——
> **①基座 ONNX 导出忠实（step-0 对官方 1.79%，余弦 0.999841）；
> ②Tier 2 的 caption 32→80 手术引入了 15.52%；
> ③三层指标面板结构上看不见这类缺陷。**
> 可复用的方法学结论已入指南 **§四十四**；本节是证据链。

### 15.44.1 逐环节结论（每条标明证据类型）

| 环节 | 结论 | 依据 |
|---|---|---|
| sigma / timestep 调度 | 🟢 **一致** | 按 C++ `ZImageFlowMatchScheduler` 公式复算 8 步 = `[1000, 954.545, 900, 833.333, 750, 642.857, 500, 300]`，与 `official_run.log` 逐项一致（差 ≤0.005 = 日志舍入）|
| CFG / 步数 / VAE `scaling·shift` | 🟢 一致 | CFG=0、8 步；`0.3611 / 0.1159` 与官方日志相同 |
| 官方 TE 两个类之间 | 🟢 无差异 | `official_run.log`：`Qwen3Model` vs `Qwen3ForCausalLM` 相对 L2 **0.0000%** |
| **文本编码器（我们 vs 官方）** | 🟡 **有差异且一直存在** | 全量 **0.8722%**、主体（\|a\|≤p99）**1.8578%**、余弦 0.999989；集中度 98.97% ⇒ 必须看主体口径（约束 7）。与 20-token 时代记录的 0.8628% 吻合 ⇒ **不是 Tier 2 引入的** |
| caption padding 内容 | 🟢 无害 | 置零后 step-0 输出 **sha256 逐字节相同**（图内被 `cap_pad_token` 覆盖）|
| **transformer（同 caption，单变量）** | 🔴 **15.52%**（主体 16.14%，余弦 0.9879）| `parity_transformer.py` + `parity_compare.py` |
| **⇒ 基座 L=32 vs 官方** | 🟢 **1.79%** | `t2_fp32_ref.py step 32` |
| **⇒ L=80 vs L=32（自己两版）** | 🔴 **15.80%** | 同上 |
| 推理引擎（HTP 跑量化 `.bin`） | 🔴 与 CPU 参考不等价 | §十五（既有结论）|
| **权重 vs 官方 checkpoint** | 🔶 **从未逐张量对照** | 全库搜索无此脚本/记录 |
| chat 模板 vs 官方 | 🔶 **未逐字节对照** | #160 只反解出我们自己的 |

### 15.44.2 装置门（约束 8：先用已知样本验证操作化）

官方臂必须复现 `official_run.log` 记的 step-0 `noise_std = 1.5485`。
**实测 1.5485，相对差 0.0022%** ⇒ 装置可信，后续数字才允许解读。

### 15.44.3 🔴 途中两个被自己判据推翻的假设（保留，免得后人重试）

**假设一：图像块位置常数 33 与官方的 23 不同 ⇒ 根因。**
🚫 **错**。我从 `patchify_and_embed_omni` 推出「官方=23」，
而实际调用的是 `patchify_and_embed`：`img_pos_start = cap_len + 1`，
`cap_len` 是**补齐后**长度（22 → 32）⇒ **官方也是 33**。
推翻它的是脚本自己的生效门：打印出官方传入的 `pos_start` **本来就是 (33,0,0)** ⇒ 实验无变量。
⊕ 单变量把它改成 81 之后**更差**（15.52% → 20.22%）。

**假设二：符号对齐前的 198.56%、余弦 −0.9879 是「实质不一致」。**
🚫 **错**，是符号约定相反。**靠已知样本挡住了**：step-0 之后的 `latents_std`
官方 **0.9516** / 我们 **0.9517** ⇒ 一致 ⇒ 只可能是约定差。对齐后才是 15.52%。

### 15.44.4 定位与机制

逐段落盘比对（**自己的 L=32 与 L=80 两版**，成本是与官方对拍的十分之一）：

| 张量 | 图像段相对 L2 |
|---|---|
| `adaln_input` / `add_131` / `tanh_19` / `select_45` / `select_46` / `unified_freqs` / `val_105` | **0.0000%** |
| **`part1a__add_138`** | **1.2381%** ← **起点** |
| `part1b__unified` | 0.7667% |
| `part2a__add_92` | 3.3820% |
| `part2b__latents`（最终） | **15.8024%** |

7 个被手术改过的常量**全部扩展正确**；位置类张量在图像段**逐字节相同**。
⇒ 机制不是常量填错，而是 **padding token 的个数**：
官方按 `ceil(n/32)*32` 动态补（22 → 32，**10 个 pad**），我们固定补到 80（**58 个 pad**），
而 padding **被值替换后照样参与注意力**（官方 batch=1 时 `attn_mask = None`，同样不掩）。
详见指南 §四十四。

🔴 **`SEQ_MULTI_OF = 32`，而 80 不是 32 的倍数 ⇒ L=80 图对任何 prompt 长度都无法与官方对齐。**

### 15.44.5 🔴 为什么三周没人发现

L1/L2/L3 面板比的是「HTP 量化 vs **我们自己的** FP32」——**两臂用同一个手术后的图，同时带伤**
⇒ 面板**在结构上**看不见导出/手术类缺陷。
`docs/reviews/REVIEW_IMAGE_QUALITY_2026-08-18.md` §8.4 早已写下规则
（「若我们的 FP32 已不如官方 ⇒ 先修流水线，再谈量化」），P0-C 也执行了（结果 22.47 dB），
**但规则本身没有被执行**。

### 15.44.6 仍未测（不得当事实用）

- ③**成图层面的 dB 损害**：15.52% 是 step-0；#121 已证 L1 不能预测 L3。
  测法：`t2_fp32_ref.py drive 32` 跑完整 8 步 + VAE，与 `official_pytorch_22tok.png` 比 PSNR
  （现网 L=80 版是 **22.47 dB**）。约 40 分钟纯宿主。**本轮因用户重排优先级而中止。**
- ③剩余 1.79% 的构成：TE 贡献（下游 2.66%）与 **bf16(官方) vs fp32(我们)** 未分离。
  官方 checkpoint 是 **fp32**、运行时 `torch_dtype=bfloat16` 转 bf16；
  输入舍入最多解释 ~1.2%（实测 latents 0.1664%、caption 0.1024%，按实测放大 3.05× 外推），
  **但 30 余层内部累积未测，且本机测不了**（官方 fp32 权重 24.6 GB > 本机 23.7 GB）。
- ③修法的取舍（L=96 / 多套图 / 回退 L=32）**未评估成本**，属产品决策。

---

## 15.45【2026-09-07~17】🟢🟢 P1 交付：mg（多图共享权重）+ 共享 spill-fill，设备占用 35 → 12.3 GiB

> 用户已重排优先级：**P1 五比例模型过大 → P2 生图速度 → P3 与官方一致性（挂起，见 §15.44）**。本节是 P1 的完整收尾。
> 方案/判据/五次运行的逐条执行记录：`scripts/EXP_PLAN_SPILLFILL.md`；原始证据：`scratch_runs/oneshot/evidence_20260917/`；台账 #146（已关闭）、#168、#169。

### 15.45.1 结果（① 实测）

| | 交付前（现网 single） | 交付后（mg + 共享 spill-fill） |
|---|---|---|
| 设备上模型文件 | 35 GiB（每比例一套）+ 历史实验残留，`models/` 共 50 个 .bin | **9 个 .bin，12.31 GiB**（`--cleanup` 后实测 `du`） |
| 手机 /data 已用 | 506 G | **423 G**（清掉 55.00 GiB 不再引用的 .bin + 27.49 GiB 暂存区） |
| ION 峰值（1:1） | 9101 MiB | **8741 / 8714**（两次） |
| MemAvail 平台期中位（ION ≥ 峰值−300） | 1009 MiB | **1369 / 1298** |
| 1:1 出图 | sha `49be8e9a4f95…` | **逐字节相同** |
| 五个比例 | 各一套文件 | 共用一套；**全部出图、PNG 宽高 == 请求、五张已人工查看**（完整橘猫坐木桌） |
| 单张 app 生成耗时 | 160 s | 146~163 s |
| 换比例 | 首次 +~100 s 校验 | 无额外校验（五比例共用同一批文件） |

设备现态：新 APK（94,334,723 B，含 `[segtime]` 埋点与「组大小读 marker」）、mg 契约 154,981 B、`files/models/ZIMAGE/SHARE_SPILLFILL` 内容 `331415552`。

### 15.45.2 过程（一次插线，5 次运行，前 4 次均按设计回滚）

| 次 | 结果 | 原因 | 性质 |
|---|---|---|---|
| 实验 1 | ✅ G2~G5 全过，**共享省 734 MiB** | — | #146 挂三周的③关闭 |
| 交付 1 | 🔴 回滚 | 传给部署脚本的路径 `/d/...`：脚本设了 `MSYS2_ARG_CONV_EXCL='*'`，原生 adb.exe 不认 | 我的执行错误 |
| 交付 2 | 🔴 回滚 | 设备侧 sha 校验用通配符 `*_ctx.SM8750.bin`，漏掉文本编码器（`_ctx_sm8750.`） | 我的执行错误 |
| 交付 3 | 🔴 回滚 | **组大小写死 302,645,248**（single 1:1 最大者）；mg part1a 图要 311~331 MB ⇒ `Failed init QNN context: transformer_part1a` | 我的设计缺陷（常数旁有我自己写的「换比例必须重读」警告） |
| 交付 4 | 🟡 回滚 | 五比例全出图，H1/H3/H4/H5 过，**H2（MemAvail 单秒最低 > 800）不过：703** | 判据操作化错误（现网自己 742 也不过） |
| 交付 5 | ✅ **已交付** | 用户决定 H2 改为「平台期中位 ≥ 同日现网」，`--smoke` 只验 1:1 | — |

之后：`--cleanup` 先试运行 → 核对 41 个待删文件在宿主均有副本（24 个 single 回滚文件**名字不同、按字节数对应**于 `D:\ZImage_Work\p0_experiments\aspect\ctx_*` 与 `p2attr\ctx_*_fp16_L80`；2 个设备独有实验文件先 `sha256` 核对拉回 `D:\ZImage_Work\device_only_backup_20260917\`）→ 真删 → 出图复验（见 15.45.5）。

### 15.45.3 经验教训（可复用的已进 `QNN_CONVERSION_GUIDE.md`）

| # | 教训 | 去处 |
|---|---|---|
| 1 | **共享 spill-fill 的组大小 = 所有成员 context 的所有图（含多图 context 的全部比例）里的最大者，且只认首个注册者的值** ⇒ 必须从交付 .bin 元数据现读，不得写死 | 指南 §38.3 |
| 2 | 共享 spill-fill 实测省量 ≈ 公式预测的 86~91% ⇒ 预算按「公式 × 0.85」 | 指南 §38.3 |
| 3 | `MSYS2_ARG_CONV_EXCL='*'` 之后传给原生程序的宿主路径只能用 `D:/a/b` | 指南 §42.5 坑 5 |
| 4 | 校验门「算哪些文件」必须从契约派生，不得另写通配符 | 指南 §42.5 坑 6 |
| 5 | **MemAvailable 单秒最低不能当门**（每臂 <800 仅 1 秒；同构两次差 345 MiB）⇒ 与同日现网配对、用平台期中位；**定门前先拿现网跑一遍** | 指南 §37.x |
| 6 | **干跑台桩掉了外部程序 ⇒ 「外部程序怎么解释参数」一类问题结构上查不出**，要真实最小探针（本次 3 个交付翻车里 2 个属于此类） | 指南 §42.5 |
| 7 | 抓 logcat 的脚本在请求返回后立刻杀 logcat ⇒ 最关键的 QNN 错误码丢失（交付 3 的错误码就这样没抓到） | `app_generate.sh` 已改；指南 §38.3 |
| 8 | 冷却基线取冷机空闲温度（38 °C）⇒ 插 USB 出过图后平台 43 °C 永远降不到 ⇒ 每臂白等 10 min | `oneshot` 阶段 4 已改；`dprime_measure.cooldown` 本体未改（见 15.45.6） |
| 9 | 干跑台必须隔离输出目录（旧版每跑一次删掉真实 state.json、覆盖当天图） | `oneshot_dryrun.py` 已改 |
| 10 | 清理前按**字节数**核对宿主副本（名字常不同）；设备独有的先拉回再删；**清理后回滚脚本必须拒绝执行**（否则把 app 弄坏） | `--rollback` 已加守卫并真机只读验证 |
| 11 | 过程教训：**自己写过的警告要回头看**；**判据定稿前用已知样本（现网）验证操作化**（约束 8 再次应验） | — |

### 15.45.4 P2 的第一份 app 内逐段耗时（① 实测，#169）

single 形态、新 APK、每段 n=8 步（`timing_breakdown.py --seg scratch_runs/oneshot/evidence_20260917/os_sf_off_logcat.txt`）：

| 段 | 中位 ms/步 | 占步循环 |
|---|---|---|
| part1a | **5744** | **37%** |
| part1b | 3331 | 21% |
| part2a | 3301 | 21% |
| part2b | 3190 | 20% |

步循环合计 124.5 s ≈ app 生成耗时 154.5 s 的 80%。③「part1a 慢是因为算量大」**未验证**（也可能是 IO 张量/搬运）。
⚠️ 这是 single 形态的数据；**mg 形态下的逐段耗时 logcat 已在证据目录（`os_mg_*_logcat.txt`），尚未解析**。

### 15.45.5 清理后复验（约束 11 铁律 3：改了设备状态就真跑一次）

`--cleanup --yes` 之后 force-stop 并经真实 app 出 1:1（`app_generate.sh os_post_cleanup 42`）：
生成 **154.4 s**、PNG **1024×1024**、sha256 **`49be8e9a4f95…` 与现网逐字节相同**；logcat（退出前等 4 s 的修复生效，本次完整）直接可见 `[spill-fill] group sharing enabled: 331415552 bytes (from marker)`、`[segtime]` 32 行（8 步 × 4 段）。证据：`scratch_runs/oneshot/evidence_20260917/os_post_cleanup*`、`cleanup_*.txt`。
⊕ `--rollback` 清理后守卫真机只读验证：single 契约需 29 个文件、设备缺 25 个 ⇒ 拒绝回滚、不改设备；反例 mg 契约引用文件缺 0 个。

### 15.45.6 🔴 未完成 / 新线程接手清单

| 优先 | 事项 | 现状与入口 |
|---|---|---|
| **P2** | **生图速度**（用户排定的下一个主线） | 起点 = 15.45.4。建议先：①解析 mg 形态的 `[segtime]`（零设备）；②查 part1a 为何 1.75 倍（层数 / IO 字节数 / const 大小，宿主元数据可读）；③再定方案并写 `EXP_PLAN_*`（约束 4）。⊕ 已关闭勿重试：O3（PD 装不下，#147）、E/F 图切换（t_switch≈装载，指南 §43）。✅ **2026-09-17：①② 已做，见 §15.46** |
| 风险 | **单秒 MemAvail 谷值**：mg 各次 496~873（现网同日 742）；方向无法判断，成因（疑 VAE 装载尖峰）**未验证** | 若用户报「生图中途闪退」，先看此项；#161 被杀时为 208 |
| 风险 | **清理后已无快速回滚**。`--rollback` 现会拒绝执行 | 恢复 single 需：把宿主 `p0_experiments/aspect/ctx_*/*.SM8750.bin`（及 `p2attr/ctx_*_fp16_L80`、`ZImage_QNN_Evidence/dlc_pipeline/vae_decoder`）**按字节数改回交付文件名**、staged 到一个目录、跑 `deploy_aspect.sh`（契约 `logs/contracts_20260906/final_qnn_contract.single.json`）。**该 staging 脚本尚未写** |
| 未验 | mg 非 1:1 比例与 single 同比例的**数值等价**：只有 1:1 做过逐字节对比，其余四比例只核了尺寸与肉眼 | 无现成 single 同种子参考图；需要时各比例在 single 与 mg 下各出一张比 sha |
| 小修 | `dprime_measure.cooldown` 的基线定义（冷机空闲温度不可达） | 只在 `oneshot` 阶段 4 绕过（`P4_WARM_BASE_C`），通用修复未做 |
| 文档 | `scripts/EXP_PLAN_OFFICIAL_PARITY.md` 里「15.52% 仍未解释」一段已被 §15.44 解释，**未加导航标注** | P3 挂起期间低优先 |
| P3 | 与官方一致性 | 挂起；根因与机制见 §15.44 |

## 15.46【2026-09-17】P2 起步：零设备诊断定出三个候选杠杆（未上机）+ 磁盘清理 47.8 GiB

> P2 = 生图速度（用户 2026-09-07 排定）。方案 `scripts/EXP_PLAN_P2_SPEED.md`（判据锁定，**D1 设备诊断待插线**）；
> 本节每个数都可由 `python scripts/p2_zero_device_evidence.py` 从盘上产物复算（输出 `logs/p2_20260917/evidence.md`）。
> 台账：#173（主）、#174（装载）、#175（HVX 线程）；#169 关闭；#162 订正一句；#34 补证。

### 15.46.1 单张时间预算（① 实测，mg 1:1 `os_post_cleanup`，后端单调时钟）

| 阶段 | 秒 | 占比 |
|---|---|---|
| 8 步循环（part1a 5827 / part1b 3183 / part2a 3231 / part2b 3155 ms/步，中位） | 122.7 | 79.5% |
| **transformer 四段装载** | **20.5** | **13.3%** |
| 文本编码阶段 | 6.0 | 3.9% |
| 后端日志时间跨度之外 | 3.6 | 2.3% |
| VAE | 1.5 | 1.0% |

⚠️ `Initializing → Initialized` 标记只覆盖装载的 7.97 s；标记**之前**的空白另有 12.19 s（见 15.46.4）。只看标记会把装载低估一半以上。

### 15.46.2 part1a「单位算量更慢」（① 测量 + ② 推算）

- ①从四段 ONNX 结构（不载权重）数出 MatMul/Gemm 乘法次数：1:1 下 part1a **8228 G**，其余 6490~6604 G（只多 25%）。
- ①**样本 20 个点**（5 比例 × 4 段，每点 n=8 步）的「段墙钟 ÷ 乘法次数」：part1b/2a/2b **0.474~0.517 ms/G**，part1a **0.709~0.750 ms/G**，5 个比例无一例外。
- ②按同比例其余三段的中位 ms/G 外推，part1a 每步超出 **1780**（1:1）/ 2063 / 2052（4:3、3:4）/ 1766 / 1729（16:9、9:16）ms。**这是模型推算。**
- ①逐条排除：IO 字节（part1a 输入输出约 35 MB，part1b/2a 约 65 MB）｜CPU 侧重量化（`add_138`、`unified`、`add_92` 生产方与消费方编码不同 ⇒ part1b/2a/2b 每步各做一次 1604 万元素的逐元素重量化，part1a 没有这一项）｜形态（single 与 mg 的 1:1 逐段耗时相同）。⇒ ②超出量在 HTP 执行内部。
- ⚠️ 我试过用「权重字节 / 算子数 / opData」做宿主回归：三者在 part1a 上同为 1.5~1.77 倍、在其余段上几乎相等 ⇒ 只有一个高值点，**变量共线，回归系数无物理意义**（曾拟合出「计算量系数 0.01」，与 16:9 快 11% 直接矛盾）。已弃用。

### 15.46.3 嫌疑：`pad_sequence` 导出的全尺寸 ScatterElements（① 结构事实 + ③ 假设）

- ①part1a 有 6 个 ScatterElements，其余段 0 个。两个作用在全尺寸激活上：`node_select_scatter` [1,4096,3840]（图像流）与 `node_select_scatter_4` [1,4176,3840]（unified 流起点，后接 RMSNorm）。
- ①结构：`ScatterElements(data=Expand(fill), indices=Expand(0), updates=Unsqueeze(Pad(x)), axis=0)` —— 这是 batch=1 时 `pad_sequence` 的导出产物；沿大小为 1 的第 0 维、索引全 0 ⇒ **FP32 下输出严格等于 updates**。
- ①部署 DLC 里 fill（uFxp_16）与索引（Int_32）均为 **STATIC**；1:1 图 `constSize` **183.84 MiB**（part1b/2a/2b 为 0），与这几个常量的字节数量级吻合。
- ①**编码核对**：`select_scatter` 与 updates `val_225`（scale 0.000361529470 / offset −31029）、`select_scatter_4` 与 `val_851`（scale 0.011286678724 / offset −7603）**逐位相同** ⇒ ②删掉这两个算子，下游读到的量化值不变，**有望逐字节不变**（须设备 sha256 证实）。
- ①量化命令 `input_list=None`、`quantization_overrides=…part1a_fp16_L80_ovr.json` ⇒ 重量化不跑校准、可确定性重建。
- ③**H1**：这两个算子是超出量的主因（若 HTP 逐元素执行约 55 ns/元素，两处合计即约 1.8 s/步）。**未验证。**
- 🚫 **被我自己推翻的推论（保留，免得重试）**：~~「非 1:1 图的索引没被常量折叠、每步现算 Expand，所以 4:3 比 1:1 每步多慢 0.26 s」~~。读 4:3 的 DLC：索引与 fill **同样是 STATIC**。⇒ 该解释不成立；**4:3/3:4 比 1:1 每步多出约 0.28 s，至今未解释**。

### 15.46.4 装载路径（① 代码 + 实测）与 #162 的订正

- ①`QnnRuntime.hpp::createAndInitContext`：`std::ifstream` 读整个 context 进 `std::vector<uint8_t>`（构造即清零，再整份拷贝）→ `createFromBuffer`。
- 🔴 **订正**：#162 写「我们的 `createFromBinary` 已经用 mmap+madvise（`QnnSampleApp.cpp:468`）」。mmap 确实在那里，但只属于 `createAndInitModel` 路径（Anima/SDXL）；**Z-Image 不走这条路径**。
- ①`os_post_cleanup` 四段装载：标记前空白 3.84 / 3.13 / 2.48 / 2.74 s（合计 12.19），`createFromBuffer` 1.79 / 1.75 / 2.93 / 1.50 s（合计 7.97）。③空白里读盘、清零、拷贝各占多少未拆。

### 15.46.5 查过、已无空间的（①）

| 项 | 证据 | 结论 |
|---|---|---|
| 功耗档 | `QnnModel.hpp`：DCVS 关、总线/核心电压角全 MAX、禁睡眠、RPC 轮询 9999、自适应轮询 | 已是最高档 |
| VTCM 大小 | 单算子探针（图名正确）：`vtcm_mb` 不写 / 0 / 8 / **64** / `--vtcm_override -1` ⇒ 读回**全部 8**、md5 相同 | SM8750 在 2.48 工具下上限即 8，部署值已在上限 |
| 宿主性能估算 | 配置确认生效（dspArch 79 / vtcm 8 / hvx 4 写入）后 basic、detailed 建图 | 不产出 Performance Estimates ⇒ 杠杆排序只能靠设备 |
| HVX 线程 | 不设 ⇒ 元数据 `numHvxThreads` **0**（文档说写入 4）；设 6 ⇒ 6、md5 变 | 运行时含义未知 ⇒ 列为 H3（#175），D1 读 profiling 事件 8001 |

### 15.46.6 磁盘清理（用户逐项勾选）

D 盘可用 **63.2 → 110.9 GiB**（删 47.83 GiB）。删前抽出 8 个量化 DLC 的转换/量化命令、存档配方引用的 62 个小文件；删后核对现网与回滚文件 63 个逐个仍在。明细、恢复方法与引用登记：`docs/DISK_INVENTORY.md`「2026-09-17 清理记录」；脚本 `scripts/cleanup_20260917.py`。

### 15.46.7 经验教训（可复用的已进指南 §45）

| # | 教训 |
|---|---|
| 1 | **图名写错时 graphs 段静默落空，我又踩了一次**（探针第一组）。抓出它的是**写了 `vtcm_mb: 8` 的对照臂读回来是 4**。更要紧的是：同一个错误装置还让我先写下了「SM8750 不产出性能估算」——结论碰巧没变，但**第一次的结论不能用**，必须重做 |
| 2 | **推论出来就找已知样本验一次**：「现算 Expand」一次 DLC 读取就被推翻（约束 8） |
| 3 | **宿主回归只有一个高值点时不要做**：共线变量会给出无物理意义的系数 |
| 4 | 装载耗时不能只看 `Initializing → Initialized`，标记之前的整文件读入另有 12 s |
| 5 | #147 离线 `qnn-net-run` 每次 15.9 s 而 app 内 3.3 s，③疑为没设功耗档 ⇒ 离线测速必须带 `--perf_profile burst`，并用「与 app 的段间耗时比」做代表性门 |
| 6 | Windows 上 `du` 统计 437 GB 超过 15 分钟，`os.scandir` 递归 2 秒；Gradle 构建树有超 260 字符路径，删除要 `\\?\` 前缀 |

### 15.46.8 未完成 / 接手清单

| 优先 | 事项 | 入口 |
|---|---|---|
| **P2-D1** | 设备诊断（约 30 分钟插线，零 APK 改动）：basic 读执行时间与 HVX 线程，detailed 读逐算子 cycles，判 H1 | `scripts/EXP_PLAN_P2_SPEED.md` §3 |
| P2-D2 | 按 D1 结果补写各杠杆 A/B 判据再执行（H1 删 ScatterElements / H2 mmap 装载 / H3 HVX 线程 / H5 CPU 侧） | 同上 §4 |
| 未解释 | 4:3/3:4 part1a 每步比 1:1 多约 0.28 s | #173 |
| 残余 | #170 单秒 MemAvail 谷值、#171 无快速回滚、#172 非 1:1 数值等价 | 不变 |

### 15.46.9【2026-09-18】D1 设备诊断结果：H1 成立，part1a 的超出量九成来自两个算子

一次插线约 35 分钟（含两次排障），装置 `scripts/p2_d1_profile.py`，原始数据 `logs/p2_20260917/d1/`，逐条判据见 `scripts/EXP_PLAN_P2_SPEED.md` §5。

| 门 / 假设 | 事前判据 | 实测 | 结论 |
|---|---|---|---|
| G1 代表性 | 加速器执行时间比 ∈ [1.55, 2.05] | **1.985**（app 内 1.72~1.83） | ✅ |
| G1c cycles 忠实性 | 总 cycles 比 ∈ [1.45, 2.15] | **1.836** | ✅ |
| **H1 ScatterElements 是主因** | S/C1a ≥ 15% 且 (C1a−S)/C1b ≤ 1.40 | **29.1%**、**1.302** | 🟢 **支持** |
| H3 HVX 线程未用满 | N < SoC 最大 | **6 vs 6** | 🔴 **关闭（已用满）** |

- ①逐算子 cycles（单次推理）：part1a 合计 **1.5768e10**（1154 条），part1b **8.5897e9**（680 条）。
  按 MatMul 算量比 1.246 外推，part1a 应为 1.071e10 ⇒ **超出 5.06e9（part1a 的 32.1%）**。
- ①两个全尺寸 ScatterElements：`node_select_scatter` **1.961e9**（12.4%）、`node_select_scatter_4` **2.625e9**（16.6%），
  合计 **29.1%**，占超出量的 **90.6%**；**第三大算子只有 1.3%**。
- ②若 app 内耗时与 cycles 成比例（③未验证）：part1a 每步 5827 → 约 4132 ms ⇒ 单张约 **−13.6 s（−8.8%）**。
- ①HVX 线程：部署 context 运行时 **6**，在线建图（SoC 最大）**6**。⊕ 顺带订正官方文档：它写「离线建图不设时写入默认值 4」，实测元数据写的是 **0**、运行时用 6。

🔴 **口径缺口（③ 未查）**：离线绝对耗时比 app 慢约 5 倍（part1a 29.0 s/次 vs app 5.83 s/步；part1b 14.5 vs 3.2）。
**比值一致所以归因可用，但"能省多少秒"不能由离线数直接换算**。疑因 shell 侧拿不到 app 那套 DCVS 电压角，未验证。
⊕ 这同时解释了 #147 记的「离线 15.9 s vs app 3.3 s」。

### 15.46.10 D1 途中查出的两个装置缺陷（一个波及历史数据）

1. 🔴🔴 **`qnn-net-run` + `.bin` 默认按 float32 解析输入**（台账 #176，已固化为 `EXECUTION_MODEL` 规则 2·补 2）。
   喂原生 uint16 字节 ⇒ 张量被**静默砍半**；而输出按 float32 写出，**字节数恰好等于契约里的原生字节数** ⇒
   **约束 3 要求的「输出字节数 == 契约」守卫报 PASS**。我被它骗过一轮，靠 `unified_mask`（`Bool_8`，原生 ≠ float32/2）交叉验证才发现。
   ⇒ 修法：显式 `--use_native_input_files --use_native_output_files`；**逐个**输出核字节数；拿 `Bool_8`/`Int_32` 张量当模式指示器。
   ⚠️ **波及 #147**：`O:3`/`dlbc` 离线测速用的就是这套装置，三臂同缺陷 ⇒ 相对比较可能仍成立，**绝对数字不得引用**。
2. profiling 日志 `adb pull` 报 `Permission denied` ⇒ `pull_robust()`：放宽权限重试 → `adb exec-out cat` 兜底（三种方式都失败才抛）。

⊕ **另一个自己的解析错误**：viewer 输出里 `Accelerator (execute) time (cycles)` 是**整段总量**行，与逐算子行同样带 `cycles`。
第一版解析把它当成一个算子，导致「单算子占比」被摊薄一半（H1 被误判为 🟡）。
**抓出它的是「出现了一个占 49% 的算子，名字叫 Accelerator (execute) time」** —— 汇总一眼就能看出的异常，比逐行读日志有效。

### 15.46.11 下一步（D2-H1，判据已锁定见方案 §6）

S1 ONNX 手术（只删两个 ScatterElements，不动 Pad 与其余 4 个小的）→ S2 照抄内嵌命令重量化 → S3 建 1:1 单图 context
→ **S4 离线 A/B：9 个输出逐字节比 sha256（硬门）+ 加速器执行时间 ≤ 0.80 倍** → 过了才 S5 建 mg 五图 → S6 真实 app 交付（出图 sha256 == 现网）。
S1~S3 纯宿主（约 1 小时），S4 需插线约 10 分钟。

## 15.47【2026-09-19】D2 收尾：A 变体定为唯一可交付手术，H2/H5 埋点就绪，S6 一次插线方案定稿

> 用户本轮的范围决定（原话）：**不做 8 步→6 步、不重导 transformer**（"这本来就是一个实验"），
> 其余按我自己的步骤推进；**并且要求把整个流程、坑、成功经验记录下来用于迁移到另一个模型** ⇒
> 迁移交付物是 `QNN_CONVERSION_GUIDE.md` **§47（新模型路线图）**，本节只记本轮过程。

### 15.47.1 单删实验的最终结论（① 实测，详见方案 §6.14）

两处结构**完全相同**的 `pad_sequence` 残留（batch=1、索引全 0、沿大小为 1 的轴 ⇒ 数学恒等）：

| 变体 | 段级数值 | 加速器 | 判定 |
|---|---|---|---|
| **A：删图像流 `node_select_scatter`** | 两组输入下 **9 个输出逐字节相同** | 0.86 倍（−14%） | 🟢 唯一可交付 |
| B：删 unified 流 `node_select_scatter_4` | 主体误差 **125%**（已毁） | — | 🔴 关闭 |

⇒ ①**语义等价推不出后端等价；同一张图里同型结构的两个实例结论可以相反**。已固化为指南 §47.3 硬规则。
收益（②推算）：part1a 每步 5827 → ~5040 ms ⇒ **单张约 −6.3 s（−4.1%）**，app 级由 S6 的交付门最终判。

### 15.47.2 本轮写的代码（都还没上设备，标记为待验证）

1. **H2 装载改 mmap**（`QnnRuntime.hpp::createAndInitContext`）：marker 文件 `LOAD_MMAP` 控制，
   **同一个 APK 跑两臂** ⇒ 严格单变量。mmap 省掉"匿名内存清零 + 一次用户态拷贝"；
   `MapGuard` 负责 `munmap`/`close`，`madvise(SEQUENTIAL|WILLNEED)`；两条路径各打一行 `[loadpath]`。
   依据 #174：四段"标记前空白"合计 **12.19 s**，远大于 `createFromBuffer` 自己的 7.97 s。
   ① 代码事实：`createFromBuffer` 把指针直接交给 `contextCreateFromBinary`，本仓库不再另做拷贝 ⇒ 映射只需活到 `initializeApp` 返回。
2. **H5 段内拆时**（`QnnModel::ExecSplit` + `PipelineZImage::logSegTiming`）：把每段墙钟拆成
   `prep / 拷入 / graphExecute / 拷出 / 落库`，按段累计、生成结束时打 `[segsplit]`。
   `ExecSplit*` 默认 `nullptr` ⇒ Anima/SDXL/SD15 的三参数调用不受影响。
3. **解析侧**：`timing_breakdown.py --split`（装载路径表 + 拆时表 + **G2 自证**：拆时合计 vs `[segtime]` 合计差须 < 5%）。
4. **`scripts/p2_s6_session.py`**：一次插线跑完「基线 B → 装新 APK → R/M/R/M → 换 noscatA → A 臂」，
   判据事前锁定在方案 §7.2，任一门不过**逆序自动回滚**。

### 15.47.3 宿主自检当场拦下的两件事（说明门是有效的，不是装饰）

1. **APK 还没重建**：`libstable_diffusion_core.so` 与现网**逐字节相同**（`f68edf747d729c28…`）⇒ 判「改动没进产物，停」。
   同时核对二进制里是否真含 `[loadpath] mmap` / `[segsplit]` / `LOAD_MMAP` 三个串 —— **"编译成功"不等于"改动进了产物"**（MAINLINE §7.4）。
2. **回滚源**：`deliver_mg\` 已在 09-17 清理中删除，但建图原件 `aspect_mg\part1a\part1a_mg.SM8750.bin`（3.151 GB）还在。
   🔴 **不假定它就是设备上那份** —— 阶段 3 动手前用**设备实测 sha256** 认领，认不上就先 `adb exec-out run-as … cat` 把设备那份拉回宿主。
   （#171 的"无快速回滚"说的是 **single 形态**，与 mg 的 part1a 不是同一条路径，本轮已查清。）

### 15.47.4 本轮的工程教训（可复用的已进指南 §47.2）

1. 🔴 **宿主被长任务占满时不要为验证一行改动去跑完整构建**：建图进程峰值 **17.84 GB / 23.7 GB**，
   起 Gradle+NDK 必然把两个任务一起拖死（约束 9 自审 #7 已经发生过两次）。
   改用 `.cxx/**/compile_commands.json` 里该 TU 的原命令起**一个** clang 做 `-fsyntax-only`：30 秒、几百 MB，同样能回答"编不编得过"。
2. ⚠️ 那份 `command` 是**已转义**的：`-DQNN_API=__attribute__((visibility(\"default\")))`。
   直接 `split()` 会让 clang 收到字面 `\"default\"`，报出一堆**指向 SDK 头文件**的假语法错 ——
   又一次「报错说谎」（约束 11·再补）：错在我的分词，不在 `QnnContext.h`。
3. ⚠️ Bash 工具会吞掉 `\\`（MAINLINE §7.4 已记）：本轮第 4 次踩到，改用 Write/Edit 写文件后不再复现。
4. 🔴 **埋点位置差一行，整个 A/B 就测不出东西**（等待期自审第 1 条抓出，**上机前**改掉）：
   `[loadpath]` 第一版打在「整文件读入 / mmap」**之后** ⇒ `[loadpath] → QNN App Initialized`
   这个区间**恰好把 H2 要省掉的那段排除在外**，两臂会量出几乎一样的数，
   而我会据此写下"mmap 无收益"——**一个看起来有数据支持的错误结论**。
   ⇒ 通则：埋点写完先问「这对标记之间，是不是正好夹着我要改的那段？」（已进指南 §47.2）

### 15.47.5 宿主侧已全部就绪（2026-09-19 04:10）

1. ✅ **mg 五图建完**（127.1 分钟）：图名五个逐字一致、dspArch/vtcm/dlbc 同部署、
   **spillFill 逐图与部署完全相同**（⇒ §6.14「A 变体不扰动元数据」的预测在 mg 形态再次成立，样本 n=3→8），
   最大值 331,415,552 == 设备 marker ⇒ **无需改 marker**；文件 2.877 GB（比部署小 274 MB）。
2. ✅ **APK 重建**：`libstable_diffusion_core.so` sha256 `4ea8568a8a2af7bc…` ≠ 现网 `f68edf747d729c28…`，
   且二进制里搜得到四个埋点串 ⇒ 改动确实进了产物。
3. ✅ **干跑台 `p2_s6_dryrun.py` 八场景全过**（含两条完整回滚链）。它查出**两个会让整次插线白费**的问题：
   ① 设备上的文件名是 `transformer_part1a_mg_ctx.SM8750.bin`（交付脚本改过名），
      按建图产物名覆盖会写到没人读的路径，**app 仍用旧文件而实验看起来"成功"**；
   ② 契约里的 sha256 是**执行时完整性检查**，换 .bin 不同步改契约 ⇒ **app 直接拒绝装载**
      （part1a 在契约里出现 5 处：基准 + 四比例，脚本改完断言命中数 == 5）。
   ⇒ 两条都已固化进指南 §47.2 坑清单。

### 15.47.6 下一步：**只差一次插线**

`python scripts/p2_s6_session.py --run`（约 50~70 分钟：5 次出图 + 每臂冷却 + 推 2.88 GB）。
不带参数先跑宿主自检（已全绿）；任一门不过自动逆序回滚（契约 → part1a → marker → APK）。
结果回填方案 §7、台账 #173、指南 §47.4 杠杆表。

### 15.47.7【2026-09-19 08:52–09:21】S6 一次插线（29 分钟，`--quick`）：H5 关闭该方向，H2 因我的操作作废

用户 09:15 要出门（宽限到 09:18）⇒ 砍掉阶段 3（换 context 是交付动作，回滚要推 3.15 GB，不能让用户带着半截状态走）。
完整数据见 `scripts/EXP_PLAN_P2_SPEED.md` §7.6。

**① 三次出图 sha256 全部 `49be8e9a4f95…`**（基线 B / R1 / M1）= 历史交付图 ⇒
铁律 1 通过，且 **H2/H5 两处埋点数值完全惰性**。

**① H5（G2 自证差 0.1%，表可用）——§2 的 H5 假设被否定**：

| 段 | prep | 拷入 | graphExecute | 拷出 | 落库 | 非算力 |
|---|---|---|---|---|---|---|
| part1a | 14 | 3 | **45495** | 185 | 226 | **0.9%** |
| part1b | 689 | 128 | **24352** | 206 | 207 | 4.8% |
| part2a | 683 | 37 | **24624** | 204 | 218 | 4.4% |
| part2b | 711 | 35 | **24465** | 4 | 5 | 3.0% |

宿主侧（重量化 + 拷贝 + 落库）只占 0.9%~4.8% ⇒ ②即使优化到零，单张最多省约 **2.3 s（1.9%）**，
而其中 0.69 s×3 是重量化、不可能归零。**「CPU 侧重量化 + 拷贝可优化」这条关闭。**

🔴 **我在本轮犯的口径错误（已改，原文留证）**：方案里曾写「D1 只能②推出 part1a 41.2% 非算力，本表①实测它落在哪一段拷贝上」——
**两处都错**：① 那 41.2% 是 D1 逐算子 profiling 的**①实测**，不是②推算；
② 它是 **HTP cycles 层**的搬运/转换算子占比，而 H5 是 **app 内墙钟层**，
**两层不能互相验证也不能互相推翻**。我据此还向用户汇报过一句"D1 的 41.2% 在宿主侧是错的"，
那是把自己写错的话又放大了一次。正确表述：H5 只回答"宿主侧不是杠杆"，
HTP 内部那部分仍然只能靠删算子（H1-A）解决。

**🔴 H2 作废 —— 原因是我的操作，不是实验设计**：
M1 臂 logcat 里装载路径 = `['mmap','read-into-vector']` 混合。
根因：09:18:21 我手动删 marker（为让用户随时能安全拔线），**而 M1 的 transformer 四段当时还没装完**；
我依据的是②推算（"+170 s 启动校验 +30 s 装载 ⇒ 09:16:30 已装完"），
**没去读 logcat 里 `[loadpath]` 到底出现了几行**——那是 2 秒的①实测。
⇒ 又一次把推算当实测用（约束 2）。**删 marker 的决定是对的，错在没先花 2 秒确认。**
🟢 脚本的 `len(set(paths)) != 1` 自证抓住了它，否则我会拿混合数据算出"mmap 省了 X 秒"并据此交付。

**⊕ 仍然拿到的①**：R1（纯 read 路径）**八段装载合计 29.58 s**，占单张 158 s 的 **18.7%**
（#174 的 12.19 s 只含 transformer 四段、且口径是日志空白）⇒ H2 的上界比原估**更大**，值得重测。

**设备结束状态**（逐项读回）：新 APK + 现网契约未动 + 现网模型未动 + marker 已删（读回 0）⇒ 完全可用。

### 15.47.8 下一次插线（约 30 分钟，两件事）

1. **H2 重测**：R/M 各两次交替，**marker 只在两臂之间切换，且切换前必须先确认 logcat 里 `[loadpath]` 已满 8 行**。
2. **H1-A 交付**（阶段 3）：推 2.88 GB + 改契约 5 处 + A 臂出图，判据见 §7.2 G3。

## 15.48【2026-09-19 20:17–20:57】第二次插线：**两个杠杆全部交付，单张 −18.8%**

完整判据与数据见 `scripts/EXP_PLAN_P2_SPEED.md` §7.8/§7.9。

### 15.48.1 归因表（① 实测，每步只差一个变量）

| 臂 | 装置 | 单张 generation | 八段装载 | 该步贡献 |
|---|---|---|---|---|
| R | read + 旧 context | 160.5 s | 29.17 s | 起点 |
| M | **mmap** + 旧 context | 143.0 s | 14.55 s | **H2：−17.5 s** |
| A | mmap + **新 context** | **130.4 s** | 12.97 s | **H1-A：−12.6 s** |

**合计 160.5 → 130.4 s（−18.8%），出图 sha256 三臂全部逐字节等于金标准 `49be8e9a4f95…`，
MemAvail 最低点 511 → 1279（+768 MiB）。**

两个意外（都是好的）：
1. ①**mmap 的内存收益**：MemAvail 最低点 +616 MiB（文件页可回收，不占匿名内存，与 #144 预期一致）
   ⇒ 对 #170「mg 交付后内存余量薄」是实打实的缓解，不只是速度。
2. ①**H1-A 实测 −12.6 s 是②预估 −6.3 s 的两倍**：约 1.6 s 来自新 context 小 274 MB 使装载变快，
   约 11 s 来自步循环 ⇒ ②该算子的真实墙钟成本**高于按 cycles 占比的线性推算**（cycles 12.4% vs 墙钟 ~19%）。③原因未查。

### 15.48.2 本轮三处**判据/归因**自身的错误（都已修，通则已入指南）

| # | 错在哪 | 后果（若没抓住） | 通则 |
|---|---|---|---|
| P12 | H2 判🟢后 A 臂开着 marker 跑，而速度对照写死用 R 臂 ⇒ 差了**两个**变量 | 把 mmap 的 17.5 s 记到删算子头上（脚本当场就报了"省 30.1 s"） | **多个优化叠加时，对照臂必须与实验臂只差一个变量** |
| P13 | 终态自检无条件把"marker 还在"判成设备不干净 | 会指示我在**交付成功之后删掉交付物本身** | **"干净"是相对于本次交付意图的状态**；写终态判据前先问"交付成功后设备应该是什么样" |
| P3 | 原本没有任何证据能区分"换成功了"与"根本没换" | 推送若没生效，会安静地给出一样的图和速度，被读成"收益未复现" | **预期结果是"没有变化"时，必须有独立于主指标的生效证据**（这次用装载字节数 2,877,374,464 vs 3,151,335,424） |

⊕ P3 是**上机前排查**加的门，本次真的用上了 —— 它在交付门里第一个打印 ✅，
把"文件确实换了"这件事从假设变成了实测。

### 15.48.3 🔴 交付范围：只验证了 1:1（台账 #177）

五比例 mg 里**只有 1:1 出过图**，其余四比例一次都没跑过，**不得由 1:1 外推**（§47.3：同型结构的两个实例结论可以相反）。
用户只用 1:1 无风险；用其他比例可能破图或装载失败（③未验证）。
回滚路径完好：`aspect_mg/part1a/part1a_mg.SM8750.bin`（实测与设备原件 sha256 相同）+ `scratch_runs/p2s6/contract_device_before.json`。

### 15.48.4 设备现态（交付形态，2026-09-19 20:57 起）

新 APK（`4ea8568a8a2af7bc`）+ **noscatA part1a**（2,877,374,464 B）+ 改过 5 处的契约（154,981 B）
+ **`LOAD_MMAP` marker 在（它就是 H2 的交付形态，不要删）** + `SHARE_SPILLFILL=331415552`。
现网原版 APK 存档：`scratch_runs/p2s6/installed_before_p2s6.apk`（`f68edf747d729c28`，**全仓库唯一**）。

## 15.49【2026-09-19 夜~09-20】P2 收官：第三个杠杆交付 + UI 收尾 + 交付态冻结

### 15.49.1 三个杠杆的总账（① 实测，同一台设备，每步只差一个变量）

| 杠杆 | 归因 | 台账 |
|---|---|---|
| **契约校验换 ARMv8 硬件 SHA-256** | **启动开销 148~210 s → 24.5 s（−124~186 s）** | #178 ✅ |
| 装载改 mmap | −17.5 s，且 MemAvail 最低点 **+616 MiB** | #174 ✅ |
| 删 part1a 图像流的 `pad_sequence` ScatterElements | −12.6 s（②预估的两倍） | #173 ✅ |

**用户实际等待：约 350 s → 167 s（−52%）**，出图 sha256 全程**逐字节等于金标准** `49be8e9a4f95…`。
完整判据与数据见 `scripts/EXP_PLAN_P2_SPEED.md` §7.8 / §7.9 / §9。

### 15.49.2 最大的一条为什么最迟才被发现

前三轮优化的都是 `generation_time_ms`（后端自报），从 160.5 压到 130.4 s；
**而用户等的是墙钟**，其中 148~210 s 花在生成开始之前（契约 sha256 过 12.04 GiB）。
那段时间**从来没进过时间预算表**——因为那张表本身就是从内部指标拆出来的。
⇒ 通则已入指南 **§47.9**：第一张时间表必须从"用户点下去"量到"图出现"。

⊕ 做法上值得复用的一点：**不直接改 app 验证**。SHA 指令写错 = 契约校验全失败 = app 起不来，
而改 app 一轮十几分钟。先把实现编成 2.5 MB 的独立可执行文件（`tools/sha_bench.cpp`
+ `scripts/build_sha_bench.py`）push 到设备，单独确认「算得对不对、快多少」，通过了再集成。
正确性做到**三重闭合**：硬件版 == 自家软件版（含分片边界）== 系统 `sha256sum`。

### 15.49.3 🔴 判据失效：同一个坑踩了两次

G1/G2 依赖 `[startup] 契约基准校验 N ms` 这行日志，**它没出现在 logcat 里**：
`app_generate.sh` 在 `logcat -c` 之后才开录，而 `initialize()`（含校验）在录制窗口**之前**。
**而我加这行日志，正是为了解决"校验耗时录不到"** —— 位置仍然没躲开窗口。
事后去环形缓冲捞，晚 3 分钟、6590 行已滚过。

⇒ 收益改用另一个①实测指标坐实（logcat 首条后端单调钟，与之前完全同口径），结论不受影响；
但「校验本身占 24.5 s 的多少」仍是③，转 **#181**。
⇒ 通则：**埋点位置要对齐采集窗口**，不只对齐被测代码。

### 15.49.4 UI 收尾：Z-Image 下隐藏负面提示词框（#180，**已改未验**）

① 代码事实：`PipelineZImage.hpp` 全文件不引用 `negative_prompt`，且强制 `cfg == 0`
（无 CFG ⇒ 负面提示词在数学上无处可用），而原 UI **无条件渲染**该框 ⇒ 用户填了完全不起作用。

🔴 **实测发现的一个坑**：隐藏输入框**不会**让上面的提示词框自动变长 ——
`PromptTagTextField` 的高度是**行数驱动**的（折叠态 `minLines = maxLines = 2`），
与下方有没有控件无关。所以同时把正向框折叠行数 2 → 4，才拿到"往下延伸"的效果。

产物核对：`.so` sha256 **未变**（`538c36c58263b074`，native 一行没动）⇒ 改动确在 dex 层。

✅ **2026-09-20 10:45 真机验证通过并交付**：装机后设备上 APK 的**整包 sha256 == 宿主 UI 版**
（`d87e9670…`）；截图 `scratch_runs/ui_zimage.png` 确认负面框消失、正向框约 4 行占下原位置；
出图 sha256 **逐字节等于金标准**，generation 135.0 s（上次 131.2 s，差 3%，热漂移范围内）。

🔴 **这一步顺带抓出一个判据缺陷**：原先按 `.so` 指纹给 APK 存档命名 —— 只改 Kotlin 时 `.so`
**完全相同**，第二版会被当成「已有存档」跳过而**丢失存档**；更巧的是那两个 APK 的**字节数也完全相同**
（94,339,337）而内容不同。⇒ 存档命名改用**整包 sha256**；**字节数不是判据，sha256 才是**（已入指南 §47.2）。

⊕ 数据层 `negativePrompt` 字段**故意保留**：67 处 / 20 个文件（含远程协议、PNG 元数据、
参数分享），删它对界面**零可见收益**，留到做纯 Z-Image 专用 app 时一并清理。

### 15.49.5 交付态冻结

`python scripts/freeze_delivery.py` **现读**生成 `docs/DELIVERY_FROZEN.md`（不抄旧值）。
本次以 `--host` 运行（设备已拔线），**设备侧那一节待下次插线补齐**。

### 15.49.6 下一步

1. ~~一次插线约 5 分钟：装 UI 版 APK → 截图确认版面 → 出图核 sha256 → 补冻结快照的设备侧（#180）~~
   ✅ **2026-09-20 10:45 已完成**：#180 关闭；`docs/DELIVERY_FROZEN.md` 的设备侧已补齐
   （12 个文件逐个现读 sha256，与契约吻合；part1a = noscatA `67b5583d…`）。
2. 用户已定的后续：跑通几个模型后再做「纯 Z-Image/Flux 专用 app」（删 SDXL/Anima/SD15）。
   评估见对话记录：**APK 体积收益很小**（94 MB 里 dex 占 76 MB，native 仅 7.3 MB），
   真正收益是 UI 简化与维护负担；分四层做，前两层性价比最高。
3. 未关闭：#181（校验耗时未直接实测）、#170、#171、#172、#179 已关闭。

## 15.50【2026-09-20】P2 全部收口：交付完成 + 模型打包验证 + 本线程复盘

### 15.50.1 交付最终态（用户已亲自验证）

| 项 | 内容 |
|---|---|
| 设备 | 新 APK（`.so` `0357a043606d4970`）+ noscatA part1a + 改过的契约 + `LOAD_MMAP` + `SHARE_SPILLFILL` + 校验缓存 |
| 模型包 | `/sdcard/Download/ZIMAGE_sm8750_v2.zip`（13.53 GB，123 文件；10 个关键文件 sha256 与设备逐字节一致）；宿主副本 `D:\ZImage_Work\package\` |
| 导入验证 | ✅ **用户已完成导入并验证通过**（2026-09-20） |
| 冻结快照 | `docs/DELIVERY_FROZEN.md`（宿主产物 + 设备 12 文件现读 sha256 + 金标准出图 sha256） |

**P2 总账（① 实测，用户实际等待）**：约 **350 s → 约 148 s（−58%）**，出图 sha256 全程逐字节不变，五比例全部验过。

| 杠杆 | 归因 | 台账 |
|---|---|---|
| 契约校验：软件 SHA-256 → ARMv8 硬件加速 | 129 s → 19.4 s | #178 ✅ |
| 契约校验：结果缓存（按文件失效） | 19.4 s → **0 ms** | #182 ✅ |
| context 装载改 mmap | −17.5 s，且 MemAvail 余量 **+616 MiB** | #174 ✅ |
| 删 part1a 图像流的 `pad_sequence` ScatterElements | −12.6 s（②预估的两倍） | #173 ✅ |
| UI：隐藏负面提示词框、提示词框 2→4 行 | 版面 | #180 ✅ |

### 15.50.2 剩余未关闭

- **#170** 单秒内存谷值成因、**#171** single 形态回滚脚本、**#172** 非 1:1 数值等价（无参照 sha）
- ⊕ 已查明无空间：HVX（已满）、VTCM（上限 8）、功耗档（已最高）、宿主侧重量化+拷贝（占段墙钟 1~5%）、
  并行装载（#179 官方文档：依赖同一 device handle 非线程安全）、
  **context 装载 12.7 s 已贴 I/O 极限**（12.93 GB ÷ 12.74 s = 1015 MB/s）

### 15.50.3 本线程复盘：五种思维模式（已固化为指南 §47.10）

本轮返工按根因归类只有五种，**每一种都重犯过至少两次**。具体见 `QNN_CONVERSION_GUIDE.md` §47.10，摘要：

| 模式 | 本轮案例 | 识别信号 |
|---|---|---|
| 1 没弄清根因就行动 | 打包时 `Permission denied`，先判"路不通"换方案、再判"权限位"加 `chmod`，**真因是 SELinux** | 正在说"换一条路"，但说不出它为什么不通 |
| 2 手搓替代成熟工具 | 不用 `adb pull`/`ZipFile.write`，手搓 `exec-out tar` 流与流式 zip ⇒ 卡死两次 + 4.34 GB 白传 | 为省一点开销把两个独立步骤耦合成管道 |
| 3 证据没看全就下结论 | `grep \| head -8` 没看到 `ModelListScreen.kt` 就断言"没有导入入口"，实际有 | 用分页结果论证"**不存在**" |
| 4 判据本身没验证 | G4 假设全量重算（实为逐文件）；用 APK **字节数**判断改动进没进产物（两个不同 APK 字节数相同）；终态把"marker 在"判成不干净（它正是交付物） | 判据含具体数值/顺序/粒度，却没拿已知样本试过 |
| 5 优化内部指标而非用户等待 | 三轮压 `generation_time`，而生成开始前还有 148~210 s，预算表里连一行都没有 | 时间预算表是从某个内部字段拆出来的 |

🔴 **共同点：都不是"不会做"，而是在该停下来确认的地方没停。**
每一次都可以用几十秒的确认避免，实际代价从几分钟到几小时。

### 15.50.4 本轮还犯的两个沟通/操作类错误

1. **整场会话对用户的汇报正文用英文**，而文档、注释、脚本输出全是中文 —— 用户爆发后才发现。
   根因：把"用中文"当成**产物的格式要求**而非**沟通对象的语言**，规则只覆盖了写文件。
   ⇒ 已加入 `CLAUDE.md` 约束 9 自审清单**第 8 条**（每次输出汇报正文前问一次）+ 跨会话记忆。
2. **Bash 工具吞反斜杠，本轮踩 7 次**（两次把 .py 改出语法错误）。
   MAINLINE §7.4 早有记录，我仍反复犯——因为"不含反斜杠时用 heredoc 没出事"赌赢过。
   ⇒ 已定为无条件规则：**改文件只用 Write/Edit，Bash 只用来跑命令**（写进跨会话记忆）。

### 15.50.5 下一个模型从哪开工

`docs/QNN_CONVERSION_GUIDE.md` **§47**（完整路线图）：
§47.6 第一天清单 → **§47.10 五种模式（第 0 条，先读）** → §47.2 坑清单 → §47.3 图手术硬规则
→ §47.4 提速动作 → §47.7 导出要求 → §47.8 mmap 必做 → §47.9 时间表从用户点击量起。

---

## 15.51【2026-09-20~21】方法验证轮：门可执行性重放 + 交付谱系 + 当前交付件实测

> **背景**：主线 B 的交付物（§47 路线图）此前只是"写下来"，从未被验证过"照着做能不能做完"。
> 本轮对它做了三件事：修自相矛盾 → 重放门的可执行性 → 在**当前交付件**上实跑。
> 方案与判据见 `scripts/EXP_PLAN_GATE_REPLAY.md`（执行前定稿，判据未改）。

### 15.51.1 门可执行性重放（判据事前锁定）

34 个门抽全，本轮实跑 21 个。**第一轮 🔴+🟡 = 42.9% > 判据的 1/3 ⇒ 判定失败**；
补完四个缺失工具后转为通过（14.3%）。失败归因：**把「这件事该做」和「这件事有工具」混为一谈**
—— §47.10 模式 4 的重犯，而且是在写"防模式 4"的卡片时犯的。

顺带查出并订正的文档错误：
- 卡 D 的 D2 命令**参数不存在**（`--against` 实为 `--diff`；`--expect-arch` 收 int `79` 不是 `v79`）
- §2.4「转换器只支持到 opset 17」**不准**：实测 `text_encoder_part1~4`、`vae_decoder` **全是 opset 18 且已交付跑通**

### 15.51.2 交付谱系（`scripts/build_lineage.py`）

从**产物自身**反推"设备上这个文件是怎么来的"：
设备 sha256（`DELIVERY_FROZEN`）→ 宿主 `.bin` → 同目录 `d.json`（图名/soc/vtcm/权重共享）
→ 图名主干 = fp32 DLC 名（§24.2）→ 量化 DLC → **内嵌 Converter/Quantizer 命令**（§二）。

**9/10 锚定成功**。两个发现：
- 🔴 **设备上的 `final_qnn_contract.json` 没有宿主备份**（台账 #187）。15 份副本无一匹配；
  最接近的 `final_qnn_contract.mg.json` **字节数完全相同（154981 B）但 sha256 不同**
  —— §47.2「别用字节数判断产物」那条坑撞在自己身上。差异是机械的：part1a 的
  `sha256`/`size_bytes` 在契约里共 **10 处**（基准 + 4 比例 × 2 字段）。
- ⚠️ TE 四段**谱系断链**（同目录无建图配置 json，台账 #186）。
  其实际危害在清理时显形：自动白名单认不出它们的来源 DLC ⇒ 差点被判成可删。

⊕ 从内嵌命令读出的**当前交付真实配方**：`use_per_row_quantization=True`、
`use_per_channel_quantization=False`、`float_bitwidth=32`、`float_fallback=True`、`input_list=None`。
🔴 **`act_bitwidth=8` 是默认值不是实际值** —— 用 overrides 时激活 encoding 是注入的（16-bit），
照字面读会误判成 A8W8。

### 15.51.3 🔴 在**当前交付件**上实跑（不是历史样本）

以谱系锚定的四个 1:1 基准 DLC 为对象（`scripts/verify_delivery_gates.py`）：

| 段 | Where 节点 | requant 极端边 |
|---|---|---|
| part1a / part1b / part2b | 0 | **0** |
| **part2a** | 1 | **18** 🔴 |

`node_Where_105` FAIL：掩码分支 `val_104` min = **−3.4028e38**（**原始 -FLT_MAX，不是 maskfix 后的 −100
⇒ maskfix 从未进入交付路径**）；输出 `val_105`（scale **1.526e-09**）成为 18 条边的共同分母，
最大 `val_1814/val_105` = **2.47e7**。

**根因当日闭合（门 B4，直接读 40 个校准样本）**：

| 输入 | 实测 | 判定 |
|---|---|---|
| `cap_pad_mask` | **全部为 0**，1 种取值组合 | ❌ 掩码分支从未激活 ⇒ `val_105` 被校准成 [0, 1e-4] ⇒ **这就是 18 条边的来源** |
| `caption` | 40 个样本**完全相同**（min −4556.2085 / max 13753.4668） | ❌ **文档从未记过**：TE 输出的动态范围在校准中根本没被覆盖 |
| `latents` | 40/40 各异 | ✅ |
| `timestep` | [0, 0.7]，8 种取值 | ✅ 门 B6 通过，与 §4.3 记载一致 |

⇒ 与 §4.2 的记载一致，但现在是**实测**不是转述；且 `caption` 固定这一条是新发现，
**独立于掩码问题**，可能影响更广（未评估）。台账 **#188**。

### 15.51.4 其余门的结论

- **D3 graphs 段** ✅：六个交付件配置齐全（5 图 / vtcm 8 / soc 69 / v79 / 权重共享 on）；
  TE 四段虽无配置文件，但**元数据反查**证明生效（dspArch 79 / vtcm 8 / 图名正确）。
  ⇒ 门 D3 的主判据应是**元数据反查**，配置文件只是辅助。
- **F2 指纹层级** ✅：`DELIVERY_FROZEN` 同时记了整包 APK 与 native `.so` 两层 sha256。
- **E5 对拍参照** ❌：按卡 A A3-e 定的格式（`fp32_ref/` + `manifest.json`）**不存在**；
  只有 10 个零散的 `.log`/`.png`，**无统一管理、无 sha256 清单** ⇒ 参照本身无法校验。

### 15.51.5 磁盘清理（原则由用户定）

判据不是"它是不是证据"，而是**"这个结论还需要被重新验证吗"**。
删 34 个文件、**释放 49.23 GB**（D 盘 69 → 115 GB），交付白名单删后逐个核对 ✅ 全部仍在。
标记存 `logs/cleanup_20260921/deleted_manifest.json`（34 条 + 8 个实验组代表 DLC 的内嵌配方）。
方法论入指南 **§21.4**，清理记录入 `DISK_INVENTORY`。

两条新纪律：**① 删除走白名单不走黑名单**（按黑名单跑时漏过两次保护：#171 回滚源、TE 来源 DLC）；
**② 谱系断链 = 清理盲区**（转新模型时每建一个 context 就把配置与日志留在产物旁）。
