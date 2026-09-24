# 对《DIAGNOSIS_HTP_ZIMAGE.md》的独立审查

日期：2026-08-15  
审查对象：`DIAGNOSIS_HTP_ZIMAGE.md`（SHA-256：`fbe504c729104d1094d4fb485a8aa83c616c0f3eff8ddae9ce248f34f2aa03c9`）

## 0. 审查边界

本审查区分四类内容：

- **报告内实测**：原报告声称已经测得，但本次只拿到了 Markdown，没有拿到第 217–220 行列出的脚本、原始张量、日志、图片和配置，因此不能独立复现。
- **可复算**：可以直接用报告中的数字复算数学关系。
- **严格推论**：在报告所述实验前提为真时，逻辑上能够推出。
- **待验证假设**：存在合理机制，但现有材料不能确认。

因此，本审查可以判断推理是否越界、数学是否自洽、还缺什么实验；不能替代对原始数据的复现审计。

## 1. 总结论

原报告的总体排查方向**对了一半，而且是重要的一半**：问题已经明显收敛到 **Z-Image transformer 的 HTP 专属执行链路**。继续反复排查 VAE、文本编码器、随机种子或普通 App UI，信息增益很低。

但原报告把这个结果写成“根因已确定为 HTP 定点计算精度不足”，证据尚不够。当前严格成立的结论应是：

> 在所测模型、输入和设备条件下，成图失败与 transformer 的 QNN/SNPE HTP 目标执行链路绑定。最终 `unified` 张量在报告所用 encoding 下的直接量化—反量化误差，远小于 HTP 输出误差，所以**该输出张量自身的 16-bit 存储分辨率**不足以单独解释失败。现有证据尚不能区分 HTP prepared graph、encoding 解释、融合/重定标、float I/O 转换、算子 kernel、SDK/runtime 版本、驱动/固件或硬件执行中的哪一层。

换句话说：**方向正确，根因层级写深了两层。** 目前不是“项目无解”，但也还没有证据支持“只剩选择性 FP16”这一条路。

还有一条会改变可行性判断的官方事实：HTP FullyConnected OpDef 列有“FP16 activation/output + 静态 8-bit 权重”的组合。因此，即使最终定位到 FC，现阶段也不能推出“61.54 亿权重必须全部翻倍、项目必然出局”；应先在首个异常真实 block 上验证这条混合类型路径。

## 2. 哪些判断成立

### 2.1 HTP transformer 路径是主要差异来源

原报告第 33–40 行的 CPU/HTP 对照，在“其余输入和流水线确实逐字节相同”的前提下，可以支持：

- 把 transformer 换入 HTP 路径，是清晰图变成色块的必要差异；
- text encoder、调度器、VAE、RNG 等两路共有组件，不像本次 CPU/HTP 差异的独立主因；
- 下一步应在 transformer 的 HTP 目标链内定位，而不是重新从整条 App 漫无目的排查。

但它定位的是一条链，不是链中的具体一层。后端切换同时改变 prepared graph、lowering、融合、encoding 使用、I/O Convert、kernel 和 target runtime，不能把“后端”当成一个原子算术变量。

### 2.2 第 71–75 行的范数加权数字自洽

按表中能量占比复算：

```text
sqrt(0.9958 × 0.003%^2 + 0.0042 × 2.903%^2) ≈ 0.188%
```

与表中的 0.189% 一致。CPU 和 HTP 两列也分别可复算为约 1.175% 和 19.238%。这组表内数学没有明显算错。

### 2.3 第 84–88 行的算术可复算，但名称不准确

```text
sqrt(14.598%^2 + 12.534%^2) = 19.2407%
16 - log2(4.558 / 2.903) ≈ 15.35
16 - log2(73.621 / 2.903) ≈ 11.34
log2(73.621 / 4.558) ≈ 4.01
```

所以“在这套自定义误差倍率口径下，HTP 相对 CPU 多约 4 bit 的误差倍率”可以复算。它不能被表述成“HTP 实际累加器只有 11.34 bit”或“硬件确定丢了 4 bit”。

### 2.4 未把两个未生效开关算作排除项，这一边界正确

第 131–137 行明确承认 `high_precision_sigmoid` 与 `advanced_activation_fusion` 没有真正完成有效 A/B，这是正确的证据边界。不过“产物 MD5 不变”本身也不能唯一证明“被静默忽略”：如果设置值等于默认行为，也可能不改产物；仍需核对默认值、prepare 日志和 prepared graph 配置。

## 3. 必须修正的关键逻辑

### 3.1 “根因已确定”应改为“故障域已定位”

对应原报告第 29–47、77–80、165–167 行。

CPU 与 HTP 的差异至少覆盖：

- 原始 DLC 与 HTP prepared graph；
- 编译器及算子 lowering；
- 融合、tiling、重定标和定点 kernel；
- scale、offset、signedness、per-axis 轴在 target graph 中的实际解释；
- float32 用户缓冲与 native quantized tensor 之间的 Convert；
- QNN backend/System/stub/skel、驱动和固件；
- 最终硬件执行。

所以现有实验只能把根因域定位为“HTP 专属编译/准备/执行链”，不能单独锁定“硬件定点算术精度”。量化配置虽然在两路中保持不变，但这只说明 **backend 在该量化模型下产生差异**；不能排除“量化尺度 × HTP lowering/kernel”的交互。

建议把标题“根因已确定：HTP 执行”改成：

> 故障域已定位：transformer 的 HTP 目标执行链；具体层级尚未定位。

### 3.2 “HTP 在环几乎逐点复现 App”没有被现有表格证明

对应第 36–42 行。

HTP 在环和 App 分别相对第三个参考结果得到 46.73 与 46.40 的平均像素差，并不等于二者逐像素接近；两个完全不同的坏图也可能对参考图有相近 MAE/PSNR。

必须直接计算：

- `HTP-in-loop 图 vs App 图` 的 MAE、PSNR、最大绝对差和像素相关性；
- 更重要的是三段 transformer 边界输出的直接差；
- 表中每个 PSNR/像素差究竟相对哪张参考图。

只有直接差很小，才能用 HTP-in-loop 真正排除 App 独有的 C++/buffer/布局问题。若两路还共享相同的边界转换代码或 tensor contract，同一种错误也可能同时存在。

### 3.3 SNPE CPU 是否真是“相同定点语义”没有说明

原报告只写“SNPE CPU 参考实现”。Qualcomm 的官方 QAIRT/SNPE 文档把 **CPU Fixed Point Mode** 单列为一个模式；因此报告必须记录：

- CPU 基线是否明确启用了 fixed-point mode；
- 若未启用，CPU 是否实际用 float32 数据/数学执行量化 DLC；
- CPU 与 HTP 是否使用同一组实际 encodings 和相同边界类型。

若 CPU 基线不是定点仿真，它仍然是有价值的模型正确性基线，但不能被称为“正确的 16-bit HTP 等价实现”，也不能据它推导 HTP 的实际 bit 数。

### 3.4 第 68–80 行只排除了一个输出 tensor 的直接 QDQ 误差

把 FP32 真值按 `unified` 自身 encoding 做一次 QDQ，可以严格支持：

> 在脚本使用的这组 encoding 和这份样本上，`unified` 最终存储为该 16-bit 表示时，直接 QDQ 误差不足以解释 HTP 的 73.62%。

它不能排除：

- 上游输入、权重和中间 tensor 的 encoding；
- HTP prepared graph 是否改写 scale、offset、signedness 或 per-axis 信息；
- FC/MatMul 的累加、饱和、舍入和输出 requant；
- Residual Add 两路尺度对齐；
- 布局、stride、权重量化轴或 float I/O Convert 错误；
- calibration 对 HTP kernel 数值路径的影响。

“量化表示极限”应改名为“该 tensor、该固定 encoding、该样本上的直接 QDQ 误差”。它不是整个量化图的理论极限。

### 3.5 “主体”被用于两个不同集合

- 第 57 行的“主体”是按元素筛选的 `|a| ≤ p99`；
- 第 75 行的“主体”是排除 ch85 后的 3839 个完整通道。

二者不是同一集合，却被共同用于后续结论。应分别命名为：

- `p99 非离群元素子集`；
- `non-ch85 通道子集`。

全量相对 L2 也不是数学上“失真”，它准确回答能量加权误差；只是可能与下游图像质量不一致。应该同时保留全量 L2、两个子集 L2、逐通道 RMSE/余弦和最终图像指标，而不是用一个口径替代另一个。

### 3.6 “随机噪声”和“HTP 逐位确定”互相矛盾

若 14.598% 是把 HTP 输出投影到参考向量得到的全局增益分量，12.534% 是正交残差，那么平方和等于总误差是投影定义导致的恒等式，不是对硬件机理的额外证明。

12.534% 不应叫“随机噪声”。相同输入逐位确定，说明它至少在这两次运行中是**确定性残差**；它仍可能来自 clipping、非线性近似、scale/offset 错误、布局、融合或重定标。

报告应补充回归公式、是否过原点、是否含截距、使用哪个子集和权重，并把 15.36/11.34 bit 改称“相对误差等效位数（启发式）”。

### 3.7 calibration 和 SmoothQuant 没有被排除

第 62–64 行先说明 MSE/能量与下游损害并不一致，第 118 行又以 MSE 最优和被裁掉能量比例排除 percentile，判据前后冲突。单个 tensor 的重建 MSE 最优，不等于最终成图最优，也不能代表所有 prompt、token 长度和去噪 timestep。

第 128 行对 SmoothQuant 的排除尤其不成立。SmoothQuant 类等价通道重缩放不只是改善 `unified` 的最终存储分辨率；它还会改变 FC 输入动态范围、整数乘法、累加器占用和 requant shift。massive activation 恰好使它成为一个高信息增益的**机理探针**。

正确实验是：只在首个异常小图中，对 ch85 做可逆缩放，并对对应 FC 权重做逆缩放；先证明 FP32 函数等价，再看 HTP 误差是否随缩放显著变化。

### 3.8 “厂商承认 v79 短板”不是严格推论

即使第 155–156 行引用的 `PRECISION_COMPENSATION` 描述完全准确，一个 v81+ 的精度补偿功能也只能说明新架构提供了相关能力，不能反推 v79 有已知缺陷，更不能证明本问题与该功能针对的是同一机理。

这应降级为“值得核对的官方产品线索”。此外，当前提供材料没有附该引文在 QAIRT 2.48 本地文档/头文件中的具体路径和上下文，应补齐后再把它列为已核实官方事实。

### 3.9 “选择性 FP16 是唯一方向”以及体积/速度结论过强

第 169–180 行按参数量估算静态权重体积的思路有价值，但参数量不能推出：

- FP16/FP32 激活和 Convert buffer；
- runtime workspace、VTCM、context 元数据；
- 浮点 type propagation 是否把相邻子图一并变成 FP16；
- 融合失效、backend 分区和 tensor copy；
- 实际延迟。

尤其 MatMul 即使“0 参数”，仍可能有巨大计算量和激活内存。因此：

- “非 FC 全浮点只增加约 1 MB”最多只是**静态参数增量**估算；
- “part1a+1b 仍为 3839 MB、速度基本不变”必须实际编译、测 context 大小、峰值 PSS/RSS 和时延；
- “FC 浮点必然出局”还隐含假设“FP16 activation/output 必须搭配 FP16 权重”。Qualcomm 当前公开 HTP FullyConnected OpDef 列有 **FP16 activation/output + 静态 SFIXED_POINT_8 weights** 的组合；至少从官方类型约束看，存在保留 W8 权重、只让 FC 激活/输出走 FP16 的可能，不必先验增加 6154 MB 权重。它能否由 QAIRT 2.48 为 SM8750/v79 实际转换、编译和正确运行，仍必须在真实小图上验证；
- 即使上述混合类型不可生成，“全部 276 个 FC 同时 FP16”也不是唯一粒度，仍可只替换首个异常 FC 或少数敏感层。

选择性 FP16 是候选 workaround，不是当前唯一方向。应先定位首个异常算子，再决定精度替换范围。

## 4. “已排除”表需要逐项降级

| 原项目 | 严格能排除的范围 | 仍未排除 |
|---|---|---|
| `requant inf` / all-ones mask | 所测短 prompt 下该运行分支可能未取另一支 | 编译器仍准备错误分支、其他 token 长度/带 padding mask、无穷 scale 对融合/prepare 的影响 |
| 在线 prepare 与离线 `.bin` 输出相同 | 宿主序列化/缓存差异不是唯一原因 | 两路共用的 target lowering/kernel/encoding bug |
| `Unsupported HTP Arch 1 79` | 告警不像一个会立即阻断运行的错误 | context 实际 target、op placement、stub/skel/driver 组合错误 |
| min-max 为局部 MSE 最优 | 所测张量/样本下的直接重建 MSE | 端到端图像最优、跨 prompt/timestep 覆盖、HTP 重定标敏感性 |
| 全局 symmetric 编译失败 | 这次全局 CLI 方案不可用 | FC-only symmetric，RmsNorm 保留支持的类型，per-op override |
| O=3 无改善 | 这一次 O=3 不是现成修复 | 具体 fusion/tiling/lowering kernel 问题；MD5 改变不证明关键节点变化 |
| Convert 前后余弦相同 | Convert 没显著改变这个余弦 | 全局 scale、bias、少数元素、bit-exact；需 MAE/max diff/直接 bypass |
| Q/K/V 入口已坏 | 错误在 attention 之前已经存在 | attention/Softmax 是否继续放大；需给 attention 相同正确 Q/K/V |
| gamma bit 数与误差不单调 | “gamma bit 数单调决定误差”不成立 | RmsNorm reduction、rsqrt、epsilon、融合和输入范围 |
| 零 bias 占位 | 模型语义上没有 bias | kernel/OpDef 对 scale=0 元数据的处理；需移除占位 A/B |
| 当前 W16A16 被 LMKD 杀 | 当前切分/并存/运行方式不可行 | 少量敏感层 W16、更多分段、load-run-release；文件大小也不等于峰值驻留内存 |
| 当前 W8A8 tensor 误差大 | 当前 encoding/该 tensor 的纯 W8A8 不合适 | 混合精度、per-channel、通道重平衡 |
| part2 CPU 替换后更差 | 这一个组合没有收益，并存在误差抵消或分布外输入 | 修复首个错误段/算子是否有效；不能泛化成“单段修复无效” |
| SD1.5/SDXL 能跑 | 该设备不是所有 W8A16 扩散模型都失败 | 当前 SDK、DiT 算子组合、massive activation、同一 lowering/kernel；版本和模型均是混杂变量 |

“拆更多段”也不能简单列为数值方向已排除：总加载字节数不变，确实未必改善延迟；但它可能显著降低**同时驻留峰值**，从而让局部 W16A16/FP16 方案从内存上变得可测。内存可行性和延迟可接受性是两个不同门禁。

## 5. 还应排查什么：按信息增益排序

### P0：先证明现有对照真的可比

1. **直接比较 HTP-in-loop 与 App**
   - 比较最终图和三个 transformer 边界张量，而不是各自只对第三个参考。
   - 记录 raw SHA-256、name、shape、dtype、layout、encoding、MAE、max error、cosine。
   - 判据：若二者边界输出不接近，App/边界 contract 尚未排除；若接近，才能把 App 独有代码降为低优先级。

2. **确认 SNPE CPU 语义**
   - 明确普通 CPU 与 CPU Fixed Point Mode 的配置和输出。
   - 判据：若只有普通 CPU 好、fixed-point CPU 已明显坏，问题可能是定点图本身/量化交互；若 fixed-point CPU 好而 HTP 坏，才更强地指向 HTP target lowering/kernel/runtime。

3. **冻结并核对完整版本包**
   - 哈希 converter、quantizer、graph prepare/context generator、`libQnnHtp.so`、`libQnnSystem.so`、v79 stub/skel；记录设备 build fingerprint、驱动/固件信息。
   - 用 SDK 自带的 context binary utility / `validate_binary`（以本机 2.48 工具帮助为准）核对产物元数据确实是 SoC model 69 / v79，并从运行日志确认 App 实际加载路径，而不是只看文件名。
   - Qualcomm 官方部署指南建议 target 使用与模型编译相同的 QAIRT 版本。当前报告只给出 host SDK 2.48，没有证明实际被加载的全部 target 库属于同一版本。

### P1：找“第一个分歧”，不要在已经毁掉的输出上猜

4. **固定相同输入，按拓扑在 part1b Id 0–306 定位**
   - 每个 shard 独立喂入同一份捕获输入，避免把上游 HTP 错误级联到下游。
   - 按数据依赖做二分，优先在 FC、FC 输出 requant、Residual Add、RmsNorm、QKV、MatMul 后取点。
   - 插探针可能改变 fusion；所以发现首个异常点后，必须切成等价小子图复验，并保留未插桩基线。

5. **去噪步 teacher-forcing**
   - 每个 timestep 都给 CPU 和 HTP 相同的 golden latent，而不是让 HTP 的上一轮输出反馈到下一轮。
   - 这能分开“单步模型误差”和“8 步反馈放大/误差抵消”。

### P2：在首个异常小图上区分机理

6. **核对 target 实际 encodings**
   - 比较 converter 后与 HTP prepare/context 中每个相关 tensor 的 bitwidth、scale、offset、signedness、per-axis 轴、权重转置后的轴和插入的 Convert/Requant。
   - 判据：只要 target metadata 与脚本 QDQ 使用值不同，第 4.2 节当前“表示 vs 计算”结论就不能直接套用。

7. **绕过 float32 I/O 桥**
   - 用明确 encoding 生成 native quantized 输入，并读取 native quantized 输出，与当前 float32 自动 Convert 路径 A/B。
   - 判据：若 native 路径恢复，问题在边界 Convert/encoding/layout；若仍坏，再进入内部 kernel。

8. **massive activation 的可逆缩放实验**
   - 对 ch85 做缩放，对相应 FC 权重做逆缩放，先验证 FP32 输出等价，再比较 HTP。
   - 再做保持数学等价的通道置换：把巨大值从 ch85 移到另一个物理通道，并对权重轴做反向置换。
   - 判据：若 HTP 误差随缩放显著下降，说明 massive activation 通过乘加、饱和或 requant 影响计算；SmoothQuant/通道迁移路线应恢复优先级。
   - 若错误跟随“巨大数值”移动，优先查动态范围/累加；若错误固定跟随某个物理通道或 tile，优先查 per-axis 轴、layout 或 kernel lane/tiling。

9. **累加器/重定标探针**
   - 对首个异常 FC/MatMul，用实际量化整数、zero-point 和 K 维计算理论累加范围；统计是否接近饱和。
   - A/B 输入整体缩放、缩小 K、分块求和、one-hot/零输入。
   - 判据：误差随 K 或输入幅值出现阈值跳变，支持 overflow/saturation；误差近似固定比例则再查 shift/rounding/scale。
   - 对报告中的 0.854 斜率按 token、channel 和数值区间分别拟合，并枚举相邻 producer/consumer 的 scale 比值与二进制 shift；若 0.854 只来自 ch85 或恰好接近某个 scale 比，不能解释成统一“少 4 bit”。

10. **融合、kernel 与 tiling**
    - 只在小图上逐一禁用相关 fusion，改变不影响数学语义的 shape/alignment/K 分块或 VTCM/tiling 条件。
    - 判据：数学等价但某个 kernel/tiling 选择改变后误差消失，优先判断 compiler/kernel regression，而不是笼统归因硬件 bit 数。

11. **选择性 symmetric / W16A16 / FP16**
    - RmsNorm 保持 OpDef 支持类型，仅对首个异常 FC 做 symmetric 或更高精度；不要一开始全图替换 276 个 FC。
    - 优先尝试官方 OpDef 所列的 `FP16 activation/output + SFIXED_POINT_8 static weights` 代表性 block；它比把所有 FC 权重升到 FP16 更符合当前体积约束。
    - 实际测量 context 文件、峰值 PSS/RSS、workspace、每步时延和最终图，不用参数量直接代替 runtime 成本。

### P3：确认版本/设备边界

12. **QAIRT 版本 A/B 应先跑小图**
    - 用成套 converter/generator/runtime/stub/skel，在同一设备和同一小图上比较可获得的受支持版本。
    - 不要只换 generator 或混用不受支持的 runtime；也不要每次重编 6B 全模型。
    - 不能把为 v73/SM8550 生成的 context 直接放到 v79/SM8750 上当版本对照；官方 HTP 文档警告跨 HTP architecture 加载 context 的结果可能不确定。每个 A/B 产物都必须为目标 arch 重新生成。

13. **第二台同 SoC 或 v81 只在小图成立后做**
    - 同一小图在第二台 SM8750 上失败：降低单机固件/硬件个例概率。
    - 同一 2.48 小图在另一台 SM8750 正常：优先查设备固件/runtime 包。
    - v81 + precision compensation 正常：只能证明该配置可绕过，不自动证明 v79 硬件缺陷。

14. **mask/token 长度覆盖**
    - 当前 all-ones mask 只覆盖所测短 prompt。应另测 padding/masked path 和最大 token shape，防止把 `Where_105` 过早定义为全局死路径。

## 6. 官方资料能支持到什么程度

1. Qualcomm 当前 Device Support Table 明列 Snapdragon 8 Elite `SM8750`、SoC model `69`、Hexagon Arch `V79`。HTP backend 文档说明其一般支持 8/16-bit 量化网络；在支持浮点的 SoC 上可执行 FP16 math，并允许 fixed/float 混合图。这证明平台能力存在，不证明本模型的全部算子组合一定能落到该路径。

2. 当前公开 HTP Supported Ops 表把 RmsNorm 的 FP16、INT16、INT8 均列为支持。RmsNorm OpDef 也证实：UFIXED16 activation/output 下，gamma 支持 UFIXED16、SFIXED16、UFIXED8，不支持 SFIXED8。因此原报告“本次全局 symmetric 8-bit gamma 配置会失败”方向基本正确，但不能扩展成“RmsNorm 没有对称/混合精度方案”。同一 OpDef 还列出 FullyConnected 可使用 FP16 activation/output 搭配静态 SFIXED8 weights，这直接否定“FC 一旦浮点，全部权重必然翻倍”的先验二分。

3. Qualcomm 当前官方 QAIRT/SNPE 文档目录明确列有 **CPU Fixed Point Mode**、**Architecture Checker (Experimental)**、**Accuracy Debugger (Experimental)** 和 profiling 相关入口。QHAS/optrace 可以核对图到 HTP 的映射、融合与 kernel 类别，但没有公开依据表明它们能直接读取原始定点累加器。

4. Qualcomm 官方 `qidk` 的 Whisper 示例明确写明 encoder 为 `Quantized-a16w8`，运行于 Snapdragon DSP，并把 SM8750/v79 列为支持设备。这是“Transformer 的 A16W8 在 SM8750/v79 上并非类别性不可用”的一手反例；它不能证明 Z-Image 这套 DiT、尺寸和 massive activation 一定可行。

5. Qualcomm 官方 GenieX QAIRT 插件把 SM8750 映射为 HTP v79 / SoC model 69，并把 runtime 版本作为单一事实来源管理。Qualcomm 官方 LLM 部署指南还建议 target 使用与编译资产相同的 QAIRT 版本。因此版本/库哈希不是可选清单，而是当前 P0 门禁。

6. Qualcomm 官方 AI Hub Apps 仓库写明 Hexagon v69+ 平台支持 FP16 NPU 路径。由此可把原报告“v79 是否支持 FP16 完全未知”修正为：**平台级 FP16 支持有官方依据，但 Z-Image 各算子、混合图、内存和性能仍须用 QAIRT 2.48 实际编译验证。**

7. 当前公开的 `QnnHtpGraph.h` 页面未找到 `PRECISION_COMPENSATION`，公开页面也未找到 `high_precision_sigmoid` / `advanced_activation_fusion`。这不能证明较新的本地 QAIRT 2.48 不含这些项；正确证据应是安装包自己的 `QnnHtpGraph.h`、`QNN_ReleaseNotes.txt`、`SNPE Releasenotes.txt` 及完整工具帮助。即使本地 2.48 确认该功能要求 `min_arch >= 81`，也只能推出它不能用于 v79，不能推出 Qualcomm 已承认 v79 普遍存在本案这种精度缺陷。

官方链接：

- [Qualcomm QAIRT Device Support Table](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/overview.md)
- [Qualcomm QNN HTP backend](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/htp_backend.md)
- [Qualcomm QNN HTP Supported Ops](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/SupportedOps.md)
- [Qualcomm QNN HTP OpDef Supplement](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/HtpOpDefSupplement.md)
- [Qualcomm QNN HTP Graph API](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/api-rst_file_include_QNN_HTP_QnnHtpGraph_h.md)
- [Qualcomm QAIRT / SNPE 文档目录](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-4/developing-apps-qualcomm-neural-processing-sdk.html?product=1601111740010412)
- [Qualcomm QIDK Whisper A16W8 示例](https://github.com/qualcomm/qidk/blob/master/Solutions/NLPSolution3-AutomaticSpeechRecognition-Whisper/README.md)
- [Qualcomm GenieX QAIRT 插件：SM8750=v79、runtime 版本管理](https://github.com/qualcomm/geniex-qairt-plugin)
- [Qualcomm AI Hub Apps：NPU 精度支持范围](https://github.com/qualcomm/ai-hub-apps)
- [Qualcomm LLM 部署指南：建议编译与 target QAIRT 版本一致](https://github.com/qualcomm/ai-hub-apps/blob/main/tutorials/llm_on_genie/README.md)

本次没有找到可公开核实的“Z-Image Turbo 在 SM8750/v79 上 W8A16 全 HTP 成功”官方配方，也没有找到公开资料证明 `PRECISION_COMPENSATION` 就是为本报告这种误差机理设计。两者都不应被写成已知事实。

## 7. 建议给 Claude 的固定执行顺序

在完成前一项门禁前，不进入下一项：

1. 补齐报告引用的脚本、原始输入/输出、图片、日志、配置和所有 runtime 库哈希。
2. 直接证明 HTP-in-loop 与 App 的边界 tensor 一致。
3. 确认 CPU 普通模式与 CPU Fixed Point Mode；明确 CPU 基线的算术语义。
4. 用成套版本核对 host 和 target，并冻结唯一版本矩阵。
5. 以同一捕获输入在 part1b Id 0–306 按拓扑定位首个分歧。
6. 把首个分歧切为可快速重编、可重复运行的小图。
7. 在小图上依次测试：target encoding → native I/O → massive-channel 可逆缩放 → accumulator/K → fusion/tiling → SDK 版本。
8. 只有定位到敏感 op/fused cluster 后，才测试最小范围的 symmetric、W16A16 或 FP16 fallback。
9. 对通过的小图方案回灌三段 transformer，做每 timestep teacher-forcing，再做完整 8 步闭环成图。
10. 最终以四个产品门禁验收：图像语义、峰值 PSS/RSS、总生图时间、固定输入可复现；仅“能编译、能加载、exit 0”不算成功。

## 8. 对项目是否“没有结果”的判断

现阶段不能判定没有结果，原因不是乐观推测，而是根因尚未定位到最小算子，且多个高判别力实验还没有做：CPU fixed-point 对照、native quantized I/O、实际 target encoding、首个分歧小图、成套 SDK 版本 A/B、massive-channel 可逆缩放。

同样也不能承诺一定能在 v79 上以当前质量、内存和 4–5 分钟时延约束完成。合理的 No-Go 条件应是：

> 首个异常小图已经稳定复现；成套受支持 SDK、正确 target encoding/native I/O、可行的重缩放/融合规避及最小范围混合精度均无法在质量、峰值内存和时延门禁内通过。

在达到这个条件前，把问题定性为“v79 天生少 4 bit、只剩 FP16、FC 出问题就项目结束”都过早。
