# legacy_export —— 2026-08-03 的原始转换脚本（**孤本存档，非本项目维护的工具**）

## 这是什么

Z-Image Turbo 最初那轮 ONNX→DLC→context 的转换脚本，由**接手之前**的人写的。
2026-09-20 发现时它们只存在于 `D:\ZImage_Work\Cloud_Package\`，**仓库里没有任何副本**，
而 `docs/QNN_CONVERSION_GUIDE.md` §3.1 是全项目唯一引用它的地方。
⇒ 按约束 11 第 2 条（覆盖/依赖任何东西前先保全原件）拷进仓库。

| 文件 | 行数 | 内容 |
|---|---|---|
| `convert_qairt_zimage.py` | 514 | 全流程驱动：校准数据转换、`freeze_dynamic_shapes`、converter/quantizer 调用、context 生成 |
| `fix_qnn_graph.py` | 84 | `fix_onnx_model_for_qnn()` |

**存档时逐字节核对**：`convert_qairt_zimage.py` 宿主原件与本副本
sha256 均为 `f80f6f2d7f15c412528da76827f1d8b023b6c2153323000bb395780a4117338c`。

## 🔴 不要直接拿来跑

1. **它是 2026-08-03 的配方，已被推翻多处**。最要命的是量化那段用的是
   `--use_per_channel_quantization` —— 在 DiT 上**完全空转**（零 Convolution），
   正解是 `--use_per_row_quantization`（指南 §11.1）。
   照它跑等于放弃本项目唯一实测有效的精度手段（E_all 19.24% → 7.87%）。
2. **路径、模型名、分段全部是 Z-Image 专属的**，换模型要重写。
3. 当前的正确流程在 **`docs/QNN_CONVERSION_GUIDE.md` §47.1.1 六张执行卡片**。

## 它唯一不可替代的价值：`freeze_dynamic_shapes`（第 234~389 行）

"固定动态 shape"是卡 B 的第一步，而**这是全项目唯一的实现**。
其算法与其中 5 个通用坑已于 2026-09-20 提炼进 **指南 §3.1**，
所以**正常情况下读 §3.1 就够了**；本文件留作实现细节的对照源。

⚠️ 若 §3.1 与本脚本冲突，**以 §3.1 为准**（它经过了后续实测订正），但请回来订正本 README。
