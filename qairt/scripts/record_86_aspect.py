# -*- coding: utf-8 -*-
"""把「不同比例生图」的实测证据写进台账 #86。

原条目只有一行来自 local-dream 的转述（且是我上次回答用户时写下的），
本项目零实测。本次补上五条一手证据。
"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

ADD = (
    u"<br>\U0001F7E2\U0001F7E2 **2026-08-30：用户明确需求 = 【不同比例】（非放大）。"
    u"本项目首次一手实测，结论比原转述乐观得多。**<br>"
    u"①**比例在运行时是免费的**：按等 token 预算分桶（1152×896 / 1344×768 / 768×1344 等）"
    u"图像 token **4032 vs 现在 4096（−1.6%）** ⇒ **速度、ION、PD 全部不变**。"
    u"（token = H×W/256：除 8 是 VAE、除 2 是 patch。1024²→64×64=4096，与契约 unified 4176−80 吻合。）<br>"
    u"①**官方机制已读代码证实**（`QnnRuntime.hpp:186` `applyZstdPatchToBuffer`）："
    u"`ZSTD_decompress_usingDict(..., 基座.bin 作字典)` ⇒ `.patch` 不是二进制 diff，"
    u"是**以基座为字典的 zstd 帧**，解压出该分辨率的**完整 context**，内存加载后立即释放。"
    u"分辨率清单就是 `<边长>.patch` / `<宽>x<高>.patch` 的文件名（`Model.kt:34`）。"
    u"⇒ **官方确实是「每比例一份 context」**，只是分发时压成增量。<br>"
    u"\U0001F534 **我原先担心的「高宽同值无法分辨」不成立** —— 实测两处都能分辨：<br>"
    u"　① `val_70 = [16,1,1,64,2,64,2]` 是 **patchify 的 reshape**"
    u"（乘积 262144 = 16×128×128，正好是 latent 元素数）⇒ H 与 W **按位置区分**"
    u"（第 3 位 / 第 5 位），改成 `[16,1,1,H/2,2,W/2,2]` 即可；<br>"
    u"　② `stack_1` int32 `[1,64,64,3]` 是**烘焙的二维位置网格**，"
    u"**实读内容 `stack_1[0,h,w,:] = [33, h, w]`**（通道 0 恒 33，通道 1/2 各 0..63）"
    u"⇒ 它把 H/W 写成**独立的数组维度**，改比例只需按 `[1,H/2,W/2,3]` **重新生成，三行 numpy**。<br>"
    u"①**part1a 里的 `128` 不是空间维，是注意力头维**（30 头 × 128 = 3840 = hidden，与 "
    u"`add_138 [1,4176,3840]` 吻合）；唯一真正的空间常量 `latents_shape [1,128,128]` **无消费者**，"
    u"只是透传输出 ⇒ ②**空间结构几乎不体现在形状常量里，主要体现在 `stack_1` 这张数据表上**。<br>"
    u"①**RoPE 频率表 `mul_7/9/65/67` 均为 `[512,24]`** ⇒ 覆盖 512 个位置，"
    u"而 1152×896 只需到 72 ⇒ **不必重新生成频率表**。<br>"
    u"⇒ **改比例 = Tier 2 那类图手术 + 重算 `stack_1` + VAE 单独处理**，"
    u"已有的 `dit_seq_surgery.py`（含 §34.1 的 `value_info` 修复）可复用。<br>"
    u"③**仍未验证**：`unsqueeze_3 [4096,1]` 与 `unified_mask [1,4128]` 的内容与改法；"
    u"VAE 的空间维手术；量程是否需重标定（#139 证过「序列变长不改量程」，"
    u"③**换 token 排布是否同理未验证**）；画质。<br>"
    u"⊕ **顺带**：app **已内置升采样器**（`Model.kt:344`，RealESRGAN x4plus anime/realistic，按芯片分发）"
    u"⇒ 「想要更大的图」**已经是出厂能力，不需要动扩散模型**。 |"
)


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    L = s.split("\n")
    i = [k for k, l in enumerate(L) if l.startswith(u"| 86 | ")]
    assert len(i) == 1, "定位 #86 失败: %r" % i
    assert u"stack_1" not in L[i[0]], "似乎已记录过"
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_86aspect.md"))
    L[i[0]] = L[i[0]][:-1].rstrip() + ADD
    io.open(MAIN, "w", encoding="utf-8", newline="").write("\n".join(L))
    t = io.open(MAIN, encoding="utf-8").read()
    for k in (u"stack_1", u"val_70", u"applyZstdPatchToBuffer", u"4032"):
        assert k in t, k
    print("OK #86 已补记五条一手实测证据")


if __name__ == "__main__":
    main()
