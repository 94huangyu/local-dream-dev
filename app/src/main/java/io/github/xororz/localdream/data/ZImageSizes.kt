package io.github.xororz.localdream.data

/**
 * 已交付的 Z-Image 生成尺寸（台账 #86）。
 *
 * 🔴 必须与 `cpp/src/Config.hpp` 的 `zimage_sizes` 一致。
 * 之所以允许这里存在第二份清单（#73 通常禁止），是因为**漂移会响而不是静默**：
 *   - UI 列了而 C++ 没列  -> RequestParser 把尺寸打回 1024，PipelineZImage 抛
 *     "outside the delivered size/8/CFG0 profile"；
 *   - C++ 列了而交付里没有对应的图 -> 抛 "delivery has no graphs for size WxH"。
 * 两条都是硬失败，不存在「按错误形状喂进 QNN 图」的可能。
 *
 * 每个非 1:1 尺寸都要求交付契约里有对应的 `size_variants` 条目（整套图规格）。
 * 1184x896 的画质依据：D2 双臂 n=3，1:1 均值 18.96 dB vs 4:3 均值 18.59 dB，
 * 差 0.37 dB（判据 <= 2 dB）。
 */
val zimageDeliveredSizes: List<Resolution> = listOf(
    Resolution(1024, 1024),   // 1:1
    Resolution(1184, 896),    // 4:3
    Resolution(896, 1184),    // 3:4
    Resolution(1280, 720),    // 16:9
    Resolution(720, 1280),    // 9:16
)

fun isZImageDeliveredSize(width: Int, height: Int): Boolean =
    zimageDeliveredSizes.any { it.width == width && it.height == height }
