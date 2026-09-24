# -*- coding: utf-8 -*-
"""#116 判据（方案 scripts/EXP_PLAN_116.md §四，事前锁定，不得修改）。

比 `part1b_amax.json`（testB = CPU 参考链血统）与 `part1b_pure32_amax.json`（纯 FP32 血统）。
唯一变量 = 段间输入的血统。
"""
import io
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
W = r"D:\ZImage_Work\p0_experiments\p2attr"
THR_CLIP, FP16MAX = 65504.0 / 1.2, 65504.0


def main():
    a = json.load(io.open(os.path.join(W, "part1b_amax.json"), encoding="utf-8"))["tensors"]
    b = json.load(io.open(os.path.join(W, "part1b_pure32_amax.json"),
                          encoding="utf-8"))["tensors"]
    pair = [(k, a[k]["amax"], b[k]["amax"]) for k in a if k in b and a[k]["amax"] > 0]
    big = [(k, x, y) for k, x, y in pair if x > 5000]
    r = np.array([y / x for _k, x, y in big])
    print("配对 %d 个；决策相关区间（testB |a|max > 5000）n=%d" % (len(pair), len(big)))
    print("pure32/testB 比值: min %.4f  中位 %.4f  max %.4f" % (r.min(), np.median(r), r.max()))
    out5 = [(k, x, y) for (k, x, y), q in zip(big, r) if q < 0.95 or q > 1.05]
    print("超出 ±5%% 的: %d 个" % len(out5))
    for k, x, y in out5[:8]:
        print("   %-22s testB %9.0f -> pure32 %9.0f  (%.4f)" % (k, x, y, y / x))
    # 是否有张量因血统不同而跨越阈值
    cross = []
    for k, x, y in pair:
        for thr, nm in ((THR_CLIP, "钳位阈值 54586.7"), (FP16MAX, "FP16 上限 65504")):
            if (x > thr) != (y > thr):
                cross.append((k, x, y, nm))
    print()
    if cross:
        print("🔴 FAIL —— 有 %d 个张量因血统不同而**跨越阈值**：" % len(cross))
        for k, x, y, nm in cross[:10]:
            print("   %-22s testB %9.0f / pure32 %9.0f  跨越 %s" % (k, x, y, nm))
        print("⇒ 基于 testB 的白名单/钳位集合不可信，#111/#114 的选择需用纯 FP32 重做")
        return 1
    if len(out5) == 0:
        print("🟢 PASS —— 决策相关区间比值全部落在 0.95~1.05，且无张量跨越阈值")
        print("⇒ #116 的担心不成立，可关闭；基于 testB 的集合不受影响")
    else:
        print("🟡 有偏但不改决策 —— 有 %d 个超出 ±5%%，但**无张量跨越阈值**" % len(out5))
        print("⇒ 结论仍可用，但引用 testB 血统的数值时须加限定")
    print()
    print("⚠️ n=1 个 caption（部署那份 20 真实槽）⇒ 结论限定在该 caption 下")
    return 0


if __name__ == "__main__":
    sys.exit(main())
