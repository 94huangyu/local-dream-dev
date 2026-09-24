# -*- coding: utf-8 -*-
"""实测「每增加一个比例，分发要多大」——用官方那套 zstd 字典机制。

机制已读代码证实（`QnnRuntime.hpp:186` `applyZstdPatchToBuffer`）：
    ZSTD_decompress_usingDict(..., patch, ..., 基座.bin 作字典)
=> `.patch` 是**以现网 context 为字典**压出来的 zstd 帧，解压得到该比例的完整 context。
权重跨比例完全相同，字典命中率应当极高。

本脚本回答 #86 里标为「③我们的产物压缩比未测」的那个缺口。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import zstandard as zstd

P2 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
ASP = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
TAG = "1152x896"
# 段 -> (基座 = 现网 1:1 context, 新比例 context)
PAIRS = [
    ("part1a", os.path.join(P2, "ctx_part1a_fp16_L80", "part1a_fp16_L80.SM8750.bin"),
     os.path.join(ASP, "ctx_part1a_%s" % TAG, "part1a_%s.SM8750.bin" % TAG)),
    ("part1b", os.path.join(P2, "ctx_part1b_fp16_L80", "part1b_fp16_L80.SM8750.bin"),
     os.path.join(ASP, "ctx_part1b_%s" % TAG, "part1b_%s.SM8750.bin" % TAG)),
    ("part2a", os.path.join(P2, "ctx_part2a_fp16_L80", "part2a_fp16_L80.SM8750.bin"),
     os.path.join(ASP, "ctx_part2a_%s" % TAG, "part2a_%s.SM8750.bin" % TAG)),
    ("part2b", os.path.join(P2, "ctx_part2b_fp16_L80", "part2b_fp16_L80.SM8750.bin"),
     os.path.join(ASP, "ctx_part2b_%s" % TAG, "part2b_%s.SM8750.bin" % TAG)),
]


def main():
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("zstd 级别 %d（官方 app 只负责解压，级别是我们分发时的选择）\n" % lvl)
    print("%-8s %10s %10s %12s %8s" % ("段", "基座MB", "新比例MB", "patch MB", "占比"))
    print("-" * 54)
    tb = tn = tp = 0
    for seg, base, new in PAIRS:
        if not (os.path.isfile(base) and os.path.isfile(new)):
            print("  %-8s 跳过（%s）" % (seg, "基座缺失" if not os.path.isfile(base)
                                         else "新比例尚未建成"))
            continue
        with open(base, "rb") as f:
            d = f.read()
        with open(new, "rb") as f:
            n = f.read()
        t0 = time.time()
        c = zstd.ZstdCompressor(level=lvl, dict_data=zstd.ZstdCompressionDict(d))
        p = c.compress(n)
        # 🔴 必须验证能还原 —— 否则「patch 很小」毫无意义
        dec = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(d))
        assert dec.decompress(p, max_output_size=len(n) + 1) == n, \
            "%s 解压结果与原文件不同，patch 无效" % seg
        tb += len(d); tn += len(n); tp += len(p)
        print("%-8s %9.1f %9.1f %11.1f %7.2f%%   (%.0fs)"
              % (seg, len(d) / 1e6, len(n) / 1e6, len(p) / 1e6,
                 100.0 * len(p) / len(n), time.time() - t0))
    if tp:
        print("-" * 54)
        print("%-8s %9.1f %9.1f %11.1f %7.2f%%" % ("合计", tb/1e6, tn/1e6, tp/1e6,
                                                   100.0*tp/tn))
        print("\n⇒ 每增加一个比例，分发增量约 **%.0f MB**（完整 context 是 %.1f GB）"
              % (tp / 1e6, tn / 1e9))
        print("  解压后仍需 %.1f GB 设备存储 —— 除非按需下载/删除。" % (tn / 1e9))


if __name__ == "__main__":
    main()
