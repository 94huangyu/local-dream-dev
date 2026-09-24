# -*- coding: utf-8 -*-
"""D 盘清理：只删**已关闭路线**的产物与**文档判定可再生**的中间品。

纪律（§7.5 + DISK_INVENTORY）：
  · `.onnx` 与 `.data` 是多对一 —— 本脚本**完全不碰** onnx 目录（已实查：
    剩下的 3 个大 .data 各有 13/32/37 个引用，无可删项；15.69 GB 那个上一轮已清）
  · fp32 DLC 是「≈2 分钟可再生」的中间品（DISK_INVENTORY §1 表），量化 DLC ≈3 小时，保留
  · 每一项都要指出**关闭它的台账编号**，没有依据的一律不删

默认 **dry-run**，加 --apply 才真删。
"""
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
P2 = os.path.join(P0, "p2attr")

# (路径, 依据)
DIRS = [
    (os.path.join(P0, "o3_all"),
     "#147 已关闭：O3 的 part1a 击穿 PD 红线(0x3ea)，且 ION 代价不可预测 ⇒ 不可交付。"
     "四段量化 DLC 仍在，需要时约 95 分钟可重建"),
    (os.path.join(P0, "odlbc_speed"),
     "§6 的 B 已关闭：dlbc 无差异；CTRL 与部署版 md5 逐字节相同（冗余）；O3 见 #147"),
    (os.path.join(P2, "ctx_part2a_dt"),
     "#123 已关闭：dtype 机制产物在设备上不兑现收益（模拟 23.75% / 设备 43.38%），不再用它做实验"),
    (os.path.join(P2, "ctx_part2b_dt"),
     "#123 同上"),
]

FILES = [
    # --- dtype 机制（#123 关闭）---
    (os.path.join(P2, "part2a_dt_quantized.dlc"), "#123 dtype 机制已关闭"),
    (os.path.join(P2, "part2b_dt_quantized.dlc"), "#123 dtype 机制已关闭"),
    (os.path.join(P2, "part2a_dt_fp32.dlc"), "#123 已关闭，且 fp32 DLC 本就 2 分钟可再生"),
    (os.path.join(P2, "part2b_dt_fp32.dlc"), "#123 已关闭，且 fp32 DLC 本就 2 分钟可再生"),
    # --- fp32 DLC：文档明定「≈2 分钟可再生的廉价中间品」---
    (os.path.join(P2, "part1a_fp16_L80_fp32.dlc"), "fp32 DLC 可再生（DISK_INVENTORY §1）"),
    (os.path.join(P2, "part1b_fp16_L80_fp32.dlc"), "fp32 DLC 可再生"),
    (os.path.join(P2, "part2a_fp16_L80_fp32.dlc"), "fp32 DLC 可再生"),
    (os.path.join(P2, "part2b_fp16_L80_fp32.dlc"), "fp32 DLC 可再生"),
    (os.path.join(P2, "part2a_ctrl_fp32.dlc"), "fp32 DLC 可再生；且 ctrl 臂实验已结束"),
]


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    t = 0
    for r, _d, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def main():
    apply = "--apply" in sys.argv
    print("模式：%s\n" % ("🔴 真删（--apply）" if apply else "dry-run（加 --apply 才真删）"))
    tot = 0
    for p, why in DIRS + FILES:
        if not os.path.exists(p):
            print("  （不存在，跳过） %s" % os.path.basename(p))
            continue
        s = size_of(p)
        tot += s
        print("  %8.2f GB  %-36s  ← %s" % (s / 1e9, os.path.basename(p), why))
        if apply:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    print("\n合计 %.2f GB%s" % (tot / 1e9, "  已删除" if apply else "  （未删）"))


if __name__ == "__main__":
    main()
