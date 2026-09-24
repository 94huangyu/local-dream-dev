#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在**当前交付件**上实跑门 B2 / B3（不是在历史样本上）

为什么单独有这个脚本
--------------------
`scan_requant_edges.py` / `scan_where_clip.py` 的 `--selftest` 用的是 2026-08-14 的历史转储
（part2 尚未拆分），只证明**判据的操作化正确**。
要回答"**当前交付件到底有没有这些缺陷**"，必须拿当前产物的转储跑一遍 —— 就是本脚本。

当前交付的 1:1 基准 DLC 由 `build_lineage.py` 从产物自身定位（谱系锚点 = 设备 sha256）。

零设备、零项目磁盘增量：转储写到 scratchpad。
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SCRATCH", REPO / "scratch_runs" / "gate_verify"))
OUT.mkdir(parents=True, exist_ok=True)

# 当前交付态的 1:1 基准（谱系确认，对应金标准出图 49be8e9a…）
SEGMENTS = {
    "part1a": r"D:\ZImage_Work\p0_experiments\h1_deliver\part1a_fp16_L80_quantized.dlc",
    "part1b": r"D:\ZImage_Work\p0_experiments\p2attr\part1b_fp16_L80_quantized.dlc",
    "part2a": r"D:\ZImage_Work\p0_experiments\p2attr\part2a_fp16_L80_quantized.dlc",
    "part2b": r"D:\ZImage_Work\p0_experiments\p2attr\part2b_fp16_L80_quantized.dlc",
}


def run(cmd, **kw):
    """🔴 透传 stderr 与返回码（约束 11·再补：包装子进程不许吞掉报错）。"""
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", **kw)


def dump(seg: str, dlc: str) -> Path | None:
    out = OUT / f"{seg}_delivery_dlcinfo.txt"
    if out.exists() and out.stat().st_size > 100_000:
        print(f"[{seg}] 复用已有转储 {out.name}（{out.stat().st_size:,} B）")
        return out
    if not Path(dlc).exists():
        print(f"[{seg}] 🔴 DLC 不存在：{dlc}")
        return None
    print(f"[{seg}] 生成转储…（{Path(dlc).stat().st_size / 1e9:.2f} GB，数分钟）")
    r = run([sys.executable, str(REPO / "scripts" / "qairt_tool.py"),
             "snpe-dlc-info", "-i", dlc], capture_output=True, timeout=3600)
    if r.returncode != 0:
        print(f"[{seg}] 🔴 snpe-dlc-info rc={r.returncode}\n{(r.stderr or '')[:400]}")
        return None
    out.write_text(r.stdout, encoding="utf-8")
    print(f"[{seg}] 转储 {out.stat().st_size:,} B -> {out}")
    return out


def main():
    dumps = {}
    for seg, dlc in SEGMENTS.items():
        d = dump(seg, dlc)
        if d:
            dumps[seg] = d

    if not dumps:
        print("\n🔴 没有可用转储，无法下结论")
        return 1

    for tool, title in (("scan_requant_edges.py", "门 B2：requant 极端边"),
                        ("scan_where_clip.py", "门 B3：Where 输出钳位")):
        print(f"\n{'=' * 78}\n{title}（当前交付件）\n{'=' * 78}")
        r = run([sys.executable, str(REPO / "scripts" / tool)] + [str(p) for p in dumps.values()],
                capture_output=True, timeout=1800)
        print(r.stdout or "")
        if r.returncode != 0:
            print(f"🔴 rc={r.returncode}\n{(r.stderr or '')[:400]}")

    print(f"\n转储保留在 {OUT}（下次复用，不重复生成）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
