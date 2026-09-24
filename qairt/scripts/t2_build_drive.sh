#!/usr/bin/env bash
# Tier 2 L=80 transformer 四段：overrides -> 转换 -> 量化 -> 建 context
# 配方来源：`seg_fp16_build.py`（**造出部署 18.67 dB 产物的那个脚本**），
# 只加 SEG_L=80（重映射序列维）与 SEG_ONNX_SUFFIX=_clip_L80（用手术后的部署源）。
# 🔴 所有产物名带 _L80，**不覆盖任何部署产物**。
# 🔴 每段单独查返回码；任一段失败立即停。不套 tail/head（管道吞返回码，§7.4）。
set -u
export PYTHONIOENCODING=utf-8
export SEG_L=80
export SEG_ONNX_SUFFIX=_clip_L80
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_build
mkdir -p "$LOG"

for s in part1a part1b part2a part2b; do
  echo "=== [$(date '+%m-%d %H:%M:%S')] $s 开始（转换+量化+建context）==="
  python scripts/seg_fp16_build.py "$s" > "$LOG/$s.log" 2>&1
  rc=$?
  echo "--- $s rc=$rc ---"
  tail -n 30 "$LOG/$s.log"
  if [ $rc -ne 0 ]; then
    echo "🔴 $s 失败 rc=$rc，整条链停止（日志 $LOG/$s.log）"
    exit $rc
  fi
  df -h /d | tail -1
done
echo "=== [$(date '+%m-%d %H:%M:%S')] 四段全部完成 ==="
