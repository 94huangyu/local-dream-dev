#!/bin/bash
# 用实测量程扩充后的白名单重建 part1b（2->5）与 part1a（8->11）
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 9
for s in part1b part1a; do
  echo "############ $s  $(date +%H:%M:%S) ############"
  python scripts/seg_fp16_build.py "$s" || exit 1
done
echo "重建完成 $(date +%H:%M:%S)"
