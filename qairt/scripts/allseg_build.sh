#!/bin/bash
# 三段串行构建（part2a 已完成并实测有效，不重建）
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 9
for s in part2b part1b part1a; do
  echo "############ $s  $(date +%H:%M:%S) ############"
  python scripts/seg_fp16_build.py "$s" || exit 1
done
echo "三段全部完成 $(date +%H:%M:%S)"
