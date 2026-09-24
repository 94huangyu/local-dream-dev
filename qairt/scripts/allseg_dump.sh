#!/bin/bash
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage
W=/d/ZImage_Work/p0_experiments/p2attr
say(){ echo "[dump $(date +%H:%M:%S)] $*"; }
say "part1a"
python scripts/qairt_tool.py snpe-dlc-info -i 'D:\ZImage_Work\p0_experiments\perrow_p1a\part1a_perrow_quantized.dlc' -d -s "$W/part1a_full_enc.csv" >/dev/null 2>&1 || exit 1
ls -la "$W/part1a_full_enc.csv"
say "part2b"
python scripts/qairt_tool.py snpe-dlc-info -i 'D:\ZImage_Work\p0_experiments\p2split\part2b_fixed_perrow.dlc' -d -s "$W/part2b_full_enc.csv" >/dev/null 2>&1 || exit 2
ls -la "$W/part2b_full_enc.csv"
say "part1b（复用已有，复制一份统一命名）"
cp /d/ZImage_Work/p0_experiments/p0b/part1b_perrow_encodings.csv "$W/part1b_full_enc.csv"
ls -la "$W/part1b_full_enc.csv"
say "三段转储完成"
