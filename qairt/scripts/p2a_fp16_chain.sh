#!/bin/bash
# EXP_PLAN_FP16_SURGICAL 无人值守链：overrides -> convert/quantize -> context -> G1/G3 门
set -o pipefail
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage
W=/d/ZImage_Work/p0_experiments/p2attr
say(){ echo "[chain $(date +%H:%M:%S)] $*"; }

say "1/5 生成 overrides"
python scripts/p2a_fp16_gen.py || exit 1

say "2/5 convert + quantize"
python scripts/p2a_fp16_build.py || exit 2

say "3/5 build context"
python scripts/p2a_fp16_ctx.py || exit 3

say "4/5 G1 注入门（读回数据类型）"
python scripts/p2a_fp16_g1.py || exit 4

say "5/5 FP32 真值 add_92"
python scripts/p2a_fp32_add92.py || exit 5

say "全部完成 —— 等设备回来跑两臂"
