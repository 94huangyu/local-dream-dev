#!/bin/bash
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage
say(){ echo "[两臂 $(date +%H:%M:%S)] $*"; }

say "=== 控制臂（零浮点，纯复现部署配置）==="
python scripts/p2a_ctrl_build.py || exit 11
python scripts/p2a_ctrl_ctx.py   || exit 12
say "控制臂完成"

say "=== FP16 臂（8 个 SwiGLU 转 FP16）==="
python scripts/p2a_fp16_gen.py   || exit 21
python scripts/p2a_fp16_build.py || exit 22
python scripts/p2a_fp16_ctx.py   || exit 23
say "FP16 臂完成"

say "=== G1 注入门 ==="
python scripts/p2a_fp16_g1.py || exit 30
say "=== FP32 真值 add_92 ==="
python scripts/p2a_fp32_add92.py || exit 31
say "=== 两臂装置画像 diff（#96）==="
python scripts/check_ctx_identity.py --diff \
  "D:\ZImage_Work\p0_experiments\p2attr\ctx_ctrl\part2a_ctrl.SM8750.bin" \
  "D:\ZImage_Work\p0_experiments\p2attr\ctx_fp16\part2a_fp16.SM8750.bin"
say "全部完成 —— 等设备"
