#!/usr/bin/env bash
# D：把 L=80 一致性从 n=1 补到 n=3。本脚本只做**宿主半边**（FP32 参考），零设备。
#
# 为什么要补：2026-08-29 报的「未发现退化」是 n=1，而 #131 已实测**同一改动在两个 prompt
# 上可差 5.6 dB** => 单样本判决功率不足（约束 7）。
# 按 #131 的发现，**约束强度**是关键变量，故一强一弱：
#   c1 弱约束「a cat on a mat」（正文 5 token，67 槽 padding）—— 自由度最大
#   c2 强约束「黄铜罗盘 + 海图 + 绳圈 + 油灯 + 舷窗光」（正文 43 token）—— 且非猫场景
# 设备半边（量化臂 in-loop）等有设备时再做，两臂共用同一 caption 与同一初始噪声。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_consist; mkdir -p "$LOG"
P0='D:\ZImage_Work\p0_experiments'
INIT="$P0\htp_inloop\s0\latents.raw"      # 与已有两臂同一初始噪声（md5 99ff38fdde…）

step () { local n="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $n ==="
  "$@" > "$LOG/$n.log" 2>&1; local rc=$?
  echo "--- $n rc=$rc ---"; tail -n 8 "$LOG/$n.log"
  if [ $rc -ne 0 ]; then echo "🔴 $n 失败 rc=$rc"; exit $rc; fi
}

for c in c1 c2; do
  step "caption_$c" python scripts/t2_fp32_ref.py caption "$P0\prompt_ids_$c.npy" 80 \
       "$P0\caption_${c}_L80.raw" "$P0\cap_pad_mask_${c}_L80.raw"
done
for c in c1 c2; do
  step "fp32_$c" python scripts/t2_fp32_ref.py drive 80 "$INIT" \
       "$P0\caption_${c}_L80.raw" "$P0\cap_pad_mask_${c}_L80.raw" \
       "D:\LocalDreamZImage\scratch_runs\fp32ref_${c}_L80.png"
  # drive 会把终点 latents 写成 fp32_final_latents_L80.raw（固定名）=> 立刻改名保存，
  # 否则第二个 prompt 会覆盖第一个（同类事故：今天我用同 basename 覆盖过 logcat）
  mv -f /d/LocalDreamZImage/scratch_runs/fp32_final_latents_L80.raw \
        /d/LocalDreamZImage/scratch_runs/fp32_final_latents_${c}_L80.raw || exit 1
  mv -f /d/ZImage_Work/p0_experiments/fp32_steps_L80 \
        /d/ZImage_Work/p0_experiments/fp32_steps_${c}_L80 || exit 1
  echo "  已归档 $c 的终点 latents 与逐步产物"
done
echo "=== [$(date +%H:%M:%S)] 两个 prompt 的 L=80 FP32 参考完成 ==="
