#!/usr/bin/env bash
# D 的设备半边：c1/c2 两个 prompt 的 L=80 量化臂 in-loop。
# 两臂共用同一初始噪声与**同一份 FP32 caption**（#131：把 TE 量化排除）。
# .bin 已在设备研究工作区（上一轮推过，大小已核对）=> 本轮不重推。
# 🔴 只动 /data/local/tmp/htpcmp，不碰 app 的模型目录。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_consist_dev; mkdir -p "$LOG"
ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
export MSYS_NO_PATHCONV=1
"$ADB" get-state >/dev/null 2>&1 < /dev/null || { echo "🔴 设备不可达，请插线"; exit 2; }
P0='D:\ZImage_Work\p0_experiments'

for c in c1 c2; do
  echo "=== [$(date +%H:%M:%S)] $c 量化臂 in-loop ==="
  INLOOP_TAG="L80_$c" \
  INLOOP_BIN_SUFFIX=_fp16_L80 \
  INLOOP_P2=split \
  INLOOP_LATENTS="$P0\htp_inloop\s0\latents.raw" \
  INLOOP_CAPTION="$P0\caption_${c}_L80.raw" \
  INLOOP_MASK="$P0\cap_pad_mask_${c}_L80.raw" \
  python scripts/htp_inloop_pipeline.py > "$LOG/$c.log" 2>&1
  rc=$?
  echo "--- $c rc=$rc ---"; tail -n 8 "$LOG/$c.log"
  if [ $rc -ne 0 ]; then echo "🔴 $c 失败 rc=$rc"; exit $rc; fi
done
echo "=== [$(date +%H:%M:%S)] 两个 prompt 的量化臂完成 ==="
