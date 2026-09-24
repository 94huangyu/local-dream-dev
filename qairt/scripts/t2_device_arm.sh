#!/usr/bin/env bash
# Tier 2 一致性测量的设备半边：把 L=80 量化四段推到**研究工作区**并跑 8 步 in-loop。
#
# 🔴 只动 /data/local/tmp/htpcmp（研究工作区），**完全不碰 app 的模型目录**
#    —— 刚交付的东西与本实验隔离。
# 🔴 两臂必须共用同一份初始噪声与同一份 caption：
#    · INLOOP_LATENTS -> 与 FP32 参考臂同一个 lat_init
#    · INLOOP_CAPTION/INLOOP_MASK -> FP32 TE 在 L=80 下算出的 caption（#131：
#      两臂都用 FP32 TE，把「TE 量化」这个变量排除掉）
# 🔴 每步单独查 rc，不套 tail/head（§7.4）；adb 路径先过 cygpath -w（#83）。
set -u
export PYTHONIOENCODING=utf-8
ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
export MSYS_NO_PATHCONV=1
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_device; mkdir -p "$LOG"
T=/data/local/tmp/htpcmp
P2A=/d/ZImage_Work/p0_experiments/p2attr

w() { cygpath -w "$1"; }

"$ADB" get-state >/dev/null 2>&1 < /dev/null || { echo "🔴 设备不可达，请插线"; exit 2; }
echo "设备: $("$ADB" shell getprop ro.serialno < /dev/null | tr -d '\r')"

# 设备上的目标名 | 宿主源
BINS="
part1a_fp16_L80.bin|$P2A/ctx_part1a_fp16_L80/part1a_fp16_L80.SM8750.bin
part1b_fp16_L80.bin|$P2A/ctx_part1b_fp16_L80/part1b_fp16_L80.SM8750.bin
part2a_fixed_fp16_L80.bin|$P2A/ctx_part2a_fp16_L80/part2a_fp16_L80.SM8750.bin
part2b_fixed_fp16_L80.bin|$P2A/ctx_part2b_fp16_L80/part2b_fp16_L80.SM8750.bin
"

echo "=== 推 4 段 L=80 到研究工作区（约 6.4 GB）==="
while IFS='|' read -r dst src; do
  [ -z "$dst" ] && continue
  [ -f "$src" ] || { echo "🔴 缺 $src"; exit 1; }
  have=$("$ADB" shell "stat -c %s $T/$dst 2>/dev/null" < /dev/null | tr -d '\r')
  want=$(stat -c %s "$src")
  if [ "$have" = "$want" ]; then echo "  跳过 $dst（已存在且大小一致 $want）"; continue; fi
  echo "--- $dst  $(( want / 1048576 )) MB ---"
  "$ADB" push "$(w "$src")" "$T/$dst" < /dev/null || exit 1
  got=$("$ADB" shell "stat -c %s $T/$dst" < /dev/null | tr -d '\r')
  [ "$got" = "$want" ] || { echo "🔴 $dst 落盘大小不符 $got != $want"; exit 1; }
  echo "    ✅ $got B"
done <<< "$BINS"

echo "=== 8 步 in-loop（四段，L=80）==="
export INLOOP_TAG=L80_fp16
export INLOOP_BIN_SUFFIX=_fp16_L80
export INLOOP_P2=split
export INLOOP_LATENTS='D:\ZImage_Work\p0_experiments\htp_inloop\s0\latents.raw'
export INLOOP_CAPTION='D:\ZImage_Work\p0_experiments\caption_cat_L80.raw'
export INLOOP_MASK='D:\ZImage_Work\p0_experiments\cap_pad_mask_cat_L80.raw'
python scripts/htp_inloop_pipeline.py > "$LOG/inloop_L80.log" 2>&1
rc=$?
echo "--- in-loop rc=$rc ---"
tail -n 30 "$LOG/inloop_L80.log"
if [ $rc -ne 0 ]; then echo "🔴 in-loop 失败 rc=$rc"; exit $rc; fi
echo "=== 完成 $(date +%H:%M:%S) ==="
