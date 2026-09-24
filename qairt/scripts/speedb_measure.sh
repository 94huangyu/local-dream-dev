#!/usr/bin/env bash
# EXP_PLAN_SPEEDB 的**唯一**测量入口。B1 / B2 / M1 必须全部走这一个脚本，
# 否则「口径相同」只是口头承诺（本项目已因多处副本各写一遍栽过四次，指南 §35.4）。
#
# 用法: bash scripts/speedb_measure.sh <tag> [--no-forcestop]
#
# 产出（logs/speedb/<tag>.*）：
#   .summary   一行式结论（时间 / RSS 峰值 / 温度 / sha256）
#   .rss       1 Hz 轮询序列：epoch VmRSS_kB
#   .temps     生成前后的全部 85 个热分区
#   .gen.log   app_generate.sh 的完整输出
# 图片与 logcat 由 app_generate.sh 落在 scratch_runs/<tag>.*
set -uo pipefail
cd /d/LocalDreamZImage || exit 90

ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
PKG=io.github.xororz.localdream.zimage
export MSYS_NO_PATHCONV=1
export PYTHONIOENCODING=utf-8

TAG="${1:?usage: speedb_measure.sh <tag> [--no-forcestop]}"
NOFS="${2:-}"
SEED=777
# 固定 prompt（与 #141 诊断同一条，便于与那批热数据对照）
PROMPT="A quiet street corner in early autumn with a bicycle parked by a lamppost and fallen leaves on the pavement"

OUT=logs/speedb
mkdir -p "$OUT"
for ext in summary rss temps gen.log; do
  if [ -e "$OUT/$TAG.$ext" ]; then
    echo "🔴 $OUT/$TAG.$ext 已存在 —— 换新 tag，不得覆盖上一次现场（#141 教训）" >&2
    exit 3
  fi
done

sh_() { "$ADB" shell "$1" < /dev/null 2>/dev/null | tr -d '\r'; }
"$ADB" get-state >/dev/null 2>&1 < /dev/null || { echo "🔴 设备不可达，请插线" >&2; exit 2; }

temps() {  # $1 = 标签
  echo "--- $1 $(date +%H:%M:%S) ---"
  sh_ 'for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done'
}
npu_max() {  # 七个 nsp 分区的最大值，单位 0.001 C
  sh_ 'cat /sys/class/thermal/thermal_zone*/type' > /tmp/_ty.$$
  sh_ 'cat /sys/class/thermal/thermal_zone*/temp' > /tmp/_tv.$$
  paste /tmp/_ty.$$ /tmp/_tv.$$ | awk '$1 ~ /^nsph/ {if ($2+0>m) m=$2+0} END{print m+0}'
  rm -f /tmp/_ty.$$ /tmp/_tv.$$
}

echo "=== speedb_measure  tag=$TAG  seed=$SEED  $(date +%F' '%H:%M:%S) ==="

# --- 步 1：force-stop，使 VmHWM 恰好覆盖这一次生成（方案 §3.2）---
if [ "$NOFS" != "--no-forcestop" ]; then
  echo "[1] force-stop $PKG（让后端成为全新进程 ⇒ VmHWM 只统计本次）"
  sh_ "am force-stop $PKG"
  sleep 3
fi
PID_BEFORE=$(sh_ "pidof libstable_diffusion_core.so" | awk '{print $1}')
echo "    force-stop 后 后端PID = '${PID_BEFORE:-无}'（应为空）"

# --- 步 2：温度与内存前置快照 ---
temps "PRE" > "$OUT/$TAG.temps"
T_PRE=$(npu_max)
MEM_PRE=$(sh_ 'grep -E "MemFree|MemAvailable" /proc/meminfo' | tr '\n' ' ')
echo "[2] PRE  NPU最高=$(awk -v v="$T_PRE" 'BEGIN{printf "%.1f",v/1000}') C   $MEM_PRE"

# --- 步 3：设备侧 1 Hz 采样（覆盖整个 app_generate.sh，含加载期）---
# 🔴 采的是 IonTotalUsed，不是进程 VmRSS。B1 实测证明 VmRSS 看不见 context：
#    生成全程 RSS 仅 ~200 MiB，峰值 2356 MiB 只是「.bin 读进 CPU 缓冲区」的瞬时值。
#    context 常驻在 DSP/ION 侧。详见 EXP_PLAN_SPEEDB §1.2。
"$ADB" push "$(cygpath -w scripts/speedb_sampler.sh)" /data/local/tmp/speedb_sampler.sh >/dev/null 2>&1 < /dev/null
sh_ 'chmod 755 /data/local/tmp/speedb_sampler.sh'
"$ADB" shell 'sh /data/local/tmp/speedb_sampler.sh' < /dev/null > "$OUT/$TAG.rss" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; "$ADB" shell "pkill -f speedb_sampler" < /dev/null >/dev/null 2>&1' EXIT
sleep 2
ION_IDLE=$(awk 'NR==1{print $2}' "$OUT/$TAG.rss")
echo "    采样器已起，空载 IonTotalUsed = $(awk -v v="${ION_IDLE:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB"

# --- 步 4：真实 app 端到端生成（不用 run-as，约束 11·④）---
echo "[3] 生成中（加载 ~1-2 分钟 + 去噪 ~3.5 分钟）..."
bash scripts/app_generate.sh "$TAG" "$SEED" "$PROMPT" > "$OUT/$TAG.gen.log" 2>&1
GEN_RC=$?

# --- 步 5：立刻读 VmHWM（进程仍在，端口保持监听）---
# ⚠️ VmHWM 只作参考：① 它看不见 DSP/ION 侧的 context；② 内核 update_hiwater_rss()
#    是惰性更新，B1 实测出现过 VmHWM(2412344) < 瞬时 VmRSS(2412992) 的滞后。
PID_AFTER=$(sh_ "pidof libstable_diffusion_core.so" | awk '{print $1}')
VMHWM=$(sh_ "awk '/VmHWM/{print \$2}' /proc/${PID_AFTER:-0}/status")
kill $SAMPLER 2>/dev/null; sh_ 'pkill -f speedb_sampler'; sleep 1

temps "POST" >> "$OUT/$TAG.temps"
T_POST=$(npu_max)
MEM_POST=$(sh_ 'grep -E "MemFree|MemAvailable" /proc/meminfo' | tr '\n' ' ')

# --- 步 6：汇总 ---
# 列: epoch ion memfree memavail swapfree pid rss npu
ION_PEAK=$(awk '{if ($2+0>m) m=$2+0} END{print m+0}' "$OUT/$TAG.rss")
ION_MIN=$(awk 'NR==1{m=$2+0} {if ($2+0<m) m=$2+0} END{print m+0}' "$OUT/$TAG.rss")
MA_MIN=$(awk 'NR==1{m=$4+0} {if ($4+0<m) m=$4+0} END{print m+0}' "$OUT/$TAG.rss")
MF_MIN=$(awk 'NR==1{m=$3+0} {if ($3+0<m) m=$3+0} END{print m+0}' "$OUT/$TAG.rss")
RSS_PEAK=$(awk '{if ($7+0>m) m=$7+0} END{print m+0}' "$OUT/$TAG.rss")
NPU_PEAK=$(awk '{if ($8+0>m) m=$8+0} END{print m+0}' "$OUT/$TAG.rss")
RSS_N=$(wc -l < "$OUT/$TAG.rss")
HTTPLINE=$(grep -a '^HTTP ' "$OUT/$TAG.gen.log")
TIME_TOTAL=$(echo "$HTTPLINE" | awk '{for(i=1;i<=NF;i++) if($i=="time") print $(i+1)}' | tr -d 's')
HTTP_CODE=$(echo "$HTTPLINE" | awk '{print $2}')
SHA=$(sha256sum "scratch_runs/$TAG.png" 2>/dev/null | awk '{print $1}')

{
echo "tag            $TAG"
echo "gen_rc         $GEN_RC"
echo "http_code      ${HTTP_CODE:-NA}"
echo "time_total_s   ${TIME_TOTAL:-NA}       <- 主时钟（判据用这个）"
echo "png_sha256     ${SHA:-NA}"
echo "pid_before     ${PID_BEFORE:-无}"
echo "pid_after      ${PID_AFTER:-无}"
echo "ion_peak_kB    ${ION_PEAK:-NA}         ($(awk -v v="${ION_PEAK:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB)  <- 内存主判据"
echo "ion_min_kB     ${ION_MIN:-NA}          ($(awk -v v="${ION_MIN:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB)  空载底"
echo "ion_delta_MiB  $(awk -v a="${ION_PEAK:-0}" -v b="${ION_MIN:-0}" 'BEGIN{printf "%.0f",(a-b)/1024}')"
echo "memavail_min   ${MA_MIN:-NA}           ($(awk -v v="${MA_MIN:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB)"
echo "memfree_min    ${MF_MIN:-NA}           ($(awk -v v="${MF_MIN:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB)"
echo "rss_peak_kB    ${RSS_PEAK:-NA}         ($(awk -v v="${RSS_PEAK:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB)  仅参考，看不见 context"
echo "vmhwm_kB       ${VMHWM:-NA}            ($(awk -v v="${VMHWM:-0}" 'BEGIN{printf "%.0f",v/1024}') MiB)  仅参考"
echo "npu_peak_C     $(awk -v v="${NPU_PEAK:-0}" 'BEGIN{printf "%.1f",v/1000}')   (采样 $RSS_N 点)"
echo "npu_pre_C      $(awk -v v="$T_PRE" 'BEGIN{printf "%.1f",v/1000}')"
echo "npu_post_C     $(awk -v v="$T_POST" 'BEGIN{printf "%.1f",v/1000}')"
echo "mem_pre        $MEM_PRE"
echo "mem_post       $MEM_POST"
} > "$OUT/$TAG.summary"

cat "$OUT/$TAG.summary"

# --- 门 0c/0d：装置自证（约束 8）---
echo "--- 装置自检 ---"
# 0c 已知样本：生成期间任一时刻至少有一个 part2 半段（1.42 GB）+ part1a/1b 常驻在 DSP 侧。
#     若 ION 增量连 1 GiB 都不到，说明这把尺子仍然看不见 context ⇒ 不得用它下内存结论。
ION_DELTA=$(awk -v a="${ION_PEAK:-0}" -v b="${ION_MIN:-0}" 'BEGIN{print int((a-b)/1024)}')
if [ "${ION_DELTA:-0}" -ge 1024 ] 2>/dev/null; then
  echo "  0c ION 增量 >= 1 GiB  : PASS (${ION_DELTA} MiB) —— 该口径确实看得见 context"
else
  echo "  0c ION 增量 >= 1 GiB  : 🔴 FAIL (${ION_DELTA} MiB) —— 这把尺子仍看不见 context，"
  echo "                          **不得用它下任何内存结论**，先换口径（约束 8）"
fi
if [ -z "$PID_BEFORE" ] && [ -n "$PID_AFTER" ]; then
  echo "  0d 后端为本次新建进程 : PASS (PID=$PID_AFTER)"
else
  echo "  0d 后端为本次新建进程 : 🔴 注意 before='$PID_BEFORE' after='$PID_AFTER'"
fi

exit $GEN_RC
