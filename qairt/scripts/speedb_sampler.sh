#!/system/bin/sh
# 设备侧 1 Hz 采样器（由 speedb_measure.sh 推送并启动）。
#
# 🔴 为什么必须采 IonTotalUsed 而不是进程 VmRSS：
#   2026-08-29 B1 实测，后端进程 VmRSS 生成全程只有 ~200 MiB，峰值 2356 MiB 出现在
#   「把 .bin 读进 CPU 缓冲区」那一瞬（≈ part1a 文件大小），context 交给 DSP 后即释放。
#   ⇒ 常驻的 context 内存在 DSP/ION 侧，**进程 RSS 根本看不见它**。
#
# 输出列: epoch ion_used_kB memfree_kB memavail_kB swapfree_kB pid rss_kB npu_mC
#
# 启动时把 nsp* 热分区的路径缓存一次。每 tick 重扫 85 个分区会把采样率压到 0.5 Hz
# （2026-08-29 实测），而 context 加载是秒级事件，采样率不足会漏掉峰值。
NSPZ=""
for Z in /sys/class/thermal/thermal_zone*; do
  case "$(cat $Z/type 2>/dev/null)" in
    nsph*) NSPZ="$NSPZ $Z/temp" ;;
  esac
done

while :; do
  TS=$(date +%s)
  ION=$(awk '/IonTotalUsed/{print $2}' /proc/meminfo)
  MF=$(awk '/^MemFree/{print $2}' /proc/meminfo)
  MA=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  SF=$(awk '/^SwapFree/{print $2}' /proc/meminfo)
  P=$(pidof libstable_diffusion_core.so 2>/dev/null)
  RSS=0
  if [ -n "$P" ]; then
    RSS=$(awk '/VmRSS/{print $2}' /proc/$P/status 2>/dev/null)
    [ -z "$RSS" ] && RSS=0
  fi
  NPU=0
  for F in $NSPZ; do
    V=$(cat $F 2>/dev/null)
    [ -n "$V" ] && [ "$V" -gt "$NPU" ] && NPU=$V
  done
  echo "$TS ${ION:-0} ${MF:-0} ${MA:-0} ${SF:-0} ${P:-0} $RSS $NPU"
  sleep 1
done
