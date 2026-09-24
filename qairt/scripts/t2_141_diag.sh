#!/usr/bin/env bash
# #141 诊断（**只诊断不修**）。方案 scripts/EXP_PLAN_141.md，判据事前锁定。
#
# 🔴 与上次失败的关键差异：
#   ① logcat 用 **-b all**（lmkd / crash 不在默认 buffer 里 —— 上次死因查不出就是因为这个）
#   ② 每张用**不同 basename**，杜绝覆盖现场（上次补跑用同名把崩溃 logcat 盖了）
#   ③ 每轮记录 MemFree/MemAvailable + 两个进程的 oom_score_adj 与 RSS，给 H1 提供曲线
#
# 🔴 oom_score_adj 同时是**游戏模式方案的直接判据**（2026-08-29 用户提议）：
#    被杀的是 phantom 子进程。若它的 oom_score_adj 本来就低（已受保护），
#    则 H1(LMK) 不太可能、游戏模式也没什么可改善；
#    若显著高于 app 主进程（典型 cached/phantom 档），它就是 LMK 首要目标，
#    "拿到更多余量"才可能有意义。**这个数不需要改任何代码就能取到。**
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
PKG=io.github.xororz.localdream.zimage
export MSYS_NO_PATHCONV=1
OUT=/d/LocalDreamZImage/logs/t2_141
mkdir -p "$OUT"
N="${1:-10}"
"$ADB" get-state >/dev/null 2>&1 < /dev/null || { echo "🔴 设备不可达，请插线"; exit 2; }

sh_() { "$ADB" shell "$1" < /dev/null 2>/dev/null | tr -d '\r'; }
mem() { sh_ "grep -E 'MemFree|MemAvailable' /proc/meminfo" | tr '\n' ' '; }
bpid() { sh_ "pidof libstable_diffusion_core.so" | awk '{print $1}'; }
apid() { sh_ "pidof $PKG" | awk '{print $1}'; }
pinfo() {   # $1=pid $2=标签
  local p="$1"
  if [ -z "$p" ]; then echo "    $2: 无进程"; return; fi
  local o r
  o=$(sh_ "cat /proc/$p/oom_score_adj")
  r=$(sh_ "awk '/VmRSS/{print \$2}' /proc/$p/status")
  echo "    $2: PID=$p  oom_score_adj=${o:-?}  RSS=${r:-?} kB"
}

echo "=== #141 诊断开始 $(date +%H:%M:%S)  最多 $N 张 ==="
echo "  平台: $(sh_ 'getprop ro.product.model') / Android $(sh_ 'getprop ro.build.version.release')"
echo "  max_phantom_processes = $(sh_ 'dumpsys activity settings' | grep -o 'max_phantom_processes=[0-9]*' | head -1)"
"$ADB" logcat -c < /dev/null 2>/dev/null
"$ADB" logcat -b all -v threadtime > "$OUT/full_logcat.txt" 2>&1 &
LP=$!
trap 'kill $LP 2>/dev/null' EXIT
sleep 2

for i in $(seq 1 "$N"); do
  tag=$(printf "d141_%02d" "$i")
  echo "--- [$(date +%H:%M:%S)] $tag  前: $(mem) ---"
  bash scripts/app_generate.sh "$tag" 777 \
     "A quiet street corner in early autumn with a bicycle parked by a lamppost and fallen leaves on the pavement" \
     > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "    rc=$rc   后: $(mem)"
  pinfo "$(apid)" "app主进程"
  pinfo "$(bpid)" "后端子进程"
  grep -a 'HTTP' "$OUT/$tag.log" 2>/dev/null | head -1
  if [ "$rc" -ne 0 ]; then
    echo "🔴🔴 第 $i 张复现失败 —— 立刻固定现场（现场只有一次）"
    sleep 3
    kill $LP 2>/dev/null
    sleep 1
    cp "$OUT/full_logcat.txt" "$OUT/CRASH_logcat_$tag.txt"
    sh_ "ls -la /data/tombstones/ 2>/dev/null | tail -5" > "$OUT/tombstones_$tag.txt"
    echo "    已存: $OUT/CRASH_logcat_$tag.txt"
    echo "=== 复现于第 $i 张 ==="
    exit 0
  fi
done
kill $LP 2>/dev/null
echo "=== $N 张全部正常，未复现 ==="
exit 0
