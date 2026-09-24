#!/usr/bin/env bash
# 一次跑完剩余三件（用户 2026-08-29「一次性做完，不要停」）。三件都在主线上：
#   ① D 的判决：c1/c2 的三层指标（§6 item D，把 n=1 补到 n=3）
#   ② #141 诊断（台账 #141，只诊断不修 —— 用户要求「必须明确是问题的前提下再修」）
#   ③ ②顺带给出游戏模式方案的直接判据（oom_score_adj 对照），无需改任何代码
#
# 🔴 等前一个设备任务真结束再开始：**同时查进程与产物**（约束 9·补 第 2 条；
#    只看日志内容会被"停在最后一行"骗过）。有超时上限，不空等。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_finish; mkdir -p "$LOG"
WAIT_MAX=2400          # 最多等 40 分钟
WANT=/d/LocalDreamZImage/scratch_runs/htp_inloop_transformer_L80_c2.png

echo "=== 等待 c2 量化臂结束（进程+产物双判，上限 ${WAIT_MAX}s）==="
t0=$(date +%s)
while :; do
  running=$(tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -c "python.exe" || true)
  if [ -f "$WANT" ]; then echo "  产物已出现：$WANT"; break; fi
  now=$(date +%s); el=$(( now - t0 ))
  if [ "$el" -ge "$WAIT_MAX" ]; then echo "🔴 等待超时 ${el}s，停止（不空等）"; exit 3; fi
  if [ "$running" -eq 0 ]; then
    echo "🔴 python 进程已全部结束但产物未出现 => 前一任务失败，停止"; exit 4
  fi
  sleep 30
done
sleep 5

step () { local n="$1"; shift
  echo ""; echo "=== [$(date +%H:%M:%S)] $n ==="
  "$@" > "$LOG/$n.log" 2>&1; local rc=$?
  echo "--- $n rc=$rc ---"; tail -n 30 "$LOG/$n.log"
  if [ "$rc" -ne 0 ] && [ "$n" != "diag141" ]; then
    echo "🔴 $n 失败 rc=$rc"; exit "$rc"
  fi
}

# ① D 的判决
for c in c1 c2; do
  T2_TAG="L80_$c" T2_REF="$c" step "l3_$c" python scripts/t2_l3.py
done

# ② + ③ #141 诊断（含 oom_score_adj 对照）—— 失败也不终止链，它本身就是在找失败
step diag141 bash scripts/t2_141_diag.sh 10

echo ""
echo "=== [$(date +%H:%M:%S)] 全链完成 ==="
