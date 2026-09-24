#!/usr/bin/env bash
# Tier 2 门 无人值守链：emit L=80 段间输入 -> 四段 amax -> 判决
# 方案 scripts/EXP_PLAN_T2_CLIPSET.md §四 步 2~4
# 🔴 每一步都单独查返回码；**任何一步失败立即停**，不许把失败拖进下一步。
# 🔴 不套 tail/head（管道会吞返回码，见 MAINLINE §7.4）。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_gate
mkdir -p "$LOG"

step () {                      # step <名字> <命令...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $name ==="
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "--- $name rc=$rc ---"
  tail -n 25 "$LOG/$name.log"
  if [ $rc -ne 0 ]; then
    echo "🔴 $name 失败 rc=$rc，整条链停止（日志 $LOG/$name.log）"
    exit $rc
  fi
}

step emit        python scripts/t2_chain_L80.py emit
for s in part1a part1b part2a part2b; do
  step "amax_$s" python scripts/maxfp16_stats.py "$s" --L80 --budget-gb 6
done
step judge       python scripts/t2_clipset_gate.py judge
echo "=== [$(date +%H:%M:%S)] 全链完成 ==="
