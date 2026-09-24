#!/usr/bin/env bash
# #116：段间输入「CPU 参考链」vs「纯 FP32 链」的单变量对照。方案 scripts/EXP_PLAN_116.md。
# 全程宿主，零设备。每步单独查 rc，不套 tail/head（§7.4）。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_116; mkdir -p "$LOG"
step () { local n="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $n ==="
  "$@" > "$LOG/$n.log" 2>&1; local rc=$?
  echo "--- $n rc=$rc ---"; tail -n 12 "$LOG/$n.log"
  if [ $rc -ne 0 ]; then echo "🔴 $n 失败 rc=$rc"; exit $rc; fi
}
step dump32 python scripts/t2_chain_L80.py dump32
step amax_pure32 python scripts/maxfp16_stats.py part1b --pure32 --budget-gb 6
step judge python scripts/t2_116_judge.py
echo "=== [$(date +%H:%M:%S)] 完成 ==="
