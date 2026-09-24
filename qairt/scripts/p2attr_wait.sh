#!/bin/bash
export MSYS_NO_PATHCONV=1
ADB="/c/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe"
SER=3B1F65EA9BBUMSHZ; T=/data/local/tmp/htpcmp
# 约束 9·补：只看进程表 + 产物，不看日志内容
# 🔴 #65 / 约束 11·再补「报错说谎」：必须把「设备够不着」与「进程没了」分开！
#    2026-08-22 实测踩过：用户拔线后 adb 失败，循环把它当成"进程已退出"并误报任务完成。
online() { "$ADB" -s $SER get-state 2>/dev/null | grep -q device; }
running() { "$ADB" -s $SER shell "ps -A -o NAME 2>/dev/null | grep -qi qnn-net-run" 2>/dev/null; }
while true; do
  if ! online; then
    echo "[wait] $(date +%H:%M:%S) ⚠️ 设备离线（拔线/休眠）—— **这不等于任务结束**。"
    echo "[wait]    设备侧是 nohup 脱离会话运行，拔 USB 不会杀它；产物写在手机存储。"
    echo "[wait]    设备回来后用 ls \$T/p2attr/out2/Result_0 核实，不要凭本次退出下结论。"
    exit 8
  fi
  running || break
  M=$("$ADB" -s $SER shell "grep MemAvailable /proc/meminfo" 2>/dev/null | awk '{print $2}')
  N=$("$ADB" -s $SER shell "ls $T/p2attr/out2/Result_0 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r')
  echo "[wait] $(date +%H:%M:%S) 仍在跑  MemAvailable=${M} kB  已产出张量=${N}"
  if [ -n "$M" ] && [ "$M" -lt 800000 ]; then
    echo "[wait] 🔴 设备可用内存 < 800 MB —— 告警并停止等待（不自动杀，交人工决定）"; exit 9
  fi
  sleep 60
done
echo "[wait] $(date +%H:%M:%S) qnn-net-run 已退出（设备在线时确认，可信）"
"$ADB" -s $SER shell "tail -6 $T/p2attr/run.log; echo '--- 产物 ---'; ls -la $T/p2attr/out2/Result_0/ 2>/dev/null | head -16"
