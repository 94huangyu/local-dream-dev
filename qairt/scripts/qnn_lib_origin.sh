#!/usr/bin/env bash
# 等本包的后端进程起来并加载 QNN 后，读它的 /proc/<pid>/maps，打印实际加载的 QNN 库路径。
# 用途：出图 sha 相同时区分"走了哪条库路径"（门 F1 的生效证据，EXP_PLAN_APK_QNNLIBS）。
# 用法: bash scripts/qnn_lib_origin.sh [超时秒数，默认 900]
set -uo pipefail
ADB="/c/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe"
PKG="io.github.xororz.localdream.zimage"
TIMEOUT="${1:-900}"

for _ in $(seq 1 $((TIMEOUT / 3))); do
  u=$("$ADB" shell "ps -A -o USER,NAME" 2>/dev/null | tr -d '\r' | awk -v p="$PKG" '$2==p{print $1; exit}')
  pid=""
  [ -n "$u" ] && pid=$("$ADB" shell "ps -A -o USER,PID,NAME" 2>/dev/null | tr -d '\r' \
      | awk -v u="$u" '$1==u && $3 ~ /^libstable_diffu/ {print $2; exit}')
  if [ -n "$pid" ]; then
    libs=$("$ADB" shell "run-as $PKG cat /proc/$pid/maps" 2>/dev/null | tr -d '\r' \
      | awk '{print $6}' | grep -E 'libQnn(Htp|System|HtpV79Stub)\.so$' | sort -u)
    if echo "$libs" | grep -q 'libQnnHtp.so'; then
      echo "backend pid $pid (user $u) loaded:"
      echo "$libs" | sed 's/^/  /'
      exit 0
    fi
  fi
  sleep 3
done
echo "FAIL: ${TIMEOUT}s 内没看到本包后端加载 libQnnHtp.so" >&2
exit 1
