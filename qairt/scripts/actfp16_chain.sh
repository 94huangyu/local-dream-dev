#!/bin/bash
# 等 ctxgen 真结束（进程消失 + 产物存在）后，自动过 G3 并上机。
# 约束 9·补：判断死活只看进程表 + 产物，不看日志内容。
export PYTHONIOENCODING=utf-8
export MSYS_NO_PATHCONV=1
CTX=/d/ZImage_Work/p0_experiments/actfp16/ctx
ADB="/c/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe"
cd /d/LocalDreamZImage

# 🔴 不要用 tasklist //FI：本脚本 export 了 MSYS_NO_PATHCONV=1，
#    它同时关掉了 //FI -> /FI 的转换，tasklist 收到非法参数、静默给出错误答案，
#    导致循环立刻退出并误报"进程已退出"（2026-08-22 实测踩过）。用无参数 tasklist + grep。
echo "[chain] 等 ctxgen 结束 $(date +%H:%M:%S)"
alive() { tasklist 2>/dev/null | grep -qi "qnn-context-binary"; }
if ! alive; then echo "[chain] ⚠️ 启动时就没检测到 ctxgen 进程，请人工确认"; fi
while alive; do
  RSS=$(tasklist 2>/dev/null | grep -i "qnn-context-binary" | awk '{gsub(",","",$(NF-1)); print $(NF-1)}')
  echo "[chain] $(date +%H:%M:%S) ctxgen 仍在跑  RSS=${RSS} KB"
  if [ -n "$RSS" ] && [ "$RSS" -gt 21000000 ]; then
    echo "[chain] 🔴 RSS 超过 21 GB（本机 25.5 GB，曾被压死）—— 停止等待并告警，不自动杀进程"
    break
  fi
  sleep 60
done
echo "[chain] ctxgen 检查结束 $(date +%H:%M:%S)"

BIN=$(ls $CTX/*.bin 2>/dev/null | head -1)
if [ -z "$BIN" ]; then
  echo "[chain] ❌ 未产出 .bin —— 查 G2 容量门"
  grep -i "available PD\|context size estimate\|error" /d/ZImage_Work/p0_experiments/actfp16/03_ctxgen.log 2>/dev/null | head -5
  exit 2
fi
echo "[chain] ✅ 产物 $(basename $BIN)  $(stat -c %s "$BIN") 字节"

echo "[chain] === G3 装置门 ==="
python scripts/check_ctx_identity.py --expect-arch 79 --expect-vtcm 8 "$BIN" || exit 3
python scripts/check_ctx_identity.py --diff /d/ZImage_Work/p0_experiments/perrow/ctx/part1b_perrow_ctx.SM8750.bin "$BIN"

if ! "$ADB" devices | grep -q "3B1F65EA9BBUMSHZ"; then
  echo "[chain] ⚠️ 设备不在线 —— G2/G3 结果已就绪，上机留待设备回来"
  exit 0
fi
echo "[chain] === 上机 actfp16 臂 $(date +%H:%M:%S) ==="
python scripts/device_run_part1b.py "$BIN" actfp16 /d/ZImage_Work/p0_experiments/actfp16/unified_actfp16.raw || exit 4
echo "[chain] === 上机 perrow 参照臂 $(date +%H:%M:%S) ==="
python scripts/device_run_part1b.py /d/ZImage_Work/p0_experiments/perrow/ctx/part1b_perrow_ctx.SM8750.bin perrowref /d/ZImage_Work/p0_experiments/actfp16/unified_perrow.raw || echo "[chain] 参照臂未完成（可明天补跑，确定性执行不影响可比性）"
echo "[chain] === 判据 ==="
python scripts/actfp16_analyze.py
