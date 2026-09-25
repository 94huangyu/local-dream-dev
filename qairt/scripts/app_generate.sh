#!/usr/bin/env bash
# 驱动真实 app 端到端生成一张图（不用 run-as —— SELinux 域不同，见 #72）。
#
# 与 zimage_device_test.sh 的区别：
#  1. 不再用 grep "Server listening" 当就绪判据 —— 2026-08-21 实测该判据会误报：
#     backend 已在 127.0.0.1:8081 监听（/proc/net/tcp 里 0100007F:1F91），
#     日志里却没有那句话，脚本据此报 "backend never came up"（又一次"报错说谎"）。
#     改为直接探测端口是否监听。
#  2. 固定 seed，使前后对照成为单变量。
#
# 用法: bash scripts/app_generate.sh <basename> [seed] [prompt]
set -uo pipefail

ADB="/c/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe"
PKG="io.github.xororz.localdream.zimage"
# 🔴 2026-09-25：本版（versionCode 101）起后端端口 8081 → 8091，与官方 Local Dream（8081）并存。
#    旧 APK（versionCode 100）仍是 8081 —— 端口必须与设备上装的版本一致。
PORT="${ZIT_PORT:-8091}"
PORT_HEX=$(printf ':%04X' "$PORT")
OUTDIR="D:/LocalDreamZImage/scratch_runs"
BASENAME="${1:?usage: app_generate.sh <basename> [seed] [prompt]}"
SEED="${2:-42}"
PROMPT="${3:-a cute orange cat sitting on a wooden table, masterpiece, best quality}"
# 尺寸（可选，第 4/5 个参数）。不传则不发 width/height ⇒ 后端回落到 1024x1024。
# 🔴 后端只放行 zimage_sizes 里已交付的尺寸，其余一律回落（RequestParser.hpp:82-92）
#    ⇒ 传错尺寸不会报错，只会**静默变成 1024**。所以出图后必须校验 PNG 实际宽高。
GEN_W="${4:-}"
GEN_H="${5:-}"

SSE="$OUTDIR/${BASENAME}.sse"
IMG="$OUTDIR/${BASENAME}.png"
LOG="$OUTDIR/${BASENAME}_logcat.txt"

DEVICE=$("$ADB" devices | awk '$2=="device"{print $1; exit}')
[ -z "$DEVICE" ] && { echo "FAIL: 没有在线设备" >&2; exit 1; }
echo "device: $DEVICE   seed=$SEED"

# --- 前置门：屏幕与前台 Activity（2026-08-25 事故，见台账 #125）---
# 不加这道门时，屏幕休眠或有系统弹窗会让 input tap 全部打空，
# 而脚本报的是「8081 始终未监听」—— 指向错误的对象。
screen_ready() {
  "$ADB" -s "$DEVICE" shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
  sleep 2
  local w
  w=$("$ADB" -s "$DEVICE" shell "dumpsys power | grep -m1 mWakefulness=" 2>/dev/null | tr -d '\r')
  case "$w" in *Awake*) return 0;; esac
  echo "FAIL: 屏幕未唤醒（$w）—— input tap 会全部打空，先解锁屏幕" >&2
  return 1
}

foreground_clear() {
  local fg
  fg=$("$ADB" -s "$DEVICE" shell "dumpsys activity activities | grep -m1 ResumedActivity" 2>/dev/null | tr -d '\r')
  case "$fg" in
    *GrantPermissionsActivity*|*permissioncontroller*)
      echo "FAIL: 前台是**权限对话框**，点击会打在它上面而不是 app：" >&2
      echo "      $fg" >&2
      echo "      => 请在手机上关掉该弹窗后重试（这不是模型或 backend 的问题）" >&2
      return 1;;
    *ResolverActivity*|*ChooserActivity*)
      echo "FAIL: 前台是系统选择器对话框：$fg" >&2
      return 1;;
  esac
  return 0
}

port_up() {
  "$ADB" -s "$DEVICE" shell "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null" \
    | awk '{print $2}' | grep -qi "$PORT_HEX"
}

# 🔴 官方 Local Dream 的后端进程**同名**（libstable_diffusion_core.so），包名也含 localdream
#    ⇒ 按进程名/子串判断会被它骗过。只认本包 uid 名下的进程。
app_user() {
  "$ADB" -s "$DEVICE" shell "ps -A -o USER,NAME" 2>/dev/null | tr -d '\r' \
    | awk -v p="$PKG" '$2==p{print $1; exit}'
}

# 先长轮询：全新安装后首启要做契约校验(~119s)再加载 11GB 模型，
# 2026-08-21 实测 120s 窗口太短，会把正在进行的加载误判成"没起来"并强杀重来。
if ! port_up; then
  # 只有在 app 进程**确实存在**时才值得长等（说明加载正在进行）；
  # 进程不在就直接进入下面的启动分支，别白等 15 分钟（2026-08-25 白等过一次）。
  if [ -n "$(app_user)" ]; then
    echo "backend 未监听但 app 进程在，等已在进行的加载（最多 15 分钟）..."
    for _ in $(seq 1 450); do port_up && break; sleep 2; done
  else
    echo "app 进程不存在 —— 跳过等待，直接启动"
  fi
fi

if ! port_up; then
  echo "仍未监听，启动 app 并点进模型页..."
  screen_ready || exit 1
  foreground_clear || exit 1
  "$ADB" -s "$DEVICE" shell am force-stop "$PKG"
  # 🔴 2026-08-26 (台账 #126)：必须带 MAIN/LAUNCHER。原来只用 `-n <组件>` 拉起，
  # 任务 root 的 intent 与桌面图标不匹配 ⇒ 用户此后每点一次图标就**再叠一个**
  # MainActivity 实例，而每个实例各持一份 ModelRunScreen composition，
  # 一次生图会被写成 N 条历史行（2026-08-26 设备实测：4 实例 ⇒ 2 行；1 实例 ⇒ 1 行）。
  "$ADB" -s "$DEVICE" shell am start -a android.intent.action.MAIN \
    -c android.intent.category.LAUNCHER \
    -n "$PKG/io.github.xororz.localdream.MainActivity" >/dev/null
  # 🔴 2026-09-07（台账 #163）：原来是 `sleep 3` + 两次固定点击。
  #    force-stop 重启后列表未渲染完就点 ⇒ 点空，随后进入「等已在进行的加载」
  #    分支空转最多 15 分钟，而日志一直显示「在等」，看不出死活。本轮踩了两次。
  #    实测：手动点同一坐标立刻生效 ⇒ **不是坐标问题，是时序**。
  #    ⇒ 改为**重试点击直到后端进程出现**（后端一起来就说明点进模型页了）。
  backend_up() {
    local u
    u=$(app_user)
    [ -n "$u" ] || return 1
    "$ADB" -s "$DEVICE" shell "ps -A -o USER,NAME" 2>/dev/null | tr -d '\r' \
      | awk -v u="$u" '$1==u && $2 ~ /^libstable_diffu/ {f=1} END{exit !f}'
  }
  tapped=0
  for i in $(seq 1 20); do          # 最多 20 次 × 3 s = 60 s
    backend_up && break
    port_up && break
    sleep 3
    "$ADB" -s "$DEVICE" shell input tap 608 1279 >/dev/null 2>&1
    tapped=$((tapped+1))
  done
  echo "点击 $tapped 次后：后端进程 $(backend_up && echo 已起 || echo 未起)"
  # 后端起来之后才是真正漫长的加载（契约校验 + 模型），这时才值得长等
  for _ in $(seq 1 450); do port_up && break; sleep 2; done
fi
port_up || { echo "FAIL: 127.0.0.1:$PORT 始终未监听" >&2; exit 1; }
echo "backend 就绪（$PORT 已监听）"

"$ADB" -s "$DEVICE" logcat -c
"$ADB" -s "$DEVICE" logcat -v time > "$LOG" 2>&1 &
LOGCAT_PID=$!
# 🔴 2026-09-17：原来是立刻 kill。后端报错后 SSE 只要 14 s 就返回，脚本随即退出并杀掉 logcat
#    ⇒ **最关键的几行（QNN 的错误码）还在管道缓冲里就被丢掉**，本地 logcat 停在出错前 1 秒；
#    事后再 `logcat -d` 捞，设备环形缓冲早已滚过去。⇒ 先等几秒让 logcat 冲刷完再杀。
trap 'sleep 4; kill $LOGCAT_PID 2>/dev/null' EXIT

"$ADB" -s "$DEVICE" forward "tcp:$PORT" "tcp:$PORT" >/dev/null

echo "请求生成（约 4 分钟）..."
BODY=$(python -c 'import json,sys
d={"prompt": sys.argv[1], "seed": int(sys.argv[2]), "output_format": "png"}
if len(sys.argv) > 4 and sys.argv[3] and sys.argv[4]:
    d["width"] = int(sys.argv[3]); d["height"] = int(sys.argv[4])
print(json.dumps(d))' "$PROMPT" "$SEED" "$GEN_W" "$GEN_H")
curl -sS -m 900 -X POST "http://127.0.0.1:$PORT/generate" \
  -H "Content-Type: application/json" -d "$BODY" -o "$SSE" \
  -w "HTTP %{http_code}  size %{size_download}B  time %{time_total}s\n"
RC=$?
[ $RC -ne 0 ] && { echo "FAIL: curl rc=$RC（这是 HTTP 层失败，不要当成模型失败）" >&2; exit $RC; }

python - "$SSE" "$IMG" <<'PYEOF'
import sys, json, base64
content = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
idx = content.rfind("event: complete")
if idx < 0:
    tail = content[-500:]
    print("FAIL: SSE 里没有 complete 事件。尾部：\n" + tail)
    sys.exit(1)
obj = json.loads(content[idx:].split("data: ", 1)[1].split("\n", 1)[0])
open(sys.argv[2], "wb").write(base64.b64decode(obj["image"]))
print("saved:", sys.argv[2])
for k in obj:
    if k != "image":
        print("  %s: %s" % (k, obj[k]))
PYEOF
