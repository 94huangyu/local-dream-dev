#!/usr/bin/env bash
# Z-Image Turbo on-device smoke test: launches the app, waits for the native
# backend to come up, POSTs one /generate request directly (bypassing the UI
# entirely), decodes the resulting PNG, and leaves logcat captured for
# analysis. Run from Git Bash on the machine with the phone attached via adb.
#
# Usage: ./zimage_device_test.sh [prompt] [output_basename]
set -uo pipefail

ADB="/c/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe"
PKG="io.github.xororz.localdream.zimage"
OUTDIR="D:/LocalDreamZImage/scratch_runs"
PROMPT="${1:-a cute orange cat sitting on a wooden table, masterpiece, best quality}"
BASENAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTDIR"
LOG="$OUTDIR/${BASENAME}_logcat.txt"
SSE="$OUTDIR/${BASENAME}.sse"
IMG="$OUTDIR/${BASENAME}.png"

DEVICE=$("$ADB" devices | awk 'NR==2{print $1}')
if [ -z "$DEVICE" ]; then
  echo "No device attached." >&2
  exit 1
fi
echo "Using device: $DEVICE"

"$ADB" -s "$DEVICE" shell am force-stop "$PKG"
"$ADB" -s "$DEVICE" logcat -c
"$ADB" -s "$DEVICE" shell am start -n "$PKG/io.github.xororz.localdream.MainActivity" >/dev/null

# Background logcat capture for the whole run.
"$ADB" -s "$DEVICE" logcat -v time > "$LOG" 2>&1 &
LOGCAT_PID=$!
trap 'kill $LOGCAT_PID 2>/dev/null' EXIT

# The model card tap coordinates are for a 1216x2640-ish display with the
# ZIMAGE card as the sole/first NPU model in the list; re-dump+adjust if the
# UI layout ever changes.
sleep 2
"$ADB" -s "$DEVICE" shell input tap 608 1279
sleep 1
"$ADB" -s "$DEVICE" shell input tap 608 1279  # tap twice: first one sometimes lands before the list settles

echo "Waiting for backend to come up (loads all 4 text-encoder parts first, ~8-10s)..."
for _ in $(seq 1 60); do
  if grep -q "Server listening" "$LOG" 2>/dev/null; then
    echo "Backend ready."
    break
  fi
  sleep 2
done
if ! grep -q "Server listening" "$LOG" 2>/dev/null; then
  echo "Backend never came up within 120s; check $LOG" >&2
  exit 1
fi

"$ADB" -s "$DEVICE" forward tcp:8081 tcp:8081 >/dev/null

echo "Requesting generation (prompt: $PROMPT)"
echo "Current transformer loading is per-step (see PipelineZImage.hpp comments) -- full run takes ~4 minutes."
curl -sS -m 600 -X POST http://127.0.0.1:8081/generate \
  -H "Content-Type: application/json" \
  -d "$(python -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "output_format": "png"}))' "$PROMPT")" \
  -o "$SSE" \
  -w "\nHTTP %{http_code}, size %{size_download} bytes, time %{time_total}s\n"

python - "$SSE" "$IMG" <<'PYEOF'
import sys, json, base64
sse_path, img_path = sys.argv[1], sys.argv[2]
content = open(sse_path, "r", encoding="utf-8", errors="replace").read()
idx = content.rfind("event: complete")
if idx < 0:
    print("No 'complete' event in response -- request likely failed or timed out.")
    sys.exit(1)
data_line = content[idx:].split("data: ", 1)[1].split("\n", 1)[0]
obj = json.loads(data_line)
with open(img_path, "wb") as f:
    f.write(base64.b64decode(obj["image"]))
print("Saved image:", img_path)
for k in obj:
    if k != "image":
        print(f"  {k}: {obj[k]}")
PYEOF

echo ""
echo "=== [diagnostic] lines from this run ==="
grep "diagnostic" "$LOG" || echo "(none -- diagnostic logging may have been removed from the build)"

echo ""
echo "Done. Logcat: $LOG"
echo "SSE stream: $SSE"
echo "Image (if generated): $IMG"
