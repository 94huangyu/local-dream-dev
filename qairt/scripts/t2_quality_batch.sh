#!/usr/bin/env bash
# 画质评测（不是一致性评测）：用**真实 app 全链路**出一批多样化 prompt 的图，人看。
#
# 🔴 为什么必须单独做这件事：
#   本项目至今所有指标（L1/L2/L3）测的都是**离 FP32 参考多远**，
#   而 #69/#70/#84 已实测这类标量与**观感画质反相关** ⇒ 手上没有任何数能支撑画质判断。
#   而且研究流水线的 PSNR **把量化 TE 与量化 VAE 排除了**（两臂都用 FP32 TE），
#   所以那个数描述的不是用户实际看到的东西。本批走真实 app，含全部量化环节。
#
# 🔴 prompt 集按**量化扩散的已知失效模式**设计，不是随便挑的：
#   手/脸/文字/多主体空间关系/中文/风格化/精细纹理/低光。
#   8 条全部 33~55 token（>旧的 32 槽上限）=> 同时也是 Tier 2 的实用验证。
#
# 固定 seed，便于将来与其他配置做单变量对照。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
SEED="${1:-1234}"
OUT=/d/LocalDreamZImage/scratch_runs
LOG=/d/LocalDreamZImage/logs/t2_quality; mkdir -p "$LOG"
ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
export MSYS_NO_PATHCONV=1
"$ADB" get-state >/dev/null 2>&1 < /dev/null || { echo "🔴 设备不可达，请插线"; exit 2; }

run () {   # run <名字> <prompt>
  local n="$1"; shift
  local p="$1"
  echo "=== [$(date +%H:%M:%S)] q_$n ==="
  bash scripts/app_generate.sh "q_$n" "$SEED" "$p" > "$LOG/$n.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "🔴 q_$n 失败 rc=$rc"
    grep -aiE 'error|fail|HTTP' "$LOG/$n.log" | head -3
    return 1     # 单张失败不终止整批，继续跑其余的（人看时缺一张比缺七张好）
  fi
  grep -a 'HTTP\|generation_time_ms' "$LOG/$n.log" | head -2
  return 0
}

fail=0
run hands    "A close-up photograph of a person's open hand holding five ripe red cherries, fingers clearly separated, natural skin texture, soft daylight" || fail=$((fail+1))
run face     "A portrait photograph of an elderly woman smiling warmly, deep wrinkles around her eyes, grey hair in a bun, window light, shallow depth of field" || fail=$((fail+1))
run text     "A rustic wooden sign hanging on a shop door with the word OPEN painted in bold black letters, weathered paint, morning light" || fail=$((fail+1))
run multi    "Three cats sitting on a windowsill, a black cat on the left, a white cat in the middle, an orange cat on the right, all facing the camera" || fail=$((fail+1))
run chinese  "水墨风格的山水画，远处是雾气缭绕的连绵山峰，近处一叶小舟漂在平静江面上，岸边有几株垂柳，留白处题一方红印" || fail=$((fail+1))
run vector   "A flat vector illustration of a red bicycle leaning against a plain blue wall, minimal geometric shapes, bold outlines, limited flat color palette" || fail=$((fail+1))
run texture  "A macro photograph of a hand-knitted wool sweater in deep forest green, showing individual yarn fibers and a raised cable stitch pattern" || fail=$((fail+1))
run lowlight "A dimly lit jazz bar late at night, a saxophone player standing alone under a single warm spotlight, cigarette smoke drifting through the beam, deep shadows in the background, amber highlights on brass, shot on 35mm film" || fail=$((fail+1))

echo
echo "=== [$(date +%H:%M:%S)] 批次结束  失败 $fail / 8 ==="
ls -la "$OUT"/q_*.png 2>/dev/null | awk '{printf "  %8.0f KB  %s\n", $5/1024, $NF}'
exit 0     # 单张失败不算整批失败；由人看图决定
