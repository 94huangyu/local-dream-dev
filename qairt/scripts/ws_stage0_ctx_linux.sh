#!/bin/bash
# EXP_PLAN_MULTIGRAPH 阶段 0 · Linux 三臂（L1/L2/L3），在 WSL 里跑。
# 与 Windows 三臂唯一的差别是宿主与库后缀；判据完全相同。
set -u
SDK=/mnt/d/qairt/2.48.0.260626
BIN=$SDK/bin/x86_64-linux-clang
LIB=$SDK/lib/x86_64-linux-clang
W=/mnt/d/ZImage_Work/p0_experiments/ws_stage0
export LD_LIBRARY_PATH=$LIB

A=$W/wsA_quant.dlc
B=$W/wsB_quant.dlc
for f in "$A" "$B"; do [ -f "$f" ] || { echo "缺 $f"; exit 2; }; done

run_arm () {           # $1=tag  $2=dlc列表  $3=ws(none/false/true)
  tag=$1; dlcs=$2; ws=$3
  od=$W/$tag; rm -rf "$od"; mkdir -p "$od"
  if [ "$ws" = "none" ]; then
    ctx=""
  else
    ctx=", \"context\": {\"weight_sharing_enabled\": $ws}"
  fi
  if [ "$dlcs" = "$A" ]; then
    gn='[{"graph_names": ["wsA_fp32"], "vtcm_mb": 8}]'
  else
    gn='[{"graph_names": ["wsA_fp32"], "vtcm_mb": 8}, {"graph_names": ["wsB_fp32"], "vtcm_mb": 8}]'
  fi
  printf '{"graphs": %s, "devices": [{"soc_model": 69, "dsp_arch": "v79"}]%s}\n' "$gn" "$ctx" > "$od/d.json"
  printf '{"backend_extensions": {"shared_library_path": "%s/libQnnHtpNetRunExtensions.so", "config_file_path": "%s/d.json"}}\n' "$LIB" "$od" > "$od/e.json"

  echo ""; echo "--- $tag  weight_sharing=$ws ---"
  cat "$od/d.json"
  "$BIN/qnn-context-binary-generator" \
      --backend "$LIB/libQnnHtp.so" \
      --dlc_path "$dlcs" --binary_file "$tag" \
      --output_dir "$od" --htp_socs sm8750 --config_file "$od/e.json" \
      > "$od/build.log" 2>&1
  rc=$?                                  # 先取返回码，别套管道（§7.4）
  echo "    rc=$rc"
  grep -iE "weight|shar|error|warn|unsupport|ignor|not support" "$od/build.log" | head -15 | sed 's/^/    | /'
  if [ $rc -ne 0 ]; then echo "    ❌ 建图失败，日志尾部："; tail -25 "$od/build.log"; return 1; fi
  b=$(ls "$od"/*.bin 2>/dev/null | head -1)
  [ -n "$b" ] || { echo "    ❌ 无产物"; return 1; }
  sz=$(stat -c%s "$b")
  echo "    产物 $(basename "$b") = $sz B ($(awk -v s=$sz 'BEGIN{printf "%.2f", s/1048576}') MB)"
  "$BIN/qnn-context-binary-utility" --context_binary "$b" --json_file "$od/info.json" > /dev/null 2>&1
  if [ -f "$od/info.json" ]; then
    echo "    图 = $(grep -o '"graphName"[^,]*' "$od/info.json" | head -8 | tr '\n' ' ')"
  fi
  echo "$tag $sz" >> "$W/stage0_linux.txt"
}

rm -f "$W/stage0_linux.txt"
run_arm L1 "$A"     none
run_arm L2 "$A,$B"  false
run_arm L3 "$A,$B"  true

echo ""; echo "=============================================================="
echo "阶段 0 · Linux 三臂结果"
cat "$W/stage0_linux.txt"
awk '{a[$1]=$2} END{
  if (a["L1"] && a["L2"] && a["L3"]) {
    printf "  G0-b 共享生效线 : (S(L3)-S(L1))/S(L1) = %.4f  < 0.30 ? %s\n", (a["L3"]-a["L1"])/a["L1"], ((a["L3"]-a["L1"])/a["L1"]<0.30?"YES":"NO");
    printf "  G0-c 开关无效线 : S(L3)/S(L2) = %.4f  > 0.95 ? %s\n", a["L3"]/a["L2"], (a["L3"]/a["L2"]>0.95?"YES":"NO");
    printf "  参考           : S(L2)/S(L1) = %.4f\n", a["L2"]/a["L1"];
  }}' "$W/stage0_linux.txt"
echo "=============================================================="
