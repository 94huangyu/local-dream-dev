#!/usr/bin/env bash
# Tier 2 L=80 text_encoder 四段：量化 + 建 context。
#
# 配方来源：**Tier 1 的 s32 量化 DLC 里记录的原始命令**（不是我编的）：
#   Converter : float_bitwidth=32, input_dim=[['input_ids','1,L'],['attention_mask','1,L']]
#               —— fp32 DLC 已由 §11.1 建好，在 TIER2_L80/ 下，本脚本不重转
#   Quantizer : --input_list <input_list_raw.txt> --act_bitwidth 16 --weights_bitwidth 8
#               --bias_bitwidth 32 --param_quantizer_calibration min-max
#               --act_quantizer_calibration min-max
# 校准数据由 `te_L80_calib.py` 生成（混合长度 5 条，已在盘）。
#
# 🔴 建 context 必须传 --config_file，否则静默编成 dspArch 68 / vtcm 4MB（#95）。
# 🔴 不套 tail/head（管道吞返回码，§7.4）；每步单独查 rc。
# ⚠️ 实测工时（#135）：量化约 20 秒/段、建 context 约 10 秒/段。
set -u
export PYTHONIOENCODING=utf-8
GEN='D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\qnn-context-binary-generator.exe'
BACKEND='D:\qairt\2.48.0.260626\lib\x86_64-windows-msvc\QnnHtp.dll'
CFG='D:\ZImage_Work\backend_ext.json'
LOG=/d/LocalDreamZImage/logs/t2_te
mkdir -p "$LOG"
cd /d/LocalDreamZImage || exit 90

for P in 1 2 3 4; do
  DW="D:\ZImage_Work\TIER2_L80\text_encoder_part${P}"
  DU="/d/ZImage_Work/TIER2_L80/text_encoder_part${P}"
  F="$DU/text_encoder_part${P}_fp32.dlc"
  Q="$DU/text_encoder_part${P}_quantized.dlc"
  if [ ! -f "$F" ]; then echo "🔴 缺 fp32 DLC: $F"; exit 91; fi
  if [ ! -f "$DU/input_list_raw.txt" ]; then echo "🔴 缺校准 input_list"; exit 92; fi

  echo "=== part${P} 量化 $(date +%H:%M:%S) ==="
  python scripts/qairt_tool.py qairt-quantizer \
      --input_dlc "$F" --output_dlc "$Q" \
      --input_list "${DW}\input_list_raw.txt" \
      --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
      --param_quantizer_calibration min-max \
      --act_quantizer_calibration min-max \
      > "$LOG/quant_part${P}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$Q" ]; then
    echo "🔴 part${P} 量化失败 rc=$rc"; grep -iE "error|fail" "$LOG/quant_part${P}.log" | head -3; exit $rc
  fi
  echo "part${P} 量化 ✅ $(( $(stat -c %s "$Q") / 1048576 )) MB"

  echo "=== part${P} 建 context $(date +%H:%M:%S) ==="
  "$GEN" --backend "$BACKEND" --dlc_path "${DW}\text_encoder_part${P}_quantized.dlc" \
         --binary_file "text_encoder_part${P}_ctx_sm8750" --output_dir "$DW" \
         --htp_socs sm8750 --config_file "$CFG" \
         > "$LOG/ctx_part${P}.log" 2>&1
  rc=$?
  out=$(ls "$DU"/text_encoder_part${P}_ctx_sm8750*.bin 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "🔴 part${P} 建 context 失败 rc=$rc"; grep -iE "error|fail" "$LOG/ctx_part${P}.log" | head -3; exit $rc
  fi
  echo "part${P} context ✅ $(( $(stat -c %s "$out") / 1048576 )) MB"
  # 装置画像必须是 v79 / 8MB（#95/#96：不看文件名，读回 dspArch/vtcmSize）
  python scripts/check_ctx_identity.py --expect-arch 79 --expect-vtcm 8 "$out"
  rc=$?
  if [ $rc -ne 0 ]; then echo "🔴 part${P} 装置画像不符 rc=$rc"; exit $rc; fi
done
echo "=== TE 四段完成 $(date +%H:%M:%S)  磁盘 $(df -h /d | tail -1 | awk '{print $4}') ==="
