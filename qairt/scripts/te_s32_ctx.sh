#!/usr/bin/env bash
# Tier 1 步5：为 4 段 s32 量化 DLC 建 SM8750 context。
#
# 🔴 §7.2 硬规则：必须传 --config_file，否则会**静默**编译成 dspArch 68 / vtcm 4MB，
#    尽管传了 --htp_socs sm8750、尽管产物文件名就带 .SM8750。同一份 DLC 数值差 15 倍（#95）。
#    判别方法不是看文件名，而是读回 dspArch/vtcmSize（check_ctx_identity.py）。
#
# §7.1：qnn-context-binary-generator 是原生 exe，**不走 qairt_tool.py**。
set -u
GEN='D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc\qnn-context-binary-generator.exe'
BACKEND='D:\qairt\2.48.0.260626\lib\x86_64-windows-msvc\QnnHtp.dll'
CFG='D:\ZImage_Work\backend_ext.json'
LOG=/d/LocalDreamZImage/logs/tier1_20260827

fail=0
for P in 1 2 3 4; do
  DW="D:\\ZImage_Work\\TIER1_S32\\text_encoder_part${P}"
  DU="/d/ZImage_Work/TIER1_S32/text_encoder_part${P}"
  echo "=== part${P} 开始 $(date +%H:%M:%S) ==="
  "$GEN" --backend "$BACKEND" \
         --dlc_path "${DW}\\text_encoder_part${P}_quantized.dlc" \
         --binary_file "text_encoder_part${P}_ctx_sm8750" \
         --output_dir "$DW" \
         --htp_socs sm8750 \
         --config_file "$CFG" \
         > "$LOG/ctx_part${P}.log" 2>&1
  rc=$?
  out=$(ls "$DU"/text_encoder_part${P}_ctx_sm8750*.bin 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "part${P} 🔴 失败 rc=$rc"
    grep -iE "error|fail" "$LOG/ctx_part${P}.log" | head -3
    fail=1
    break
  fi
  new=$(stat -c %s "$out")
  old=$(stat -c %s "/d/ZImage_Work/ZImage_QNN_Evidence/dlc_pipeline/text_encoder_part${P}/text_encoder_part${P}_ctx_sm8750.SM8750.bin" 2>/dev/null || echo 0)
  echo "part${P} ✅ rc=0  新=$((new/1048576)) MB  部署版=$((old/1048576)) MB  $(date +%H:%M:%S)"
  # T2 判据：装置画像必须是 v79 / 8MB
  PYTHONIOENCODING=utf-8 python /d/LocalDreamZImage/scripts/check_ctx_identity.py \
      --expect-arch 79 --expect-vtcm 8 "$out" 2>&1 | tail -3
done
echo "=== 建图结束 fail=$fail  磁盘: $(df -h /d | tail -1 | awk '{print $4}') ==="
exit $fail
