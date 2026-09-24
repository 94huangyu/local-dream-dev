#!/bin/bash
# #30 单 MatMul 探针：转换 -> 量化 -> 建 context binary，K = 64 / 1024 / 3840。
# 量化配置 = 主模型基线四项（方案 EXP_PLAN_MATMUL_PROBE 第 1 节），不加 per-row。
# 理由见 matmul_probe_gen.py 文档：判据是 E_htp/E_fxp 比值，encoding 在比值里对消。
set -u
ROOT="D:/ZImage_Work/p0_experiments/matmul_probe"
SDK="D:/qairt/2.48.0.260626"
export PYTHONPATH="$SDK/lib/python"
TOOL="D:/LocalDreamZImage/scripts/qairt_tool.py"
CHAIN="$ROOT/chain.log"
: > "$CHAIN"

cat > "$ROOT/cfg.json" <<'EOF'
{"devices":[{"soc_model":69,"dsp_arch":"v79"}]}
EOF
cat > "$ROOT/backend.json" <<EOF
{"backend_extensions":{"shared_library_path":"$SDK/lib/x86_64-windows-msvc/QnnHtpNetRunExtensions.dll","config_file_path":"$ROOT/cfg.json"}}
EOF

for K in 64 1024 3840; do
  D="$ROOT/k$K"
  echo "[$(date +%H:%M:%S)] K=$K 转换" | tee -a "$CHAIN"
  python "$TOOL" qairt-converter --input_network "$D/probe_k$K.onnx" \
      --output_path "$D/probe_k${K}_fp32.dlc" > "$D/01_conv.log" 2>&1
  echo "[$(date +%H:%M:%S)] K=$K 转换 rc=$?" | tee -a "$CHAIN"

  echo "[$(date +%H:%M:%S)] K=$K 量化" | tee -a "$CHAIN"
  python "$TOOL" qairt-quantizer --input_dlc "$D/probe_k${K}_fp32.dlc" \
      --output_dlc "$D/probe_k${K}_quant.dlc" --input_list "$D/input_list.txt" \
      --weights_bitwidth 8 --act_bitwidth 16 --bias_bitwidth 32 \
      --act_quantizer_calibration min-max --param_quantizer_calibration min-max \
      > "$D/02_quant.log" 2>&1
  echo "[$(date +%H:%M:%S)] K=$K 量化 rc=$?" | tee -a "$CHAIN"

  echo "[$(date +%H:%M:%S)] K=$K 建图" | tee -a "$CHAIN"
  "$SDK/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe" \
      --backend "$SDK/lib/x86_64-windows-msvc/QnnHtp.dll" \
      --dlc_path "$D/probe_k${K}_quant.dlc" --binary_file "probe_k${K}_ctx" \
      --output_dir "$D" --config_file "$ROOT/backend.json" \
      --htp_socs sm8750 > "$D/03_ctx.log" 2>&1
  # `--htp_socs` 是生成 **HTP Offline Cache**（`*.SM8750.bin`）的开关。
  # 漏掉它产出的是不带 SoC 后缀的通用 context binary —— 项目里 part1a/part2a 等
  # 全部是 `--htp_socs` 产物，探针必须走同一条路径，否则测的不是同一个东西。
  echo "[$(date +%H:%M:%S)] K=$K 建图 rc=$?" | tee -a "$CHAIN"
done

echo "=== 产物 ===" | tee -a "$CHAIN"
ls -la "$ROOT"/k*/probe_*_ctx.SM8750.bin 2>&1 | tee -a "$CHAIN"
