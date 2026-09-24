#!/usr/bin/env bash
# Tier 1 步7：把 seq-32 的 text_encoder 交付到设备（台账 #134）。
#
# 🔴 约束 11 四条铁律：
#   ① 先证明现状可用 —— 2026-08-27 00:32 已在旧版上跑通一次生图（id=47），基线成立
#   ② 先备份原件     —— 已完成，且**逐位核对过**：
#        · 4 段旧 .bin 的宿主副本 sha256 与设备在用的完全一致
#          (dlc_pipeline/text_encoder_partN/text_encoder_partN_ctx_sm8750.SM8750.bin)
#        · 旧 APK: logs/tier1_20260827/installed_now.apk (sha 485899f0…)
#        · 旧契约: logs/tier1_20260827/contract.json
#   ③ 交付后必须自己跑一次真实链路（本脚本只做交付，验收见末尾提示）
#   ④ 不得用 run-as 代替真实 app 验证（SELinux 域不同，台账 #72）
#
# 用法： bash scripts/te_s32_deliver.sh          # 交付
#        bash scripts/te_s32_deliver.sh rollback # 回滚到 20 槽版本
set -u
ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
PKG=io.github.xororz.localdream.zimage
export MSYS_NO_PATHCONV=1
MODE="${1:-deliver}"
NEW_ROOT=/d/ZImage_Work/TIER1_S32
OLD_ROOT=/d/ZImage_Work/ZImage_QNN_Evidence/dlc_pipeline
APK=/d/LocalDreamZImage/scratch_runs/apk_backup/built_2026-08-27_s32.apk
OLD_APK=/d/LocalDreamZImage/logs/tier1_20260827/installed_now.apk
NEW_CONTRACT=/d/LocalDreamZImage/logs/tier1_20260827/final_qnn_contract.s32.json
OLD_CONTRACT=/d/LocalDreamZImage/logs/tier1_20260827/contract.json
TMP=/data/local/tmp/te_s32

"$ADB" get-state >/dev/null 2>&1 || { echo "🔴 设备不可达，请插线"; exit 2; }

# 🔴 adb.exe 是 Windows 程序，**不认 MSYS 的 /d/... 绝对路径**（2026-08-27 实测：
# `adb: error: cannot stat '/d/...'`，交付在第一个 push 就停住）。凡是交给 adb 的
# 宿主侧路径一律先过 cygpath -w；留给 bash 用的（stat/sha256sum）仍用 MSYS 路径。
w() { cygpath -w "$1"; }

# contract-only：设备上模型与 APK 都已是 32 槽版，只有契约是坏的 ⇒ 只推 JSON，不动其余。
# （2026-08-27 事故修复路径：app 报 "manifest input byte mismatch: input_ids"）
if [ "$MODE" = contract-only ]; then
  echo "=== 只推契约（不动模型、不重装 APK）==="
  "$ADB" shell "mkdir -p $TMP"
  "$ADB" push "$(w "$NEW_CONTRACT")" "$TMP/final_qnn_contract.json" >/dev/null || exit 1
  "$ADB" shell "run-as $PKG cp $TMP/final_qnn_contract.json files/models/ZIMAGE/final_qnn_contract.json" || exit 1
  want=$(sha256sum "$NEW_CONTRACT" | cut -d' ' -f1)
  got=$("$ADB" shell "run-as $PKG sha256sum files/models/ZIMAGE/final_qnn_contract.json" | tr -d '\r' | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "🔴 落盘 sha 不符"; exit 1; }
  "$ADB" shell "rm -rf $TMP"
  echo "   ✅ sha 一致 ${want:0:16}…"
  echo "现在可直接在 app 里跑 T3 / T4，无需重装。"
  exit 0
fi

if [ "$MODE" = rollback ]; then
  SRC_DIR_FMT="$OLD_ROOT/text_encoder_part%s/text_encoder_part%s_ctx_sm8750.SM8750.bin"
  USE_APK="$OLD_APK"; USE_CONTRACT="$OLD_CONTRACT"; echo "=== 回滚到 20 槽版本 ==="
else
  SRC_DIR_FMT="$NEW_ROOT/text_encoder_part%s/text_encoder_part%s_ctx_sm8750.SM8750.bin"
  USE_APK="$APK"; USE_CONTRACT="$NEW_CONTRACT"; echo "=== 交付 32 槽版本 ==="
fi

"$ADB" shell "mkdir -p $TMP"
for P in 1 2 3 4; do
  f=$(printf "$SRC_DIR_FMT" $P $P)
  [ -f "$f" ] || { echo "🔴 缺少 $f"; exit 1; }
  n=$(basename "$f")
  echo "-- push $n ($(( $(stat -c %s "$f") / 1048576 )) MB)"
  "$ADB" push "$(w "$f")" "$TMP/$n" >/dev/null || { echo "🔴 push 失败"; exit 1; }
  "$ADB" shell "run-as $PKG cp $TMP/$n files/models/ZIMAGE/models/$n" || { echo "🔴 cp 失败"; exit 1; }
  # 逐位校验：设备上的副本必须与宿主源文件同 sha256
  want=$(sha256sum "$f" | cut -d' ' -f1)
  got=$("$ADB" shell "run-as $PKG sha256sum files/models/ZIMAGE/models/$n" | tr -d '\r' | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "🔴 $n 落盘后 sha 不符：$want vs $got"; exit 1; }
  echo "   ✅ sha 一致 ${want:0:16}…"
  "$ADB" shell "rm -f $TMP/$n"
done

echo "-- push 契约"
"$ADB" push "$(w "$USE_CONTRACT")" "$TMP/final_qnn_contract.json" >/dev/null
"$ADB" shell "run-as $PKG cp $TMP/final_qnn_contract.json files/models/ZIMAGE/final_qnn_contract.json" || exit 1
"$ADB" shell "rm -rf $TMP"

echo "-- 安装 APK $(basename "$USE_APK")"
rc_i=0; "$ADB" install -r "$(w "$USE_APK")" | tail -1; rc_i=${PIPESTATUS[0]}
[ $rc_i -eq 0 ] || { echo "🔴 安装失败 rc=$rc_i"; exit 1; }

echo
echo "=== 交付完成。现在必须在真实 app 里验收（约束 11·③④，不得用 run-as）==="
echo "  T3 短 prompt 回归：用 12 token 以内的 prompt + 固定 seed 生成一次"
echo "     预测：与旧版**逐字节相同**（因果性 + encoding 未变 + cap_pad_mask 相同）"
echo "     若不同 ⇒ 说明改动影响了本不该变的输入，按 rollback 回滚"
echo "  T4 长 prompt 生效：用 13~24 token 的 prompt"
echo "     预测：计数器显示 n/32；出图与截断版**不同**"
echo "  T5 内存：生成期间系统可用内存不见底、app 不被杀"
echo
echo "  回滚： bash scripts/te_s32_deliver.sh rollback"
