#!/usr/bin/env bash
# Tier 2 交付：caption 槽 32 -> 80（可用正文 24 -> 72 token）。台账 #134 / #139。
#
# 🔴🔴 **非破坏性交付**（2026-08-29，用户明确要求「别破坏之前的成果」）：
#   L=80 的八段用**新文件名** `*_L80_ctx_sm8750.SM8750.bin` 下发，
#   **设备上的 L=32 产物一个字节都不动**。设备本来就并存多代
#   （`_ctx_` / `_perrow_ctx_` / `_clip_ctx_`，实查 24 GB），/data 剩 529 GB。
#   => **回滚 = 推回 40 KB 的旧契约 + 装回旧 APK，不需要恢复 10 GB 模型。**
#
# 🔴 约束 11 四条铁律：
#   ① 先证明现状可用 —— precheck 比对设备 APK 与设备上在用的八段 .bin 的 sha256
#   ② 先备份原件   —— precheck 存下设备 APK 与设备契约到 logs/tier2_20260829/backup/
#   ③ 交付后必须自己跑一次真实链路（静态校验不算验证）
#   ④ 不得用 run-as 代替真实 app 验证（SELinux 域不同，#72）
#
# 🔴 adb.exe 不认 MSYS 的 /d/... 路径 => 一律先过 cygpath -w（#83）
# 🔴 循环里的 adb 调用必须 `< /dev/null` —— adb shell 会吃掉 while read 的 stdin，
#    2026-08-29 实测：precheck 的八段循环只跑了 1 条就静默退出。
# 🔴 不套 tail/head（管道返回码是最后一段的，会把失败伪装成成功，§7.4）
#
# 用法：
#   bash scripts/t2_deliver.sh precheck   # 只读：验证基线 + 备份，不动设备
#   bash scripts/t2_deliver.sh deliver    # 推 8 段(新名) + 契约 + 装 APK
#   bash scripts/t2_deliver.sh rollback   # 回滚：只换契约 + 装回旧 APK
set -u
ADB='C:/Users/sinai/AppData/Local/Android/Sdk/platform-tools/adb.exe'
PKG=io.github.xororz.localdream.zimage
export MSYS_NO_PATHCONV=1
MODE="${1:-precheck}"

ROOT=/d/LocalDreamZImage
OUT=$ROOT/logs/tier2_20260829
BK=$OUT/backup
NEW_APK=$ROOT/local-dream/app/build/outputs/apk/basic/debug/LocalDreamZImage_armv8a_2.8.1-zimage-mvp.apk
OLD_APK=$ROOT/scratch_runs/apk_backup/built_2026-08-27_s32.apk
NEW_CONTRACT=$OUT/final_qnn_contract.L80.json
ZDIR=files/models/ZIMAGE
MDIR=$ZDIR/models
TMP=/data/local/tmp/t2
mkdir -p "$BK"

w() { cygpath -w "$1"; }

# 设备上**当前在用**的文件名（L=32） | L=80 的新文件名 | 新 .bin 宿主路径 | 旧 .bin 宿主副本
MAP="
text_encoder_part1|text_encoder_part1_ctx_sm8750.SM8750.bin|text_encoder_part1_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER2_L80/text_encoder_part1/text_encoder_part1_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER1_S32/text_encoder_part1/text_encoder_part1_ctx_sm8750.SM8750.bin
text_encoder_part2|text_encoder_part2_ctx_sm8750.SM8750.bin|text_encoder_part2_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER2_L80/text_encoder_part2/text_encoder_part2_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER1_S32/text_encoder_part2/text_encoder_part2_ctx_sm8750.SM8750.bin
text_encoder_part3|text_encoder_part3_ctx_sm8750.SM8750.bin|text_encoder_part3_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER2_L80/text_encoder_part3/text_encoder_part3_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER1_S32/text_encoder_part3/text_encoder_part3_ctx_sm8750.SM8750.bin
text_encoder_part4|text_encoder_part4_ctx_sm8750.SM8750.bin|text_encoder_part4_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER2_L80/text_encoder_part4/text_encoder_part4_ctx_sm8750.SM8750.bin|/d/ZImage_Work/TIER1_S32/text_encoder_part4/text_encoder_part4_ctx_sm8750.SM8750.bin
transformer_part1a|transformer_part1a_clip_ctx_sm8750.SM8750.bin|transformer_part1a_clip_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part1a_fp16_L80/part1a_fp16_L80.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part1a_fp16/part1a_fp16.SM8750.bin
transformer_part1b|transformer_part1b_clip_ctx_sm8750.SM8750.bin|transformer_part1b_clip_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part1b_fp16_L80/part1b_fp16_L80.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part1b_fp16/part1b_fp16.SM8750.bin
transformer_part2a|transformer_part2a_clip_ctx_sm8750.SM8750.bin|transformer_part2a_clip_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part2a_fp16_L80/part2a_fp16_L80.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part2a_fp16/part2a_fp16.SM8750.bin
transformer_part2b|transformer_part2b_clip_ctx_sm8750.SM8750.bin|transformer_part2b_clip_L80_ctx_sm8750.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part2b_fp16_L80/part2b_fp16_L80.SM8750.bin|/d/ZImage_Work/p0_experiments/p2attr/ctx_part2b_fp16/part2b_fp16.SM8750.bin
"

need_device() {
  "$ADB" start-server >/dev/null 2>&1
  "$ADB" wait-for-device >/dev/null 2>&1 &
  local wp=$!
  ( sleep 15; kill $wp 2>/dev/null ) >/dev/null 2>&1 &
  wait $wp 2>/dev/null
  "$ADB" get-state >/dev/null 2>&1 < /dev/null || { echo "🔴 设备不可达，请插线"; exit 2; }
  echo "设备: $("$ADB" shell getprop ro.serialno < /dev/null | tr -d '\r')"
}

dev_sha() {   # $1 = models/ 下的文件名
  "$ADB" shell "run-as $PKG sha256sum $MDIR/$1" < /dev/null 2>/dev/null | tr -d '\r' | cut -d' ' -f1
}

precheck() {
  need_device
  local bad=0
  echo "=== ① 证明现状可用 + ② 备份原件 ==="
  "$ADB" shell "pm path $PKG" < /dev/null >/dev/null 2>&1 || { echo "🔴 app 未安装"; exit 3; }
  local apath a b
  apath=$("$ADB" shell "pm path $PKG" < /dev/null | tr -d '\r' | head -1 | sed 's/^package://')
  "$ADB" pull "$apath" "$(w "$BK/installed_before_L80.apk")" < /dev/null >/dev/null 2>&1 \
    || { echo "🔴 拉 APK 失败"; exit 3; }
  a=$(sha256sum "$BK/installed_before_L80.apk" | cut -d' ' -f1)
  b=$(sha256sum "$OLD_APK" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "  ✅ 设备 APK == built_2026-08-27_s32.apk (${a:0:16}…)"
  else echo "  ⚠️ 设备 APK 与宿主备份不同（回滚将用 $BK/installed_before_L80.apk）"; fi
  "$ADB" shell "run-as $PKG cat $ZDIR/final_qnn_contract.json" < /dev/null > "$BK/contract_before_L80.json" 2>/dev/null
  [ -s "$BK/contract_before_L80.json" ] \
    && echo "  ✅ 已备份设备契约 ($(stat -c %s "$BK/contract_before_L80.json") B)" \
    || { echo "  🔴 拉设备契约失败"; bad=1; }

  echo "=== 设备上**在用的 L=32 八段** vs 宿主副本（证明回滚基线可信）==="
  while IFS='|' read -r n cur new src old; do
    [ -z "$n" ] && continue
    [ -f "$old" ] || { echo "  🔴 $n 宿主 L=32 副本不存在: $old"; bad=1; continue; }
    local ds hs
    ds=$(dev_sha "$cur"); hs=$(sha256sum "$old" | cut -d' ' -f1)
    if [ "$ds" = "$hs" ]; then echo "  ✅ $n  ${ds:0:16}…"
    else echo "  🔴 $n 设备 ${ds:0:16}… != 宿主 ${hs:0:16}…"; bad=1; fi
  done <<< "$MAP"

  echo "=== 新产物完整性（宿主）==="
  while IFS='|' read -r n cur new src old; do
    [ -z "$n" ] && continue
    [ -f "$src" ] || { echo "  🔴 缺 $src"; bad=1; }
  done <<< "$MAP"
  [ -f "$NEW_CONTRACT" ] || { echo "  🔴 缺 $NEW_CONTRACT"; bad=1; }
  [ -f "$NEW_APK" ]      || { echo "  🔴 缺 $NEW_APK"; bad=1; }
  [ $bad -eq 0 ] && echo "  ✅ 齐全"

  echo "=== 设备空间（非破坏性交付要多放 10.3 GB）==="
  "$ADB" shell "df -h /data" < /dev/null | tr -d '\r'

  if [ $bad -ne 0 ]; then echo; echo "🔴 precheck 未通过，**不要交付**"; exit 1; fi
  echo; echo "✅ precheck 通过：基线已证、原件已备份、新产物齐全"
  echo "   下一步： bash scripts/t2_deliver.sh deliver"
}

deliver() {
  need_device
  echo "=== 推送八段 L=80（新文件名，不覆盖 L=32；逐段暂存->拷贝->删暂存）==="
  "$ADB" shell "mkdir -p $TMP" < /dev/null || exit 1
  while IFS='|' read -r n cur new src old; do
    [ -z "$n" ] && continue
    echo "--- $n  $(( $(stat -c %s "$src") / 1048576 )) MB  -> $new ---"
    "$ADB" push "$(w "$src")" "$TMP/$new" < /dev/null || exit 1
    "$ADB" shell "run-as $PKG cp $TMP/$new $MDIR/$new" < /dev/null || exit 1
    "$ADB" shell "rm -f $TMP/$new" < /dev/null || exit 1
    local want got
    want=$(sha256sum "$src" | cut -d' ' -f1); got=$(dev_sha "$new")
    [ "$want" = "$got" ] || { echo "🔴 $n 落盘 sha 不符"; exit 1; }
    echo "    ✅ ${got:0:16}…"
  done <<< "$MAP"

  echo "=== 推契约 ==="
  "$ADB" push "$(w "$NEW_CONTRACT")" "$TMP/c.json" < /dev/null || exit 1
  "$ADB" shell "run-as $PKG cp $TMP/c.json $ZDIR/final_qnn_contract.json" < /dev/null || exit 1
  local want got
  want=$(sha256sum "$NEW_CONTRACT" | cut -d' ' -f1)
  got=$("$ADB" shell "run-as $PKG sha256sum $ZDIR/final_qnn_contract.json" < /dev/null | tr -d '\r' | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "🔴 契约落盘 sha 不符"; exit 1; }
  "$ADB" shell "rm -rf $TMP" < /dev/null
  echo "  ✅ 契约 ${want:0:16}…"

  echo "=== 装 APK ==="
  "$ADB" install -r "$(w "$NEW_APK")" < /dev/null || exit 1
  echo
  echo "✅ 交付完成（L=32 产物仍在设备上，未被触碰）"
  echo "🔴 约束 11 第 3 条：现在必须在真实 app 里跑一次出图，静态校验不算验证。"
  echo "   验收点：① 能出图；② prompt 计数器上限应为 72；③ >24 token 的 prompt 后半段应体现在图里"
}

rollback() {
  need_device
  echo "=== 回滚：只换契约 + 装回旧 APK（L=32 模型一直都在，无需恢复）==="
  local C="$BK/contract_before_L80.json"
  [ -s "$C" ] || { echo "🔴 没有设备契约备份，先跑 precheck"; exit 1; }
  "$ADB" shell "mkdir -p $TMP" < /dev/null || exit 1
  "$ADB" push "$(w "$C")" "$TMP/c.json" < /dev/null || exit 1
  "$ADB" shell "run-as $PKG cp $TMP/c.json $ZDIR/final_qnn_contract.json" < /dev/null || exit 1
  "$ADB" shell "rm -rf $TMP" < /dev/null
  local A="$BK/installed_before_L80.apk"
  [ -s "$A" ] || A="$OLD_APK"
  "$ADB" install -r "$(w "$A")" < /dev/null || exit 1
  echo "✅ 已回滚到 L=32"
}

case "$MODE" in
  precheck) precheck ;;
  deliver)  deliver ;;
  rollback) rollback ;;
  *) echo "用法: bash scripts/t2_deliver.sh [precheck|deliver|rollback]"; exit 64 ;;
esac
