#!/usr/bin/env bash
# Tier 2 一致性测量的宿主半边（零设备）。
# 🔴 G0-known 用**产出盘上参考的那份 caption**（htp_inloop/const/caption.raw，20 真实槽 + 复制补齐到 32），
#    而不是重新生成 —— 只有这样才能与 fp32_final_latents_inloopref_vaefix.raw 对得上。
# 🔴 不过 G0-known 就不跑 L=80。每步单独查 rc，不套 tail/head（§7.4）。
set -u
export PYTHONIOENCODING=utf-8
cd /d/LocalDreamZImage || exit 90
LOG=/d/LocalDreamZImage/logs/t2_fp32; mkdir -p "$LOG"
P0='D:\ZImage_Work\p0_experiments'
INIT="$P0\htp_inloop\s0\latents.raw"

step () { local n="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $n ==="
  "$@" > "$LOG/$n.log" 2>&1; local rc=$?
  echo "--- $n rc=$rc ---"; tail -n 14 "$LOG/$n.log"
  if [ $rc -ne 0 ]; then echo "🔴 $n 失败 rc=$rc"; exit $rc; fi
}

# ① 验证驱动：L=32 + 部署那份 caption，必须复现盘上终点 latents
step drive_L32 python scripts/t2_fp32_ref.py drive 32 "$INIT" \
     "$P0\htp_inloop\const\caption.raw" "$P0\htp_inloop\const\cap_pad_mask.raw" \
     "D:\LocalDreamZImage\scratch_runs\fp32ref_cat_L32.png"
# ② L=80 的 caption（用 _L80 的 FP32 TE）
step caption_L80 python scripts/t2_fp32_ref.py caption "$P0\prompt_ids.npy" 80 \
     "$P0\caption_cat_L80.raw" "$P0\cap_pad_mask_cat_L80.raw"
# ③ L=80 的 FP32 参考
step drive_L80 python scripts/t2_fp32_ref.py drive 80 "$INIT" \
     "$P0\caption_cat_L80.raw" "$P0\cap_pad_mask_cat_L80.raw" \
     "D:\LocalDreamZImage\scratch_runs\fp32ref_cat_L80.png"
echo "=== [$(date +%H:%M:%S)] 全部完成 ==="
