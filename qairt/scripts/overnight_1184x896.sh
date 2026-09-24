#!/bin/bash
# 今晚无人值守链：把 1184x896 做到「只差插线」为止。
# 顺序是按主机争用排的：先便宜的手术/量化，再 2 小时的建图，最后 ORT 参考图。
# 🔴 全程串行，绝不并发建图（本项目两次把 23.7 GB 打穿都是并发造成的）。
set -u
cd /d/LocalDreamZImage
export PYTHONIOENCODING=utf-8
PY=python
VPY=/d/ZImage_Work/venv-official/Scripts/python.exe
T=1184x896; W=1184; H=896

step () { echo ""; echo "########## $1  $(date +%H:%M)"; shift; "$@"; rc=$?; echo "   rc=$rc"; [ $rc -ne 0 ] && { echo "!!! 失败，链条停止"; exit 1; }; return 0; }

step "1/6 手术 deployed"  $VPY scripts/dit_aspect_surgery.py $T $W $H --variant=deployed
step "2/6 手术 base"      $VPY scripts/dit_aspect_surgery.py $T $W $H --variant=base
step "3/6 结构门 S2/S4/S5" $VPY scripts/aspect_surgery_check.py $T $W $H
step "4/6 四段转换+量化"   $PY scripts/aspect_build.py $T --segs part1a,part1b,part2a,part2b --skip-build
step "5/6 VAE FP32 手术"   $VPY scripts/vae_aspect_surgery.py $T $W $H
step "6/6 四段双比例建图"  $PY scripts/aspect_multigraph_build.py $T

echo ""; echo "########## 全部完成 $(date +%H:%M)"
echo "下一步需要插线：D1 复验 + D2 画质"
