#!/bin/sh
# 交付 1184x896 支持（台账 #86）。只增不删：1:1 的 .bin 与契约条目逐字未动。
#
# 🔴 2026-09-04 第一版踩了三个坑，这一版逐个堵住（全部是实测，不是预防性猜测）：
#  1) **Git Bash 的 MSYS 路径转换**把 `/data/local/tmp/x` 改写成
#     `C:/Program Files/Git/data/local/tmp/x`，push 全部失败。
#     => 全脚本用 `MSYS2_ARG_CONV_EXCL='*'`，并且**推完必须比对设备侧字节数**。
#  2) **`adb push` 失败后照样打印 "1 file pushed ... 43.6 MB/s"**（约束 11·再补
#     「报错说谎」）。第一版还用 `| tail -1` 恰好只留下那句谎话，`set -e` 因而没触发。
#     => 不再对 push 接管道；查返回码 + 查 stderr 里的 "error:" + 查设备侧字节数。
#  3) `run-as sh -c 'cat > f'` 在**源不存在时照样建出 0 字节文件**，
#     配上已经换好的新契约，设备当场变成坏状态。
#     => 每一步写入后立刻校验字节数，不合就退出；契约**最后**才换。
#  4) `adb shell "... cat > f" < 本地文件` 在 Windows 上**不是二进制安全的**
#     （实测契约 64854 B 传成 62572 B）。=> 一律先 push 到暂存区，再设备本地 cat。
set -e
export MSYS2_ARG_CONV_EXCL='*'
ADB="$1"; PKG="$2"; SRC="$3"; CONTRACT="$4"
# 🔴 中间文件必须落在 **Windows Python 也能打开** 的路径。
#    第一版用 `${TMPDIR:-/tmp}`：Git Bash 的 /tmp 映射到某个 Windows 目录，
#    但传给 Windows 版 python 的字符串仍是字面量 "/tmp/..."，于是
#    FileNotFoundError，sha256 门直接崩掉（按设计中止了交付，设备没坏，
#    但每次都要手工补做最后两步）。用契约所在目录最稳。
WORK="$(dirname "$4")"
APPDIR=files/models/ZIMAGE
D="$APPDIR/models"
TMP=/data/local/tmp/aspect
RA="run-as $PKG"

dev_size() { "$ADB" shell "stat -c %s '$1' 2>/dev/null || echo 0" | tr -d '\r\n'; }
app_size() { "$ADB" shell "$RA stat -c %s '$1' 2>/dev/null || echo 0" | tr -d '\r\n'; }

push_verified() {   # $1=本地文件 $2=设备目标
  want=$(stat -c %s "$1")
  if [ "$(dev_size "$2")" = "$want" ]; then echo "    暂存区已有且大小一致，跳过 push"; return 0; fi
  err=$("$ADB" push "$1" "$2" 2>&1 >/dev/null) || { echo "    🔴 push 返回码非零: $err"; return 1; }
  case "$err" in *error:*) echo "    🔴 push 报错: $err"; return 1;; esac
  got=$(dev_size "$2")
  [ "$got" = "$want" ] || { echo "    🔴 设备侧字节数 $got != 期望 $want"; return 1; }
  echo "    ok $want B"
}

echo "=== 1/5 推到设备暂存区（USB）==="
"$ADB" shell "mkdir -p $TMP"
for f in "$SRC"/*.bin; do
  b=$(basename "$f"); echo "  $b"; push_verified "$f" "$TMP/$b"
done
echo "  $(basename "$CONTRACT")"; push_verified "$CONTRACT" "$TMP/contract.json"

echo "=== 2/5 设备内复制进 app 私有目录（本地 cat，不经 USB）==="
for f in "$SRC"/*.bin; do
  b=$(basename "$f"); want=$(stat -c %s "$f")
  echo "  $b"
  "$ADB" shell "cat $TMP/$b | $RA sh -c 'cat > $D/$b'"
  got=$(app_size "$D/$b")
  [ "$got" = "$want" ] || { echo "    🔴 写入后字节数 $got != $want，中止（契约未动，设备仍是好的）"; exit 1; }
  echo "    ok $want B"
done

echo "=== 3/5 sha256 现算，与契约声明逐一比对 ==="
# 🔴 第一版这里【只打印不比对】——判据写了却没执行，等于没有这道门。
#    本项目已因「门是装饰性的」栽过（约束 8）。现在真比，不符即退出。
# 🔴 2026-09-17：原来是 `sha256sum *_ctx.SM8750.bin` —— 漏掉了文本编码器
#    （`..._ctx_sm8750.SM8750.bin`），mg 交付因此回滚。文件清单改为从契约派生。
NAMES=$(python "$(dirname "$0")/check_deploy_sha.py" --names "$CONTRACT" | tr -d '\r')
[ -n "$NAMES" ] || { echo "  🔴 契约里取不到要校验的文件名，中止"; exit 1; }
"$ADB" shell "$RA sh -c 'cd $D && sha256sum $NAMES'" | tr -d '\r' > "$WORK/dev_sha.txt"
python "$(dirname "$0")/check_deploy_sha.py" "$CONTRACT" "$WORK/dev_sha.txt" || exit 1

echo "=== 4/5 备份当前契约（设备本地）==="
"$ADB" shell "$RA sh -c 'cp -f $APPDIR/final_qnn_contract.json $APPDIR/final_qnn_contract.pre-aspect-backup.json'"

echo "=== 5/5 换契约（一切校验通过后才做这一步）==="
want=$(stat -c %s "$CONTRACT")
"$ADB" shell "cat $TMP/contract.json | $RA sh -c 'cat > $APPDIR/final_qnn_contract.json'"
got=$(app_size "$APPDIR/final_qnn_contract.json")
[ "$got" = "$want" ] || { echo "  🔴 契约字节数 $got != $want，回滚"; \
  "$ADB" shell "$RA sh -c 'cp -f $APPDIR/final_qnn_contract.pre-aspect-backup.json $APPDIR/final_qnn_contract.json'"; exit 1; }
echo "  契约 ok $want B"
echo "完成。暂存区 $TMP 保留（回滚/重试可复用），交付后再删。"
