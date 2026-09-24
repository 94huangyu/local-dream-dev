"""在设备 HTP 上跑 part1b 的 `.bin`，输入取 testB s0（与实验 A 逐字节相同）。

方案 `EXP_PLAN_SYM_FC_OVR` 第 2.3 步 ④。

纪律：
  · 输入必须与实验 A 的那一份**逐字节相同** —— 脚本自己核 md5，不一致就退出
  · 不加 `--use_native_input_files` / `--use_native_output_files`
    （非 native 模式下 qnn-net-run 按 float32 读写，与宿主口径一致）
  · 输出字节数必须是 15,851,520 × 4 = 63,406,080 —— 防静默缩批
  · 表述纪律：本脚本产出的结论才可以写"设备实测"；
    `snpe-net-run` 跑 `.dlc` 的只能写"在 SNPE CPU 参考实现上"

用法: python device_run_part1b.py <ctx.bin> <设备子目录名> <本地输出.raw>
"""
import hashlib
import os
import subprocess
import sys

ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
SRC = r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b"
NAMES = ("add_138", "add_131", "tanh_19", "adaln_input", "select_45", "select_46")
REMOTE = "/data/local/tmp/htpcmp"
EXPECT_BYTES = 63406080
# 实验 A 用的那一份输入的 md5（①实测，2026-08-15 由本机核出）
EXPECT_MD5 = {"add_138": "5cd1cabfed4e684eec4c461de882d165"}


def sh(*args, check=True):
    r = subprocess.run([ADB] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"adb {' '.join(args)} 失败:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    binp, tag, out_local = sys.argv[1], sys.argv[2], sys.argv[3]
    rdir = f"{REMOTE}/{tag}"

    devs = [l for l in sh("devices").splitlines()[1:] if l.strip()]
    if not devs:
        sys.exit("没有检测到设备，请插线并确认 USB 调试已授权")
    print("设备:", devs)

    # 有效性自检 1：输入必须与实验 A 逐字节相同
    for n, want in EXPECT_MD5.items():
        got = md5(os.path.join(SRC, n + ".raw"))
        print(f"  输入 {n} md5 = {got}  {'OK' if got == want else '✗ 不一致'}")
        if got != want:
            sys.exit("输入与实验 A 不同，拒绝继续（会破坏单变量对照）")

    sh("shell", f"rm -rf {rdir}; mkdir -p {rdir}/in {rdir}/out")
    print(f"推送 .bin ({os.path.getsize(binp)/1e6:.0f} MB) ...")
    sh("push", binp, f"{rdir}/ctx.bin")
    # 有效性自检 2：设备上的 .bin 与宿主同 md5
    host_md5 = md5(binp)
    dev_md5 = sh("shell", f"md5sum {rdir}/ctx.bin").split()[0]
    print(f"  .bin md5 宿主 {host_md5}  设备 {dev_md5}  "
          f"{'OK' if host_md5 == dev_md5 else '✗ 不一致'}")
    if host_md5 != dev_md5:
        sys.exit("推送后 md5 不一致")

    for n in NAMES:
        sh("push", os.path.join(SRC, n + ".raw"), f"{rdir}/in/{n}.raw")
    line = " ".join(f"{n}:={rdir}/in/{n}.raw" for n in NAMES)
    sh("shell", f"printf '%s\\n' '{line}' > {rdir}/in/list.txt")
    print("input_list:", sh("shell", f"cat {rdir}/in/list.txt")[:160], "...")

    cmd = (f"cd {REMOTE} && export LD_LIBRARY_PATH={REMOTE} && "
           f"export ADSP_LIBRARY_PATH={REMOTE} && "
           f"./qnn-net-run --retrieve_context {rdir}/ctx.bin --backend libQnnHtp.so "
           f"--input_list {rdir}/in/list.txt --output_dir {rdir}/out --log_level info")
    print("设备执行 ...")
    print(sh("shell", cmd)[-1500:])

    sh("pull", f"{rdir}/out/Result_0/unified.raw", out_local)
    n = os.path.getsize(out_local)
    print(f"拉回 {out_local}: {n} 字节 "
          f"{'OK' if n == EXPECT_BYTES else f'✗ 期望 {EXPECT_BYTES}（静默缩批？）'}")
    if n != EXPECT_BYTES:
        sys.exit(1)


if __name__ == "__main__":
    main()
