"""#111 Clip 版四段的设备端实测（一条命令跑完，减少现场操作失误）。

流程（每步都有门，失败即停）：
  1. 设备可达检查（`lab_dev`，不可达抛 DeviceOffline，**不会退化成「任务完成」**）
  2. 🔴 **先在设备侧改名备份**旧的 `_fp16` 四段（约束 11 铁律 2：覆盖前先备份）
  3. 推 4 段新 context，**逐个核对字节数**
  4. 端到端 8 步 + VAE（`htp_inloop_pipeline.py`，TAG=clip，SUFFIX=_fp16，P2=split）
  5. `panel3.py` 三层指标

判据（事前锁定，`EXP_PLAN_SIM_CEILING` / §6 P0）：
  L3 PSNR 相对当前最优 **15.53 dB**：≥1.0 dB 🟢 ／ 0.2~1.0 🟡 ／ <0.2 或倒退 ❌
  模拟预测 L1 由 30.68% 降到约 **22%**（mode H = 18.18% x 1.22）；
  ⚠️ #121 已证 L1 不能换算 L3 ⇒ **不对 L3 做数值预测**。

用法: python clip_device_run.py [--push-only|--run-only]
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
ADB = None
for c in [r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe", "adb"]:
    if c == "adb" or os.path.exists(c):
        ADB = c
        break

# (本地 context, 设备上的名字)
SEGS = [("ctx_part1a_fp16/part1a_fp16.SM8750.bin", "part1a_fp16.bin"),
        ("ctx_part1b_fp16/part1b_fp16.SM8750.bin", "part1b_fp16.bin"),
        ("ctx_part2a_fp16/part2a_fp16.SM8750.bin", "part2a_fixed_fp16.bin"),
        ("ctx_part2b_fp16/part2b_fp16.SM8750.bin", "part2b_fixed_fp16.bin")]


def sh(args, **kw):
    """透传 stderr 与返回码 —— 包装子进程必须这样（约束 11·再补：报错说谎比报错本身贵）"""
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", **kw)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def adb(*a, **kw):
    return sh([ADB, "-s", DEV] + list(a), **kw)


def main():
    import lab_dev
    if not lab_dev.online():
        sys.exit("FAIL 设备不可达 —— 请先插线")
    print("✅ 设备在线", flush=True)

    only = sys.argv[1] if len(sys.argv) > 1 else ""

    if only != "--run-only":
        # --- 2. 备份（改名，不复制：瞬时且不占空间）---
        for _, dn in SEGS:
            bak = dn.replace(".bin", "_old26.bin")
            rc, out = adb("shell", "cd %s && [ -f %s ] && mv %s %s || true" % (T, dn, dn, bak))
            if rc:
                sys.exit("FAIL 备份 %s: %s" % (dn, out[-300:]))
        print("✅ 旧 _fp16 四段已在设备侧改名为 *_old26.bin（可回滚）", flush=True)

        # --- 3. 推送并逐个核对字节数 ---
        for lp, dn in SEGS:
            p = os.path.join(W, lp.replace("/", os.sep))
            if not os.path.exists(p):
                sys.exit("FAIL 本地缺 %s" % p)
            n = os.path.getsize(p)
            t0 = time.time()
            rc, out = adb("push", p, "%s/%s" % (T, dn))
            if rc:
                sys.exit("FAIL push %s: %s" % (dn, out[-300:]))
            rc, out = adb("shell", "stat -c %%s %s/%s" % (T, dn))
            got = out.strip().replace("\r", "")
            if got != str(n):
                sys.exit("FAIL %s 字节数不符：设备 %s vs 本地 %d" % (dn, got, n))
            print("  ✅ %-24s %10d bytes  %.1f min" % (dn, n, (time.time() - t0) / 60),
                  flush=True)

    if only == "--push-only":
        print("（--push-only：到此为止）")
        return 0

    # --- 4. 端到端 ---
    print("\n=== 端到端 8 步 + VAE（约 12 分钟）===", flush=True)
    env = dict(os.environ, INLOOP_TAG="clip", INLOOP_BIN_SUFFIX="_fp16",
               INLOOP_P2="split", PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "htp_inloop_pipeline.py")], env=env)
    if r.returncode != 0:
        sys.exit("FAIL 端到端 rc=%d" % r.returncode)

    # --- 5. 三层指标 ---
    print("\n=== 三层指标 ===", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "panel3.py")],
                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print("\n🔌 设备任务结束 —— 可以拔线", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
