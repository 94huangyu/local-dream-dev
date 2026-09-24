# -*- coding: utf-8 -*-
"""用 NDK 把 `tools/sha_bench.cpp` 编成 arm64 可执行文件（台账 #178）。

为什么走这条路而不是直接改 app：改 app 要 gradle 构建 + 装 APK + 起后端，一轮十几分钟；
而 SHA 指令写错的后果是契约校验全失败、app 起不来。先用几十 KB 的可执行文件把
「算得对不对、快多少」单独确认掉，通过了再集成。

用法: python scripts/build_sha_bench.py          # 只编译
      python scripts/build_sha_bench.py --push   # 编译 + 推到设备并运行
"""
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join("D:", os.sep, "LocalDreamZImage")
SRC = os.path.join(REPO, "tools", "sha_bench.cpp")
INC = os.path.join(REPO, "local-dream", "app", "src", "main", "cpp", "src")
OUT = os.path.join(REPO, "scratch_runs", "sha_bench")


def find_clang():
    """在 Android SDK 的 NDK 里找 aarch64 的 clang++。**找不到就报清楚**，不猜路径。"""
    roots = [os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\ndk"),
             os.path.join("C:", os.sep, "Android", "Sdk", "ndk")]
    cands = []
    for r in roots:
        if os.path.isdir(r):
            for v in sorted(os.listdir(r), reverse=True):
                p = os.path.join(r, v, "toolchains", "llvm", "prebuilt", "windows-x86_64",
                                 "bin", "aarch64-linux-android30-clang++.cmd")
                cands += glob.glob(p)
                cands += glob.glob(p.replace("android30", "android3*"))
    if not cands:
        raise SystemExit("🔴 找不到 NDK 的 aarch64 clang++。已找过：\n  " + "\n  ".join(roots))
    return cands[0]


def main():
    cxx = find_clang()
    print("NDK clang++: %s" % cxx, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = [cxx, "-O2", "-std=c++17", "-static-libstdc++",
           "-I", INC, SRC, "-o", OUT]
    print("编译……", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    if r.returncode != 0:
        raise SystemExit("🔴 编译失败 rc=%d" % r.returncode)
    print("✅ 产出 %s（%.1f KB）" % (OUT, os.path.getsize(OUT) / 1024.0), flush=True)
    if "--push" not in sys.argv:
        return 0

    import lab_dev as dev
    dev.require_online("sha_bench")
    dst = "/data/local/tmp/sha_bench"
    subprocess.run([dev.ADB, "push", OUT, dst], check=True, timeout=300)
    dev.sh("chmod 755 %s" % dst)
    print("\n== 设备实测 ==", flush=True)
    out = dev.sh("%s" % dst, timeout=1800)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
