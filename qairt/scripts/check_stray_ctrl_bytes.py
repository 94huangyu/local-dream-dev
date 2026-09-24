# -*- coding: utf-8 -*-
"""扫描源码/脚本里的【裸控制字节】——转义序列被展开成真字节的痕迹。

## 为什么要有这个检查（不是预防性设计，是实测反复发生）
本项目 §7.4 记过：通过 heredoc 写含反斜杠的字符串时，`\\n` / `\\r` / `\\0`
会在传递链的某一环被展开成**真的控制字节**写进文件。2026-09-04 一天内发生三次：

  1. `QnnModel.hpp` 里 `'\\0'` 变成裸 NUL —— Clang 恰好按值 0 解释，
     **语义没错所以不报错**，但让 grep 把整个文件当二进制、排查时误导了我。
  2. `deploy_aspect.sh` 里 `tr -d '\\r'` 变成 `tr -d '<CR>'` —— 行为恰好相同，
     **同样不报错**，但下一个读代码的人看到的是一对空引号。
  3. 同一处，在我写「修复上一次」的补丁时**又犯了一遍**。

⇒ 三次都不会被编译器或测试抓到。**靠纪律防不住，只能靠检查。**

## 判据
源文件（.cpp/.hpp/.py/.sh/.kt/.json）里不允许出现：
  - NUL (0x00)
  - CR (0x0D)，除非该文件整体是 CRLF 行尾（此时 CR 只允许紧邻 LF）
  - 其他 C0 控制字符（除 TAB 0x09 / LF 0x0A）
用法: python check_stray_ctrl_bytes.py [根目录...]   非零退出 = 有问题
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

EXTS = (".cpp", ".hpp", ".h", ".py", ".sh", ".kt", ".json", ".cmake", ".md")
SKIP_DIRS = {".git", "build", "intermediates", "node_modules", "__pycache__",
             "3rdparty", ".gradle", ".idea"}
ALLOWED = {0x09, 0x0A}


def scan(path):
    with open(path, "rb") as f:
        b = f.read()
    # CRLF 文件：CR 只允许出现在 LF 之前
    crlf = b.count(b"\r\n")
    bare_cr = b.count(b"\r") - crlf
    bad = []
    if bare_cr > 0:
        bad.append(("CR", bare_cr))
    counts = {}
    for byte in b:
        if byte < 0x20 and byte not in ALLOWED and byte != 0x0D:
            counts[byte] = counts.get(byte, 0) + 1
    for k, v in sorted(counts.items()):
        bad.append(("0x%02X" % k, v))
    if not bad:
        return None
    # 给出第一处的行号与上下文，便于直接定位
    first = None
    for i, byte in enumerate(b):
        if (byte < 0x20 and byte not in ALLOWED and byte != 0x0D) or \
           (byte == 0x0D and b[i + 1:i + 2] != b"\n"):
            first = i
            break
    line = b[:first].count(b"\n") + 1 if first is not None else 0
    ctx = repr(b[max(0, first - 40):first + 20]) if first is not None else ""
    return bad, line, ctx


def main():
    roots = sys.argv[1:] or ["."]
    hits = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(EXTS):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    r = scan(p)
                except OSError:
                    continue
                if r:
                    bad, line, ctx = r
                    hits += 1
                    print("🔴 %s:%d  %s" % (p, line,
                                            " ".join("%s x%d" % kv for kv in bad)))
                    print("     %s" % ctx)
    print("\n%s" % ("✅ 未发现裸控制字节" if hits == 0
                    else "🔴 %d 个文件含裸控制字节 —— 多半是转义被展开了" % hits))
    return 1 if hits else 0


sys.exit(main())
