# -*- coding: utf-8 -*-
"""把文件里的裸控制字节换回两字符转义写法。

🔴 这个脚本必须用 Write 工具落盘，**不能用 heredoc 写**：
   本项目 2026-09-06 一天内触发 9 次转义展开，其中第 9 次就发生在
   「用 heredoc 写一段修复裸控制字节的代码」时 —— 源码里的 `b"\\0"`
   在传递链上被展开成真 NUL，于是修复脚本自己又写进去一个。
   规律：**含反斜杠的字符串一律用 Edit/Write 工具，不走 heredoc。**

用法: python fix_ctrl_bytes.py <文件> [<文件>...]
"""
import io
import sys

sys.stdout.reconfigure(encoding="utf-8")

REP = {0: "\\" + "0", 8: "\\" + "b", 13: "\\" + "r"}


def fix(path):
    b = io.open(path, "rb").read()
    out = bytearray()
    n = 0
    i = 0
    while i < len(b):
        c = b[i]
        # CRLF 里的 CR 是合法行尾，不动
        if c in REP and not (c == 13 and i + 1 < len(b) and b[i + 1] == 10):
            out += REP[c].encode("ascii")
            n += 1
        else:
            out.append(c)
        i += 1
    if n:
        io.open(path, "wb").write(bytes(out))
    c2 = io.open(path, "rb").read()
    left = c2.count(bytes([0])) + c2.count(bytes([8])) + (
        c2.count(b"\r") - c2.count(b"\r\n"))
    print("  %-55s 替换 %d 处，剩余 %d" % (path, n, left))
    return left


bad = sum(fix(p) for p in sys.argv[1:])
sys.exit(1 if bad else 0)
