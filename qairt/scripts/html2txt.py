#!/usr/bin/env python3
"""把 QAIRT SDK 的 HTML 文档转成纯文本，便于按关键词带上下文检索。

用法:
    python scripts/html2txt.py <html文件> [-o 输出.txt]
    python scripts/html2txt.py <html文件> --grep 关键词 [-C 行数]

为什么需要它：排查闭源后端问题的第一步是读官方文档（约束 5），
而 SDK 文档只有 HTML。本脚本只做去标签 + 空白归一，不做任何语义改写。
"""
import argparse
import html
import re
import sys
from pathlib import Path

DROP_TAGS = ("script", "style", "head", "nav", "footer")


def to_text(raw: str) -> str:
    s = raw
    for tag in DROP_TAGS:
        s = re.sub(rf"<{tag}\b.*?</{tag}>", " ", s, flags=re.S | re.I)
    # 块级标签换成换行，避免整篇挤成一行
    s = re.sub(r"<(br|/p|/div|/li|/tr|/h[1-6]|/pre)\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<li\b[^>]*>", "\n- ", s, flags=re.I)
    s = re.sub(r"<td\b[^>]*>", " | ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return "\n".join(line.strip() for line in s.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html_file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--grep", help="只打印命中该正则的行及上下文")
    ap.add_argument("-C", "--context", type=int, default=6)
    args = ap.parse_args()

    text = to_text(Path(args.html_file).read_text(encoding="utf-8", errors="replace"))

    if args.grep:
        pat = re.compile(args.grep, re.I)
        lines = text.splitlines()
        hits = [i for i, ln in enumerate(lines) if pat.search(ln)]
        if not hits:
            print(f"(无命中: {args.grep})")
            return 1
        shown = set()
        for i in hits:
            lo, hi = max(0, i - args.context), min(len(lines), i + args.context + 1)
            if shown and lo > max(shown) + 1:
                print("...")
            for j in range(lo, hi):
                if j not in shown:
                    print(f"{j + 1}: {lines[j]}")
                    shown.add(j)
        return 0

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"写出 {args.out}（{len(text)} 字符）")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
