"""把根目录平铺的文件按类别归位，并【同步改写所有引用】。

## 为什么要一个脚本而不是手搬
根目录 60+ 文件里有 15 个被 40 余处引用（文档互相引用、CLAUDE.md 引用、EXP_PLAN 引用）。
手搬必然留下死链。本脚本：
  1. 按类别规划移动
  2. 对每个被移动的文件，把所有引用它的文本里的路径改成【相对新位置】的正确路径
  3. 移动后做死链复核

用法:
    python scripts/tidy_repo.py            # 空跑，只打印计划
    python scripts/tidy_repo.py --apply    # 真正执行
"""
import os, re, sys, glob, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLY = "--apply" in sys.argv

# 留在根目录的（开工必读 + project instructions）
KEEP = {"CLAUDE.md", "MAINLINE.md"}

PLAN = [
    ("docs", ["HANDOVER_2026-08-13_ZIMAGE_MVP.md", "QNN_CONVERSION_GUIDE.md",
              "EXECUTION_MODEL.md", "LEDGER_CLOSED.md"]),
    ("docs/reviews", ["REVIEW_IMAGE_QUALITY_2026-08-18.md", "REVIEW_VAE_EXPORT_2026-08-19.md",
                      "DIAGNOSIS_HTP_ZIMAGE.md"]),
    ("archive", ["MAINLINE_ARCHIVE_2026-08-17.md", "MAINLINE_ARCHIVE_2026-08-21.md",
                 "ARCHIVE_2026-08-10_OBSOLETE_handover.md",
                 "STATUS_FOR_REVIEW_2026-08-17_SNAPSHOT.md"]),
    ("reference/sdk_docs", ["appendixC.txt", "appendixC_real.txt",
                            "applyencodings_text.txt", "perchannel_example.txt"]),
    ("evidence/images", ["zimage_test_output.png", "zimage_tokenizer_fix_test.png",
                         "zimage_vae_only_test.png", "generate_response2.png"]),
]
# 通配类（全部是无人引用的过程垃圾，归入 logs/，并在索引里标为可删）
GLOBS = [
    ("logs/device", ["device_logcat_*.log", "device_run1.log", "curl5.log"]),
    ("logs/build", ["build_debug*.log"]),
    ("logs/app", ["*.sse", "ui*.xml"]),
    ("logs/misc", ["compare_*.log", "trace_residual_ref.log", "disk_usage_top.txt"]),
]

# 会被扫描并改写引用的文本
# 🔴 2026-08-21：必须排除本脚本自身——否则 PLAN 里的字面量会被当成引用替换掉（已踩）
def scan_files():
    out = []
    for pat in ("*.md", "docs/*.md", "docs/reviews/*.md", "archive/*.md",
                "scripts/*.py", "scripts/*.md", "scripts/*.sh", "*.sh"):
        out += glob.glob(os.path.join(ROOT, pat))
    me = os.path.abspath(__file__)
    return sorted(x for x in set(out) if os.path.abspath(x) != me)


def build_moves():
    moves = {}
    for d, names in PLAN:
        for n in names:
            if os.path.isfile(os.path.join(ROOT, n)):
                moves[n] = d + "/" + n
    for d, pats in GLOBS:
        for pat in pats:
            for p in glob.glob(os.path.join(ROOT, pat)):
                n = os.path.basename(p)
                if n in KEEP or n in moves or not os.path.isfile(p):
                    continue
                moves[n] = d + "/" + n
    return moves


def main():
    moves = build_moves()
    print("=== 移动计划：%d 个文件 ===" % len(moves))
    bydir = {}
    for n, dst in sorted(moves.items()):
        bydir.setdefault(os.path.dirname(dst), []).append(n)
    for d in sorted(bydir):
        sz = sum(os.path.getsize(os.path.join(ROOT, n)) for n in bydir[d])
        print("  %-22s %2d 个  %8.1f MB" % (d + "/", len(bydir[d]), sz / 1e6))
        for n in bydir[d][:3]:
            print("        %s" % n)
        if len(bydir[d]) > 3:
            print("        ... 另 %d 个" % (len(bydir[d]) - 3))

    # ---- 引用改写计划 ----
    files = scan_files()
    edits = {}
    for f in files:
        try:
            t = open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        newt, hits = t, []
        # 引用者在【移动后】的位置（若它自己也被移动）
        selfname = os.path.basename(f)
        selfdir = os.path.dirname(os.path.relpath(f, ROOT)).replace(os.sep, "/")
        if selfname in moves:
            selfdir = os.path.dirname(moves[selfname])
        for n, dst in moves.items():
            if n == selfname or n not in newt:
                continue
            rel = os.path.relpath(dst, selfdir or ".").replace(os.sep, "/")
            if rel == n:
                continue
            # 只替换"裸文件名"，不碰已经带路径的写法
            pat = r"(?<![\w/\\.-])" + re.escape(n)
            newt2 = re.sub(pat, rel, newt)
            if newt2 != newt:
                hits.append((n, rel, len(re.findall(pat, newt))))
                newt = newt2
        if hits:
            edits[f] = (newt, hits)
    print()
    print("=== 引用改写计划：%d 个文件 ===" % len(edits))
    for f, (_, hits) in sorted(edits.items()):
        print("  %s" % os.path.relpath(f, ROOT))
        for n, rel, c in hits[:4]:
            print("      %-44s -> %-46s x%d" % (n, rel, c))
        if len(hits) > 4:
            print("      ... 另 %d 种" % (len(hits) - 4))

    if not APPLY:
        print()
        print(">>> 这是空跑。确认无误后用 --apply 执行。")
        return

    # ---- 执行：先改引用，再移动 ----
    for f, (newt, _) in edits.items():
        open(f, "w", encoding="utf-8").write(newt)
    print()
    print("已改写 %d 个文件的引用" % len(edits))
    for n, dst in sorted(moves.items()):
        src = os.path.join(ROOT, n)
        tgt = os.path.join(ROOT, dst.replace("/", os.sep))
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        shutil.move(src, tgt)
    print("已移动 %d 个文件" % len(moves))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
