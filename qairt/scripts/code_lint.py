"""代码体检器：扫 scripts/ 找出**已知会致命的写法**。

存在理由：CLAUDE.md 的约束是散文，写在文档里；我写代码时会退回"显然的实现"。
今天（2026-08-24）代价一整夜：`overnight.py` 的等待循环只查产物、不查进程，
构建 02:45 失败、链子空等到 06:37 —— 而约束 9·补 早写明要「进程表 + 产物」两样都看。

⇒ 把每条致命写法做成**可机检的规则**，长任务启动前必须先跑本脚本。

用法: python code_lint.py [--strict]     # --strict: 有命中则退出码 1
"""
import io
import os
import re
import sys


# 🔴 本工具是强制门（MAINLINE §0.4）。在 GBK 控制台上打印 emoji 会抛
#    UnicodeEncodeError 而整个崩掉——门崩掉和门放行一样危险，都不报问题。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))

# (规则号, 说明, 命中正则, 豁免正则——同一文件里出现即认为已处理)
RULES = [
    ("C1", "等待循环只查产物、不查进程（约束 9·补）"
           " => 构建失败时会空等到超时。用 wait_for.wait_build()",
     re.compile(r"while[^\n]*\n(?:[^\n]*\n){0,6}?[^\n]*os\.path\.exists", re.M),
     re.compile(r"wait_build|_proc_alive|tasklist|wmic process")),

    ("C2", "用 `adb shell ps | grep` 判断进程存活，却没先确认设备可达（#65）"
           " => 拔线会被误判成「进程结束」。用 lab_dev",
     re.compile(r"shell[^\n]*ps\s+-A[^\n]*grep"),
     re.compile(r"lab_dev|get-state|require_online|online\(\)")),

    ("C3", "`tasklist //FI` 与 MSYS_NO_PATHCONV=1 同时出现"
           " => //FI 不被转换，tasklist 静默给出错误答案",
     re.compile(r"tasklist\s+//FI"),
     re.compile(r"(?!)")),          # 无豁免：一律不该用

    ("C4", "Python 源里出现 `nohup ... &` 且未见任何等待/监听"
           " => 通知对应的是「启动」而非「完成」（约束 9·补 规则 1）",
     re.compile(r"nohup[^\n]*&"),
     re.compile(r"wait_build|wait_done|until |Monitor")),

    ("C5", "subprocess 调用未捕获/未检查返回码"
           " => 报错会被吞掉，掉线伪装成模型失败（#65、约束 11·再补）",
     re.compile(r"subprocess\.run\((?:(?!returncode)[\s\S]){0,400}?\)\s*\n(?![\s\S]{0,200}returncode)"),
     re.compile(r"check=True|returncode")),
]

SKIP = {"code_lint.py"}


def main():
    hits = []
    for fn in sorted(os.listdir(ROOT)):
        if not fn.endswith(".py") or fn in SKIP:
            continue
        p = os.path.join(ROOT, fn)
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for rid, desc, pat, exempt in RULES:
            m = pat.search(src)
            if not m:
                continue
            if exempt.pattern != "(?!)" and exempt.search(src):
                continue
            ln = src[:m.start()].count("\n") + 1
            hits.append((fn, ln, rid, desc, src.split("\n")[ln - 1].strip()[:90]))

    print("代码体检：扫描 %d 个脚本"
          % sum(1 for f in os.listdir(ROOT) if f.endswith(".py") and f not in SKIP))
    if not hits:
        print("✅ 无命中")
        return 0
    print("\n命中 %d 处：\n" % len(hits))
    cur = None
    for fn, ln, rid, desc, snippet in hits:
        if fn != cur:
            print("  %s" % fn)
            cur = fn
        print("    [%s] 第 %d 行  %s" % (rid, ln, desc))
        print("         %s" % snippet)
    print("\n🔴 启动任何长任务 / 无人值守链之前，本表必须为空或每条都已确认无害。")
    return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if "--strict" in sys.argv else 0)
