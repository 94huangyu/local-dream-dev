# -*- coding: utf-8 -*-
"""#126 重复历史行 —— 判据自动判读。

判据来自 scripts/EXP_PLAN_DUP_HISTORY.md §三，**已于用户点击生成前定稿**，本脚本
只做机械判读，不得在此处修改判据。

用法：
    python scripts/dup126_analyze.py --baseline-id 43 --instances 4 \
        --log logs/dup126_20260826/live3.txt
"""
import argparse
import os
import re
import sqlite3
import subprocess
import sys

ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
PKG = "io.github.xororz.localdream.zimage"
DB_FILES = ("local_dream.db", "local_dream.db-wal", "local_dream.db-shm")


def sh(cmd, **kw):
    """透传返回码与 stderr（指南 §15.3：包装子进程必须不吞错）。"""
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(
            "cmd failed rc=%d\nCMD: %s\nSTDERR: %s"
            % (r.returncode, cmd, (r.stderr or b"")[-400:].decode("utf-8", "replace"))
        )
    return r.stdout


def pull_db(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name in DB_FILES:
        data = sh([ADB, "exec-out", "run-as %s cat databases/%s" % (PKG, name)])
        with open(os.path.join(outdir, name), "wb") as f:
            f.write(data)
    return os.path.join(outdir, DB_FILES[0])


def list_pngs():
    out = sh([ADB, "shell", "run-as %s ls -l files/history/ZIMAGE" % PKG])
    txt = out.decode("utf-8", "replace")
    return re.findall(r"(\d{13})\.png", txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-id", type=int, required=True)
    ap.add_argument("--instances", type=int, required=True,
                    help="点击生成前实测的活着的 MainActivity 实例数")
    ap.add_argument("--log", required=True)
    ap.add_argument("--outdir", default="logs/dup126_20260826/after")
    a = ap.parse_args()

    db = pull_db(a.outdir)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cols = [r[1] for r in cur.execute("pragma table_info(generation_history)")]
    rows = [dict(zip(cols, r)) for r in cur.execute(
        "select * from generation_history where id>? order by id", (a.baseline_id,))]

    # 🔴 2026-08-26 修：原来用 "...start generation\n" 计数，两处都错——
    # ① logcat 经 Windows 重定向是 CRLF，"\n" 匹配不上行尾；
    # ② 只喂了一个日志文件，而点击时刻可能落在另一个文件里。
    # 现在改为按行正则、并接受多个日志文件。
    log = ""
    for path in a.log.split(","):
        if os.path.exists(path):
            log += open(path, encoding="utf-8", errors="replace").read()
    # ③ 多份抓取在时间上重叠 ⇒ 同一条日志会出现多次，必须先按整行去重，
    #    否则计数会翻倍（本脚本第一版就因此把 P3 报成 2）。
    seen, lines = set(), []
    for ln in log.replace("\r\n", "\n").split("\n"):
        ln = ln.rstrip()
        if ln and ln not in seen:
            seen.add(ln)
            lines.append(ln)
    n_start = sum(1 for ln in lines if ln.endswith("ModelRunScreen: start generation"))
    n_bitmap = sum(1 for ln in lines if ln.endswith("ModelRunScreen: update bitmap"))
    n_params = sum(1 for ln in lines if "ModelRunScreen: params update" in ln)

    N = len(rows)
    print("=" * 62)
    print("点击前活着的 MainActivity 实例数 N_alive = %d" % a.instances)
    print("新增历史行数                    = %d" % N)
    print("logcat: start generation        = %d" % n_start)
    print("logcat: update bitmap           = %d" % n_bitmap)
    print("logcat: params update           = %d" % n_params)
    print("=" * 62)
    for d in rows:
        print("id=%-3d ts=%-14d steps=%-3s cfg=%-4s seed=%s img=%s"
              % (d["id"], d["timestamp"], d["steps"], d["cfg"], d["seed"],
                 d["imagePath"].split("/")[-1]))
        print("       prompt=%r" % d["prompt"][:50])

    stamps = set(str(d["timestamp"]) for d in rows)
    pngs = [p for p in list_pngs() if p in stamps]
    print("对应磁盘 PNG 数 = %d %s" % (len(pngs), sorted(pngs)))

    print("-" * 62)
    # P3
    print("P3 start generation == 1 : %s (%d)" % ("PASS" if n_start == 1 else "FAIL", n_start))
    # P2
    print("P2 update bitmap == 行数 : %s (%d vs %d)"
          % ("PASS" if n_bitmap == N else "FAIL", n_bitmap, N))
    # P1 分支
    if N == a.instances:
        print("P1a 新增行数 == 实例数   : HIT  -> 机制全对：所有活着的实例都写")
    elif N == 1:
        print("P1b 只有 1 行            : HIT  -> 停止态不参与，机制部分错，需重查")
    else:
        print("P1c 新增 %d 行(实例 %d)   : HIT  -> 只有部分实例参与，需按 state 细分"
              % (N, a.instances))
    # P5 除 prompt 外全同
    if N > 1:
        keys = [k for k in cols if k not in ("id", "prompt", "timestamp", "imagePath")]
        same = all(all(r[k] == rows[0][k] for k in keys) for r in rows)
        span = max(d["timestamp"] for d in rows) - min(d["timestamp"] for d in rows)
        print("P5 除 prompt 外字段全同  : %s | 时间戳跨度 %d ms"
              % ("PASS" if same else "FAIL", span))
        if not same:
            for k in keys:
                vals = set(r[k] for r in rows)
                if len(vals) > 1:
                    print("     差异字段 %s -> %s" % (k, vals))
        # P4
        newp = [r for r in rows if r["prompt"] == "一只戴帽子的企鹅"]
        print("P4 恰好 1 行是新 prompt  : %s (%d)"
              % ("PASS" if len(newp) == 1 else "FAIL", len(newp)))
        # P6 指纹
        virgin = [r for r in rows if r["prompt"] == "" and r["steps"] == 0]
        print("P6 未生成过实例的指纹行  : %d 行 (prompt='' 且 steps=0)" % len(virgin))
    return 0


if __name__ == "__main__":
    sys.exit(main())
