"""part2a dtype 构建完成后，自动继续建另外三段（纯宿主，不需要设备）。"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from wait_for import wait_build

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
PY = sys.executable
ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")

print("[chain] 等 part2a dtype context（进程+产物双判）", flush=True)
st = wait_build(os.path.join(W, "ctx_part2a_dt"), marker="seg_dtype_build",
                logfile=os.path.join(W, "p2a_dt3.log"), timeout=5 * 3600)
print("[chain] part2a =>", st, flush=True)
if st != "done":
    sys.exit("part2a 构建未成功（%s）—— 见 p2a_dt3.log" % st)

for seg in ("part2b", "part1b", "part1a"):
    d = os.path.join(W, "ctx_%s_dt" % seg)
    if os.path.isdir(d) and [x for x in os.listdir(d) if x.endswith(".bin")]:
        print("[chain] 跳过已建好的", seg, flush=True); continue
    print("[chain] 建 %s  %s" % (seg, time.strftime("%H:%M:%S")), flush=True)
    r = subprocess.run([PY, "scripts/seg_dtype_build.py", seg], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=28800)
    open(os.path.join(W, "chain_%s.log" % seg), "w", encoding="utf-8").write(
        (r.stdout or "") + (r.stderr or ""))
    print("[chain] %s rc=%d" % (seg, r.returncode), flush=True)
    if r.returncode:
        for l in ((r.stdout or "") + (r.stderr or "")).splitlines()[-12:]:
            print("    " + l.strip()[:160], flush=True)
        sys.exit("%s 构建失败" % seg)
print("[chain] 四段全部就绪 —— 等设备回来做端到端", flush=True)
