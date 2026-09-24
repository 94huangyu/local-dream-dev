"""#114 步 1 的其余三段：part2b 输入 -> part1a -> part1b -> part2b，一条链跑完。

为什么是一条链而不是四条后台命令（约束 9·补 规则 1）：
  链在**前台**依次执行，调用方把整条链挂后台 => **通知到达 == 整条链真的结束**。
  分开挂四次的话，第一条的完成通知会被误当成整体完成。

每一步都检查返回码；任何一步失败立刻停下并报出是哪一步（约束 11·再补：报错不许说谎）。

用法: python maxfp16_chain.py
"""
import os
import sys
import time
import subprocess

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
PY = sys.executable

STEPS = [
    ("part2b 输入", [PY, os.path.join(HERE, "maxfp16_p2b_inputs.py")],
     os.path.join(W, "mf16_p2b_inputs.log"), 3600),
    # 🔴 part1b 必须排在 part1a 前面：#112 的三个反例张量（mul_406/431/456）在这一段，
    #    G0-contra 决策点就在它跑完的那一刻（EXP_PLAN §3.3）。早 1 小时拿到决策。
    ("part1b", [PY, os.path.join(HERE, "maxfp16_stats.py"), "part1b", "--budget-gb", "4"],
     os.path.join(W, "mf16_part1b.log"), 14400),
    ("part1a", [PY, os.path.join(HERE, "maxfp16_stats.py"), "part1a", "--budget-gb", "4"],
     os.path.join(W, "mf16_part1a.log"), 14400),
    ("part2b", [PY, os.path.join(HERE, "maxfp16_stats.py"), "part2b", "--budget-gb", "4"],
     os.path.join(W, "mf16_part2b.log"), 14400),
]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t00 = time.time()
    for name, cmd, log, tmo in STEPS:
        print("[%s] 开始 %s" % (time.strftime("%H:%M:%S"), name), flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fo:
            r = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT, env=env, timeout=tmo)
        mins = (time.time() - t0) / 60
        # 失败时把日志尾巴打出来 —— 不许只报「失败」而不给证据
        if r.returncode != 0:
            try:
                tail = open(log, encoding="utf-8", errors="replace").read()[-2000:]
            except Exception:
                tail = "(日志读不出来)"
            print("[FAIL] %s rc=%d  %.1f min\n%s" % (name, r.returncode, mins, tail), flush=True)
            return r.returncode
        print("[OK] %s  %.1f min  （累计 %.1f min）"
              % (name, mins, (time.time() - t00) / 60), flush=True)
    print("[DONE] 四段步 1 全部完成，累计 %.1f min" % ((time.time() - t00) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
