# -*- coding: utf-8 -*-
"""SmoothQuant / 重标定 encoding 的**端到端** QDQ 模拟（EXP_PLAN_SMOOTHQUANT §三）。

三臂，每臂「四段建图 -> sim_runner drive（8 步 + VAE）」：
  H   部署配置（出厂 encoding）                 —— 基线
  S0  重标定 encoding，s≡1（不平滑）            —— 分离出「重标定」的效应
  S5  重标定 encoding + SmoothQuant alpha=0.5   —— 再分离出「平滑」的净效应

🔴 S(alpha=0) 与 S(alpha=0.5) 共用同一份重标定 encoding
   => 「S0 -> S5」是 SmoothQuant 的**净效应**，与重标定解耦。

🔴 措辞纪律（沿用 sim_runner）：结果一律写「**在 QDQ 模拟上**」，不得写成设备实测。
⚠️ #121 已证 L1 无法换算成 L3；#123 出现过模拟/设备比值 1.826 的离群
   => 模拟只用于排序与量级，最终仍必须实测。

用法: python sq_e2e.py [--arms H,S0,S5]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

SEGS = ["part1a", "part1b", "part2a", "part2b"]
ARMS = {"H": ("H", None), "S0": ("S", "0"), "S5": ("S", "0.5")}


def sh(cmd, env=None, tag=""):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        out = ((r.stdout or "") + (r.stderr or ""))[-1200:]
        sys.exit("FAIL %s (rc=%d):\n%s" % (tag, r.returncode, out))
    return (r.stdout or "") + (r.stderr or "")


def main():
    arms = ["H", "S0", "S5"]
    if "--arms" in sys.argv:
        arms = sys.argv[sys.argv.index("--arms") + 1].split(",")
    t00 = time.time()
    for arm in arms:
        mode, alpha = ARMS[arm]
        env = dict(os.environ)
        if alpha is not None:
            env["SQ_ALPHA"] = alpha
        print("\n" + "=" * 60, flush=True)
        print("=== 臂 %s（mode=%s, alpha=%s）建四段图 ===" % (arm, mode, alpha), flush=True)
        for s in SEGS:
            o = sh([sys.executable, os.path.join(HERE, "sim_qdq.py"), s, mode],
                   env=env, tag="build %s/%s" % (arm, s))
            last = [x for x in o.strip().split("\n") if x.strip()][-1]
            print("   %s" % last[:150], flush=True)
        print("=== 臂 %s 跑 8 步 + VAE（约 70 分钟）===" % arm, flush=True)
        t0 = time.time()
        o = sh([sys.executable, os.path.join(HERE, "sim_runner.py"), "drive", mode,
                "sq_%s" % arm], env=env, tag="drive %s" % arm)
        print(o[-2500:], flush=True)
        print("=== 臂 %s 完成，用时 %.0f 分钟 ===" % (arm, (time.time() - t0) / 60.0), flush=True)
    print("\n全部完成，总用时 %.1f 小时" % ((time.time() - t00) / 3600.0))


if __name__ == "__main__":
    main()
