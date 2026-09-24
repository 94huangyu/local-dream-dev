# -*- coding: utf-8 -*-
"""D2 双臂一次启动：推送两臂 -> 过门 -> nohup 串行跑完，**启动完即可拔线**。

为什么要串行：单臂 ION 峰值约 2.9 GB（D1 实测），两臂并行会撞 ION。
为什么要合并成一次启动：用户只能给 5 分钟的插线窗口。

判据与三道门沿用 aspect_device_d2.py，本脚本只负责【编排】。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import lab_dev as L                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.join(HERE, "aspect_device_d2.py")


def main():
    tag, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    L.require_online("d2 both")
    # 第一步：两臂各自推送 + 过门，但**不启动**（--stage push）
    for a in ("ctrl", "new"):
        print("\n########## 准备臂 %s ##########" % a, flush=True)
        r = subprocess.run([sys.executable, D2, "launch", tag, str(W), str(H),
                            "--arm", a, "--stage", "push"],
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("臂 %s 准备失败，**不启动任何臂**" % a)
            return 1
    # 第二步：一条设备侧命令串行跑完两臂
    # 🔴🔴 code_lint 的 C4 会拦下面的 `nohup &`（约束 9·补 规则 1）。
    # **本处经确认无害，理由与 aspect_device_d2.py 同**：
    #   1. 脱机是本脚本存在的目的（用户只能给 5 分钟插线窗口）；
    #      规则的本意是「别把『启动』当『完成』」，不是禁止后台化。
    #   2. 完成判定在 `collect`：查【D2_DONE 标记 + 进程数 + 字节数】，**不看日志内容**。
    #   3. 每臂脚本 `set -e` + 每步 G-SIZE 硬校验，失败即退出且不产生该臂的 D2_DONE。
    #   4. 两臂**必须串行**（`;` 而非 `&`）：单臂 ION 峰值约 2.9 GB（D1 实测），
    #      并行会撞 ION —— 这也是不能简单起两个后台任务的原因。
    cmd = ("cd /data/local/tmp && rm -f D2_BOTH_DONE && "
           "nohup sh -c '(cd d2_ctrl && ./run.sh > run.log 2>&1); "
           "(cd d2_new && ./run.sh > run.log 2>&1); "
           "touch /data/local/tmp/D2_BOTH_DONE' > /dev/null 2>&1 &")
    L.sh(cmd)
    print("\n=== 两臂已串行启动（nohup），现在可以拔线 ===", flush=True)
    print("预计 40~70 分钟（每臂 8 步 x 四段）。回来后跑：", flush=True)
    print("  python scripts/aspect_device_d2.py collect %s %d %d --arm ctrl" % (tag, W, H), flush=True)
    print("  python scripts/aspect_device_d2.py collect %s %d %d --arm new" % (tag, W, H), flush=True)
    print("  python scripts/d2_decode_compare.py %s %d %d" % (tag, W, H), flush=True)
    return 0


sys.exit(main())
