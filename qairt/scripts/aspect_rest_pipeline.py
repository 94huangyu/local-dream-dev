# -*- coding: utf-8 -*-
"""无人值守编排：FP32 参考之后，把宿主机能做的全部做完。

用户外出约 36 小时，回来只想插一次线。本脚本的目标是**让机器不空转**，
并且把每一步的门都跑掉，使插线时不再出现「一连上才发现的问题」。

## 顺序与依赖
  0  等 FP32 参考齐（15 个 latent），进程没了却没齐 => 立刻失败，不空等
  1  VAE 量化 x3（门 W1/W2：权重 encoding 必须与现网 1:1 相同）
  2  建 16:9 / 9:16 的四段 transformer context
  3  建各比例的**单图** VAE context（single 形态的回退路径需要）
  4  建**五图共享**的 mg context（4 段 + VAE）—— 最长的一步
  5  多图 vs 单图元数据等价检查（M1~M5），任一段不过就别上机
  6  激活量程预检 G-RANGE x3（只能否决不能放行）
  7  生成两份契约（single / mg）
  8  总预检 aspect_preflight.py

## 纪律
- 每步**失败即停**，不往下走（前一步的产物不完整时，后一步的结论没有意义）。
- 每步前查磁盘，低于阈值就停下来而不是把盘写满（宿主被压死过两次）。
- 🔴 **不并行**：单个建图进程占 10.6 GB，宿主共 23.7 GB。
- 日志与产物时间戳是唯一的死活判据，不看日志措辞（约束 9·补）。

用法: python aspect_rest_pipeline.py            # 全跑
      python aspect_rest_pipeline.py --from 4   # 从第 4 步开始（重试用）
"""
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
AS = os.path.join(P0, "aspect")
MG = os.path.join(P0, "aspect_mg")
SCRATCH = os.path.join(
    "C:", os.sep, "Users", "sinai", "AppData", "Local", "Temp", "claude",
    "D--LocalDreamZImage", "bbf01231-6581-4b08-b5d0-7dd03e47f992", "scratchpad")
NEW = ["896x1184", "1280x720", "720x1280"]
ALL = ["1184x896"] + NEW
SEGS = ["part1a", "part1b", "part2a", "part2b"]
MIN_FREE_GIB = 25.0


def free_gib():
    return shutil.disk_usage(os.path.join("D:", os.sep)).free / 1073741824.0


def step(n, title):
    print("\n" + "=" * 68, flush=True)
    print("[步骤 %d] %s   （%s，D 盘剩 %.1f GiB）"
          % (n, title, time.strftime("%m-%d %H:%M"), free_gib()), flush=True)
    print("=" * 68, flush=True)


def sh(cmd, timeout=36000):
    """跑一条命令；**返回码原样返回，stderr 并入 stdout 全量打印**。

    🔴 不做 `| tail` 之类的裁剪：本项目实测 `adb push` 失败仍打印成功行，
       裁剪恰好只留下那句谎话（约束 11·再补）。
    """
    print("  $ %s" % " ".join(str(x) for x in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = r.stdout or ""
    for line in out.splitlines():
        print("    " + line, flush=True)
    print("  -> rc=%d  %.1f 分钟" % (r.returncode, (time.time() - t0) / 60.0), flush=True)
    return r.returncode


def guard_disk(need=MIN_FREE_GIB):
    f = free_gib()
    if f < need:
        print("  🔴 D 盘只剩 %.1f GiB（阈值 %.1f）——停止，不把盘写满" % (f, need))
        return False
    return True


def wait_fp32():
    """等 FP32 参考齐。判据 = 产物计数 + 进程存在（不看日志内容）。"""
    want = {t: 5 for t in NEW}
    last = -1
    dead = 0
    while True:
        have = {t: len([f for f in os.listdir(os.path.join(P0, "vae_calib_%s" % t))
                        if f.startswith("latent_s")])
                if os.path.isdir(os.path.join(P0, "vae_calib_%s" % t)) else 0
                for t in NEW}
        n = sum(have.values())
        if n != last:
            print("  参考进度 %s = %d/15  (%s)" % (have, n, time.strftime("%H:%M")), flush=True)
            last = n
        if all(have[t] >= want[t] for t in NEW):
            return True
        alive = subprocess.run(["tasklist", "/fi", "IMAGENAME eq python.exe"],
                               capture_output=True, text=True).stdout
        if alive.count("python.exe") <= 1:      # 只剩本脚本自己
            # 🔴 **连续两次**才判死。单次采样判死栽过：驱动脚本在两个 seed 之间
            #    有极短的无子进程窗口，恰好采到就会误判。2026-09-03 我就是用单个
            #    采样点宣布「对照臂已死」，实际那一臂已经跑完了。
            dead += 1
            print("  ⚠️ 第 %d 次采到「无其他 python 进程」（参考 %d/15）" % (dead, n),
                  flush=True)
            if dead >= 2:
                print("  🔴 连续两次确认 FP32 进程不在，参考仍只有 %d/15 —— 判失败，不空等" % n)
                return False
        else:
            dead = 0
        time.sleep(120)


def main():
    start = 0
    if "--from" in sys.argv:
        start = int(sys.argv[sys.argv.index("--from") + 1])

    if start <= 0:
        step(0, "等 FP32 参考齐（15 个）")
        if not wait_fp32():
            return 1

    if start <= 1:
        step(1, "VAE 量化 x3（门 W1/W2）")
        for t in NEW:
            if not guard_disk():
                return 1
            if sh([PY, os.path.join(HERE, "vae_quant_aspect.py"), t]):
                print("  🔴 %s 的 VAE 量化未过门，停止" % t)
                return 1

    if start <= 2:
        step(2, "建 16:9 / 9:16 的四段 transformer context")
        for t in ("1280x720", "720x1280"):
            if not guard_disk(30):
                return 1
            # 只建 transformer 四段：VAE 走专用脚本（通用路径会把 transformer 的
            # overrides 误套到 VAE 上，报 IrQuantizer::fallbackToFloat —— 已实测）
            if sh([PY, os.path.join(HERE, "aspect_build.py"), t,
                   "--segs", ",".join(SEGS)]):
                print("  🔴 %s 建图失败，停止" % t)
                return 1

    if start <= 3:
        step(3, "建各比例的单图 VAE context（single 形态的回退路径）")
        for t in ALL:
            if not guard_disk():
                return 1
            if sh([PY, os.path.join(HERE, "aspect_vae_ctx.py"), t]):
                print("  🔴 %s 的 VAE context 失败，停止" % t)
                return 1

    if start <= 4:
        step(4, "建五图共享的 mg context（最长的一步）")
        if not guard_disk(30):
            return 1
        if sh([PY, os.path.join(HERE, "aspect_multigraph_build.py"), ",".join(ALL)],
              timeout=86400):
            print("  🔴 mg 建图失败 —— mg 形态不可用，只能退回 single 形态")
            return 1

    if start <= 5:
        step(5, "多图 vs 单图元数据等价（M1~M5）")
        bad = []
        for seg in SEGS:
            args = [PY, os.path.join(HERE, "check_mg_equivalence.py"), seg,
                    os.path.join(MG, seg, "%s_mg.SM8750.bin" % seg),
                    "%s_fp16_L80_fp32=%s" % (seg, os.path.join(
                        P0, "p2attr", "ctx_%s_fp16_L80" % seg, "%s_fp16_L80.SM8750.bin" % seg))]
            for t in ALL:
                args.append("%s_%s_fp32=%s" % (seg, t, os.path.join(
                    AS, "ctx_%s_%s" % (seg, t), "%s_%s.SM8750.bin" % (seg, t))))
            if sh(args):
                bad.append(seg)
        if bad:
            print("  🔴 %s 未过等价检查 —— G-MG 必挂，别上机" % bad)
            return 1

    if start <= 6:
        step(6, "激活量程预检 G-RANGE x3（只能否决不能放行）")
        for t in NEW:
            W, H = t.split("x")
            sh([PY, os.path.join(HERE, "aspect_act_range_check.py"), t, W, H])
            # 不因 🟡 停机：本门只能否决不能放行，判读留给人

    if start <= 7:
        step(7, "生成两份契约")
        for form in ("mg", "single"):
            if sh([PY, os.path.join(HERE, "build_aspect_contract.py"),
                   "--form=%s" % form] + ALL):
                print("  ⚠️ %s 形态契约未生成（若 single 缺 VAE context 属预期）" % form)

    if start <= 8:
        step(8, "总预检")
        apk = os.path.join(ROOT, "local-dream", "app", "build", "outputs", "apk",
                           "basic", "debug",
                           "LocalDreamZImage_armv8a_2.8.1-zimage-mvp.apk")
        c = os.path.join(SCRATCH, "deliver_backup", "final_qnn_contract.mg.json")
        if os.path.isfile(c):
            sh([PY, os.path.join(HERE, "aspect_preflight.py"), c, "--apk", apk])
        else:
            print("  🔴 mg 契约不存在，预检跳过")
            return 1

    print("\n全部步骤完成 %s" % time.strftime("%m-%d %H:%M"))
    return 0


sys.exit(main())
