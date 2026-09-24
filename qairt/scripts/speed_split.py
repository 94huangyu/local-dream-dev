"""分离「context 加载」与「纯计算」—— EXP_PLAN_SPEED §二。

方法：同一段分别跑 N=1 与 N=5 次推理
  斜率 (T5-T1)/4 = 纯计算/次
  截距 T1 - 斜率  = 加载 + 初始化

判据（事前锁定，EXP_PLAN_SPEED §二）：
  加载占比 >= 40%  => 🟢 值得把 loadGraph 提出步循环
  15% ~ 40%        => 🟡 与内存风险一起权衡
  < 15%            => ❌ 加载不是瓶颈，放弃这条

🔴 措辞纪律：这是 **qnn-net-run 单段离线执行**的计时，
   **不得**直接等同于 app 内每步耗时（app 复用进程、不走 qnn-net-run）。

用法: python speed_split.py [段名...]   默认 part1a
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
if not os.path.exists(ADB):
    ADB = "adb"

# 段 -> (设备上的 .bin 名, {张量名: 设备上的 .raw 路径})
# 中间输入取自上一次在环运行留下的 Result_0，与真实喂料逐字节相同。
IN = T + "/loop/in"
O1A = T + "/loop/o1a/Result_0"
O1B = T + "/loop/o1b/Result_0"
O2A = T + "/loop/o2a/Result_0"
SEGS = {
    "part1a": ("part1a_fp16.bin", {k: IN + "/" + k + ".raw" for k in
                                   ["latents", "timestep", "caption", "cap_pad_mask"]}),
    "part1b": ("part1b_fp16.bin", {k: O1A + "/" + k + ".raw" for k in
                                   ["adaln_input", "add_131", "add_138",
                                    "select_45", "select_46", "tanh_19"]}),
    "part2a": ("part2a_fixed_fp16.bin", dict(
        [("unified", O1B + "/unified.raw")] +
        [(k, O1A + "/" + k + ".raw") for k in
         ["unified_mask", "unified_freqs", "adaln_input"]])),
    "part2b": ("part2b_fixed_fp16.bin", dict(
        [(k, O2A + "/" + k + ".raw") for k in
         ["add_92", "select", "select_1", "split_7_split_2",
          "split_7_split_3", "val_105"]] +
        [("adaln_input", O1A + "/adaln_input.raw")])),
}


def sh(cmd):
    """透传返回码与 stderr（约束 11·再补：包装子进程必须这样）"""
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def run_n(binname, names, n, tag):
    line = " ".join("%s:=%s" % (k, v) for k, v in names.items())
    od = "spd_%s" % tag
    rc, out = sh("mkdir -p %s/%s && printf '%%s\\n' '%s' > %s/%s/list.txt" % (T, od, line, T, od))
    if rc:
        sys.exit("FAIL 准备 list.txt: %s" % out[-300:])
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context %s --backend libQnnHtp.so "
           "--input_list %s/list.txt --output_dir %s/%s --log_level error "
           "--num_inferences %d 2>&1 | tail -3" % (T, T, T, binname, od, T, od, n))
    t0 = time.time()
    rc, out = sh(cmd)
    el = time.time() - t0
    if "Finished Executing Graphs" not in out:
        sys.exit("FAIL qnn-net-run [n=%d]:\n%s" % (n, out[-500:]))
    return el


def main():
    import lab_dev
    if not lab_dev.online():
        sys.exit("FAIL 设备不可达 —— 请插线")
    segs = sys.argv[1:] or ["part1a"]
    print("%-10s %10s %10s %12s %12s %10s" %
          ("段", "T(N=1)", "T(N=5)", "计算/次", "加载+初始化", "加载占比"))
    print("-" * 70)
    for s in segs:
        if s not in SEGS:
            print("  跳过未登记的段: %s" % s)
            continue
        binname, names = SEGS[s]
        t1 = run_n(binname, names, 1, s + "_1")
        t5 = run_n(binname, names, 5, s + "_5")
        comp = (t5 - t1) / 4.0
        load = t1 - comp
        frac = 100.0 * load / t1 if t1 > 0 else 0.0
        print("%-10s %9.1fs %9.1fs %11.1fs %11.1fs %9.1f%%"
              % (s, t1, t5, comp, load, frac))
        print()
        v = ("🟢 >=40%% ⇒ 值得把 loadGraph 提出步循环" if frac >= 40 else
             "🟡 15~40%% ⇒ 与内存风险一起权衡" if frac >= 15 else
             "❌ <15%% ⇒ 加载不是瓶颈，放弃这条")
        print("  判定（EXP_PLAN_SPEED §二 事前锁定）: %s" % v)
    print()
    print("⚠️ 这是 qnn-net-run 单段离线计时，**不等于** app 内每步耗时（app 复用进程）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
