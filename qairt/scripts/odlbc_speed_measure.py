# -*- coding: utf-8 -*-
"""EXP_PLAN_ODLBC_SPEED 阶段 1 测速：CTRL / O3 / DLBC 三臂的 part1b 纯计算时间。

方法：同一臂跑 N=1 与 N=9，斜率 (T9-T1)/8 = 纯计算/次，截距 = 加载+初始化。

🔴 三条纪律，都是踩过坑换来的：
 1. **输入字节数走契约**（约束 3）。`.bin` 路径的宿主契约是**声明的量化尺寸**，
    不是 `.dlc` 路径的「一律 float32」—— 见 EXECUTION_MODEL 规则 2·补。
    跑完**校验输出字节数**，对了才说明输入被正确接受。
 2. **不套 `| tail`**（§7.4）—— 管道会把返回码换成 tail 的 rc=0，失败伪装成成功。
 3. **正反两趟**（CTRL→O3→DLBC，再 DLBC→O3→CTRL）。#144 刚实测热会让耗时单调漂移，
    单向顺序会把热漂移算进臂间差异。取两趟均值可抵消单调漂移。
"""
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

DEV = "3B1F65EA9BBUMSHZ"
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
T = "/data/local/tmp/htpcmp"
W = "/data/local/tmp/odlbc"
SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "odlbc_speed")
CONTRACT = os.path.join("D:", os.sep, "LocalDreamZImage", "logs", "tier2_20260829",
                        "final_qnn_contract.L80.json")
HOSTIN = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "odlbc_speed", "in")
ARMS = ["CTRL", "O3", "DLBC"]
NS = [1, 9]


def sh(cmd, timeout=1800):
    """透传返回码与 stderr（约束 11·再补：包装子进程必须这样）"""
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout,
                       env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def push(host, dev):
    r = subprocess.run([ADB, "-s", DEV, "push", host, dev], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600,
                       env=dict(os.environ, MSYS_NO_PATHCONV="1"))
    if r.returncode != 0:
        sys.exit("FAIL push %s -> %s\n%s" % (host, dev, (r.stdout or "") + (r.stderr or "")))


def npu_c():
    rc, o = sh("M=0; for z in /sys/class/thermal/thermal_zone*; do "
               "case \"$(cat $z/type)\" in nsph*) V=$(cat $z/temp); "
               "[ \"$V\" -gt \"$M\" ] && M=$V;; esac; done; echo $M")
    try:
        return int(o.strip()) / 1000.0
    except Exception:
        return -1.0


def contract_part1b():
    d = json.load(open(CONTRACT, encoding="utf-8"))
    for e in d["models"]:
        if e["internal_graph_name"] == "transformer_part1b":
            return e
    sys.exit("FAIL 契约里没有 transformer_part1b")


def make_inputs(spec):
    """按契约 exact_bytes 生成确定性输入（固定种子）。三臂共用同一份 ⇒ 臂间可比。"""
    import numpy as np
    os.makedirs(HOSTIN, exist_ok=True)
    rng = np.random.default_rng(20260829)
    names = []
    for t in spec["inputs"]:
        p = os.path.join(HOSTIN, t["name"] + ".raw")
        nb = t["exact_bytes"]
        if not os.path.isfile(p) or os.path.getsize(p) != nb:
            rng.integers(0, 65536, size=nb // 2, dtype=np.uint16).tofile(p)
        assert os.path.getsize(p) == nb, "生成的 %s 字节数不符契约" % t["name"]
        names.append(t["name"])
    print("  输入已备（契约字节全部吻合）: %s" % ", ".join(names))
    return names


def run_n(arm, n, names, out_bytes_expect):
    od = "%s/o_%s_%d" % (W, arm, n)
    line = " ".join("%s:=%s/in/%s.raw" % (k, W, k) for k in names)
    rc, o = sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' '%s' > %s/list.txt" %
               (od, od, line, od))
    if rc:
        sys.exit("FAIL 准备 list.txt: %s" % o[-400:])
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context %s/%s.bin --backend libQnnHtp.so "
           "--input_list %s/list.txt --output_dir %s --log_level error "
           "--num_inferences %d" % (T, T, T, W, arm, od, od, n))
    t0 = time.time()
    rc, o = sh(cmd)
    el = time.time() - t0
    if "Finished Executing Graphs" not in o:
        sys.exit("FAIL qnn-net-run [%s n=%d] rc=%d:\n%s" % (arm, n, rc, o[-700:]))
    # 🔴 约束 3 式自证：输出字节数必须等于契约值，否则输入被静默缩批/错解
    rc2, o2 = sh("stat -c %%s %s/Result_0/unified.raw 2>/dev/null" % od)
    got = o2.strip()
    if got != str(out_bytes_expect):
        sys.exit("🔴 FAIL 输出字节数 %s ≠ 契约 %d ⇒ 输入未被正确接受，本轮数据作废"
                 % (got or "读不到", out_bytes_expect))
    return el


def main():
    spec = contract_part1b()
    out_bytes = spec["outputs"][0]["exact_bytes"]
    print("契约 part1b：输入 %d 个 / 输出 unified %d B" % (len(spec["inputs"]), out_bytes))

    rc, o = sh("echo ok")
    if rc or "ok" not in o:
        sys.exit("FAIL 设备不可达 —— 请插线")

    names = make_inputs(spec)
    if "--no-push" in sys.argv:
        # 复用上一次已推送的产物。🔴 但必须逐个核对字节数，不得凭文件存在就信
        # （#83「grep 当就绪判据会误报」同类：存在 ≠ 正确）。
        print("\n=== --no-push：核对设备上已有产物的字节数 ===", flush=True)
        for arm in ARMS:
            d = os.path.join(SRC, arm)
            b = [f for f in os.listdir(d) if f.endswith(".bin")][0]
            want = os.path.getsize(os.path.join(d, b))
            rc, o = sh("stat -c %%s %s/%s.bin 2>/dev/null" % (W, arm))
            if o.strip() != str(want):
                sys.exit("🔴 %s.bin 设备 %s ≠ 宿主 %d ⇒ 不得复用，去掉 --no-push"
                         % (arm, o.strip() or "缺失", want))
            print("  %-5s %d B  ✅" % (arm, want))
        for k in names:
            want = os.path.getsize(os.path.join(HOSTIN, k + ".raw"))
            rc, o = sh("stat -c %%s %s/in/%s.raw 2>/dev/null" % (W, k))
            if o.strip() != str(want):
                sys.exit("🔴 输入 %s 设备 %s ≠ 宿主 %d" % (k, o.strip() or "缺失", want))
        print("  输入 6 个字节数全部吻合 ✅")
    else:
        print("\n=== 推送（三臂 .bin 约 4.4 GB + 输入 33 MB）===", flush=True)
        sh("rm -rf %s && mkdir -p %s/in" % (W, W))
        for k in names:
            push(os.path.join(HOSTIN, k + ".raw"), "%s/in/%s.raw" % (W, k))
        for arm in ARMS:
            d = os.path.join(SRC, arm)
            b = [f for f in os.listdir(d) if f.endswith(".bin")][0]
            print("  push %-5s %s" % (arm, b), flush=True)
            push(os.path.join(d, b), "%s/%s.bin" % (W, arm))

    order = ARMS + ARMS[::-1]          # 正反两趟，抵消热漂移
    res = {a: [] for a in ARMS}
    print("\n=== 测量（正反两趟）===", flush=True)
    print("%-6s %-5s %8s %8s" % ("趟", "臂", "NPU前", "耗时"))
    for i, arm in enumerate(order):
        row = {}
        for n in NS:
            t0c = npu_c()
            el = run_n(arm, n, names, out_bytes)
            row[n] = el
            print("%-6d %-5s %7.1fC %7.2fs  (N=%d)" % (i // 3 + 1, arm, t0c, el, n),
                  flush=True)
        res[arm].append(row)

    print("\n=== 结果（两趟均值）===")
    print("%-6s %9s %9s %11s %11s" % ("臂", "T(N=1)", "T(N=9)", "计算/次", "加载+初始化"))
    S = {}
    for a in ARMS:
        t1 = sum(r[1] for r in res[a]) / len(res[a])
        t9 = sum(r[9] for r in res[a]) / len(res[a])
        comp = (t9 - t1) / 8.0
        load = t1 - comp
        S[a] = comp
        d1 = abs(res[a][0][1] - res[a][1][1])
        d9 = abs(res[a][0][9] - res[a][1][9])
        print("%-6s %8.2fs %8.2fs %10.3fs %10.2fs   (两趟差 N=1 %.2fs / N=9 %.2fs)"
              % (a, t1, t9, comp, load, d1, d9))

    print("\n=== G-screen 判据（事前锁定，EXP_PLAN_ODLBC_SPEED §四）===")
    base = S["CTRL"]
    verdict = []
    for a in ("O3", "DLBC"):
        r = S[a] / base if base else 0
        v = ("🟢 有收益 ⇒ 进阶段 2" if r <= 0.95 else
             "⚠️ 有害" if r >= 1.05 else "❌ 无差异")
        verdict.append(r <= 0.95)
        print("  %-5s 计算/次 = %.4f × CTRL   %s" % (a, r, v))
    if not any(verdict):
        print("\n⇒ ❌ **B 关闭**：两臂都落在 0.95~1.05，不做阶段 2（省约 3 小时）")
    print("\n⚠️ 措辞：这是 qnn-net-run **单段离线**计时，不等于 app 内每步耗时（#132）。"
          "只用于臂间相对比较。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
