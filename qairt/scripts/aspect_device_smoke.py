# -*- coding: utf-8 -*-
"""插线后的**第一个**测试：五图 context 能不能装载（秒级出结果）。

## 为什么它必须排在最前
宿主预测：part1a 的 **1:1 图**在五图 context 下 ION = **3320 MiB**，
而 unsigned PD 红线是 **3506~3535 MiB**（#57 实测），余量只有 5.3%。
其余 15 个图都在 1872~3144，安全。
⇒ 整个 mg 形态的成败，压在这一个图上。
它要么成功、要么以 `0x3ea` 硬失败，**几秒就知道**。
先跑它，再决定后面是走 mg 还是退回 single —— 不要先花二十分钟出图才发现装不上。

## 判据（执行前锁定）
S1 每个待测图**单独** enable 时，`qnn-net-run --retrieve_context` 能建起 context
   （不要求跑完推理，只要求 context 建成 —— PD 容量是在建 context 时判的）
S2 失败时日志里若出现 `0x3ea` / `available PD`，判定为**PD 容量**问题，
   而不是别的错（约束 11·再补：报错指向的对象未必是真凶）
S3 ION 实测：从 `/proc/meminfo` 的 `IonTotalUsed` 取 装载前/后 之差，
   与宿主预测比。**这是给预测器标定用的，不作放行判据**（#144：只有 ION 能与内存压力比）

## 用法
  python aspect_device_smoke.py push      # 只推 part1a 的 mg context（约 3 GiB）
  python aspect_device_smoke.py run       # 逐图装载测试
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
MG = os.path.join(P0, "aspect_mg")
T = "/data/local/tmp/mgsmoke"
SEG = "part1a"
GRAPHS = ["part1a_fp16_L80_fp32", "part1a_1184x896_fp32", "part1a_896x1184_fp32",
          "part1a_1280x720_fp32", "part1a_720x1280_fp32"]
# 宿主预测（MiB），用于与实测 ION 对照
PRED = {"part1a_fp16_L80_fp32": 3320, "part1a_1184x896_fp32": 3144,
        "part1a_896x1184_fp32": 3143, "part1a_1280x720_fp32": 3053,
        "part1a_720x1280_fp32": 3054}
PD_RED = 3506


def adb(cmd, **kw):
    """🔴 返回 (rc, 合并后的输出)。**不裁剪、不吞返回码**（约束 11·再补）。"""
    import lab_dev as L
    return L.sh_rc(cmd, **kw) if hasattr(L, "sh_rc") else _raw(cmd)


def _raw(cmd):
    exe = os.environ.get("ADB", "adb")
    r = subprocess.run([exe, "shell", cmd], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout or ""


def ion():
    rc, out = _raw("grep IonTotalUsed /proc/meminfo")
    m = re.search(r"IonTotalUsed:\s+(\d+)", out)
    return int(m.group(1)) / 1024.0 if m else None      # kB -> MiB


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    exe = os.environ.get("ADB", "adb")
    binp = os.path.join(MG, SEG, "%s_mg.SM8750.bin" % SEG)

    if stage == "push":
        if not os.path.isfile(binp):
            print("🔴 缺 %s" % binp)
            return 1
        subprocess.run([exe, "shell", "mkdir -p %s" % T])
        want = os.path.getsize(binp)
        r = subprocess.run([exe, "push", binp, "%s/%s_mg.bin" % (T, SEG)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           encoding="utf-8", errors="replace")
        # 🔴 push 失败仍会打印成功行 —— 一律以设备侧字节数为准
        rc, out = _raw("stat -c %%s %s/%s_mg.bin 2>/dev/null || echo 0" % (T, SEG))
        got = int((out or "0").strip() or 0)
        print("  设备侧 %d B / 期望 %d B  %s" % (got, want, "OK" if got == want else "🔴 不符"))
        return 0 if got == want else 1

    fails = []
    base = ion()
    print("装载前 IonTotalUsed = %s MiB" % ("%.0f" % base if base else "读不到"))
    print("PD 红线 %d MiB（#57）\n" % PD_RED)
    print("  %-26s %7s %7s %8s  %s" % ("图", "预测", "实测ΔION", "结果", "备注"))
    for g in GRAPHS:
        cfg = ('{"graphs":[{"graph_names":["%s"]}],'
               '"devices":[{"soc_model":69,"dsp_arch":"v79"}]}') % g
        _raw("echo '%s' > %s/d.json" % (cfg, T))
        ext = ('{"backend_extensions":{"shared_library_path":'
               '"libQnnHtpNetRunExtensions.so","config_file_path":"%s/d.json"}}') % T
        _raw("echo '%s' > %s/e.json" % (ext, T))
        b0 = ion()
        rc, out = _raw(
            "cd %s && LD_LIBRARY_PATH=%s ADSP_LIBRARY_PATH=%s "
            "./qnn-net-run --retrieve_context %s_mg.bin --backend libQnnHtp.so "
            "--config_file e.json --input_list /dev/null 2>&1 | tail -40" % (T, T, T, SEG))
        b1 = ion()
        d = (b1 - b0) if (b0 and b1) else None
        pd = ("0x3ea" in out) or ("available PD" in out)
        # context 建成的判据：不报 PD、也不报 create context 失败
        ok = not pd and ("Could not create context" not in out) and \
             ("failed" not in out.lower() or "input_list" in out.lower())
        note = "PD 容量(0x3ea)" if pd else ("" if ok else out.strip().splitlines()[-1][:60])
        print("  %-26s %7d %7s %8s  %s"
              % (g, PRED.get(g, 0), ("%.0f" % d) if d else "-",
                 "PASS" if ok else "FAIL", note))
        if not ok:
            fails.append(g)
    print("\n%s" % ("✅ 五个图全部可装载 => 单段 PD 容量没问题" if not fails else
                    "🔴 %s 装不上 => **退回 single 形态**（回退产物已在盘上）" % fails))
    print("⚠️ 本测试只判「装得上」，不判数值正确 —— 后者由出图与画质判据负责。")
    if fails:
        return 1

    # ---- 第二阶段：四段**同时常驻**的总 ION（#145 方案 B 就是这么跑的）----
    # 🔴 单段过了不等于四段一起过。宿主外推：现网四段实测 ION 9040 MiB
    #    （quadctx 探针，峰值时余 MemAvailable 2010）；五图·出 1:1 时预测合计 9165，
    #    按同一标定比值（1.029）外推 ≈ **9432 MiB**，即 **+392 MiB**，
    #    余量从 2010 掉到约 1617 MiB。台账 #141 记过后端被系统当 phantom process 杀掉，
    #    所以余量变小是**真风险**，必须实测。
    print("\n=== 第二阶段：四段同时常驻的总 ION ===")
    print("  宿主外推：现网 9040 MiB -> 新配置约 9432 MiB（+392），余量 2010 -> 约 1617 MiB")
    print("  🔴 这是外推不是实测。真正的判据是**装完四段后 app 还能出图**（第 5 步），")
    print("     本阶段只把数字量出来，供与外推对照、并给预测器标定。")
    print("  做法：app 里出一张 1:1 的图，同时采 /proc/meminfo：")
    print("    while true; do grep -E 'IonTotalUsed|MemAvailable' /proc/meminfo; sleep 2; done")
    print("  判据：IonTotalUsed 峰值 < 11000 MiB 且 MemAvailable 峰值时 > 800 MiB")
    print("        （低于此值就接近 #141 的被杀区间 => 退回 single 形态或不常驻四段）")
    return 0


sys.exit(main())
