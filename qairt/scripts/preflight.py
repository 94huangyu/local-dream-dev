"""开跑任何实验前的**机械化门禁**。过不了就返回非零退出码，不由我口头判断。

存在的理由（2026-08-16，用户质问后加）：
本项目的约束此前是"记得去遵守"型的，长会话里**必然失效**。当天的每一次违规
都不是不知道，而是没被触发——需要的事实全在手上：
  · 引用了一整天的"unified 能量 99.83% 集中在前 1%"，没用到 encoding 选择上
    ⇒ percentile 钳掉 0.0261% 的元素、丢掉 98.75% 的能量，白花 80 分钟
  · 一小时前刚杀掉一个 5.76 GB 的设备任务并写下"先估内存峰值"
    ⇒ 转头跑了个更重的（约 8.7 GB > 可用 8.15 GB），把用户手机压到重启
  · 十分钟前自己挂的链式脚本，没查就重复启动量化器 ⇒ 内存打穿，多耗 5 小时

⇒ 规则必须**机械化 + 产出物证**。本脚本就是那个物证。

用法:
  python preflight.py --plan scripts/EXP_PLAN_XXX.md --peak-gb 12
  python preflight.py --plan ... --peak-gb 12 --device --device-peak-gb 3
  python preflight.py --plan ... --peak-gb 12 --cost-gate 32.02 --current-error 7.87

硬性条件（任一不满足 ⇒ exit 1）:
  1. --plan 指向的方案文件必须存在（约束 4）
  2. 宿主无同类重任务在跑（python.exe / qnn-context-binary-generator 计数为 0）
  3. 宿主空闲内存 >= --peak-gb * 1.2（留 20% 余量）
  4. --device 时：设备在线、无残留 qnn 进程、MemAvailable >= --device-peak-gb * 1.5
  5. 给了 --cost-gate 时：纯表示误差必须 < --current-error（约束 4·补 G0-cost）

软性输出（必须读，不阻断）:
  · 台账里所有 🔄进行中 / 🔶未查 条目（约束 6：转向前先读表）
  · 标定尺提醒（约束 7）
"""
import argparse
import os
import re
import subprocess
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import sys

ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
ROOT = r"D:\LocalDreamZImage"
MAINLINE = os.path.join(ROOT, "MAINLINE.md")
HEAVY = ["python.exe", "qnn-context-binary-generator.exe"]

fails, warns = [], []


def host_free_gb():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$o=Get-CimInstance Win32_OperatingSystem; "
         "'{0:N2}' -f ($o.FreePhysicalMemory/1MB)"],
        capture_output=True, text=True).stdout.strip()
    return float(out)


def host_proc_count(name):
    """统计同类重任务进程数。**必须排除本脚本自己**，否则 python.exe 永远 >= 1（自指假阳性）。"""
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True).stdout
    me = os.getpid()
    n = 0
    for line in out.splitlines():
        m = re.match(r'"[^"]+","(\d+)"', line.strip())
        if m and int(m.group(1)) != me:
            n += 1
    return n


def adb(*a, timeout=60):
    return subprocess.run([ADB] + list(a), capture_output=True, text=True,
                          timeout=timeout).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="scripts/EXP_PLAN_*.md（约束 4）")
    p.add_argument("--peak-gb", type=float, required=True, help="宿主预期内存峰值")
    p.add_argument("--device", action="store_true", help="本实验要用设备")
    p.add_argument("--device-peak-gb", type=float, default=3.0)
    p.add_argument("--cost-gate", type=float, default=None,
                   help="新 encoding 的纯表示误差 %%（约束 4·补）")
    p.add_argument("--current-error", type=float, default=None,
                   help="当前配置的端到端实测误差 %%")
    a = p.parse_args()

    print("=" * 72)
    print("PREFLIGHT —— 机械化门禁（约束 4 / 4·补 / 6 / 7 / 9）")
    print("=" * 72)

    # 1 方案存在
    pl = a.plan if os.path.isabs(a.plan) else os.path.join(ROOT, a.plan)
    ok = os.path.isfile(pl)
    print(f"[{'PASS' if ok else 'FAIL'}] 约束4 方案文件存在: {pl}")
    if not ok:
        fails.append("方案文件不存在——先写方案再执行")

    # 2 宿主无同类重任务
    for h in HEAVY:
        n = host_proc_count(h)
        ok = (n == 0)
        print(f"[{'PASS' if ok else 'FAIL'}] 宿主 {h} 进程数 = {n}（要求 0）")
        if not ok:
            fails.append(f"{h} 已有 {n} 个在跑——重复启动会打穿内存（2026-08-15 实测代价 5 小时）")

    # 3 宿主内存
    free = host_free_gb()
    need = a.peak_gb * 1.2
    ok = free >= need
    print(f"[{'PASS' if ok else 'FAIL'}] 宿主空闲内存 {free:.2f} GB "
          f">= 峰值 {a.peak_gb:.1f} x1.2 = {need:.2f} GB")
    if not ok:
        fails.append(f"宿主内存不足（空闲 {free:.2f} GB < 需要 {need:.2f} GB）")

    # 4 设备
    if a.device:
        devs = [l for l in adb("devices").splitlines()[1:] if l.strip()]
        ok = bool(devs)
        print(f"[{'PASS' if ok else 'FAIL'}] 设备在线: {devs}")
        if not ok:
            fails.append("设备不在线")
        else:
            # 🔴 必须先确认设备可达，否则拔线会被误判成「进程已结束」（#65 / code_lint C2）
            import lab_dev as _L
            if not _L.online():
                raise RuntimeError('设备不可达 —— 这不等于进程结束，请核实产物后再判断')
            ps = adb('shell', "ps -A -o RSS,NAME | grep -iE 'qnn-net|context-binary' || true")
            left = [l for l in ps.splitlines() if l.strip()]
            ok = not left
            print(f"[{'PASS' if ok else 'FAIL'}] 设备无残留 qnn 进程: {left or '无'}")
            if not ok:
                fails.append("设备上还有残留进程，先清掉")
            m = re.search(r"MemAvailable:\s+(\d+)", adb("shell", "cat /proc/meminfo"))
            avail = int(m.group(1)) / 1048576 if m else 0.0
            need_d = a.device_peak_gb * 1.5
            ok = avail >= need_d
            print(f"[{'PASS' if ok else 'FAIL'}] 设备可用内存 {avail:.2f} GB "
                  f">= 峰值 {a.device_peak_gb:.1f} x1.5 = {need_d:.2f} GB")
            if not ok:
                fails.append(f"设备内存不足（可用 {avail:.2f} GB < 需要 {need_d:.2f} GB）"
                             "——2026-08-16 就是漏算这一步把用户手机压到重启")
            if a.device_peak_gb > 4.0:
                warns.append(f"设备峰值预估 {a.device_peak_gb} GB > 4 GB："
                             "此类任务已实测会拖垮整机，**必须先取得用户明确同意**")

    # 5 代价门
    if a.cost_gate is not None:
        if a.current_error is None:
            fails.append("给了 --cost-gate 就必须给 --current-error")
        else:
            ok = a.cost_gate < a.current_error
            print(f"[{'PASS' if ok else 'FAIL'}] 约束4·补 代价门: "
                  f"新 encoding 纯表示误差 {a.cost_gate:.2f}% "
                  f"< 当前端到端误差 {a.current_error:.2f}%")
            if not ok:
                fails.append(f"代价门不过：表示误差 {a.cost_gate:.2f}% 已 >= "
                             f"端到端 {a.current_error:.2f}%，该实验不可能有收益，不许上机")
    else:
        warns.append("未提供 --cost-gate：若本实验会改 encoding/量化参数，"
                     "**必须**先算纯表示误差再来（约束 4·补）")

    # 软性：台账未关闭条目
    print("-" * 72)
    print("约束6 台账里未关闭的条目（转向前必须读）：")
    if os.path.isfile(MAINLINE):
        for l in open(MAINLINE, encoding="utf-8"):
            if l.startswith("|") and ("🔄" in l or "🔶" in l or "⛔" in l):
                cols = [c.strip() for c in l.strip().strip("|").split("|")]
                if len(cols) >= 2 and cols[0].isdigit():
                    print(f"   #{cols[0]:<4} {cols[1][:66]}")
    print("-" * 72)
    print("约束7 标定尺提醒：噪声预测相对 FP32 —— 15.99% 成图完好 / 47.07% 橙色色块；")
    print("               `unified` 能量 99.83% 集中在前 1% ⇒ 必须用主体口径。")

    print("=" * 72)
    for w in warns:
        print(f"[WARN] {w}")
    if fails:
        for f in fails:
            print(f"[FAIL] {f}")
        print("\n>>> PREFLIGHT 未通过，禁止执行。")
        return 1
    print(">>> PREFLIGHT 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
