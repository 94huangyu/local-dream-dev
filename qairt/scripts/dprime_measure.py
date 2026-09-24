# -*- coding: utf-8 -*-
"""台账 #162 · 方案 D' 实测 —— 「只把 part1a 改成每步装卸」的真实代价与内存收益。

## D' 是什么、为什么是它（两张拆解表导出，见 scripts/EXP_PLAN_MEMSPEED.md）
`part1a` 是**占 ION 最多（2967 MiB）但装载并不最慢（中位 1.93 s，part2b 才 2.62 s）**
的那一段 ⇒ 每 MiB 的时间代价只有方案原文 D（装卸 part2 两半，实测 +43.4 s）的 **1/1.7**。
③预测代价 **+18.7 s**（用 #145/F 的实测 43.4 s 标定出的模型外推）—— **本脚本就是来证伪它的**。

## 单变量怎么保证
`LOWMEM_EVICT_PART1A` 是 app 私有目录里的 marker 文件，**同一个 APK 跑两条臂**：
文件不在 = 现网行为；文件在 = D'。⇒ 两臂之间**只差这一个文件**，
不需要为基线单独编一版（约束 3.6：归因模式必须单变量）。

## 判据（执行前锁定，事后不得改）
| 门 | 判据 | 不过怎么办 |
|---|---|---|
| G0 铁律 1 | 换 APK **之前**，现网 APK 当天实跑出一张图 | 停止，先查现状为什么坏 |
| G1 铁律 2 | 已安装 APK 已 `adb pull` 存档，字节数核对 | 停止 |
| G2 等价性 | 新 APK **不带 marker** 出图 sha256 与 G0 **逐字节相同** | 说明我的改动动了数值，停止 |
| G3 精度 | 带 marker 出图 sha256 与 G0 **逐字节相同** | 装卸引入了数值差异 ⇒ D' 判负 |
| G4 内存 | 带 marker 的 ION 峰值比不带低 **≥ 2500 MiB** | 收益不及预测，重新解释 |
| G5 速度 | 带 marker 的耗时增量 **≤ +30 s** | 超出则与原 D（+43.4 s）比较后再定 |

🔴 所有对比一律**同一台设备、同一天、同 seed、同 prompt**，
且每臂前 force-stop + 静置（热历史会造成 +16%，#141）。

用法:
  python scripts/dprime_measure.py --check     # 只做宿主自检，不碰设备
  python scripts/dprime_measure.py --run       # 需要设备，约 25 分钟
"""
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join("D:", os.sep, "LocalDreamZImage")
APK = os.path.join(REPO, "local-dream", "app", "build", "outputs", "apk", "basic",
                   "debug", "LocalDreamZImage_armv8a_2.8.1-zimage-mvp.apk")
RUNS = os.path.join(REPO, "scratch_runs")
OUT = os.path.join(RUNS, "dprime")
PKG = "io.github.xororz.localdream.zimage"
MARKER = "files/models/ZIMAGE/LOWMEM_EVICT_PART1A"
SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def check():
    ok = True
    if not os.path.isfile(APK):
        print("🔴 找不到新 APK：%s" % APK)
        ok = False
    else:
        print("新 APK  %s  %d 字节  %s"
              % (os.path.basename(APK), os.path.getsize(APK),
                 time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(APK)))))
    src = os.path.join(REPO, "local-dream", "app", "src", "main", "cpp", "src",
                       "PipelineZImage.hpp")
    txt = open(src, encoding="utf-8", errors="replace").read()
    for token in ("LOWMEM_EVICT_PART1A", "evict_part1a", "part1a_step"):
        hit = txt.count(token)
        print("   源码含 %-22s %d 处" % (token, hit))
        if hit == 0:
            ok = False
    print("✅ 宿主自检通过" if ok else "🔴 宿主自检未过")
    return 0 if ok else 1


def sample_start(dev, tag):
    dev.sh("rm -f /data/local/tmp/ion_%s.txt /data/local/tmp/avail_%s.txt "
           "/data/local/tmp/stop_%s" % (tag, tag, tag))
    # 🔴 每秒采一次。抽稀采样会漏掉真峰值（#161 实测漏成 7557 vs 真值 9486）。
    dev.sh("sh -c 'while [ ! -f /data/local/tmp/stop_%s ]; do "
           "grep IonTotalUsed /proc/meminfo >> /data/local/tmp/ion_%s.txt; "
           "grep MemAvailable /proc/meminfo >> /data/local/tmp/avail_%s.txt; "
           "sleep 1; done' >/dev/null 2>&1 &" % (tag, tag, tag))


def sample_stop(dev, tag):
    dev.sh("touch /data/local/tmp/stop_%s" % tag)

    def series(fn):
        txt = dev.sh("cat /data/local/tmp/%s_%s.txt 2>/dev/null || true" % (fn, tag))
        v = []
        for ln in txt.splitlines():
            f = [x for x in ln.replace("kB", "").split() if x.isdigit()]
            if f:
                v.append(int(f[0]) // 1024)
        return v
    ion, av = series("ion"), series("avail")
    return {"n": len(ion), "ion_peak": max(ion) if ion else 0,
            "avail_low": min(av) if av else 0,
            "avail_plateau_med": avail_plateau_median(ion, av)}


PLATEAU_BAND_MIB = 300


def avail_plateau_median(ion, av):
    """ION 高位平台（ION >= 峰值 − 300 MiB）期间 MemAvailable 的中位数（MiB）。

    🔴 为什么不用「单秒最低」（2026-09-17 实测，`EXP_PLAN_SPILLFILL.md`「H2 这把尺子本身的问题」）：
       每臂低于 800 的都只有 1 秒；ION 几乎相同的两臂（9101 vs 9129）单秒最低相差 345 MiB
       （742 vs 397）⇒ 次间噪声比要分辨的差距还大；现网自己当天就过不了「> 800」。
    ⇒ 取平台期中位数：同样反映「峰值那段时间还剩多少」，但不被单个尖峰主导。
    """
    n = min(len(ion), len(av))
    if n == 0:
        return 0
    peak = max(ion[:n])
    band = sorted(av[i] for i in range(n) if ion[i] >= peak - PLATEAU_BAND_MIB)
    m = len(band)
    return band[m // 2] if m % 2 else (band[m // 2 - 1] + band[m // 2]) // 2


def npu_c(dev):
    """七个 nsp* 热分区的最大值（摄氏度）。取法照抄 speedb_measure.sh 的 npu_max。"""
    out = dev.sh("for z in /sys/class/thermal/thermal_zone*; do "
                 "t=$(cat $z/type 2>/dev/null); v=$(cat $z/temp 2>/dev/null); "
                 "case $t in nsph*) echo $v;; esac; done")
    vals = [int(x) for x in out.split() if x.strip().lstrip("-").isdigit()]
    return max(vals) / 1000.0 if vals else 0.0


def cooldown(dev, base_c, tol=3.0, max_wait_s=600):
    """等 NPU 降到 base+tol 以内。

    🔴 为什么必须有：#141 实测连续 6 张耗时单调爬升 **+16%**（185.8 -> 215.0 s），
       而本实验要测的 Δt 只有约 19 s / 180 s ≈ 11% ⇒ **不控热就没有信噪比**，
       测了也不能用（约束 7：尺子没标定之前不许下结论）。
    """
    t0 = time.time()
    while True:
        c = npu_c(dev)
        if c <= base_c + tol or time.time() - t0 > max_wait_s:
            print("     NPU %.1f C（基线 %.1f，等了 %.0f s）%s"
                  % (c, base_c, time.time() - t0,
                     "" if c <= base_c + tol else "⚠️ 超时仍未降到位，Δt 解读需谨慎"),
                  flush=True)
            return c
        time.sleep(20)


def generate(dev, tag):
    """走真实 app 全链路（不用 run-as —— SELinux 域不同，#72）。"""
    sample_start(dev, tag)
    t0 = time.time()
    r = subprocess.run(["bash", os.path.join("scripts", "app_generate.sh"),
                        "dprime_" + tag, str(SEED)],
                       cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=2400)
    el = time.time() - t0
    mem = sample_stop(dev, tag)
    img = os.path.join(RUNS, "dprime_%s.png" % tag)
    if r.returncode != 0 or not os.path.isfile(img):
        print(r.stdout[-1500:])
        print(r.stderr[-800:])
        raise SystemExit("🔴 %s 臂生成失败 rc=%d" % (tag, r.returncode))
    os.makedirs(OUT, exist_ok=True)
    info = {"tag": tag, "wall_s": el, "sha256": sha256(img), "img": img,
            "npu_after_c": npu_c(dev)}
    info.update(mem)
    # app 自己报的耗时（比墙钟干净：不含 app 启动与 curl 建连）
    sse = os.path.join(RUNS, "dprime_%s.sse" % tag)
    if os.path.isfile(sse):
        import re
        t = open(sse, encoding="utf-8", errors="replace").read()
        m = re.search(r"generation_time_ms\D{0,4}(\d+)", t)
        if m:
            info["generation_s"] = int(m.group(1)) / 1000.0
    print("  %-10s 生成 %.1f s（app 报 %s）  ION 峰值 %d MiB  MemAvail 最低 %d MiB  "
          "采样 %d 点  sha %s"
          % (tag, el, ("%.1f s" % info["generation_s"]) if "generation_s" in info else "—",
             info["ion_peak"], info["avail_low"], info["n"], info["sha256"][:16]), flush=True)
    if info["n"] < 30:
        print("     ⚠️ 采样点仅 %d ⇒ 采样器可能没起来，内存数字不得引用" % info["n"])
    return info


def main():
    if "--check" in sys.argv or len(sys.argv) == 1:
        return check()
    if "--run" not in sys.argv:
        raise SystemExit("用法: --check | --run")
    if check() != 0:
        return 1
    import lab_dev as dev
    os.makedirs(OUT, exist_ok=True)
    res = {}
    # 热基线：先 force-stop 让设备安静下来，再取一次 NPU 温度做后续冷却门的基准。
    dev.sh("am force-stop %s || true" % PKG)
    time.sleep(5)
    base_c = npu_c(dev)
    print("热基线：NPU %.1f C（此后每臂前都等回 +3 C 以内）" % base_c, flush=True)
    res["npu_base_c"] = base_c

    print("")
    print("=== G0 铁律 1：先证明现网 APK 当天可用（不许拿几天前的 history 当证据）===", flush=True)
    cooldown(dev, base_c)
    res["G0_baseline"] = generate(dev, "g0_baseline")

    print("")
    print("=== G1 铁律 2：存档当前已安装的 APK ===", flush=True)
    path = dev.sh("pm path %s" % PKG).strip().replace("package:", "").splitlines()[0].strip()
    bak = os.path.join(OUT, "BACKUP_installed_%s.apk" % time.strftime("%Y%m%d_%H%M%S"))
    dev.pull(path, bak)
    if not os.path.isfile(bak) or os.path.getsize(bak) < 10 << 20:
        raise SystemExit("🔴 APK 存档失败或过小")
    print("  已存档 %s（%d 字节）" % (bak, os.path.getsize(bak)), flush=True)
    res["G1_backup"] = {"device_path": path, "backup": bak, "bytes": os.path.getsize(bak)}

    print("")
    print("=== 安装带 D' 开关的新 APK（marker 不在 ⇒ 行为应与现网完全一致）===", flush=True)
    r = subprocess.run([dev.ADB, "-s", dev.SERIAL, "install", "-r", APK],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800)
    if "Success" not in (r.stdout or ""):
        print(r.stdout, r.stderr)
        raise SystemExit("🔴 安装失败")
    print("  安装成功", flush=True)

    print("")
    print("=== G2 等价性臂（新 APK，无 marker）===", flush=True)
    dev.sh("run-as %s rm -f %s || true" % (PKG, MARKER))
    cooldown(dev, base_c)
    res["G2_newapk_off"] = generate(dev, "g2_off")

    print("")
    print("=== G3/G4/G5 D' 臂（同一 APK，只多一个 marker 文件）===", flush=True)
    dev.sh("run-as %s sh -c 'mkdir -p files/models/ZIMAGE && : > %s'" % (PKG, MARKER))
    got = dev.sh("run-as %s ls -l %s 2>&1 || true" % (PKG, MARKER)).strip()
    print("  marker: %s" % got, flush=True)
    if "No such file" in got:
        raise SystemExit("🔴 marker 没建上 —— D' 臂不会生效，停止（报错说谎防线）")
    cooldown(dev, base_c)
    res["G3_newapk_on"] = generate(dev, "g3_on")
    dev.sh("run-as %s rm -f %s || true" % (PKG, MARKER))
    print("  marker 已删除（设备恢复现网行为）", flush=True)

    b, off, on = res["G0_baseline"], res["G2_newapk_off"], res["G3_newapk_on"]
    print("")
    print("=== 判据 ===", flush=True)
    g2 = off["sha256"] == b["sha256"]
    g3 = on["sha256"] == b["sha256"]
    dion = off["ion_peak"] - on["ion_peak"]
    dt = on.get("generation_s", on["wall_s"]) - off.get("generation_s", off["wall_s"])
    print("  G2 等价性  新APK无marker vs 现网 sha256 %s" % ("✅ 相同" if g2 else "🔴 不同"))
    print("  G3 精度    D' vs 现网 sha256 %s" % ("✅ 相同" if g3 else "🔴 不同"))
    print("  G4 内存    ION 峰值 %d -> %d MiB，降 **%d MiB**（判据 >=2500）%s"
          % (off["ion_peak"], on["ion_peak"], dion, "✅" if dion >= 2500 else "🔴"))
    print("     峰值时 MemAvail %d -> %d MiB" % (off["avail_low"], on["avail_low"]))
    print("  G5 速度    耗时 +%.1f s（判据 <=+30；原方案 D 实测 +43.4）%s"
          % (dt, "✅" if dt <= 30 else "🔴"))
    print("")
    print("  ③预测是 +18.7 s / −2967 MiB；实测 +%.1f s / −%d MiB ⇒ 预测%s"
          % (dt, dion, "成立" if abs(dt - 18.7) < 8 and abs(dion - 2967) < 600 else "**偏离，需解释**"))
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("")
    print("✅ D' 测量结束。**设备已空闲，可以拔线。**")
    print("   如需回滚到现网 APK：adb install -r %s" % res["G1_backup"]["backup"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
