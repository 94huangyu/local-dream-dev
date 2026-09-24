# -*- coding: utf-8 -*-
"""实验 1 · 跨 context 共享 spill-fill 的实际省量（`scripts/EXP_PLAN_SPILLFILL.md`）。

## 为什么这是 P1 的生死点
mg 形态（五图共享权重）能把设备占用从 **35 GiB 降到 12.3 GiB**，但常驻多 **375 MiB**，
实测被系统杀（#161：ION 9486 / MemAvail 208，两次复现）。
共享 spill-fill ③预测省 **~854 MiB ION**，够把这 375 MiB 补回来还有余。
但 #146 自己标着「③共享后系统 ION 总量是否真的下降 —— 未验证」。**本脚本就是问这一句。**

## 单变量
同一个 APK，marker 文件 `SHARE_SPILLFILL` 控制开关 ⇒ 两臂**只差这一个文件**。
组大小 302,645,248 B（= 组内最大者 part1a，由 `mem_breakdown.py` 从元数据现读）。

## 判据（`EXP_PLAN_SPILLFILL.md` §4 事前锁定，不得改）
| 门 | 判据 |
|---|---|
| G0 | 换 APK 前，现网 APK 当天实跑出图 |
| G1 | 已装 APK 已存档且字节数核对 |
| G2 | 关臂 sha256 与 G0 逐字节相同 |
| G3 | 开臂 sha256 与 G0 逐字节相同 |
| **G4** | 开臂 ION 峰值比关臂低 **>= 600 MiB** |
| G5 | 开臂耗时增量 <= **+5 s** |

用法: python scripts/spillfill_measure.py --check | --run
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D          # 复用已验证的冷却门 / ION 采样 / sha256

REPO = D.REPO
APK = D.APK
RUNS = D.RUNS
OUT = os.path.join(RUNS, "spillfill")
PKG = D.PKG
MARKER = "files/models/ZIMAGE/SHARE_SPILLFILL"
SEED = 42
G4_MIN_SAVE = 600      # MiB
G5_MAX_SLOW = 5.0      # s


def check():
    ok = True
    if not os.path.isfile(APK):
        print("🔴 找不到 APK：%s" % APK); ok = False
    else:
        print("APK  %d 字节  %s" % (os.path.getsize(APK),
              time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(APK)))))
    for f, toks in ((os.path.join(REPO, "local-dream", "app", "src", "main", "cpp", "src",
                                  "PipelineZImage.hpp"),
                     ("SHARE_SPILLFILL", "spillFillGroupBytes", "302645248")),
                    (os.path.join(REPO, "local-dream", "app", "src", "main", "cpp", "src",
                                  "QnnRuntime.hpp"),
                     ("sfBytes", "setSpillFillGroup"))):
        txt = open(f, encoding="utf-8", errors="replace").read()
        for t in toks:
            n = txt.count(t)
            print("   %-22s 含 %-22s %d 处" % (os.path.basename(f), t, n))
            if n == 0:
                ok = False
    print("✅ 宿主自检通过" if ok else "🔴 宿主自检未过")
    return 0 if ok else 1


def generate(dev, tag):
    D.sample_start(dev, tag)
    t0 = time.time()
    r = subprocess.run(["bash", os.path.join("scripts", "app_generate.sh"),
                        "sf_" + tag, str(SEED)],
                       cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=2400)
    el = time.time() - t0
    mem = D.sample_stop(dev, tag)
    img = os.path.join(RUNS, "sf_%s.png" % tag)
    if r.returncode != 0 or not os.path.isfile(img):
        print(r.stdout[-1500:]); print(r.stderr[-800:])
        raise SystemExit("🔴 %s 臂生成失败 rc=%d" % (tag, r.returncode))
    info = {"tag": tag, "wall_s": el, "sha256": D.sha256(img), "img": img,
            "npu_after_c": D.npu_c(dev)}
    info.update(mem)
    sse = os.path.join(RUNS, "sf_%s.sse" % tag)
    if os.path.isfile(sse):
        import re
        m = re.search(r"generation_time_ms\D{0,4}(\d+)",
                      open(sse, encoding="utf-8", errors="replace").read())
        if m:
            info["generation_s"] = int(m.group(1)) / 1000.0
    print("  %-10s 生成 %.1f s（app 报 %s）  ION 峰值 %d  MemAvail 最低 %d MiB  采样 %d 点  sha %s"
          % (tag, el, ("%.1f s" % info["generation_s"]) if "generation_s" in info else "—",
             info["ion_peak"], info["avail_low"], info["n"], info["sha256"][:16]), flush=True)
    if info["n"] < 30:
        print("     ⚠️ 采样点仅 %d ⇒ 采样器可能没起来，内存数字不得引用" % info["n"])
    return info


def main():
    if "--run" not in sys.argv:
        return check()
    if check() != 0:
        return 1
    import lab_dev as dev
    os.makedirs(OUT, exist_ok=True)
    res = {}
    dev.sh("am force-stop %s || true" % PKG)
    time.sleep(5)
    base_c = D.npu_c(dev)
    print("热基线：NPU %.1f C" % base_c, flush=True)

    print("\n=== G0 铁律 1：现网 APK 当天实跑 ===", flush=True)
    D.cooldown(dev, base_c)
    res["G0"] = generate(dev, "g0")

    print("\n=== G1 铁律 2：存档已装 APK ===", flush=True)
    path = dev.sh("pm path %s" % PKG).strip().replace("package:", "").splitlines()[0].strip()
    bak = os.path.join(OUT, "BACKUP_installed_%s.apk" % time.strftime("%Y%m%d_%H%M%S"))
    dev.pull(path, bak)
    if not os.path.isfile(bak) or os.path.getsize(bak) < (10 << 20):
        raise SystemExit("🔴 存档失败或过小")
    print("  已存档 %s（%d 字节）" % (os.path.basename(bak), os.path.getsize(bak)), flush=True)
    res["backup"] = bak

    print("\n=== 安装新 APK ===", flush=True)
    r = subprocess.run([dev.ADB, "-s", dev.SERIAL, "install", "-r", APK],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800)
    if "Success" not in (r.stdout or ""):
        print(r.stdout, r.stderr); raise SystemExit("🔴 安装失败")
    print("  安装成功", flush=True)

    print("\n=== G2 关臂（无 marker，应与现网逐字节相同）===", flush=True)
    dev.sh("run-as %s rm -f %s || true" % (PKG, MARKER))
    D.cooldown(dev, base_c)
    res["off"] = generate(dev, "off")

    print("\n=== G3/G4/G5 开臂（同一 APK，只多一个 marker）===", flush=True)
    dev.sh("run-as %s sh -c 'mkdir -p files/models/ZIMAGE && : > %s'" % (PKG, MARKER))
    got = dev.sh("run-as %s ls -l %s 2>&1 || true" % (PKG, MARKER)).strip()
    print("  marker: %s" % got, flush=True)
    if "No such file" in got:
        raise SystemExit("🔴 marker 没建上 ⇒ 开臂不会生效，停止")
    D.cooldown(dev, base_c)
    res["on"] = generate(dev, "on")
    dev.sh("run-as %s rm -f %s || true" % (PKG, MARKER))

    b, off, on = res["G0"], res["off"], res["on"]
    save = off["ion_peak"] - on["ion_peak"]
    dt = on.get("generation_s", on["wall_s"]) - off.get("generation_s", off["wall_s"])
    print("\n=== 判据 ===")
    print("  G2 等价性  关臂 vs 现网 sha256 %s" % ("✅ 相同" if off["sha256"] == b["sha256"] else "🔴 不同"))
    print("  G3 精度    开臂 vs 现网 sha256 %s" % ("✅ 相同" if on["sha256"] == b["sha256"] else "🔴 不同"))
    print("  G4 省量    ION 峰值 %d -> %d，降 **%d MiB**（判据 >=%d）%s"
          % (off["ion_peak"], on["ion_peak"], save, G4_MIN_SAVE,
             "✅" if save >= G4_MIN_SAVE else "🔴"))
    print("     峰值 MemAvail %d -> %d MiB" % (off["avail_low"], on["avail_low"]))
    print("  G5 速度    %+.1f s（判据 <=+%.0f）%s" % (dt, G5_MAX_SLOW,
                                                  "✅" if dt <= G5_MAX_SLOW else "🔴"))
    print("\n  ③预测省 ~854 MiB；实测 %d MiB ⇒ %s"
          % (save, "与预测相符" if abs(save - 854) < 250 else "**偏离预测，需解释**"))
    ok = (off["sha256"] == b["sha256"] and on["sha256"] == b["sha256"]
          and save >= G4_MIN_SAVE and dt <= G5_MAX_SLOW)
    print("\n  ⇒ %s" % ("🟢 全过，可以进实验 2（mg + spill-fill）"
                        if ok else "🔴 未全过，**不得直接切 mg**"))
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    print("\n✅ 结束。**设备已空闲，可以拔线。**")
    print("   回滚现网 APK：adb install -r %s" % bak)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
