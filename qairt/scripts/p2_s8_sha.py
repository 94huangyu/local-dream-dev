# -*- coding: utf-8 -*-
"""S8：交付 #178（契约校验换 ARMv8 硬件 SHA-256）。一次插线约 10 分钟。

## 判据（事前锁定，执行后不得改）
| 门 | 内容 | 不过怎么办 |
|---|---|---|
| **G0 数值** | 出图 sha256 == 金标准 `49be8e9a4f95…` | 回滚 APK |
| **G1 自检** | `[startup]` 日志显示 **ARMv8 硬件加速**（不是"软件回落"） | 记录；回落说明实现有误，**不算交付** |
| **G2 校验耗时** | `[startup] 契约基准校验 N ms` 的 **N ≤ 30000**（②预估 12 s，给 2.5 倍余量） | 记录实测值，不达标则收益未复现 |
| **G3 内存** | ION 峰值 ≤ 上一次交付态 8447 + 200；MemAvail 最低 ≥ 400 | 回滚 APK |
| **G4 终态** | marker 仍在（H2 交付形态）、part1a 仍是 noscatA | 修正后再拔线 |

🔴 本次**只换 APK**，不碰契约、不碰模型文件 ⇒ 回滚只需装回旧 APK（一条命令）。

用法: python scripts/p2_s8_sha.py --run
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D
import oneshot_mg_session as S
import p2_s6_session as P

OUT = os.path.join(D.RUNS, "p2s8")
STATE = os.path.join(OUT, "state.json")
G2_MAX_MS = 30000
G3_ION_REF = 8447          # 2026-09-19 A 臂（交付态）的 ION 峰值
G3_AVAIL_MIN = 400
STARTUP = re.compile(r"\[startup\] 契约基准校验 (\d+) ms（sha256: ([^）]+)）")


def main():
    if "--run" not in sys.argv:
        print(__doc__)
        return 0
    import lab_dev as dev
    os.makedirs(OUT, exist_ok=True)
    dev.require_online("p2s8")
    print("\n⚠️ 占用设备约 10 分钟（装 APK + 一次出图）。**结束前请勿拔线。**", flush=True)

    # 铁律 2：先存档设备上现在这版（按指纹命名，不覆盖）
    path = dev.sh("pm path %s" % S.PKG).strip().replace("package:", "").splitlines()[0].strip()
    tmp = os.path.join(OUT, "_pulled.apk")
    import subprocess
    subprocess.run([dev.ADB, "pull", path, tmp], check=True, timeout=900)
    fp = P.so_sha(tmp)[:16]
    bak = os.path.join(OUT, "installed_%s.apk" % fp)
    if os.path.isfile(bak):
        os.remove(tmp)
    else:
        os.replace(tmp, bak)
    print("  ✅ 设备当前版已存档：%s" % os.path.basename(bak), flush=True)

    st = {"rollback_apk": bak, "prev_so": fp}
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    dev.sh("am force-stop %s || true" % S.PKG)
    time.sleep(3)
    subprocess.run([dev.ADB, "install", "-r", P.NEW_APK], check=True, timeout=1800)
    print("  ✅ 新 APK 已装（.so %s）" % P.so_sha(P.NEW_APK)[:16], flush=True)

    base_c = D.npu_c(dev)
    info = S.gen(dev, "s8_sha", base_c=base_c)
    st["run"] = info

    # 解析 [startup]
    log = os.path.join(D.RUNS, "os_s8_sha_logcat.txt")
    ms, mode = None, None
    if os.path.isfile(log):
        import io as _io
        for line in _io.open(log, encoding="utf-8", errors="replace"):
            m = STARTUP.search(line)
            if m:
                ms, mode = int(m.group(1)), m.group(2)
                break
    st["startup_ms"], st["sha_mode"] = ms, mode
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n== #178 交付门（判据事前锁定）==")
    g0 = info["sha256"] == P.GOLDEN_SHA
    print("  G0 数值：出图 sha256 %s 金标准 %s" % ("==" if g0 else "≠", "✅" if g0 else "🔴"))
    if ms is None:
        print("  G1/G2：🔴 **没读到 `[startup]` 日志** ⇒ 说明这行没执行或 logcat 没抓到")
        g1 = g2 = False
    else:
        g1 = "硬件" in (mode or "")
        g2 = ms <= G2_MAX_MS
        print("  G1 自检：sha256 走的是 **%s** %s" % (mode, "✅" if g1 else "🔴 回落了，实现有误"))
        print("  G2 校验耗时：**%d ms**（门限 %d）%s ⇒ 相对旧版约 129 s，省约 %.0f s"
              % (ms, G2_MAX_MS, "✅" if g2 else "🔴", (129000 - ms) / 1000.0))
    g3 = info["ion_peak"] <= G3_ION_REF + 200 and info["avail_low"] >= G3_AVAIL_MIN
    print("  G3 内存：ION %d（参照 %d+200）｜MemAvail 最低 %d（门限 %d）%s"
          % (info["ion_peak"], G3_ION_REF, info["avail_low"], G3_AVAIL_MIN, "✅" if g3 else "🔴"))
    print("  ⊕ 本次墙钟 %.0f s，app 内 generation %.1f s"
          % (info["wall_s"], info.get("generation_s", 0)))

    ok = g0 and g3
    if not ok:
        print("\n🔴 G0/G3 未过 ⇒ 回滚 APK")
        subprocess.run([dev.ADB, "install", "-r", bak], timeout=900)
        return 1
    P.final_state(dev, {"h2_deliver": True})
    print("\n%s" % ("✅ #178 交付成立（数值不变 + 硬件加速生效 + 内存达标）" if (g1 and g2) else
                    "🟡 数值与内存达标，但加速未按预期生效 —— 见上面 G1/G2"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
