# -*- coding: utf-8 -*-
"""S9：交付 #182（校验结果缓存）。一次插线约 6 分钟。

判据见 `EXP_PLAN_P2_SPEED.md` §10.3（事前锁定；G4 在执行前已按约束 8 修正为 G4′，原文保留）：
  G1  删缓存后首次启动：`[startup] 契约基准校验 N ms` 的 N ≥ 10000（确实算了）
  G2  第二次启动：N ≤ 1000（缓存生效）
  G4′ `touch` part1a（2.88 GiB）后启动：N ∈ [2500, 8000]（按文件粒度失效）
  G3  出图 sha256 == 金标准（数值不变）

🔴 读 `[startup]` 必须让 `logcat -c` 发生在 `am start` **之前** ——
   本项目为此白跑两次（`initialize()` 在录制窗口之外，#181）。

用法: python scripts/p2_s9_cache.py --run
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D
import p2_s6_session as P

OUT = os.path.join(D.RUNS, "p2s9")
CACHE = "files/models/ZIMAGE/.verified_cache"
PART1A = "files/models/ZIMAGE/models/transformer_part1a_mg_ctx.SM8750.bin"
STARTUP = re.compile(r"\[startup\] 契约基准校验 (\d+) ms（sha256: ([^）]+)）")


def boot_and_read(dev, tag):
    """冷启动一次并读回校验耗时。**清空 logcat 必须在 am start 之前**。"""
    dev.sh("am force-stop %s || true" % D.PKG)
    dev.sh("sleep 3")
    dev.sh("logcat -c")
    dev.sh("am start -a android.intent.action.MAIN -c android.intent.category.LAUNCHER "
           "-n %s/io.github.xororz.localdream.MainActivity" % D.PKG)
    dev.sh("sleep 4; input tap 608 1279")
    dev.sh("sleep 45")
    out = dev.sh("logcat -d")
    for line in out.splitlines():
        m = STARTUP.search(line)
        if m:
            print("  [%s] 校验 %s ms（%s）" % (tag, m.group(1), m.group(2)), flush=True)
            return int(m.group(1)), m.group(2)
    print("  [%s] 🔴 没读到 [startup] 日志" % tag, flush=True)
    return None, None


def main():
    if "--run" not in sys.argv:
        print(__doc__)
        return 0
    import lab_dev as dev
    os.makedirs(OUT, exist_ok=True)
    dev.require_online("p2s9")
    print("\n⚠️ 占用设备约 6 分钟（3 次冷启动 + 1 次出图）。**结束前请勿拔线。**", flush=True)

    # 铁律 2：先存档设备当前版（按**整包** sha256 命名，不用 .so 指纹——见 #180 的教训）
    path = dev.sh("pm path %s" % D.PKG).strip().replace("package:", "").splitlines()[0].strip()
    tmp = os.path.join(OUT, "_pull.apk")
    subprocess.run([dev.ADB, "pull", path, tmp], check=True, timeout=900, stdout=subprocess.DEVNULL)
    bak = os.path.join(OUT, "installed_%s.apk" % P.apk_sha(tmp)[:16])
    if os.path.isfile(bak):
        os.remove(tmp)
    else:
        os.replace(tmp, bak)
    print("  ✅ 旧版已存档：%s" % os.path.basename(bak), flush=True)

    subprocess.run([dev.ADB, "install", "-r", P.NEW_APK], check=True, timeout=1800,
                   stdout=subprocess.DEVNULL)
    print("  ✅ 新 APK 已装（整包 sha256 %s）" % P.apk_sha(P.NEW_APK)[:16], flush=True)

    st = {"rollback_apk": bak}
    print("\n-- G1：删缓存后首次启动（应真的算，N ≥ 10000）--", flush=True)
    dev.sh("run-as %s rm -f %s || true" % (D.PKG, CACHE))
    n1, mode1 = boot_and_read(dev, "G1")
    st["n1"], st["mode"] = n1, mode1

    print("\n-- G2：第二次启动（缓存应命中，N ≤ 1000）--", flush=True)
    n2, _ = boot_and_read(dev, "G2")
    st["n2"] = n2

    print("\n-- G4′：touch part1a（2.88 GiB）后启动（应只重算它，N ∈ [2500, 8000]）--", flush=True)
    dev.sh("run-as %s touch %s" % (D.PKG, PART1A))
    n3, _ = boot_and_read(dev, "G4")
    st["n3"] = n3

    print("\n-- G3：出图核 sha256 --", flush=True)
    r = subprocess.run(["bash", os.path.join("scripts", "app_generate.sh"), "s9_cache", "42"],
                       cwd=D.REPO, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=2700)
    img = os.path.join(D.RUNS, "s9_cache.png")
    sha = D.sha256(img) if os.path.isfile(img) else None
    st["img_sha"] = sha
    # 🔴 G5（2026-09-20 补，执行前锁定）：跳过校验后，装载时的读变成**冷读** ——
    #   原先那 19.4 s 的校验顺带把部分数据带进了页缓存。若装载因此变慢，净收益会被吃掉一部分，
    #   而只看出图 sha256 的判据**发现不了**。所以必须同时记 generation 与八段装载耗时。
    sse = os.path.join(D.RUNS, "s9_cache.sse")
    gen_s = None
    if os.path.isfile(sse):
        m = re.search(r"generation_time_ms\D{0,4}(\d+)",
                      open(sse, encoding="utf-8", errors="replace").read())
        if m:
            gen_s = int(m.group(1)) / 1000.0
    split = P.split_of("s9_cache") if os.path.isfile(
        os.path.join(D.RUNS, "s9_cache_logcat.txt")) else {}
    st["generation_s"], st["load_total_s"] = gen_s, split.get("load_total_s")
    if sha is None:
        print("  🔴 出图失败 rc=%d" % r.returncode)
        print((r.stdout or "")[-1000:])

    cache_txt = dev.sh("run-as %s cat %s 2>/dev/null | wc -l" % (D.PKG, CACHE)).strip()
    print("\n== #182 交付门 ==")
    g1 = n1 is not None and n1 >= 10000
    g2 = n2 is not None and n2 <= 1000
    g4 = n3 is not None and 2500 <= n3 <= 8000
    g3 = sha == P.GOLDEN_SHA
    print("  G1  首次仍校验：%s ms %s" % (n1, "✅" if g1 else "🔴"))
    print("  G2  二次跳过  ：%s ms %s" % (n2, "✅" if g2 else "🔴"))
    print("  G4′ 按文件失效：%s ms %s（期望 2500~8000）" % (n3, "✅" if g4 else "🔴"))
    print("  G3  数值不变  ：%s %s" % ((sha or "无")[:16], "✅" if g3 else "🔴"))
    # G5 只记录不设门：没有同日同装置的对照臂，设门就是拿没刻度的尺子下结论（约束 7）。
    print("  G5  generation %s s｜八段装载 %s s（参照：09-19 A 臂 130.4 s / 12.97 s）"
          % (gen_s, ("%.2f" % st["load_total_s"]) if st.get("load_total_s") else "—"))
    if gen_s and st.get("load_total_s") and st["load_total_s"] > 16.0:
        print("      ⚠️ 装载比 09-19 的 12.97 s 明显变慢 ⇒ 疑似冷读代价，需单独查")
    print("  ⊕ 缓存文件行数：%s（应为 12）｜sha256 模式：%s" % (cache_txt, mode1))
    ok = g1 and g2 and g4 and g3
    json.dump(st, open(os.path.join(OUT, "state.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if not ok:
        print("\n🔴 有门未过 ⇒ 回滚 APK")
        subprocess.run([dev.ADB, "install", "-r", bak], timeout=900, stdout=subprocess.DEVNULL)
        return 1
    print("\n✅ #182 交付成立：冷启动校验 %d ms → %d ms，数值不变，失效机制按文件粒度有效" % (n1, n2))
    P.final_state(dev, {"h2_deliver": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
