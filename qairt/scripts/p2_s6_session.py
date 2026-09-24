# -*- coding: utf-8 -*-
"""S6：一次插线跑完 H5 拆时 + H2 装载 A/B + H1-A 交付（判据见 `EXP_PLAN_P2_SPEED.md` §7，事前锁定）。

## 为什么合成一次
手机是用户的（MAINLINE §7.0）。三件事共用同一个 APK、同一次热机状态，分三次插线只会多两次冷却与两套基线。
顺序按**风险递增**排：先只测不改的（H5）→ 只改装载路径的（H2）→ 最后才换模型文件（H1-A）。

## 阶段（`--phase` 可续跑）
| 阶段 | 内容 | 不过怎么办 |
|---|---|---|
| 0 | 宿主自检（不碰设备）：APK 产物核对、noscatA 产物核对、spillFill vs marker | 停 |
| 1 | 铁律 1/2：**存档现网 APK** + 现网基线出图 B | 停 |
| 2 | 装新 APK → R/M/R/M 交替四次（H2 A/B + H5 拆时） | 回滚 APK |
| 3 | 推 noscatA context → A 臂出图（H1-A 交付门） | 回滚 context（+ marker） |

## 全程纪律（沿用 oneshot_mg_session 已验证的那套）
· ION 每秒采样（抽稀漏真峰值，#161）· 每臂前等 NPU 回到基线 +3 °C（#141 热漂移 +16%）
· 一切以字节数/产物为准，不看日志内容（#176「报错说谎」）· 任一门不过 ⇒ 逆序回滚，不留半截

用法:
    python scripts/p2_s6_session.py                 # 纯宿主自检（阶段 0）
    python scripts/p2_s6_session.py --run           # 全流程（需插线）
    python scripts/p2_s6_session.py --run --phase 2 # 续跑
    python scripts/p2_s6_session.py --rollback      # 紧急回滚
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D
import oneshot_mg_session as S          # 复用 gen()/write_marker()/PKG/BIN_DIR 等已验证件

REPO = D.REPO
RUNS = D.RUNS
PKG = D.PKG
OUT = os.path.join(RUNS, "p2s6")
STATE = os.path.join(OUT, "state.json")

NEW_APK = D.APK                          # local-dream 的构建产物（basic/debug）
OLD_SO_SHA = "f68edf747d729c28"          # 现网 APK 里 libstable_diffusion_core.so 的 sha256 前 16 位
MMAP_MARKER = "files/models/ZIMAGE/LOAD_MMAP"
SF_MARKER = "files/models/ZIMAGE/SHARE_SPILLFILL"
# 🔴 设备上的文件名**不是**建图产物的名字：交付脚本会改名。以契约里的 `context_binary` 为准。
DEV_PART1A_REL = "models/transformer_part1a_mg_ctx.SM8750.bin"
DEV_PART1A = "files/models/ZIMAGE/" + DEV_PART1A_REL
DEV_CONTRACT = S.DEV_CONTRACT
NEW_PART1A = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_deliver",
                          "ctx_mg", "part1a_mg_noscatA.SM8750.bin")
NEW_INFO = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_deliver",
                        "info_mg_noscatA.json")
# 🔴 回滚源：`deliver_mg/` 已在 2026-09-17 清理中删除（#171「清理后无快速回滚」），
#    但**建图原件还在** `aspect_mg/part1a/`（3.15 GB，09-04）。它能不能当回滚源，
#    不许假定 —— 阶段 3 动手前必须「设备端 sha256 == 宿主端 sha256」实测（约束 11 第 2 条）。
#    不相等 ⇒ 先把设备上那份拉回宿主当备份，再动它。
OLD_PART1A = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg",
                          "part1a", "part1a_mg.SM8750.bin")

# ---- 判据（EXP_PLAN_P2_SPEED.md §7.2，事前锁定，本脚本不得改）----
G1_MMAP_GAIN_S = 2.0        # 🟢 交付：mmap 臂装载总和 ≤ read 臂 − 2.0 s
G1_ION_TOL_MB = 200         # 🔴 关闭：mmap 臂 ION 峰值 > read 臂 + 200 MiB
G2_EXEC_TOL = 0.05          # H5 自证：拆时合计 vs [segtime] 合计相差须 < 5%
G3_SPEEDUP_S = 4.0          # H1-A 速度下限（②推算 −6.3 s 打 ~65% 折）
G3_AVAIL_LOW_MB = 400       # MemAvailable 最低点下限（#161 口径）
# 金标准出图：2026-08~09 交付门用的那张，2026-09-19 三次（现网 APK / 新 APK read / 新 APK mmap）
# 全部逐字节等于它 ⇒ 可直接当绝对参照，**不必每次再花 6 分钟单独跑一次现网基线**。
# 铁律 1「先证明现状可用」由本次第一臂兼任：它等于金标准就说明现状可用。
GOLDEN_SHA = "49be8e9a4f95b85e320a9c3caf466c7078f315e5242f7c35981f929529f83ad6"


def save(st):
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load():
    return json.load(open(STATE, encoding="utf-8")) if os.path.isfile(STATE) else {}


def so_sha(apk):
    import hashlib
    import zipfile
    with zipfile.ZipFile(apk) as z:
        return hashlib.sha256(z.read("lib/arm64-v8a/libstable_diffusion_core.so")).hexdigest()


def apk_sha(apk):
    """整个 APK 的 sha256。

    🔴 存档命名必须用它，**不能用 `.so` 指纹**：只改 Kotlin 的版本 `.so` 完全相同
    （2026-09-20 实测：sha256 版与 UI 版 `.so` 同为 538c36c58263b074），
    按 `.so` 命名会让第二个版本被当成"已有存档"跳过 ⇒ **丢失存档**。
    ⊕ 同一天还实测到：那两个 APK 的**字节数也完全相同**（94339337）而内容不同
    ⇒ **字节数不是判据，sha256 才是**。
    """
    import hashlib
    h = hashlib.sha256()
    with open(apk, "rb") as f:
        while True:
            b = f.read(4 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def max_spillfill(info_json):
    j = json.load(open(info_json, encoding="utf-8", errors="replace"))
    return max(g["info"]["graphBlobInfo"]["info"]["spillFillBufferSize"]
               for g in j["info"]["graphs"])


def check():
    """阶段 0：能在宿主判死的全部判死（约束 4·补：不把可预见的失败带到设备上）。"""
    ok = True
    print("# S6 宿主自检\n")
    if not os.path.isfile(NEW_APK):
        print("  🔴 新 APK 不存在：%s\n     ⇒ 先构建（MAINLINE §7.4：编译成功 ≠ 改动进了产物）" % NEW_APK)
        return 1
    sha = so_sha(NEW_APK)
    same = sha.startswith(OLD_SO_SHA)
    print("  %s libstable_diffusion_core.so %s（现网 %s…）"
          % ("🔴" if same else "✅", sha[:16], OLD_SO_SHA))
    if same:
        print("     ⇒ 与现网**逐字节相同**：改动没进产物，本次交付无意义，停。")
        ok = False
    print("  ✅ APK %.1f MB  %s" % (os.path.getsize(NEW_APK) / 1e6,
                                    time.strftime("%Y-%m-%d %H:%M",
                                                  time.localtime(os.path.getmtime(NEW_APK)))))
    # 埋点必须真的在二进制里（否则 A/B 拿不到数，等于白插一次线）
    import zipfile
    with zipfile.ZipFile(NEW_APK) as z:
        blob = z.read("lib/arm64-v8a/libstable_diffusion_core.so")
    # 🔴 找的是**格式串与字面量**，不是拼好的日志行：`[loadpath] %s %s (%llu bytes)`
    #    里的 "mmap"/"read-into-vector" 是另外两个独立字符串，搜 "[loadpath] mmap" 永远搜不到。
    for mark in (b"[loadpath] %s %s", b"read-into-vector", b"[segsplit]", b"LOAD_MMAP"):
        hit = blob.find(mark) >= 0
        print("  %s 二进制含 %-18s" % ("✅" if hit else "🔴", mark.decode()))
        ok &= hit
    # H1-A 产物
    if not os.path.isfile(NEW_PART1A):
        print("  ⚠️ noscatA context 还没建完：%s ⇒ 阶段 3 不可用（阶段 1/2 仍可跑）" % NEW_PART1A)
    else:
        print("  ✅ noscatA context %.3f GB" % (os.path.getsize(NEW_PART1A) / 1e9))
        if os.path.isfile(NEW_INFO):
            mx = max_spillfill(NEW_INFO)
            print("  ℹ️ 新 context 最大 spillFillBufferSize = %d（设备 marker 需 ≥ 此值）" % mx)
    if not os.path.isfile(OLD_PART1A):
        print("  🔴 回滚源缺失：%s ⇒ 阶段 3 禁止执行" % OLD_PART1A)
    else:
        print("  ✅ 回滚源在盘 %.3f GB（阶段 3 会先与设备实测 sha256 比对后才认它）"
              % (os.path.getsize(OLD_PART1A) / 1e9))
    print("\n%s" % ("✅ 阶段 0 通过" if ok else "🔴 阶段 0 未过，不得插线"))
    return 0 if ok else 1


# ---------------- 设备侧小工具 ----------------

def dev_marker(dev, path, on):
    """放/删 marker。🔴 放完必须读回确认——写没写进去只能由设备说了算。"""
    if on:
        dev.sh("run-as %s sh -c 'mkdir -p files/models/ZIMAGE && : > %s'" % (PKG, path))
    else:
        dev.sh("run-as %s rm -f %s || true" % (PKG, path))
    back = dev.sh("run-as %s sh -c 'ls %s 2>/dev/null | wc -l'" % (PKG, path)).strip()
    got = back.endswith("1")
    if got != on:
        raise SystemExit("🔴 marker %s 期望 %s，读回 %s" % (path, on, back))
    print("  marker %s ⇒ %s" % (os.path.basename(path), "有" if on else "无"), flush=True)


def split_of(tag):
    """从该臂的 logcat 解析装载总和与拆时表（口径与 timing_breakdown.py --split 一致）。"""
    import timing_breakdown as T
    import io as _io
    p = os.path.join(RUNS, "os_%s_logcat.txt" % tag)
    if not os.path.isfile(p):
        return {}
    paths, clk, load, segsplit, segt = [], {}, {}, {}, 0
    nbytes = {}
    for line in _io.open(p, encoding="utf-8", errors="replace"):
        mc = T.CLK.search(line)
        t = float(mc.group(1)) if mc else None
        mp = T.LOADPATH.search(line)
        if mp:
            paths.append(mp.group(1))
            nbytes[mp.group(2)] = int(mp.group(3))
            if t is not None:
                clk[mp.group(2)] = t
        mb = T.INIT_B.search(line)
        if mb and t is not None and mb.group(1) in clk:
            load[mb.group(1)] = t - clk.pop(mb.group(1))
        m = T.SPLIT.search(line)
        if m:
            a = segsplit.setdefault(m.group(1), [0.0] * 5)
            for i in range(5):
                a[i] += float(m.group(3 + i))
        ms = T.SEGT.search(line)
        if ms:
            segt += int(ms.group(2))
    return {"paths": sorted(set(paths)), "load_total_s": sum(load.values()) / 1000.0,
            "load_n": len(load), "segsplit": segsplit, "segtime_total_ms": segt,
            # P3：part1a 这一段实际装载了多少字节 —— 用来证明"换的文件真的被用了"
            "part1a_bytes": nbytes.get("transformer_part1a"),
            "day": time.strftime("%Y-%m-%d")}


def arm(dev, tag, base_c, mmap_on):
    dev_marker(dev, MMAP_MARKER, mmap_on)
    info = S.gen(dev, tag, base_c=base_c)
    info.update(split_of(tag))
    if len(info.get("paths", [])) != 1:
        # 🔴 P5：作废就必须**停下**，不能只打印然后接着跑——后面的判定会拿废数据算出
        #    一个看起来正常的结论（2026-09-19 H2 就是这么废的，当时只打印没停）。
        info["void"] = True
        print("     🔴 这一臂的 logcat 里装载路径 = %s ⇒ 不是单变量，作废" % info.get("paths"))
        print("     ⇒ 原因几乎总是「marker 在装载未完成时被改动」。重跑本臂前先确认没有别的进程在动 marker。")
    elif info.get("load_n") != 8:
        info["void"] = True
        print("     🔴 只读到 %s 段装载（应为 8）⇒ 本臂数据不完整，作废" % info.get("load_n"))
    else:
        print("     路径 %s｜八段装载合计 %.2f s（%d 段）"
              % (info["paths"][0], info["load_total_s"], info["load_n"]), flush=True)
    return info


def rollback(dev, why, st):
    print("\n🔴 回滚：%s" % why, flush=True)
    done = st.get("done", [])
    for act in reversed(done):
        try:
            if act == "marker":
                dev_marker(dev, MMAP_MARKER, False)
            elif act == "apk":
                # 回滚一律装**现网原版**（不是"上次拉到的那版"——第二次插线时那版就是新 APK）
                bak = st.get("apk_production") or st.get("apk_backup")
                if bak and os.path.isfile(bak):
                    subprocess.run([dev.ADB, "install", "-r", bak], timeout=900)
                    print("  已装回 %s" % os.path.basename(bak))
            elif act == "contract":
                bak = st.get("contract_backup")
                if bak and os.path.isfile(bak):
                    S.push_contract(dev, bak, "rollback")
                    print("  已推回原契约")
            elif act == "part1a":
                push_part1a(dev, st.get("rollback_src") or OLD_PART1A)
                print("  已推回现网 part1a")
        except Exception as exc:                        # 回滚不许中途炸掉（#171 教训）
            print("  ⚠️ 回滚步骤 %s 失败：%s" % (act, exc))
    dev.sh("am force-stop %s || true" % PKG)
    print("已回到已知良好态。", flush=True)


def patch_contract(src, dst, new_sha, new_bytes):
    """把契约里所有指向 part1a 的 `sha256`/`size_bytes` 换成新文件的。

    🔴 契约里的 sha256 是**执行时完整性检查**（`ZImageQnnContract::validateSize`
       在 QNN 装载前现算并比对），不是溯源信息 ⇒ **换了 .bin 不改契约，app 直接拒绝装载**。
       part1a 在契约里出现 5 次（基准 1 + 四个比例各 1，都指向同一个多图文件）。
    """
    j = json.load(open(src, encoding="utf-8"))
    n = 0

    def fix(models):
        nonlocal n
        for e in models:
            if e.get("context_binary") == DEV_PART1A_REL:
                e["sha256"] = new_sha
                e["size_bytes"] = new_bytes
                n += 1

    fix(j.get("models", []))
    for v in j.get("size_variants", {}).values():
        fix(v.get("models", []))
    if n != 5:
        raise SystemExit("🔴 契约里 part1a 条目 %d 处（应为 5：基准 + 四比例）⇒ 结构与预期不符，停" % n)
    json.dump(j, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  ✅ 契约已改 %d 处（sha256 + size_bytes）" % n, flush=True)
    return dst


def ensure_rollback_source(dev, st):
    """覆盖设备上的 part1a **之前**，先确定手里有一份能装回去的原件。

    🔴 约束 11 第 2 条：2026-08-17 那次 `install -r` 没存档，丢掉了唯一能回答
       「是不是我弄坏的」的证据。这里不接受"我以为宿主那份就是设备那份"——
       用设备**实测** sha256 说话；不一致就把设备那份拉回来（3.15 GB，走
       `exec-out`，不在设备上再占一份空间）。
    """
    if st.get("rollback_src_ok"):
        return
    print("  核对回滚源（设备端算 sha256，约 1 分钟）……", flush=True)
    dsha = dev.sh("run-as %s sh -c 'sha256sum %s' " % (PKG, DEV_PART1A)).strip().split()[0]
    hsha = D.sha256(OLD_PART1A) if os.path.isfile(OLD_PART1A) else ""
    print("  设备 %s…｜宿主 aspect_mg %s…" % (dsha[:16], (hsha or "（缺）")[:16]), flush=True)
    if dsha and dsha == hsha:
        st["rollback_src"] = OLD_PART1A
        st["rollback_src_ok"] = True
        print("  ✅ 宿主原件与设备逐字节相同 ⇒ 可作回滚源", flush=True)
        save(st)
        return
    bak = os.path.join(OUT, "part1a_mg_device_backup.bin")
    print("  ⚠️ 不一致（或宿主缺件）⇒ 先把设备那份拉回宿主（约 3.15 GB，数分钟）", flush=True)
    with open(bak, "wb") as f:
        r = subprocess.run([dev.ADB, "exec-out", "run-as", PKG, "cat", DEV_PART1A],
                           stdout=f, timeout=5400)
    if r.returncode != 0:
        raise SystemExit("🔴 拉取设备 part1a 失败 rc=%d" % r.returncode)
    got = D.sha256(bak)
    if got != dsha:
        raise SystemExit("🔴 拉回来的 sha256 %s… ≠ 设备 %s… ⇒ 备份不可信，停"
                         % (got[:16], dsha[:16]))
    st["rollback_src"] = bak
    st["rollback_src_ok"] = True
    print("  ✅ 已存档 %.3f GB 且 sha256 与设备一致" % (os.path.getsize(bak) / 1e9), flush=True)
    save(st)


def push_part1a(dev, local):
    """推 2.4 GB context 到 app 私有目录：先 push 暂存区，再设备本地 cat（adb shell 非二进制安全）。"""
    tmp = "/data/local/tmp/p2s6_part1a.bin"
    dev.sh("rm -f %s || true" % tmp)
    subprocess.run([dev.ADB, "push", local, tmp], check=True, timeout=3600)
    n_host = os.path.getsize(local)
    n_tmp = int(dev.sh("stat -c %%s %s" % tmp).strip())
    if n_tmp != n_host:
        raise SystemExit("🔴 push 后字节数不符：%d vs %d" % (n_tmp, n_host))
    dev.sh("run-as %s sh -c 'cat %s > %s'" % (PKG, tmp, DEV_PART1A))
    back = int(dev.sh("run-as %s stat -c %%s %s" % (PKG, DEV_PART1A)).strip())
    if back != n_host:
        raise SystemExit("🔴 落地字节数不符：%d vs %d" % (back, n_host))
    # 🔴 P4：字节数相同 ≠ 内容相同。而契约里钉着 sha256，app 装载前会现算比对 ——
    #    等它在装载时报 "SHA-256 mismatch" 才发现，已经白烧一次启动 + 校验（约 6 分钟）。
    #    在这里现算一次（设备约 62 MiB/s ⇒ 2.88 GB 约 46 秒）便宜得多。
    print("  核对落地内容（设备端算 sha256，约 1 分钟）……", flush=True)
    dsha = dev.sh("run-as %s sh -c 'sha256sum %s'" % (PKG, DEV_PART1A), timeout=1800).strip().split()[0]
    hsha = D.sha256(local)
    if dsha != hsha:
        raise SystemExit("🔴 落地 sha256 %s… ≠ 宿主 %s… ⇒ 推送损坏，停" % (dsha[:16], hsha[:16]))
    dev.sh("rm -f %s || true" % tmp)
    print("  ✅ part1a 已换：%d 字节，sha256 %s… 与宿主一致" % (back, dsha[:16]), flush=True)
    return hsha


# ---------------- 主流程 ----------------

def main():
    if "--run" not in sys.argv and "--rollback" not in sys.argv:
        return check()
    import lab_dev as dev
    st = load()
    if "--rollback" in sys.argv:
        rollback(dev, "手工触发", st)
        return 0
    if check() != 0:
        return 1
    phase = int(sys.argv[sys.argv.index("--phase") + 1]) if "--phase" in sys.argv else 1
    os.makedirs(OUT, exist_ok=True)
    dev.require_online("p2s6")
    print("\n⚠️ 本脚本全程占用设备。**结束前请勿拔线。**", flush=True)
    dev.sh("am force-stop %s || true" % PKG)
    time.sleep(5)
    base_c = st.get("base_c") or D.npu_c(dev)
    st["base_c"] = base_c
    st.setdefault("done", [])
    print("热基线 NPU %.1f C" % base_c, flush=True)
    # 🔴 P6：时间预估必须把**冷却**算进去。2026-09-19 我按"每臂 6 分钟"估，实际
    #    冷却 1 s → 132 s → 284 s **逐臂递增**（机器越跑越热），M1 比我说的晚了 5 分钟，
    #    直接卡到用户的出门时间。每臂 = 启动校验 170 s + 生成 ~160 s + 冷却（递增）。
    n_arms = (2 if "--quick" in sys.argv else 4) + (0 if "--quick" in sys.argv else 1)
    cool = [0, 130, 280, 300, 300][:n_arms]
    est = sum(330 + c for c in cool) / 60.0
    print("⏱️ 预估：%d 臂 × (启动校验 170 s + 生成 ~160 s) + 冷却 %s s ⇒ **约 %.0f 分钟**"
          % (n_arms, "/".join(str(c) for c in cool), est), flush=True)
    if "--quick" not in sys.argv:
        print("   再加阶段 3（推 2.88 GB + 设备端 sha256 + 改契约 + A 臂出图）约 12 分钟", flush=True)
    print("   ⚠️ 冷却是纪律不是等待（#141 热漂移 ±16%），不能省。", flush=True)

    try:
        # ---- 阶段 1：铁律 1/2 ----
        if phase <= 1:
            print("\n== 阶段 1：存档 APK%s ==" % ("" if "--no-baseline" in sys.argv else " + 现网基线"),
                  flush=True)
            path = dev.sh("pm path %s" % PKG).strip().replace("package:", "").splitlines()[0].strip()
            # 🔴 P1：**存档文件名按 .so 指纹区分，绝不覆盖**。
            #    第二次插线时设备上装的已经是新 APK，原来的固定文件名会被新 APK 覆盖
            #    ⇒ 抹掉**全仓库唯一**的那份现网 APK（f68edf747d729c28），回滚源就没了。
            #    （约束 11 第 2 条：备份的意义是"能回答是不是我弄坏的"，被覆盖就等于没备份。）
            tmp = os.path.join(OUT, "_pulled.apk")
            subprocess.run([dev.ADB, "pull", path, tmp], check=True, timeout=900)
            fp = so_sha(tmp)[:16]                 # 仍用于 P8「要不要重装」的判断
            bak = os.path.join(OUT, "installed_%s.apk" % apk_sha(tmp)[:16])
            if os.path.isfile(bak):
                os.remove(tmp)
                print("  ✅ 设备上这版已有存档（%s）⇒ 不重复存" % os.path.basename(bak), flush=True)
            else:
                os.replace(tmp, bak)
                print("  ✅ 存档 %.1f MB（%s）" % (os.path.getsize(bak) / 1e6, os.path.basename(bak)), flush=True)
            st["apk_backup"] = bak
            st["device_apk_so"] = fp
            # 现网原版必须一直在，回滚要装的是它。
            # 🔴 按**指纹**在存档目录里找，不按文件名——文件名会变，指纹不会。
            import glob as _glob
            prod = None
            for cand in sorted(_glob.glob(os.path.join(OUT, "installed*.apk"))):
                try:
                    if so_sha(cand).startswith(OLD_SO_SHA):
                        prod = cand
                        break
                except Exception:
                    continue
            if prod:
                st["apk_production"] = prod
                print("  ✅ 现网原版 APK 在档：%s" % os.path.basename(prod), flush=True)
            else:
                print("  🔴 存档目录里**找不到现网原版 APK**（.so %s…）⇒ 回滚将无法还原到现网。"
                      % OLD_SO_SHA, flush=True)
                if "--allow-no-rollback-apk" not in sys.argv:
                    raise SystemExit("🔴 没有现网 APK 回滚源，停（要强行继续加 --allow-no-rollback-apk）")
            if "--no-baseline" in sys.argv:
                # 省一次出图（约 6 分钟）：金标准 sha 已由 09-19 三臂坐实，
                # 铁律 1 改由本次**第一臂**兼任（它等于金标准即证明现状可用）。
                st["B"] = {"sha256": GOLDEN_SHA, "tag": "GOLDEN(2026-09-19 三臂坐实)"}
                print("  ⏭️ 跳过单独基线：用金标准 %s… 作绝对参照，铁律 1 由第一臂兼任"
                      % GOLDEN_SHA[:12], flush=True)
            else:
                b = S.gen(dev, "s6_B", base_c=base_c)
                st["B"] = b
                if not b["size_ok"]:
                    raise SystemExit("🔴 基线出图尺寸就不对，现状不可用 ⇒ 停（铁律 1）")
            save(st)

        # ---- 阶段 2：装新 APK → R/M/R/M ----
        if phase <= 2:
            print("\n== 阶段 2：装新 APK（H5 埋点 + H2 开关）→ R/M ==", flush=True)
            # P8：设备上已经是这版就不重装（省约 1 分钟）。比的是 .so 指纹，不是版本号。
            if st.get("device_apk_so") and so_sha(NEW_APK).startswith(st["device_apk_so"]):
                print("  ⏭️ 设备上已是本版（.so %s…）⇒ 跳过安装" % st["device_apk_so"], flush=True)
            else:
                subprocess.run([dev.ADB, "install", "-r", NEW_APK], check=True, timeout=1800)
                st["done"].append("apk")
            save(st)
            # --quick：用户时间不够时只跑一对（R1/M1）。代价写在报告里：
            # 配对样本从 2 对降到 1 对 ⇒ H2 的差值**不取中位数**，抗噪能力下降，
            # 结论只能作为"方向"，不得当作最终交付依据。
            arms = (("s6_R1", False), ("s6_M1", True))
            pairs = 1 if "--quick" in sys.argv else int(
                sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 2
            if pairs >= 2:
                arms += (("s6_R2", False), ("s6_M2", True))
            for tag, on in arms:
                if on and "marker" not in st["done"]:
                    st["done"].append("marker")
                st[tag] = arm(dev, tag, base_c, on)
                save(st)
                if st[tag]["sha256"] != st["B"]["sha256"]:
                    raise SystemExit("🔴 G0 数值门：%s 出图 sha256 与基线不同 ⇒ 关闭" % tag)
                if st[tag].get("void"):
                    raise SystemExit("🔴 %s 臂作废（装载路径/段数不对）⇒ 停，不拿废数据往下判" % tag)
            dev_marker(dev, MMAP_MARKER, False)
            verdict_h2(st)
            verdict_h5(st)
            save(st)

        # ---- 阶段 3：H1-A 交付 ----
        if "--quick" in sys.argv:
            print("\n⏭️ --quick：跳过阶段 3（换 context 是交付类动作，时间不足时不做——"
                  "一旦要回滚就得推 3.15 GB，会让用户带着半截状态出门）", flush=True)
        elif phase <= 3:
            if not os.path.isfile(NEW_PART1A):
                print("\n⚠️ noscatA context 未就绪 ⇒ 跳过阶段 3")
            else:
                print("\n== 阶段 3：换 part1a（noscatA）→ A 臂 ==", flush=True)
                ensure_rollback_source(dev, st)
                need = max_spillfill(NEW_INFO)
                cur = dev.sh("run-as %s cat %s 2>/dev/null || echo 0" % (PKG, SF_MARKER)).strip()
                print("  spillFill 需 %d｜设备 marker %s" % (need, cur), flush=True)
                if int(cur or 0) < need:
                    S.write_marker(dev, need)
                    st["sf_marker_changed"] = [cur, need]
                # 🔴 契约必须与 .bin 同步换（否则 sha256 校验直接拒绝装载）。
                #    先把**设备当前那份**取回来做基线与回滚源——不用宿主的副本猜。
                cur_ct = os.path.join(OUT, "contract_device_before.json")
                dev.sh("run-as %s cat %s > /data/local/tmp/_ct_cur.json" % (PKG, DEV_CONTRACT))
                subprocess.run([dev.ADB, "pull", "/data/local/tmp/_ct_cur.json", cur_ct],
                               check=True, timeout=300)
                st["contract_backup"] = cur_ct
                new_ct = patch_contract(cur_ct, os.path.join(OUT, "contract_noscatA.json"),
                                        D.sha256(NEW_PART1A), os.path.getsize(NEW_PART1A))
                push_part1a(dev, NEW_PART1A)
                st["done"].append("part1a")
                save(st)
                S.push_contract(dev, new_ct, "noscatA")
                st["done"].append("contract")
                save(st)
                st["A"] = arm(dev, "s6_A", base_c, st.get("h2_deliver", False))
                save(st)
                verdict_h1a(st)
        final_state(dev, st)
        save(st)
        return 0
    except BaseException as exc:
        print("\n🔴 中断：%s" % exc, flush=True)
        save(st)
        rollback(dev, str(exc), st)
        return 1


def final_state(dev, st):
    """🔴 P11：结束时**逐项读回**设备终态，再说"可以拔线"。

    2026-09-19 我是凭"脚本应该删了 marker"说的可以拔线 —— 那是②推算。
    marker 留在设备上会让用户的 app 走未验证的 mmap 装载路径，**这种话必须由设备来说**。
    """
    print("\n== 设备终态自检（逐项读回，不靠推算）==", flush=True)
    ok = True
    mk = dev.sh("run-as %s sh -c 'ls %s 2>/dev/null | wc -l'" % (PKG, MMAP_MARKER)).strip()
    present = mk.endswith("1")
    # 🔴 P13（2026-09-19 20:57 发现）：marker 在不在，对错取决于 **H2 判了什么**。
    #    H2 🟢 交付时 **marker 就是交付形态**，删掉等于撤销这次优化；
    #    H2 未交付时 marker 必须不在（否则用户会走未验证的装载路径）。
    #    第一版无条件把"marker 在"判成不干净 —— 那会让我在交付成功后去删掉交付物。
    want = bool(st.get("h2_deliver"))
    good = present == want
    print("  %s LOAD_MMAP marker：%s（H2 %s ⇒ 应%s）"
          % ("✅" if good else "🔴", "在" if present else "无",
             "已交付" if want else "未交付", "保留" if want else "删除"))
    ok &= good
    n = dev.sh("run-as %s stat -c %%s %s 2>/dev/null || echo 0" % (PKG, DEV_PART1A)).strip()
    which = {str(os.path.getsize(NEW_PART1A)): "noscatA（新）",
             "3151335424": "现网原版"}.get(n, "未知(%s)" % n)
    print("  ℹ️ part1a 当前 = %s，%s 字节" % (which, n))
    ct = dev.sh("run-as %s stat -c %%s %s 2>/dev/null || echo 0" % (PKG, DEV_CONTRACT)).strip()
    print("  ℹ️ 契约 %s 字节" % ct)
    sf = dev.sh("run-as %s cat %s 2>/dev/null || echo ?" % (PKG, SF_MARKER)).strip()
    print("  ℹ️ SHARE_SPILLFILL = %s" % sf)
    print("\n%s" % ("✅ 设备终态干净，可以拔线了。" if ok else
                    "🔴 设备终态不干净 —— **先处理再拔线**"))
    return ok


def verdict_h2(st):
    r = [st[k] for k in ("s6_R1", "s6_R2") if k in st]
    m = [st[k] for k in ("s6_M1", "s6_M2") if k in st]
    if any(len(x.get("paths", [])) != 1 for x in r + m):
        print("\n🔴 H2 作废：某一臂的装载路径不唯一")
        return
    if len(r) == 1:
        print("\n⚠️ 只有一对样本（--quick）⇒ 下面的差值没有中位数抗噪，只能当方向，不作交付依据")
    lr = statistics.median([x["load_total_s"] for x in r])
    lm = statistics.median([x["load_total_s"] for x in m])
    ion = statistics.median([x["ion_peak"] for x in m]) - statistics.median([x["ion_peak"] for x in r])
    gain = lr - lm
    print("\n== H2 判定（判据 §7.2 G1，事前锁定）==")
    print("  read 臂装载中位 %.2f s｜mmap 臂 %.2f s ⇒ 省 %.2f s" % (lr, lm, gain))
    print("  ION 峰值差（mmap − read）= %+d MiB（门限 +%d）" % (ion, G1_ION_TOL_MB))
    if ion > G1_ION_TOL_MB:
        v = "🔴 关闭：mmap 反而挤压 ION"
    elif gain >= G1_MMAP_GAIN_S:
        v = "🟢 交付：省 %.2f s ≥ %.1f s" % (gain, G1_MMAP_GAIN_S)
    elif gain > 0:
        v = "🟡 保留不交付：省 %.2f s < %.1f s" % (gain, G1_MMAP_GAIN_S)
    else:
        # gain == 0 也走这里：措辞必须是"未更快"，不能说成"更慢"（约束 2：不超出证据）
        v = "🔴 关闭：mmap %s（%+.2f s）" % ("更慢" if gain < 0 else "未更快", gain)
    print("  ⇒ %s" % v)
    st["h2_gain_s"] = gain
    st["h2_verdict"] = v
    st["h2_deliver"] = v.startswith("🟢")


def verdict_h5(st):
    a = st["s6_R1"]
    ss = a.get("segsplit") or {}
    if not ss:
        print("\n🔴 H5 无数据（logcat 里没有 [segsplit]）")
        return
    print("\n== H5 段内拆时（R1 臂，单位 ms）==")
    print("| 段 | prep | 拷入 | graphExecute | 拷出 | 落库 | 合计 | 非算力 |")
    print("|---|---|---|---|---|---|---|---|")
    tot = ex = 0.0
    for name in sorted(ss):
        prep, tin, e, out, post = ss[name]
        s = prep + tin + e + out + post
        tot += s
        ex += e
        print("| `%s` | %.0f | %.0f | **%.0f** | %.0f | %.0f | %.0f | %.1f%% |"
              % (name, prep, tin, e, out, post, s, 100.0 * (s - e) / max(s, 1e-9)))
    segt = a.get("segtime_total_ms", 0)
    if segt:
        d = abs(tot - segt) / segt
        print("  G2 自证：拆时合计 %.0f vs [segtime] %d ⇒ 差 %.1f%% %s"
              % (tot, segt, 100 * d, "✅ 可用" if d < G2_EXEC_TOL else "🔴 **作废**"))
    st["h5_ok"] = bool(segt) and abs(tot - segt) / max(segt, 1) < G2_EXEC_TOL


def verdict_h1a(st):
    a = st["A"]
    # 🔴 P12（2026-09-19 20:35 自审发现）：对照臂必须与 A 臂**只差 context 这一个变量**。
    #    H2 判 🟢 后 A 臂是开着 marker 跑的（mmap + 新 context），此时拿 R 臂
    #    （read + 旧 context）做对照，差值里会混进 mmap 的 ~17 s ⇒
    #    **把 H2 的功劳记到 H1-A 头上**。按 A 臂实际走的装载路径选同路径的那一臂。
    a_path = (a.get("paths") or ["read-into-vector"])[0]
    pool = [k for k in ("s6_R1", "s6_R2", "s6_M1", "s6_M2") if k in st
            and (st[k].get("paths") or [None])[0] == a_path]
    if not pool:
        raise SystemExit("🔴 找不到与 A 臂同装载路径（%s）的对照臂 ⇒ 速度门不成立" % a_path)
    r = st[pool[0]]
    print("\n== H1-A 交付门（判据 §7.2 G3，事前锁定）==")
    print("  对照臂 = %s（与 A 臂同为 %s，只差 context）" % (r["tag"], a_path))
    # 🔴 P2：速度必须**同日配对**（项目纪律：与同日现网配对；#141 热漂移 ±16%，
    #    跨日的后台状态、电量、温度都不同）。续跑时 state 里可能躺着上一次插线的 R 臂。
    if a.get("day") != r.get("day"):
        raise SystemExit("🔴 R 臂（%s）与 A 臂（%s）不同日 ⇒ 速度门不成立。"
                         "请先在本次插线里重跑一次 R 臂再判。" % (r.get("day"), a.get("day")))
    # 🔴 P3：**证明 A 臂真的装载了新 context**。出图 sha256 相同是**预期**结果，
    #    因此它恰恰无法区分"换成功了"与"根本没换成"——后者会安静地给出一模一样的图和速度，
    #    我会误判成"收益未复现"。`[loadpath]` 行里的字节数是完美区分器：
    #    新 2,877,374,464 vs 旧 3,151,335,424。
    want = os.path.getsize(NEW_PART1A)
    got = a.get("part1a_bytes")
    ok_bytes = got == want
    print("  装载核对：part1a 实际装载 %s 字节（新件 %d）%s"
          % (got if got is not None else "未读到", want, "✅" if ok_bytes else "🔴"))
    if not ok_bytes:
        raise SystemExit("🔴 A 臂装载的不是新 context（或没读到 [loadpath]）⇒ 本臂无意义，停")
    same = a["sha256"] == st["B"]["sha256"]
    print("  G0 数值：出图 sha256 %s 基线 %s" % ("==" if same else "≠", "✅" if same else "🔴"))
    # 🔴 必须用 SSE 的 `generation_time_ms`，不许回落到墙钟：A 变体的 context 小了 274 MB，
    #    而 app 启动时要现算基准文件的 sha256（约 62 MiB/s）⇒ 墙钟里白送约 4.4 s，
    #    那不是提速，会把收益虚报出来。拿不到就报缺失（约束 2：不填假数据）。
    if "generation_s" not in a or "generation_s" not in r:
        raise SystemExit("🔴 缺 generation_time_ms（SSE 没解析到）⇒ 速度门无法判，停")
    ra, aa = r["generation_s"], a["generation_s"]
    print("  速度：%.1f s → %.1f s ⇒ 省 %.1f s（门限 %.1f）%s"
          % (ra, aa, ra - aa, G3_SPEEDUP_S, "✅" if ra - aa >= G3_SPEEDUP_S else "⚠️ 收益未复现"))
    print("  内存：ION 峰值 %d（R 臂 %d，门限 +%d）｜MemAvail 最低 %d（门限 %d）"
          % (a["ion_peak"], r["ion_peak"], G1_ION_TOL_MB, a["avail_low"], G3_AVAIL_LOW_MB))
    ok = same and a["ion_peak"] <= r["ion_peak"] + G1_ION_TOL_MB and a["avail_low"] >= G3_AVAIL_LOW_MB
    print("  ⇒ %s" % ("🟢 交付成立（数值不变 + 内存达标）" if ok else "🔴 不得交付，必须回滚"))
    st["h1a_ok"] = ok
    if not ok:
        raise SystemExit("🔴 H1-A 交付门未过")


if __name__ == "__main__":
    sys.exit(main())
