# -*- coding: utf-8 -*-
"""干跑台：不碰设备，把 `p2_s6_session.py` 的阶段 1~3 完整走一遍。

## 为什么需要它
`p2_s6_session.py` 的**设备路径一次都没执行过**。本项目的教训是：这类脚本靠"读"能查出
路径与顺序问题，但查不出 **shell 引号、判据算术、KeyError、正则对不对**——而那几类恰好
只会在用户插着线的时候爆。⇒ 把 `lab_dev`/`subprocess`/`D.sample_*`/`S.gen` 换成桩，
让整条逻辑真的跑一遍，并**打印每一条会发给设备的命令**供逐条核对。

同时它顺带**端到端测了日志解析**：桩会写出带 `[loadpath]`/`[segsplit]`/`[segtime]` 的假
logcat，`split_of()` 与两个判定函数都吃真数据。

## 它能查什么、不能查什么
✅ 能查：命令成形、阶段顺序、判据算术、回滚路径、正则、KeyError、契约改写是否命中 5 处。
❌ **不能查**：run-as 权限、adb 引号解析、mmap 在真机上到底快不快。
   ⇒ 干跑全过**不等于**设备上能跑通，只是把"一定跑不通"的部分提前排掉。

## 场景
  pass    : 全门通过                      ⇒ 走到「S6 全部阶段结束」
  h2none  : mmap 与 read 一样快           ⇒ H2 判 🔴（不更快），但**不中止**，继续阶段 3
  h2slow  : mmap 反而更慢                 ⇒ H2 判 🔴（更慢）
  h2ion   : mmap 省时间但 ION 涨 300 MiB  ⇒ H2 判 🔴（挤压 ION）
  badsplit: 拆时与 [segtime] 差 20%       ⇒ H5 判「作废」，不中止
  sha     : R1 臂出图 sha 与基线不同      ⇒ G0 中止 + 回滚（APK）
  aslow   : A 臂只快 1 s                  ⇒ 「收益未复现」但仍交付
  asha    : A 臂 sha 不同                 ⇒ 交付门中止 + 回滚（契约→part1a→marker→APK）
  nopush  : 推送"看似成功"但装载的仍是旧件 ⇒ P3 装载核对必须抓住并中止

用法: python scripts/p2_s6_dryrun.py [pass|h2none|h2ion|badsplit|sha|aslow|asha|all]
"""
import io
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CMDS = []
BASE_SHA = "a" * 64
OTHER_SHA = "b" * 64


def stub_devmod():
    m = types.ModuleType("lab_dev")
    m.ADB = r"C:\fake\adb.exe"
    m.SERIAL = "FAKE"

    class DeviceOffline(Exception):
        pass
    m.DeviceOffline = DeviceOffline

    def sh(cmd, timeout=1800):
        CMDS.append(cmd)
        if "pm path" in cmd:
            return "package:/data/app/~~x==/base.apk\n"
        if "cat " in cmd and "part1a" in cmd and ">" in cmd:
            PUSHED[0] = LAST_PUSH[0]                  # 刚才推上去的那个宿主文件已落地
        if "sha256sum" in cmd:
            # 设备上这个文件的哈希 = 最后一次推上去的那个宿主文件的哈希。
            # 没推过时返回 "c"（≠ 宿主 aspect_mg 的 "d" ⇒ 走「拉回设备那份」再校验）。
            import dprime_measure as _D
            return "%s  file\n" % (_D.sha256(PUSHED[0]) if PUSHED[0] else "c" * 64)
        if "ls " in cmd and "wc -l" in cmd:
            return "1\n" if "LOAD_MMAP" in cmd and MARKER[0] else "0\n"
        if "cat" in cmd and "SHARE_SPILLFILL" in cmd:
            return "331415552\n"
        if "stat -c" in cmd:
            return "%d\n" % STAT_RET[0]
        return ""

    m.sh = sh
    m.online = lambda: True
    m.require_online = lambda where="": None
    return m


MARKER = [False]
STAT_RET = [0]
PUSHED = [None]      # 设备上 part1a 现在是哪个宿主文件（None = 原件）
LAST_PUSH = [None]   # 最后一次 adb push 的源文件


def fake_logcat(path, tag, mmap_on, scen):
    """写一份最小但**格式真实**的 logcat：装载标记对 + 段级 + 段内拆时。"""
    segs = ["text_encoder_part%d" % i for i in (1, 2, 3, 4)] + \
           ["transformer_part1a", "transformer_part1b", "transformer_part2a", "transformer_part2b"]
    kind = "mmap" if mmap_on else "read-into-vector"
    # h2none：两臂一样快（mmap 无收益）；h2slow：mmap 反而慢；其余：八段共省 4.0 s
    if scen == "h2none":
        load_ms = 1400
    elif scen == "h2slow":
        load_ms = 1600 if mmap_on else 1400
    else:
        load_ms = 900 if mmap_on else 1400
    t = 1000.0
    lines = []
    # part1a 的字节数 = 设备上那个文件的真实大小 ⇒ 干跑台能真的测到 P3 那道门：
    # scen="nopush" 模拟"推送没生效"（装载的仍是旧件），P3 必须抓住它。
    newp = PUSHED[0] and str(PUSHED[0]).endswith("noscatA.SM8750.bin") and scen != "nopush"
    for s in segs:
        nb = 2877374464 if (s == "transformer_part1a" and newp) else 3151335424
        lines.append("09-19 03:00:00.000 I/QNN: Backend: %10.1fms [ INFO ] "
                     "[loadpath] %s %s (%d bytes)" % (t, kind, s, nb))
        t += load_ms
        lines.append("09-19 03:00:00.000 I/QNN: Backend: %10.1fms [ INFO ] "
                     "QNN App Initialized from Buffer: %s" % (t, s))
        t += 50
    # 步循环：每段 8 步，段级墙钟 = 五段之和（G2 自证应当吻合）
    per = {"transformer_part1a": (60.0, 120.0, 5000.0, 180.0, 40.0),
           "transformer_part1b": (20.0, 60.0, 3200.0, 90.0, 20.0),
           "transformer_part2a": (15.0, 50.0, 3900.0, 70.0, 15.0),
           "transformer_part2b": (15.0, 50.0, 3900.0, 70.0, 15.0)}
    for step in range(8):
        for s, v in per.items():
            lines.append("09-19 03:00:00.000 I/QNN: Backend: %10.1fms [ INFO ] "
                         "[segtime] %s %d ms" % (t, s, int(round(sum(v)))))
            t += sum(v)
    for s, v in per.items():
        tot = [x * 8 for x in v]
        if scen == "badsplit" and s == "transformer_part1a":
            tot[2] *= 1.6                      # 让合计比 [segtime] 大 ⇒ G2 应判作废
        lines.append("09-19 03:00:00.000 I/QNN: Backend: %10.1fms [ INFO ] "
                     "[segsplit] %s n=8 prep %.0f | in %.0f | exec %.0f | out %.0f | post %.0f ms"
                     "（合计 %.0f，非算力占比 9.9%%）"
                     % (t, s, tot[0], tot[1], tot[2], tot[3], tot[4], sum(tot)))
    io.open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def run(scen):
    global CMDS
    CMDS = []
    import importlib
    import dprime_measure as D
    sys.modules["lab_dev"] = stub_devmod()
    import oneshot_mg_session as S
    import p2_s6_session as P
    importlib.reload(P)

    # 🔴 隔离输出目录：否则干跑会删掉真实 state.json、用假 PNG 覆盖当天证据
    #    （`oneshot_dryrun.py` 2026-09-17 踩过，这里直接照抄那条教训）。
    P.OUT = os.path.join(D.RUNS, "p2s6_dryrun", scen)
    P.STATE = os.path.join(P.OUT, "state.json")
    P.RUNS = os.path.join(P.OUT, "runs")
    os.makedirs(P.RUNS, exist_ok=True)
    if os.path.isfile(P.STATE):
        os.remove(P.STATE)
    D.npu_c = lambda dev: 40.0
    D.cooldown = lambda dev, base_c, tol=3.0, max_wait_s=600: 40.0
    # 设备上是 "c"，宿主 aspect_mg 是 "d" ⇒ 认不上 ⇒ 拉回设备那份，
    # 拉回来的副本再算应当是 "c" ⇒ 与设备一致 ⇒ 认领成功。这条路径每个场景都会走到。
    D.sha256 = lambda p: "c" * 64 if "device_backup" in str(p) else "d" * 64

    def gen(dev, tag, w=1024, h=1024, base_c=None, restart=True):
        mmap_on = MARKER[0]
        sha = BASE_SHA
        if scen == "sha" and tag == "s6_R1":
            sha = OTHER_SHA
        if scen == "asha" and tag == "s6_A":
            sha = OTHER_SHA
        gs = 148.0
        if tag == "s6_A":
            gs = 147.0 if scen == "aslow" else 141.7
        ion = 9116 + (300 if (scen == "h2ion" and mmap_on) else 0)
        fake_logcat(os.path.join(P.RUNS, "os_%s_logcat.txt" % tag), tag, mmap_on, scen)
        print("  [桩] 出图 %-6s mmap=%-5s sha=%s… gen=%.1f s" % (tag, mmap_on, sha[:6], gs))
        return {"tag": tag, "wall_s": gs + 12, "sha256": sha, "generation_s": gs,
                "ion_peak": ion, "avail_low": 828, "n": 300, "size_ok": True,
                "png_wh": [1024, 1024], "img": "fake.png"}
    S.gen = gen
    S.push_contract = lambda dev, local, tag: CMDS.append(
        "PUSH_CONTRACT %s (%s)" % (os.path.basename(local), tag))
    S.write_marker = lambda dev, n: CMDS.append("WRITE_SF_MARKER %d" % n)

    # marker 开关要让桩的 `ls` 与 gen 看到同一个状态
    real_marker = P.dev_marker

    def dev_marker(dev, path, on):
        MARKER[0] = on
        return real_marker(dev, path, on)
    P.dev_marker = dev_marker

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    def sub_run(cmd, **kw):
        CMDS.append("SUBPROC " + " ".join(str(c) for c in cmd[:4]))
        if "push" in cmd:
            STAT_RET[0] = os.path.getsize(cmd[-2])   # 桩：设备上的字节数 = 刚推上去那个文件的
            LAST_PUSH[0] = cmd[-2]
        if "pull" in cmd:
            dst = cmd[-1]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if str(dst).endswith(".json"):
                # 契约：给一份真的，patch_contract 才能真的被测到
                import shutil
                shutil.copy(S.MG_CONTRACT, dst)
            else:
                io.open(dst, "wb").write(b"x" * (11 << 20))
        if "exec-out" in cmd:
            kw.get("stdout").write(b"x" * 4096)   # 回滚源备份（内容无所谓，长度由 D.sha256 桩兜住）
        return R()
    P.subprocess.run = sub_run
    P.os.path.getsize = (lambda real: (lambda p: 2877374464 if str(p).endswith("noscatA.SM8750.bin")
                                       else real(p)))(os.path.getsize)
    STAT_RET[0] = 2877374464
    PUSHED[0] = None
    LAST_PUSH[0] = None
    P.check = lambda: 0                      # 阶段 0 已由真脚本单独验证过
    # 桩 so_sha：模拟**第二次插线**（设备上已经是新 APK），并在存档目录里预置一份
    # "现网原版"，用来验证 apk_production 是按指纹找到的、回滚装的是它而不是新 APK。
    prod = os.path.join(P.OUT, "installed_before_p2s6.apk")
    os.makedirs(P.OUT, exist_ok=True)
    io.open(prod, "wb").write(b"PROD")
    P.so_sha = lambda q: (P.OLD_SO_SHA + "0" * 48) if "installed_before" in str(q) else "4ea8568a8a2af7bc" + "0" * 48

    print("\n=== 场景 %s ===" % scen)
    sys.argv = ["x", "--run"]
    rc = P.main()
    print("  返回码 %d" % rc)
    dev_cmds = [c for c in CMDS if not c.startswith(("SUBPROC", "PUSH_CONTRACT", "WRITE_SF"))]
    print("  发给设备的命令 %d 条，其余操作 %d 条" % (len(dev_cmds), len(CMDS) - len(dev_cmds)))
    return rc


EXPECT = {"pass": 0, "h2none": 0, "h2slow": 0, "h2ion": 0, "badsplit": 0,
          "aslow": 0, "sha": 1, "asha": 1, "nopush": 1}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    scens = list(EXPECT) if which == "all" else [which]
    bad = 0
    for s in scens:
        rc = run(s)
        ok = rc == EXPECT.get(s)
        print("  %s 期望返回码 %s，实得 %d" % ("✅" if ok else "🔴", EXPECT.get(s), rc))
        bad += 0 if ok else 1
    print("\n%s" % ("✅ 干跑全过（不等于设备能跑通）" if not bad else "🔴 %d 个场景不符预期" % bad))
    sys.exit(1 if bad else 0)
