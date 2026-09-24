# -*- coding: utf-8 -*-
"""干跑台：不碰设备，把 `oneshot_mg_session.py` 的 phase 1~4 完整走一遍。

## 为什么需要它
那个脚本的**设备路径从未执行过**。今天光靠「读」就查出 8 个真问题
（路径、force-stop、通配符、H4 口径…），但**读不出来的还有一类：
shell 引号、执行顺序、判据算术、KeyError**。
⇒ 把 `lab_dev` 与 `subprocess` 换成桩，让整条逻辑真的跑一遍，
   并**打印每一条会发给设备的命令**供逐条核对。

## 它能查什么、不能查什么
✅ 能查：命令字符串是否成形、阶段顺序、判据算术、回滚路径、KeyError、格式化错误。
❌ **不能查**：设备上的真实行为（ION 到底降不降、run-as 权限、adb 引号解析）。
   ⇒ 干跑全过**不等于**设备上能跑通，只是把「一定跑不通」的那部分提前排掉。

## 场景
  pass  : spill-fill 省 800 MiB，mg 各门全过  ⇒ 应走到「已交付」
  mid   : 省 450 MiB（落在 375~600）          ⇒ 应仍然交付，靠 H1/H2 兜底
  fail  : 省 200 MiB（< 375）                 ⇒ 应中止并回滚
  sha   : 关臂 sha256 与基线不同              ⇒ 应在 G2 中止并回滚
  size  : 某个比例出图尺寸回落到 1024          ⇒ 应在 H5 中止并回滚

用法: python scripts/oneshot_dryrun.py [pass|mid|fail|sha|size|all]
"""
import io
import os
import struct
import sys
import types
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CMDS = []


def make_png(path, w, h):
    """写一张最小的合法 PNG（只需要头里的宽高是真的）。"""
    raw = b"".join(b"\x00" + b"\x7f" * (3 * w) for _ in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def build_stub_devmod(scen, sizes_seen):
    """假的 lab_dev。所有 dev.sh 命令都记录下来，并按模式返回可信的回值。"""
    m = types.ModuleType("lab_dev")
    m.ADB = r"C:\fake\adb.exe"
    m.SERIAL = "FAKESERIAL"

    class DeviceOffline(Exception):
        pass
    m.DeviceOffline = DeviceOffline

    def sh(cmd, timeout=1800):
        CMDS.append(cmd)
        if "pm path" in cmd:
            return "package:/data/app/~~x==/base.apk\n"
        if "stat -c" in cmd:
            # 契约 / .bin 的字节数：返回「期望值」让流程继续
            import oneshot_mg_session as O
            for n in O.referenced_files(O.MG_CONTRACT):
                if n in cmd:
                    return "%d\n" % SIZES_OF_CONTRACT[n]
            # 🔴 按「最近一次 push 的那个契约」返回大小。
            #    第一版固定返回 mg 契约的大小 ⇒ 回滚时推 single 契约会被判成
            #    「字节不符」。那本身是桩的错，但它**顺带查出真代码的一个缺陷**：
            #    push_contract 抛的是 SystemExit，而 rollback 的 except Exception
            #    抓不住它 ⇒ 回滚会在中途炸掉、留下半截状态。真代码已修。
            if "final_qnn_contract.json" in cmd or "_ct_" in cmd:
                return "%d\n" % os.path.getsize(LAST_PUSHED[0] or CURRENT_CONTRACT[0])
            return "1\n"
        if "SHARE_SPILLFILL" in cmd:
            # 2026-09-17：marker 内容 = 组大小。写（echo N >）记下来，读（cat）原样返回，
            # 以便真代码的「读回核对」走通；scen=marker_bad 模拟写进去读回不对。
            import re as _re
            mw = _re.search(r"echo (\d+) >", cmd)
            if mw:
                MARKER_VAL[0] = mw.group(1)
                return ""
            if "cat " in cmd:
                return ("garbage\n" if scen == "marker_bad" else MARKER_VAL[0] + "\n")
            if "ls -l" in cmd:
                return "-rw------- 1 u0 u0 10 2026-09-07 10:00 SHARE_SPILLFILL\n"
            return ""
        if "thermal_zone" in cmd:
            return "40000\n40000\n"
        if "grep IonTotalUsed" in cmd or "grep MemAvailable" in cmd:
            return ""
        if "cat" in cmd and "/data/local/tmp/_cur.json" in cmd:
            return ""
        return ""

    def push(a, b):
        CMDS.append("PUSH %s -> %s" % (os.path.basename(str(a)), b))
        if str(a).endswith(".json"):
            LAST_PUSHED[0] = str(a)

    def pull(a, b):
        CMDS.append("PULL %s -> %s" % (a, os.path.basename(str(b))))
        os.makedirs(os.path.dirname(b), exist_ok=True)
        with open(b, "wb") as f:
            f.write(b"x" * (11 << 20))       # 让「>10 MiB」的存档检查通过

    m.sh, m.push, m.pull = sh, push, pull
    m.online = lambda: True
    m.require_online = lambda where="": None
    return m


MARKER_VAL = [""]
SIZES_OF_CONTRACT = {}
CURRENT_CONTRACT = [None]
LAST_PUSHED = [None]


def run_scenario(scen):
    global CMDS
    CMDS = []
    import importlib
    import dprime_measure as D
    sys.modules["lab_dev"] = build_stub_devmod(scen, [])
    import oneshot_mg_session as O
    importlib.reload(O)
    # 🔴 2026-09-17 补：隔离输出目录。第一版直接用 O.OUT / O.RUNS ⇒
    #    **每跑一次干跑就删掉真实的 state.json**（下面有 os.remove），
    #    还会用假 PNG 覆盖 `scratch_runs/os_*.png`、往真实 oneshot/ 里写 11 MiB 的假「APK 存档」。
    #    真实会话中途想重跑干跑验证修复时，这会毁掉续跑所需的 state 与当天的证据。
    O.OUT = os.path.join(O.RUNS, "oneshot_dryrun")
    O.STATE = os.path.join(O.OUT, "state.json")
    O.RUNS = os.path.join(O.OUT, "runs")
    os.makedirs(O.RUNS, exist_ok=True)
    # 读 4 个 transformer .bin 的元数据要一两分钟，且会被下面的 subprocess 桩截走 ⇒ 桩成已知值。
    # 真值（2026-09-17 由 qnn-context-binary-utility 现读 deliver_mg/）= 331,415,552 B。
    O.sf_group_bytes = lambda contract, local_dir: 331415552
    MARKER_VAL[0] = ""

    CURRENT_CONTRACT[0] = O.MG_CONTRACT
    import json
    c = json.load(open(O.MG_CONTRACT, encoding="utf-8"))
    for e in c["models"]:
        SIZES_OF_CONTRACT[e["actual_filename"]] = e.get("size_bytes", 1)
    for sv in c.get("size_variants", {}).values():
        for e in sv.get("models", []):
            SIZES_OF_CONTRACT[e["actual_filename"]] = e.get("size_bytes", 1)

    # --- 桩：ION / 温度 / 冷却 ---
    plan = {"g0_live": (9116, 828), "sf_off": (9116, 828)}
    plan["sf_on"] = {"pass": (8316, 1628), "mid": (8666, 1278),
                     "fail": (8916, 1028)}.get(scen, (8316, 1628))
    for w, h in O.SIZES:
        plan["mg_%dx%d" % (w, h)] = (8649, 1290)
    D.npu_c = lambda dev: 40.0
    D.cooldown = lambda dev, base_c, tol=3.0, max_wait_s=600: 40.0

    def sample_start(dev, tag):
        CMDS.append("SAMPLE_START %s" % tag)

    def sample_stop(dev, tag):
        ion, av = plan.get(tag, (8649, 1290))
        # 平台期中位数：默认 = 单秒最低 + 300（高于 resume 基线 1009）；smoke_h2bad 让 mg 低于基线
        med = 900 if (scen == "smoke_h2bad" and tag.startswith("mg_")) else av + 300
        return {"n": 300, "ion_peak": ion, "avail_low": av, "avail_plateau_med": med}
    D.sample_start, D.sample_stop = sample_start, sample_stop

    # --- 桩：外部脚本 ---
    real_run = O.subprocess.run
    BASE_SHA = ["deadbeef"]

    def fake_run(cmd, **kw):
        s = " ".join(str(x) for x in cmd)
        CMDS.append("RUN " + s)
        rc = 0
        if "app_generate.sh" in s:
            tag = cmd[2]
            if scen == "crash" and tag == "os_mg_1184x896":
                # 模拟交付 mg 之后 app 被系统杀掉（#161 的原样）：curl 断开、没有 PNG
                class R:
                    returncode = 18
                    stdout = "FAIL: curl rc=18（传输中断）"
                    stderr = ""
                return R()
            w, h = 1024, 1024
            if len(cmd) >= 7:
                w, h = int(cmd[5]), int(cmd[6])
            if scen == "size" and (w, h) == (1280, 720):
                w, h = 1024, 1024          # 模拟后端静默回落
            img = os.path.join(O.RUNS, "%s.png" % tag)
            os.makedirs(O.RUNS, exist_ok=True)
            make_png(img, w, h)
            if scen == "sha" and tag == "os_sf_off":
                make_png(img, w, h + 8)     # 造一张不同的图 => sha256 不同
            sse = os.path.join(O.RUNS, "%s.sse" % tag)
            io.open(sse, "w", encoding="utf-8").write('"generation_time_ms": 156500')
        elif "aspect_preflight" in s or "stage_mg_delivery" in s:
            class R:
                returncode = 0
                stdout = "✅ ok"
                stderr = ""
            return R()
        elif "install" in s:
            class R:
                returncode = 0
                stdout = "Success\n"
                stderr = ""
            return R()

        class R:
            returncode = rc
            stdout = "ok"
            stderr = ""
        return R()
    O.subprocess.run = fake_run

    st = O.STATE
    if os.path.isfile(st):
        os.remove(st)
    old_argv = sys.argv[:]
    sys.argv = ["oneshot", "--run"]
    if scen.startswith(("resume", "smoke")):
        # 续跑场景：造一份「阶段 1、2 已完成」的 state（与 2026-09-17 真实 state 同构），
        # 然后 `--run --phase 3`。g0 的 sha 必须等于桩生成的 1024 假图，否则 H3 必挂。
        import hashlib
        probe = os.path.join(O.RUNS, "_probe.png")
        make_png(probe, 1024, 1024)
        sha = hashlib.sha256(open(probe, "rb").read()).hexdigest()
        bak = os.path.join(O.OUT, "BACKUP_apk_fake.apk")
        open(bak, "wb").write(b"x" * (11 << 20))
        one = {"sha256": sha, "ion_peak": 9101, "avail_low": 742, "avail_plateau_med": 1009, "n": 300,
               "wall_s": 300, "png_wh": [1024, 1024], "size_ok": True}
        state = {"apk_installed": True, "g0": dict(one), "apk_backup": bak, "base_c": 38.0,
                 "sf_off": dict(one, ion_peak=9129), "sf_on": dict(one, ion_peak=8395),
                 "sf_save": 734, "sf_dt": -0.5}
        if scen == "resume_bad":
            state["sf_save"] = 200                 # 实验 1 没过 ⇒ 续跑必须拒绝
        os.makedirs(O.OUT, exist_ok=True)
        json.dump(state, open(st, "w"))
        sys.argv = ["oneshot", "--run", "--phase", "3"]
        if scen.startswith("smoke"):
            sys.argv.append("--smoke")
    print("\n" + "=" * 78)
    print("场景 %s" % scen)
    print("=" * 78)
    try:
        rc = O.guarded_main()          # 🔴 必须走真实入口，否则测不到入口兜底
    except SystemExit as e:
        rc = e.code
    except Exception as e:
        import traceback
        traceback.print_exc()
        rc = "EXC:%s" % e
    finally:
        sys.argv = old_argv
        O.subprocess.run = real_run
    print("\n--- 发给设备的命令共 %d 条，抽样 ---" % len(CMDS))
    for c in CMDS[:6] + ["..."] + CMDS[-6:]:
        print("   " + str(c)[:150])
    if scen == "smoke":
        n_mg = sum(1 for c in CMDS if "app_generate.sh" in c and " os_mg_" in c)
        print("   smoke 检查：mg 出图次数 %d（应为 1）%s" % (n_mg, "✅" if n_mg == 1 else "🔴"))
        if n_mg != 1:
            return "SMOKE_COUNT"
    if scen == "resume":
        # 续跑必须「先装新 APK，再部署」，且部署参数不得是 /d/ 形式（09-17 翻车点）
        i_inst = next((i for i, c in enumerate(CMDS) if "RUN " in c and " install " in c), -1)
        i_dep = next((i for i, c in enumerate(CMDS) if "deploy_aspect.sh" in c), -1)
        dep = CMDS[i_dep] if i_dep >= 0 else ""
        ok = 0 <= i_inst < i_dep and " /d/" not in dep and " /c/" not in dep and "D:/" in dep
        print("   续跑检查：装 APK 在部署之前 %s ｜ 部署参数为 D:/ 形式 %s"
              % ("✅" if 0 <= i_inst < i_dep else "🔴", "✅" if ok else "🔴"))
        print("   部署命令: " + dep[:220])
        if not ok:
            return "ORDER_OR_PATH"
    return rc


# crash：交付 mg 后出图本身失败（app 被杀）⇒ 应被入口兜底并回滚，rc=2
EXPECT = {"pass": 0, "mid": 0, "fail": 2, "sha": 2, "size": 2, "crash": 2,
          # resume：2026-09-17 阶段 3 翻车回滚后的续跑（--phase 3）⇒ 应重装新 APK 并交付
          # resume_bad：state 里实验 1 没过 ⇒ 续跑必须拒绝（rc=1），不得绕过 G4
          "resume": 0, "resume_bad": 1,
          # marker_bad：marker 读回不对（app 会因此不共享）⇒ 必须在阶段 2 就回滚，不得带病进入 G4
          "marker_bad": 2,
          # smoke：2026-09-17 用户决定的「改 H2 + 只验 1:1」续跑 ⇒ 应只出 1 张 mg 图并交付
          # smoke_h2bad：mg 平台期中位 900 < 现网 1009 ⇒ H2' 不过 ⇒ 回滚
          "smoke": 0, "smoke_h2bad": 2}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    scens = list(EXPECT) if which == "all" else [which]
    bad = []
    got = {}
    for s in scens:
        rc = run_scenario(s)
        got[s] = rc
        if rc != EXPECT[s]:
            bad.append((s, rc, EXPECT[s]))
    print("\n" + "=" * 78)
    print("干跑结果（rc 应符合预期）")
    for s in scens:
        print("  %-6s rc=%-8s 期望 %s  %s"
              % (s, got[s], EXPECT[s], "✅" if got[s] == EXPECT[s] else "🔴"))
    print("\n🔴 干跑全过不等于设备上跑得通 —— 它只排掉「一定跑不通」的那部分。")
    sys.exit(1 if bad else 0)
