# -*- coding: utf-8 -*-
"""一次插线跑完：验证 spill-fill → 交付 mg → 五比例验收（`scripts/EXP_PLAN_SPILLFILL.md`）。

## 设计目标
用户出差回来只插一次线。**能在宿主判死的都已判死**（`aspect_preflight.py` 全过、
交付物 sha256 已逐个核对）。本脚本只做必须在设备上做的事，且：

  · 每一步判据**事前锁定**，写在 `EXP_PLAN_SPILLFILL.md` §4，本脚本只读不改；
  · 任何一门不过 ⇒ **自动回滚到已知良好状态**，不留半截；
  · 通过 ⇒ **直接交付**，不需要第二次插线。

## 阶段（`--phase` 可从任一阶段续跑）
| 阶段 | 内容 | 不过怎么办 |
|---|---|---|
| 0 | 宿主自检（不碰设备）| 停 |
| 1 | 铁律 1/2：现网基线出图 + 存档 APK 与契约 | 停 |
| 2 | **实验 1**：spill-fill A/B（single 形态，单变量）| 回滚 APK，停 |
| 3 | **交付 mg**：推 8.26 GiB + 换契约 + 开 marker | 回滚契约与 APK |
| 4 | 五比例验收 | 回滚契约与 APK |

## 🔴 全程纪律
  · ION 每秒采样（抽稀会漏掉真峰值，#161 实测漏成 7557 vs 真值 9486）
  · 每个计时臂前等 NPU 回到基线 +3 °C（热漂移 +16%，#141）
  · 一切以字节数/产物为准，不看日志内容（报错会说谎）
  · 设备占用与释放**由调用方明确告知用户**（MAINLINE §7.0）

用法:
  python scripts/oneshot_mg_session.py --check          # 纯宿主
  python scripts/oneshot_mg_session.py --run            # 全流程
  python scripts/oneshot_mg_session.py --run --phase 3  # 从阶段 3 续
  python scripts/oneshot_mg_session.py --rollback       # 紧急回滚
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D          # 复用已验证的 ION 采样 / NPU 冷却 / sha256

REPO = D.REPO
RUNS = D.RUNS
PKG = D.PKG
APK = D.APK
OUT = os.path.join(RUNS, "oneshot")
STATE = os.path.join(OUT, "state.json")

MG_DIR = os.path.join("D:", os.sep, "ZImage_Work", "deliver_mg")
MG_CONTRACT = os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.mg.json")
SINGLE_CONTRACT = os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.single.json")
MARKER = "files/models/ZIMAGE/SHARE_SPILLFILL"      # marker 在 model_dir_ 下
BIN_DIR = "files/models/ZIMAGE/models"              # 🔴 .bin 在 models/ 子目录
DEV_TMP = "/data/local/tmp/aspect"                  # deploy_aspect.sh 的暂存区
DEV_CONTRACT = "files/models/ZIMAGE/final_qnn_contract.json"
ADB_SH = os.path.join(REPO, "scripts", "deploy_aspect.sh")

SEED = 42
# 🔴 必须与 app_generate.sh 的默认 prompt **逐字一致**：宿主所有 FP32 参考
#    都用它算（#160），换词则一切 PSNR 比较失去意义。
DEFAULT_PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"
SIZES = [(1024, 1024), (1184, 896), (896, 1184), (1280, 720), (720, 1280)]

# ---- 判据（EXP_PLAN_SPILLFILL.md §4，事前锁定，本脚本不得改）----
# G4 三档（2026-09-07 **执行前**修订，理由见 EXP_PLAN_SPILLFILL.md「判据修订」）：
#   >=600 直接交付｜375~600 仍交付、由真实 mg 运行的 H1/H2 兜底｜<375 中止。
#   原来的单一 600 是我拍的（「预测 854 打三折」），会把够用的 375~600 区间
#   也砍掉 ⇒ 白白浪费一次插线。
G4_COMFORT = 600         # MiB
G4_MIN_SAVE = 375        # MiB = mg 的实测缺口（9486 − 9111）
G5_MAX_SLOW = 5.0        # s
H2_MIN_AVAIL = 800       # MiB，mg 交付后峰值 MemAvail 下限
H4_MAX_GIB = 13.0        # 设备占用上限

# 阶段 4 的冷却目标（**程序参数，不是判据**；H1~H5 都不含耗时）。
# 🔴 2026-09-17 实测：热基线取自开机空闲的 38.0 C，但出过一张图后插着 USB 的平台温度
#    是 43.0 / 43.8 C，两次都等满 600 s 仍降不到 41 C ⇒ 每张白等 10 分钟。
#    阶段 4 只看内存/sha/尺寸，冷却只为防 #141 那种连续出图过热
#    ⇒ 以实测平台 43.0 C 为基线（+3 C 容差 = 46 C）。
P4_WARM_BASE_C = 43.0


def mixed(p):
    """Windows 路径 -> 「盘符 + 正斜杠」形式（`D:\\a\\b` -> `D:/a/b`，即 `cygpath -m`）。

    🔴 `deploy_aspect.sh` 里是 `for f in "$SRC"/*.bin`、`stat -c %s "$f"`——
       传反斜杠路径进去**通配符不展开**（`\\` 在模式里是转义符）。

    🔴🔴 2026-09-17 设备实测翻车：上一版转成 `/d/a/b`（POSIX 形式），
       实验 1 全过之后，交付第一步 `adb push` 报
       `cannot stat '/d/ZImage_Work/deliver_mg/...': No such file or directory` 并回滚。
       原因：`deploy_aspect.sh` 开头 `export MSYS2_ARG_CONV_EXCL='*'`（为了不让
       `/data/local/tmp` 被改写）⇒ **`/d/...` 也不再被转换**，原样交给
       **Windows 原生程序** adb.exe 和 python.exe，它们不认识 `/d/`。
       `D:/a/b` 形式三方都认：Git Bash 的 glob/stat/重定向、adb.exe、python.exe。
       已在宿主用真实 adb push（15 字节探针文件推到手机）逐项实测两种形式：
       `/d/` 形式复现同一报错，`D:/` 形式 glob/stat/push/python/dirname/重定向全过。
       ⇒ 教训：干跑台把 subprocess 桩掉了，**路径形式这类「外部程序怎么解释参数」
          的问题干跑查不出来**，必须拿真实外部程序跑一个最小探针。
    """
    return str(p).replace("\\", "/")


# 阶段 2（single 形态、只出 1:1）用的组大小 = single 四段 1:1 图的最大者（mem_breakdown 实测）。
# 🔴 只对「single + 1:1」成立；single 契约的其它比例、以及 mg，都**不能**用它（见 sf_group_bytes）。
SINGLE_1024_SF_BYTES = 302645248


def sf_group_bytes(contract, local_dir):
    """共享 spill-fill 的组大小 = 契约引用的**全部 transformer 图（全部比例）**里 spillFill 的最大者。

    🔴 2026-09-17 设备实测：APK 里写死 302,645,248（single 1:1 的最大者），交付 mg 后
       part1a 的图要 311,492,608~331,415,552 B ⇒ `Failed init QNN context: transformer_part1a`。
       官方（htp_backend.html）要求组大小为所有成员的最大者，且只取第一个注册者的值。
    ⇒ 从**实际要交付的 .bin 元数据**现读，写进 marker 内容，app 读 marker。
       组成员只有四个 transformer 段（`PipelineZImage.hpp::hoist`），文本编码器与 VAE 不入组。
    """
    import mem_breakdown as M
    names = sorted(n for n in referenced_files(contract) if n.startswith("transformer_"))
    best = 0
    for n in names:
        p = os.path.join(local_dir, n)
        if not os.path.isfile(p):
            raise SystemExit("🔴 算组大小需要本地 %s，但不存在" % p)
        for g in M.dump(p)["info"]["graphs"]:
            m, _ = M.graph_mem(g["info"])
            best = max(best, m["spillFillBufferSize"])
    if best <= 0:
        raise SystemExit("🔴 组大小算出来是 0 —— 拒绝写进 marker")
    return best


def referenced_files(contract):
    """契约实际引用的文件名集合（顶层 models + 各 size_variants，去重）。"""
    c = json.load(open(contract, encoding="utf-8"))
    out = set(e["actual_filename"] for e in c["models"])
    for sv in c.get("size_variants", {}).values():
        for e in sv.get("models", []):
            out.add(e["actual_filename"])
    return out


def save_state(d):
    os.makedirs(OUT, exist_ok=True)
    old = load_state()
    old.update(d)
    json.dump(old, open(STATE, "w"), indent=1, default=str)


def load_state():
    if os.path.isfile(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return {}


def check():
    """阶段 0：宿主自检。不碰设备，可反复跑。"""
    ok = True
    print("=== 阶段 0 · 宿主自检 ===")
    r = subprocess.run([sys.executable, os.path.join("scripts", "aspect_preflight.py"),
                        MG_CONTRACT, "--apk", APK], cwd=REPO, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-3:]
    print("  预检: " + " | ".join(t.strip() for t in tail))
    if r.returncode != 0:
        ok = False
    r2 = subprocess.run([sys.executable, os.path.join("scripts", "stage_mg_delivery.py")],
                        cwd=REPO, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    print("  交付物: " + (r2.stdout or "").strip().splitlines()[-1])
    if r2.returncode != 0:
        ok = False
    for p, what in ((MG_CONTRACT, "mg 契约"), (SINGLE_CONTRACT, "single 契约（回滚用）"),
                    (APK, "APK"), (MG_DIR, "staged 目录")):
        e = os.path.exists(p)
        print("  %-18s %s %s" % (what, "✅" if e else "🔴 缺", p))
        ok = ok and e
    # 🔴 2026-09-17 补：组大小从交付文件现读（理由见 sf_group_bytes），并确认 APK 里是「读 marker」的新代码
    if ok:
        sfb = sf_group_bytes(MG_CONTRACT, MG_DIR)
        save_state({"mg_sf_bytes": sfb})
        print("  mg 组大小（全部 transformer 图的 spillFill 最大者）%d B = %.1f MiB"
              % (sfb, sfb / 2**20))
        import zipfile
        with zipfile.ZipFile(APK) as z:
            so = [n for n in z.namelist() if n.endswith("libstable_diffusion_core.so")]
            has = bool(so) and b"(from marker)" in z.read(so[0])
        print("  APK 含「组大小读 marker」的代码 %s" % ("✅" if has else "🔴 没有 ⇒ APK 不是新编的"))
        ok = ok and has
    print("  ⇒ %s" % ("✅ 可以插线" if ok else "🔴 先在宿主修掉"))
    return 0 if ok else 1


def gen(dev, tag, w=1024, h=1024, base_c=None, restart=True):
    """真实 app 出图 + 每秒 ION 采样。

    🔴 `restart=True` 会先 force-stop。**换契约后必须重启**，否则
    `app_generate.sh` 见 8081 还在监听就直接复用**内存里的旧契约**——
    不报错，只会给出「看起来正常但没意义」的数字。
    两个计时臂之间也重启，保证互相独立、ION 从同一本底起算。
    """
    if restart:
        dev.sh("am force-stop %s || true" % PKG)
        time.sleep(5)
    # 🔴 先删同名旧产物：否则 app_generate 若「rc=0 却没写图」，会拿上一次（甚至干跑桩）的 PNG 冒充
    #    （2026-09-17 发现 scratch_runs/os_mg_1024x1024.png 是 09-17 19:01 干跑留下的 5 KB 假图）。
    for ext in (".png", ".sse"):
        stale = os.path.join(RUNS, "os_%s%s" % (tag, ext))
        if os.path.isfile(stale):
            os.remove(stale)
    if base_c is not None:
        D.cooldown(dev, base_c)
    D.sample_start(dev, tag)
    t0 = time.time()
    cmd = ["bash", os.path.join("scripts", "app_generate.sh"), "os_" + tag, str(SEED)]
    if (w, h) != (1024, 1024):
        # 第 3 个参数是 prompt，必须显式带上默认值才能把 4/5 位的宽高送进去
        cmd += [DEFAULT_PROMPT, str(w), str(h)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=2700)
    el = time.time() - t0
    mem = D.sample_stop(dev, tag)
    img = os.path.join(RUNS, "os_%s.png" % tag)
    if r.returncode != 0 or not os.path.isfile(img):
        print((r.stdout or "")[-1800:]); print((r.stderr or "")[-800:])
        raise SystemExit("🔴 %s 出图失败 rc=%d" % (tag, r.returncode))
    info = {"tag": tag, "wall_s": el, "sha256": D.sha256(img), "img": img}
    info.update(mem)
    # 🔴 校验实际出图尺寸：RequestParser.hpp:82-92 对**未交付**的尺寸是
    #    「静默回落到 1024x1024」，不报错。不校验的话，一个没生效的比例
    #    会以「1024 的图」冒充通过。
    with open(img, "rb") as f:
        head = f.read(33)
    got_w = int.from_bytes(head[16:20], "big")
    got_h = int.from_bytes(head[20:24], "big")
    info["png_wh"] = [got_w, got_h]
    info["size_ok"] = (got_w == w and got_h == h)
    if not info["size_ok"]:
        print("     🔴 出图尺寸 %dx%d ≠ 请求的 %dx%d（后端静默回落）" % (got_w, got_h, w, h))
    sse = os.path.join(RUNS, "os_%s.sse" % tag)
    if os.path.isfile(sse):
        import re
        m = re.search(r"generation_time_ms\D{0,4}(\d+)",
                      open(sse, encoding="utf-8", errors="replace").read())
        if m:
            info["generation_s"] = int(m.group(1)) / 1000.0
    print("  %-12s %.0f s（app %s）ION峰值 %d  MemAvail最低 %d  采样 %d 点  sha %s"
          % (tag, el, ("%.0f s" % info["generation_s"]) if "generation_s" in info else "—",
             info["ion_peak"], info["avail_low"], info["n"], info["sha256"][:12]), flush=True)
    if info["n"] < 30:
        print("     ⚠️ 采样点仅 %d ⇒ 内存数字不得引用" % info["n"])
    return info


def push_contract(dev, local, tag):
    """推契约到 app 私有目录。🔴 先 push 到暂存区再设备本地 cat（adb shell 非二进制安全）。"""
    tmp = "/data/local/tmp/_ct_%s.json" % tag
    dev.push(local, tmp)
    got = int(dev.sh("stat -c %%s %s" % tmp).strip())
    want = os.path.getsize(local)
    if got != want:
        raise SystemExit("🔴 契约推送字节不符 %d != %d" % (got, want))
    dev.sh("run-as %s sh -c 'cat %s > %s'" % (PKG, tmp, DEV_CONTRACT))
    back = dev.sh("run-as %s stat -c %%s %s" % (PKG, DEV_CONTRACT)).strip()
    if int(back) != want:
        raise SystemExit("🔴 契约落地字节不符 %s != %d" % (back, want))
    print("  契约已换 -> %s（%d 字节，已核对）" % (os.path.basename(local), want), flush=True)


def write_marker(dev, nbytes):
    """写 marker（内容 = 组大小字节数）并**读回核对**。

    🔴 旧版只建空文件、且阶段 3 建完不核对。现在 app 从内容读组大小，
       写错/读不出 ⇒ app 会**不共享**（见 PipelineZImage.hpp），mg 必然 H1/H2 不过 ——
       那会被误读成「共享不够」，所以必须在这里就拦下。
    """
    dev.sh("run-as %s sh -c 'mkdir -p files/models/ZIMAGE && echo %d > %s'"
           % (PKG, int(nbytes), MARKER))
    back = dev.sh("run-as %s cat %s 2>&1 || true" % (PKG, MARKER)).strip()
    ok = back == str(int(nbytes))
    print("  marker 写入 %d，读回 %r %s" % (int(nbytes), back[:40], "✅" if ok else "🔴"), flush=True)
    return ok


def single_files_missing(dev):
    """设备上 single 契约引用的 .bin 缺哪些（`rollback` 的前置守卫，只读）。"""
    need = sorted(referenced_files(SINGLE_CONTRACT))
    miss = []
    for n in need:
        s = dev.sh("run-as %s stat -c %%s %s/%s 2>/dev/null || echo 0" % (PKG, BIN_DIR, n)).strip()
        if not s.isdigit() or int(s) == 0:
            miss.append(n)
    return need, miss


def rollback(dev, why):
    """回滚到交付前状态。

    🔴 **每一步独立 try，且捕获 `BaseException`**。
       原因（2026-09-07 由 `oneshot_dryrun.py` 的失败场景查出）：
       `push_contract` 在字节不符时抛 **`SystemExit`**，而它继承自 `BaseException`
       ——`except Exception` **抓不住** ⇒ 回滚会**在中途炸掉**：
       marker 已删、契约可能没换回、APK 一定没装回，
       **正好违背「任何一门不过都不留半截状态」这个承诺**。
       ⇒ 一步失败不得跳过后面的步骤；最后如实汇报每一步的结果。
    """
    print("\n🔴 回滚：%s" % why, flush=True)
    st = load_state()
    done = []
    # 🔴 2026-09-17 补（`--cleanup --yes` 已执行之后）：single 契约引用的 .bin 已从设备删掉。
    #    此时照旧「换回 single 契约 + 装回现网 APK」会让 app 引用一堆不存在的文件 ⇒ **回滚反而把 app 弄坏**。
    #    ⇒ 先核对设备上 single 契约引用的文件是否齐全；不齐就**什么都不改**，只报告怎么恢复。
    need, miss = single_files_missing(dev)
    if miss:
        print("  🔴 设备上缺 single 契约引用的 %d/%d 个文件（多半已执行过 --cleanup）⇒ **拒绝回滚，设备未做任何改动**。"
              % (len(miss), len(need)))
        print("     恢复 single 需先把这些文件重新交付（宿主副本名字不同，按字节数对应，见 HANDOVER §15.45.6），"
              "再跑 --rollback。")
        for n in miss[:6]:
            print("       缺: %s" % n)
        return

    def step(name, fn):
        try:
            fn()
            done.append((name, "✅"))
        except BaseException as e:                      # noqa: BLE001 —— 见上
            done.append((name, "🔴 %s" % str(e)[:120]))

    step("删 marker", lambda: dev.sh("run-as %s rm -f %s || true" % (PKG, MARKER)))
    if os.path.isfile(SINGLE_CONTRACT):
        step("换回 single 契约", lambda: push_contract(dev, SINGLE_CONTRACT, "rb"))
    else:
        done.append(("换回 single 契约", "🔴 本地找不到 %s" % SINGLE_CONTRACT))
    bak = st.get("apk_backup")
    if bak and os.path.isfile(bak):
        def _apk():
            r = subprocess.run([dev.ADB, "-s", dev.SERIAL, "install", "-r", bak],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=1800)
            if "Success" not in (r.stdout or ""):
                raise RuntimeError((r.stdout or "")[-160:])
        step("装回存档 APK", _apk)
    else:
        done.append(("装回存档 APK", "⏭ 没有存档（阶段 1 之前失败）"))
    step("force-stop", lambda: dev.sh("am force-stop %s || true" % PKG))

    print("  回滚各步：")
    for n, r in done:
        print("     %-16s %s" % (n, r))
    if all(r.startswith(("✅", "⏭")) for _, r in done):
        print("  ⇒ 设备已回到交付前状态。**可以拔线。**")
    else:
        print("  🔴 **回滚未全部成功，设备可能处于中间状态，请勿直接使用。**")
        print("     手工恢复：adb install -r %s" % (bak or "<阶段1的存档APK>"))
        print("     契约：把 %s 写回 %s" % (SINGLE_CONTRACT, DEV_CONTRACT))


def cleanup(dev):
    """删掉**当前设备契约不再引用**的 .bin，以及部署脚本留在 /data/local/tmp 的暂存区。

    🔴 **必须等五个比例都验收完再做**：删完之后设备上就没有 single 形态的文件了，
       回滚要重新推 25.5 GiB（约 11 分钟）。宿主上产物仍在
       （`p0_experiments/aspect/` 与 `p2attr/`），所以只是慢，不是不可逆。
    🔴 默认**只列不删**；真要删要加 `--yes`。
    """
    do = "--yes" in sys.argv
    dev.sh("run-as %s cat %s > /data/local/tmp/_cur.json" % (PKG, DEV_CONTRACT))
    cur = os.path.join(OUT, "device_contract_now.json")
    os.makedirs(OUT, exist_ok=True)
    dev.pull("/data/local/tmp/_cur.json", cur)
    ref = referenced_files(cur)
    print("设备当前契约引用 %d 个文件" % len(ref))
    listing = dev.sh("run-as %s sh -c 'cd %s && ls -1 *.bin 2>/dev/null'" % (PKG, BIN_DIR))
    names = [x.strip() for x in listing.splitlines() if x.strip().endswith(".bin")]
    dead = [n for n in names if n not in ref]
    tot = 0
    for n in dead:
        s = dev.sh("run-as %s stat -c %%s %s/%s 2>/dev/null || echo 0"
                   % (PKG, BIN_DIR, n)).strip()
        tot += int(s) if s.isdigit() else 0
    print("  目录里 %d 个 .bin，其中 **%d 个不再被引用**，合计 %.2f GiB"
          % (len(names), len(dead), tot / 2**30))
    for n in sorted(dead):
        print("     %s" % n)
    tmp = dev.sh("du -sk %s 2>/dev/null | awk '{print $1}' || echo 0" % DEV_TMP).strip()
    tmp_gib = (int(tmp) / 1048576.0) if tmp.isdigit() else 0.0
    print("  部署暂存区 %s ≈ %.2f GiB" % (DEV_TMP, tmp_gib))
    if not do:
        print("\n  ⚠️ 这是**试运行**，什么都没删。确认后加 --yes 再跑一次。")
        return 0
    for n in dead:
        dev.sh("run-as %s rm -f %s/%s" % (PKG, BIN_DIR, n))
    dev.sh("rm -rf %s || true" % DEV_TMP)
    left = 0
    for n in sorted(ref):
        s = dev.sh("run-as %s stat -c %%s %s/%s 2>/dev/null || echo 0"
                   % (PKG, BIN_DIR, n)).strip()
        left += int(s) if s.isdigit() else 0
    print("\n  ✅ 已删。剩余（契约引用的）合计 **%.2f GiB**" % (left / 2**30))
    print("  🔴 此后回滚 single 需重新推 25.5 GiB：宿主产物在 p0_experiments/aspect/ 与 p2attr/")
    return 0


def main():
    if "--cleanup" in sys.argv:
        import lab_dev as dev
        return cleanup(dev)
    if "--run" not in sys.argv and "--rollback" not in sys.argv:
        return check()
    import lab_dev as dev
    if "--rollback" in sys.argv:
        rollback(dev, "手工触发")
        return 0
    if check() != 0:
        return 1
    phase = int(sys.argv[sys.argv.index("--phase") + 1]) if "--phase" in sys.argv else 1
    os.makedirs(OUT, exist_ok=True)
    dev.require_online("oneshot")
    print("\n⚠️ 本脚本全程占用设备。**结束前请勿拔线。**", flush=True)
    dev.sh("am force-stop %s || true" % PKG)
    time.sleep(5)
    base_c = D.npu_c(dev)
    print("热基线 NPU %.1f C" % base_c, flush=True)
    st = load_state()
    # 🔴 续跑守卫：`--phase 3` 而没有阶段 1 的 state 时，后面取 st["g0"] 会 KeyError
    #    ——那会在设备已占用的情况下崩掉，是最不该发生的失败方式。
    if phase > 1:
        need = ["g0", "apk_backup", "base_c"]
        miss = [k for k in need if k not in st]
        if miss:
            print("🔴 --phase %d 需要阶段 1 的状态，但 %s 缺失（%s）"
                  % (phase, miss, STATE))
            print("   ⇒ 请从 --phase 1 开始，或先补齐 state.json")
            return 1
        base_c = st["base_c"]
    # 🔴 续跑守卫（2026-09-17 补）：`--phase 3/4` 跳过了阶段 2 ⇒
    #    ① 必须确认实验 1 **在 state 里真的全过**，否则等于绕过 G2~G5 直接交付；
    #    ② 阶段 2 失败/回滚后设备上是**现网 APK**（回滚会装回存档）。
    #       现网 APK 没有共享 spill-fill ⇒ 直接交付 mg 就是 #161 的原样（app 被杀）。
    #       ⇒ 跳过阶段 2 时必须**重新装新 APK**（安装幂等）。
    if phase > 2:
        g0 = st["g0"]
        exp1 = ("sf_off" in st and "sf_on" in st
                and st["sf_off"]["sha256"] == g0["sha256"]
                and st["sf_on"]["sha256"] == g0["sha256"]
                and st.get("sf_save", -1) >= G4_MIN_SAVE
                and st.get("sf_dt", 1e9) <= G5_MAX_SLOW)
        if not exp1:
            print("🔴 --phase %d 要求实验 1（G2~G5）在 state 里已全过，但没有 ⇒ 请从 --phase 1 开始"
                  % phase)
            return 1
        print("  续跑：实验 1 已全过（省 %d MiB，Δt %+.1f s）" % (st["sf_save"], st["sf_dt"]),
              flush=True)
        r = subprocess.run([dev.ADB, "-s", dev.SERIAL, "install", "-r", APK],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
        if "Success" not in (r.stdout or ""):
            print(r.stdout, r.stderr); raise SystemExit("🔴 安装失败")
        save_state({"apk_installed": True})
        print("  新 APK 已（重新）安装", flush=True)

    # ---------- 阶段 1：铁律 1 / 2 ----------
    if phase <= 1:
        print("\n=== 阶段 1 · 现网基线 + 存档（约束 11 铁律 1/2）===", flush=True)
        save_state({"apk_installed": False})     # 新一轮从干净状态起算
        g0 = gen(dev, "g0_live", base_c=base_c)
        path = dev.sh("pm path %s" % PKG).strip().replace("package:", "").splitlines()[0].strip()
        bak = os.path.join(OUT, "BACKUP_apk_%s.apk" % time.strftime("%Y%m%d_%H%M%S"))
        dev.pull(path, bak)
        if not os.path.isfile(bak) or os.path.getsize(bak) < (10 << 20):
            raise SystemExit("🔴 APK 存档失败")
        ct = os.path.join(OUT, "BACKUP_contract_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        dev.sh("run-as %s cat %s > /data/local/tmp/_ct.json" % (PKG, DEV_CONTRACT))
        dev.pull("/data/local/tmp/_ct.json", ct)
        print("  已存档 APK %d 字节｜契约 %d 字节"
              % (os.path.getsize(bak), os.path.getsize(ct)), flush=True)
        save_state({"g0": g0, "apk_backup": bak, "contract_backup": ct, "base_c": base_c})
        st = load_state()

    # ---------- 阶段 2：实验 1（spill-fill A/B，single 形态）----------
    if phase <= 2:
        print("\n=== 阶段 2 · 实验 1：spill-fill A/B（严格单变量）===", flush=True)
        r = subprocess.run([dev.ADB, "-s", dev.SERIAL, "install", "-r", APK],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1800)
        if "Success" not in (r.stdout or ""):
            print(r.stdout, r.stderr); raise SystemExit("🔴 安装失败")
        print("  新 APK 已安装", flush=True)
        # 从这一刻起设备已偏离现网 ⇒ 任何未预期的中断都必须回滚（见 guarded_main）
        save_state({"apk_installed": True})
        dev.sh("run-as %s rm -f %s || true" % (PKG, MARKER))
        off = gen(dev, "sf_off", base_c=st["base_c"])
        if not write_marker(dev, SINGLE_1024_SF_BYTES):
            rollback(dev, "marker 没建上"); return 2
        on = gen(dev, "sf_on", base_c=st["base_c"])
        g0 = st["g0"]
        save = off["ion_peak"] - on["ion_peak"]
        dt = on.get("generation_s", on["wall_s"]) - off.get("generation_s", off["wall_s"])
        g2 = off["sha256"] == g0["sha256"]
        g3 = on["sha256"] == g0["sha256"]
        print("\n  G2 等价性 %s ｜ G3 精度 %s ｜ G4 省量 %d MiB（>=%d）%s ｜ G5 %+.1f s（<=%.0f）%s"
              % ("✅" if g2 else "🔴", "✅" if g3 else "🔴", save, G4_MIN_SAVE,
                 "✅" if save >= G4_MIN_SAVE else "🔴", dt, G5_MAX_SLOW,
                 "✅" if dt <= G5_MAX_SLOW else "🔴"))
        print("     ION 峰值 %d -> %d ｜ MemAvail 最低 %d -> %d"
              % (off["ion_peak"], on["ion_peak"], off["avail_low"], on["avail_low"]))
        save_state({"sf_off": off, "sf_on": on, "sf_save": save, "sf_dt": dt})
        if not (g2 and g3):
            rollback(dev, "G2/G3 未过：改动影响了数值"); return 2
        if save < G4_MIN_SAVE:
            print("")
            print("  🔴 G4 未过：实际省量 %d MiB < mg 的实测缺口 %d"
                  % (save, G4_MIN_SAVE))
            print("     ⇒ **数学上覆盖不了，不切 mg**。#146 的③就此有了答案：省量不足。")
            rollback(dev, "G4 未过"); return 2
        if save < G4_COMFORT:
            print("  ⚠️ 省量 %d MiB 落在 %d~%d 之间：**仍然交付**，够不够由真实 mg 运行的"
                  " H1/H2 说了算（判据修订，执行前定稿）" % (save, G4_MIN_SAVE, G4_COMFORT))
        if dt > G5_MAX_SLOW:
            rollback(dev, "G5 未过：共享有隐藏的速度代价"); return 2
        print("  ⇒ 🟢 实验 1 全过，进入交付", flush=True)
        st = load_state()

    # ---------- 阶段 3：交付 mg ----------
    if phase <= 3:
        print("\n=== 阶段 3 · 交付 mg（推 8.26 GiB + 换契约）===", flush=True)
        # 🔴 路径必须转成 `D:/...` 形式（不是 `/d/...`），理由见 mixed() 的文档
        r = subprocess.run(["sh", mixed(ADB_SH), mixed(dev.ADB), PKG,
                            mixed(MG_DIR), mixed(MG_CONTRACT)],
                           cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=5400)
        print((r.stdout or "").strip()[-1500:])
        if r.returncode != 0:
            print((r.stderr or "")[-800:])
            rollback(dev, "部署脚本失败 rc=%d" % r.returncode); return 2
        sfb = load_state().get("mg_sf_bytes")
        if not sfb:
            rollback(dev, "state 里没有 mg 的组大小（阶段 0 应已算出）"); return 2
        if not write_marker(dev, sfb):
            rollback(dev, "marker 没写对"); return 2
        # 🔴 H4 只能量「契约引用的文件」，不能量整个目录。
        #    `deploy_aspect.sh` 是**只增不删**，交付后目录里同时存在旧 single（~30 GiB）
        #    与新 mg（8.26 GiB）⇒ 量整个目录会得到 ~38 GiB，
        #    **把一次成功的交付判成失败并触发回滚**。
        #    真正的「设备占用降下来」由 `--cleanup` 完成（见文件末尾），
        #    那一步会删掉不再被引用的旧文件，但**必须等五个比例都验完再做**。
        ref = referenced_files(MG_CONTRACT)
        tot = 0
        miss = []
        for n in sorted(ref):
            s = dev.sh("run-as %s stat -c %%s %s/%s 2>/dev/null || echo 0"
                       % (PKG, BIN_DIR, n)).strip()
            if not s.isdigit() or int(s) == 0:
                miss.append(n)
            else:
                tot += int(s)
        gib = tot / 2**30
        print("  契约引用的 %d 个文件合计 **%.2f GiB**（H4 判据 <=%.0f）%s"
              % (len(ref), gib, H4_MAX_GIB, "✅" if (not miss and gib <= H4_MAX_GIB) else "🔴"))
        if miss:
            print("     🔴 设备上缺: %s" % miss)
        save_state({"deploy_gib": gib, "missing": miss})
        if miss or gib > H4_MAX_GIB:
            rollback(dev, "H4 未过：引用文件 %.2f GiB / 缺 %d 个" % (gib, len(miss))); return 2

    # ---------- 阶段 4：五比例验收 ----------
    if phase <= 4:
        print("\n=== 阶段 4 · 五比例验收 ===", flush=True)
        st = load_state()
        res = {}
        # `--smoke`（2026-09-17 用户决定）：五比例已在 20:21~20:58 那轮以同一 APK 逻辑、同一交付文件全部验过
        # （H5 ✅，五张图已人工查看），重新交付后只出 1:1 确认交付态可用。
        sizes_run = [(1024, 1024)] if "--smoke" in sys.argv else SIZES
        if "avail_plateau_med" not in st["g0"]:
            rollback(dev, "state 里的现网基线缺平台期中位数，H2' 无从判"); return 2
        for (w, h) in sizes_run:
            tag = "mg_%dx%d" % (w, h)
            res[tag] = gen(dev, tag, w=w, h=h, base_c=max(st["base_c"], P4_WARM_BASE_C))
        save_state({"mg_runs": res})
        base = st["g0"]
        k = "mg_1024x1024"
        if k in res:
            h1 = res[k]["ion_peak"] <= base["ion_peak"]
            # 🔴 H2 修订（2026-09-17，用户决定，**本轮执行前**定稿；理由见 EXP_PLAN_SPILLFILL.md「判据修订 2」）：
            #    原 H2「单秒最低 > 800」没有分辨力，现网自己当天就不过（742）。
            #    改为与**同日现网**配对：ION 高位平台期 MemAvailable 中位数 >= 现网的。
            h2 = res[k]["avail_plateau_med"] >= base["avail_plateau_med"]
            h3 = res[k]["sha256"] == base["sha256"]
            print("\n  H1 ION 峰值 %d <= 现网 %d %s" % (res[k]["ion_peak"], base["ion_peak"],
                                                    "✅" if h1 else "🔴"))
            print("  H2' MemAvail 平台期中位 %d >= 现网 %d %s（参考：单秒最低 %d，现网 %d）"
                  % (res[k]["avail_plateau_med"], base["avail_plateau_med"],
                     "✅" if h2 else "🔴", res[k]["avail_low"], base["avail_low"]))
            print("  H3 1:1 出图 sha256 与现网逐字节相同 %s" % ("✅" if h3 else "🔴"))
            bad_size = [t for t, v in res.items() if not v.get("size_ok")]
            h5 = (len(res) == len(sizes_run)) and not bad_size
            print("  H5 五个比例都出图且尺寸正确 %s%s"
                  % ("✅" if h5 else "🔴", ("  尺寸错的: %s" % bad_size) if bad_size else ""))
            for t in sorted(res):
                v = res[t]
                print("     %-16s %sx%s  %.0f s  ION峰值 %d  MemAvail最低 %d"
                      % (t, v["png_wh"][0], v["png_wh"][1],
                         v.get("generation_s", v["wall_s"]), v["ion_peak"], v["avail_low"]))
            if not (h1 and h2 and h3 and h5):
                rollback(dev, "H1/H2/H3/H5 未过"); return 2

    print("\n=== 结论 ===")
    st = load_state()
    print("  spill-fill 实测省量 %s MiB｜mg 设备占用 %s GiB"
          % (st.get("sf_save"), round(st.get("deploy_gib", -1), 2)))
    print("  ✅ 全部通过，**mg + spill-fill 已交付**。设备空闲，可以拔线。")
    print("  回滚：python scripts/oneshot_mg_session.py --rollback")
    json.dump(st, open(os.path.join(OUT, "final.json"), "w"), indent=1, default=str)
    return 0


def guarded_main():
    """入口兜底：装上新 APK 之后发生**任何未预期的中断**都自动回滚。

    🔴 2026-09-17 插线前核对时查出（干跑五场景恰好都没覆盖）：
       `gen()` 出图失败时直接抛 `SystemExit`，而阶段 2~4 只对**判据不过**做了回滚，
       **没兜住「出图本身失败」**。最可能触发它的正是本实验要探的风险——
       mg 形态下 app 被系统杀掉（#161 的原样），curl 断开 ⇒ `SystemExit`
       ⇒ 脚本直接退出，把手机留在「mg 契约 + 新 APK」的半截状态。
       Ctrl+C（KeyboardInterrupt）同样走这里，与手册「中途停下要回滚」一致。
    """
    try:
        return main()
    except BaseException as e:                          # noqa: BLE001 —— 见上
        st = load_state()
        if "--run" in sys.argv and st.get("apk_installed"):
            try:
                import lab_dev as dev
                rollback(dev, "未预期的中断 %s: %s" % (type(e).__name__, str(e)[:200]))
            except BaseException as e2:                  # noqa: BLE001
                print("🔴 兜底回滚本身失败：%s ⇒ 请手工 --rollback" % e2)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(guarded_main())
