# -*- coding: utf-8 -*-
"""插线前总预检：把**所有不需要设备**就能查的东西一次查完，给一个总判决。

## 存在理由
用户外出，回来后只想插一次线、很短时间验完。
⇒ **凡是能在宿主上判死的，都不许留到插线时才发现。**
2026-09-04 的两次真实教训：
  ① UI 的分辨率选择器没打开（`isSdxl` 参数名误导）—— 在真机上才发现，本可在宿主查出；
  ② 交付脚本的 sha256 门只打印不比对 —— 门是装饰性的，等于没有。

## 本脚本只做「能否决」的事
⚠️ 全过**不等于**设备上一定成功（指南 §41：代理指标只能否决不能放行）。
真正的验收仍是设备上出图 + 画质判据。本脚本的价值是**把失败提前到不花设备时间的地方**。

用法: python aspect_preflight.py <契约.json> [--apk <apk路径>]
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
SCRATCH = os.path.join(
    "C:", os.sep, "Users", "sinai", "AppData", "Local", "Temp", "claude",
    "D--LocalDreamZImage", "bbf01231-6581-4b08-b5d0-7dd03e47f992", "scratchpad")
PD_RED_MIB = 3506.0          # #57 实测红线下沿（3506~3535 MB）
HASH_MIB_PER_S = 62.0        # 2026-09-04 实测：设备现算 sha256 约 62 MiB/s

results = []


def gate(name, ok, detail, fatal=True):
    results.append((name, bool(ok), detail, fatal))
    print("  %-22s %s  %s" % (name, "PASS" if ok else ("FAIL" if fatal else "WARN"), detail),
          flush=True)
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace")
    return r.returncode, r.stdout or ""


def main():
    # 🔴 必须先转绝对路径：C2 把 Windows 路径按 `D:\x` -> `/mnt/d/x` 手工拼给 WSL，
    #    传相对路径时 `cpath[0]` 取到的是路径首字母、`cpath[2:]` 把前两个字符切掉
    #    ⇒ `logs/a.json` 被拼成 `/mnt/lgs/a.json`，报成「契约找不到」。
    #    2026-09-07 实测踩到：预检报 FAIL 指向契约，实际契约完好，是工具自己的路径 bug。
    #    （同「报错说谎」一类：报错指向的对象不是真正出问题的那个。）
    cpath = os.path.abspath(sys.argv[1])
    apk = None
    if "--apk" in sys.argv:
        apk = sys.argv[sys.argv.index("--apk") + 1]

    print("=== A. 源码层（不碰产物）===")
    rc, out = run([sys.executable, os.path.join(HERE, "check_stray_ctrl_bytes.py"),
                   os.path.join(ROOT, "scripts"),
                   os.path.join(ROOT, "local-dream", "app", "src", "main", "cpp", "src"),
                   os.path.join(ROOT, "local-dream", "app", "src", "main", "java")])
    gate("裸控制字节", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
    rc, out = run([sys.executable, os.path.join(HERE, "check_size_lists.py"), cpath])
    gate("尺寸清单三处一致", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")

    print("\n=== B. 契约与产物 ===")
    c = json.load(io.open(cpath, encoding="utf-8"))
    variants = c.get("size_variants") or {}
    gate("契约含 size_variants", bool(variants), "比例 %s" % sorted(variants))

    # 去重后的**新增**文件清单（mg 形态下多比例共用同一个 .bin）。
    # 🔴 只看新增：变体里的 text_encoder 四条是从基准逐字复制的，那些 .bin 早已在设备上、
    #    宿主上未必留有副本。把它们算进「宿主缺文件」会造成假失败（第一版就是这样）。
    # 🔴 「新增」的基准必须是**设备上现在那份契约**，不是本契约的 models ——
    #    mg 形态下本契约的 models 已经是 mg 文件，拿它去减会算出「新增 0 个」。
    #    与生成器里刚修的是同一个 bug 类型：拿错了参照物。
    dep = os.path.join(SCRATCH, "deliver_backup", "BACKUP_final_qnn_contract.json")
    if os.path.isfile(dep):
        base_names = {m["actual_filename"]
                      for m in json.load(io.open(dep, encoding="utf-8"))["models"]}
    else:
        base_names = {m["actual_filename"] for m in c["models"]}
        print("  ⚠️ 找不到设备侧契约备份，退回用本契约的 models 作参照（新增数会偏小）")
    files = {}
    conflict = []
    for v in list(variants.values()) + [{"models": c["models"]}]:
        for m in v["models"]:
            k = m["actual_filename"]
            if k in base_names:
                continue
            if files.setdefault(k, (m.get("host_source"), m["sha256"],
                                    m["size_bytes"]))[1] != m["sha256"]:
                conflict.append(k)
    gate("同名文件 sha256 无冲突", not conflict, "冲突 %s" % conflict)
    gate("变体确有新增文件", bool(files),
         ("新增 %d 个" % len(files)) if files else
         "变体只引用了基准文件 —— 它没带任何新图，必然是生成器出错")

    missing, badsha, badsize = [], [], []
    for k, (src, sha, size) in sorted(files.items()):
        if not src or not os.path.isfile(src):
            missing.append(k)
            continue
        if os.path.getsize(src) != size:
            badsize.append(k)
            continue
        if sha256(src) != sha:
            badsha.append(k)
    gate("宿主 .bin 齐全", not missing, "缺 %s" % missing)
    gate("size_bytes 与文件相符", not badsize, "不符 %s" % badsize)
    gate("sha256 与文件相符", not badsha, "不符 %s" % badsha)

    # 段间切口闭合（镜像 C++ validatePart2Split），逐比例
    bad_cut = []
    for tag, v in variants.items():
        by = {m["internal_graph_name"]: m for m in v["models"]}
        if "transformer_part2a" in by and "transformer_part2b" in by:
            o = {t["name"] for t in by["transformer_part2a"]["outputs"]}
            i = {t["name"] for t in by["transformer_part2b"]["inputs"]}
            if not o <= i:
                bad_cut.append((tag, sorted(o - i)))
    gate("part2 切口逐比例闭合", not bad_cut, "断口 %s" % bad_cut)

    # 几何自洽
    bad_geo = []
    for tag, v in variants.items():
        W, H = [int(x) for x in tag.lower().split("x")]
        by = {m["internal_graph_name"]: m for m in v["models"]}
        want = {"latents": [1, 16, H // 8, W // 8],
                "pixels": [1, 3, H, W],
                "vae_latents": [1, 16, H // 8, W // 8]}
        for gname, kind, tname in (("transformer_part1a", "inputs", "latents"),
                                   ("vae_decoder", "outputs", "pixels"),
                                   ("vae_decoder", "inputs", "vae_latents")):
            e = by.get(gname)
            if not e:
                bad_geo.append((tag, gname, "缺"))
                continue
            got = next((t["fixed_shape"] for t in e[kind] if t["name"] == tname), None)
            if got != want[tname]:
                bad_geo.append((tag, tname, "%s != %s" % (got, want[tname])))
    gate("几何自洽（latents/pixels）", not bad_geo, "不符 %s" % bad_geo)

    print("\n=== C. 设备侧可行性（用元数据预测，不占设备）===")
    tot = sum(v[2] for v in files.values())
    # 交付后设备要哈希的总量 = 基准里的文件 + 变体新增的文件（按路径去重）
    allf = {m["actual_filename"]: m["size_bytes"] for m in c["models"]}
    allf.update({k: v[2] for k, v in files.items()})
    hash_gib = sum(allf.values()) / 1073741824.0
    gate("后端启动校验耗时", hash_gib * 1024 / HASH_MIB_PER_S < 360,
         "需哈希 %.2f GiB -> 约 %.0f 秒（阈值 360 秒）" % (hash_gib, hash_gib * 1024 / HASH_MIB_PER_S),
         fatal=False)
    gate("需推送体积", True, "去重后 %d 个文件 / %.2f GiB（约 %.0f 分钟 @40MB/s）"
         % (len(files), tot / 1073741824.0, tot / 1048576.0 / 40 / 60), fatal=False)

    # 预测 ION：逐比例、逐段，只计该图 + 共享权重一次
    from check_mg_equivalence import dump, pick, mem   # 复用同一份取字段逻辑
    worst = 0.0
    bad_ion = []
    seen = {}
    # 🔴 必须把**基准**也算进来：mg 形态下最高的那个正是基准的 1:1 图（3320 MiB），
    #    只遍历 variants 会漏掉它，报出偏低的 3144 —— 漏掉的恰好是风险最大的那个。
    for tag, v in list(variants.items()) + [("1024x1024", {"models": c["models"]})]:
        for m in v["models"]:
            if not m["internal_graph_name"].startswith("transformer"):
                continue
            src = m.get("host_source")
            if not src or not os.path.isfile(src):
                continue
            key = (src, m["qnn_graph_name"])
            if key in seen:
                ion = seen[key]
            else:
                g = pick(dump(src), m["qnn_graph_name"])
                if g is None:
                    bad_ion.append((tag, m["internal_graph_name"], "图名不在 .bin 里"))
                    continue
                mm, sh = mem(g)
                ion = (sum(mm.values()) + sh) * 1.06 / 1048576.0
                seen[key] = ion
            worst = max(worst, ion)
            if ion > PD_RED_MIB:
                bad_ion.append((tag, m["internal_graph_name"], "%.0f MiB" % ion))
    gate("预测 ION 低于 PD 红线", not bad_ion and worst > 0,
         "最大 %.0f MiB（红线 %.0f）%s" % (worst, PD_RED_MIB, bad_ion or ""))

    print("")
    print("=== C2. 用 app 真正的 C++ 解析器解一遍契约（WSL 编译，不占设备）===")
    # 🔴 这一项抓到过真 bug：mg 契约的基准 models 仍指向旧的单图文件，
    #    而生成器自己的汇总打印用的是内存变量、与落盘内容不一致，**自证不了**。
    #    症状是「app 照常工作、不报错」，只是多占 6.4 GiB、启动多 100 秒 ——
    #    只有把**落盘的**契约喂给 ZImageQnnContract::load 才看得见。
    tbin = os.path.join(SCRATCH, "ctest", "t")
    if os.path.isfile(tbin):
        wp = "/mnt/" + cpath[0].lower() + cpath[2:].replace("\\", "/")
        wt = "/mnt/" + tbin[0].lower() + tbin[2:].replace("\\", "/")
        rc, out = run(["wsl.exe", "-d", "Ubuntu2404", "-e", "bash", "-lc",
                       "'%s' '%s'" % (wt, wp)])
        out = out.replace("\x00", "")
        gate("C++ 解析器可解析", rc == 0 and "PASS" in out,
             (out.strip().splitlines() or ["无输出"])[-1])
        base_line = [l for l in out.splitlines()
                     if "transformer_part1a" in l and "file=" in l]
        if base_line and variants:
            gate("基准 1:1 的路由", "mg_ctx" in base_line[0],
                 "base -> %s" % base_line[0].split("file=")[1].split()[0], fatal=False)
    else:
        gate("C++ 解析器可解析", False,
             "未编译：在 WSL 里 g++ 编 scratchpad/ctest/t.cpp", fatal=False)

    print("")
    print("=== D. APK ===")
    if apk and os.path.isfile(apk):
        with zipfile.ZipFile(apk) as z:
            so = z.read("lib/arm64-v8a/libstable_diffusion_core.so")
        cur = hashlib.sha256(so).hexdigest()
        bak = os.path.join(SCRATCH, "deliver_backup", "BACKUP_installed.apk")
        old = None
        if os.path.isfile(bak):
            with zipfile.ZipFile(bak) as z:
                old = hashlib.sha256(z.read("lib/arm64-v8a/libstable_diffusion_core.so")).hexdigest()
        gate("APK 的 .so 与现网不同", old is None or cur != old,
             "新 %s / 现网 %s" % (cur[:12], (old or "无备份")[:12]))
        gate("APK 体积", len(so) > 1_000_000, "libstable_diffusion_core.so %d B" % len(so),
             fatal=False)
    else:
        gate("APK", False, "未提供或不存在：%s" % apk, fatal=False)

    print("\n=== 总判决 ===")
    fatal_bad = [n for n, ok, _, f in results if not ok and f]
    warn = [n for n, ok, _, f in results if not ok and not f]
    if warn:
        print("  ⚠️ 警告（不阻塞）: %s" % ", ".join(warn))
    if fatal_bad:
        print("  🔴 %d 项未过: %s\n  => **不要插线**，先在宿主上修掉。" % (len(fatal_bad),
                                                                  ", ".join(fatal_bad)))
        return 1
    print("  ✅ 宿主侧全部通过。")
    print("  ⚠️ 但这只能否决不能放行（指南 §41）：设备上仍须验"
          "①出图 ②画质判据 ③多图与单图数值一致。")
    return 0


sys.exit(main())
