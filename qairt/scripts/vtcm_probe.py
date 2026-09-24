"""#34：VTCM 分块策略影响算子级精度吗？

方案：scripts/EXP_PLAN_VTCM.md（判据事前锁定）
用 P0-B 的 baseline 单算子重放（已实测 E_standalone = 1.5544%），
只改 `qnn-context-binary-generator --vtcm_override`。

🔴 V0 生效门最重要：本项目 #10/#27 都是「选项被静默忽略、产物 md5 逐位相同」。
"""
import os, sys, json, hashlib, subprocess, shutil
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
QDLC = os.path.join(P0B, "standalone_baseline", "fc99_quantized.dlc")
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
WORK = os.path.join(P0B, "vtcm")
M, K, N, ROWS = 4128, 3840, 3840, 512
BYTES = M * N * 4
VTCMS = [0, 8, 4, 2]           # 0 = SoC 最大
E_REF = 1.5544                 # P0-B 实测（默认档，即 --vtcm_override 未指定）


def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def adb(cmd, timeout=1800):
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("adb shell rc=%d: %s\n%s" % (r.returncode, cmd[:100], r.stderr))
    return r.stdout


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bulk(a, b):
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return 100.0 * np.linalg.norm(y - x) / np.linalg.norm(x)


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)

    # ---- 建各档位 context ----
    hashes, ctxs = {}, {}
    for v in VTCMS:
        d = os.path.join(WORK, "v%d" % v)
        os.makedirs(d)
        rc, out = sh([os.path.join(SDK, "bin", "x86_64-windows-msvc",
                                   "qnn-context-binary-generator.exe"),
                      "--backend", os.path.join(SDK, "lib", "x86_64-windows-msvc", "QnnHtp.dll"),
                      "--dlc_path", QDLC, "--binary_file", "fc99_v%d" % v,
                      "--output_dir", d, "--htp_socs", "sm8750",
                      "--vtcm_override", str(v)])
        bins = [f for f in os.listdir(d) if f.endswith(".bin")]
        if not bins:
            print("  ❌ vtcm=%d context 生成失败 rc=%d\n%s" % (v, rc, out[-800:]))
            return
        p = os.path.join(d, bins[0])
        hashes[v] = md5(p)
        ctxs[v] = p
        print("  vtcm=%-2d  %s  %.2f MB  md5=%s" % (v, bins[0], os.path.getsize(p) / 1e6, hashes[v][:16]))

    # ---- V0 生效门 ----
    print("")
    print("=== V0 生效门：不同 vtcm_override 的产物必须互不相同（#10/#27 的坑）===")
    uniq = len(set(hashes.values()))
    print("  %d 个档位产出 %d 种不同 md5" % (len(VTCMS), uniq))
    if uniq == 1:
        print("  🔴 **全部相同 ⇒ --vtcm_override 被静默忽略**")
        print("  ⇒ 本实验无效。#34 标为 ⛔受阻，**不得**据此下「VTCM 无影响」的结论（方案 §二）")
        return
    if uniq < len(VTCMS):
        same = {}
        for v, h in hashes.items():
            same.setdefault(h, []).append(v)
        print("  ⚠️ 部分档位产物相同：%s ⇒ 这些档位实际是同一配置，判读时合并"
              % [g for g in same.values() if len(g) > 1])
    print("  ✅ V0 通过")

    # ---- 逐档上机 ----
    enc = json.load(open(os.path.join(P0B, "linear_99_encodings_baseline.json"), encoding="utf-8"))
    ex, ew, ey = enc["node_linear_99_pre_reshape"], enc["val_1800"], enc["linear_99_fc"]
    x = np.fromfile(X_RAW, np.float32, count=ROWS * K).reshape(ROWS, K).astype(np.float64)
    W = np.load(os.path.join(P0B, "val_1800_fp32.npy")).astype(np.float64)

    def q(v, s, o, bw):
        return np.clip(np.rint(v / s) - o, 0, (1 << bw) - 1)
    qx = q(x, ex["scale"], ex["offset"], ex["bitwidth"]) + ex["offset"]
    qw = q(W, ew["scale"], ew["offset"], ew["bitwidth"]) + ew["offset"]
    y_fxp = ex["scale"] * ew["scale"] * (qx @ qw)
    y_fxp = ey["scale"] * (q(y_fxp, ey["scale"], ey["offset"], ey["bitwidth"]) + ey["offset"])

    subprocess.run([ADB, "-s", DEV, "push", X_RAW, "%s/p0b_x.raw" % T],
                   capture_output=True, timeout=3600)
    print("")
    print("=== 逐档实测 ===")
    res = {}
    for v in VTCMS:
        subprocess.run([ADB, "-s", DEV, "push", ctxs[v], "%s/fc99_v%d.bin" % (T, v)],
                       capture_output=True, timeout=3600)
        od = "%s/vt%d" % (T, v)
        adb("mkdir -p %s && rm -rf %s/Result_0" % (od, od))
        adb("printf '%%s\\n' 'x:=%s/p0b_x.raw' > %s/list.txt" % (T, od))
        o = adb("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
                "./qnn-net-run --retrieve_context fc99_v%d.bin --backend libQnnHtp.so "
                "--input_list vt%d/list.txt --output_dir %s --log_level error 2>&1 | tail -3"
                % (T, T, T, v, v, od))
        if "Finished Executing Graphs" not in o:
            print("  vtcm=%-2d  ❌ 执行失败：%s" % (v, o.strip()[:200]))
            continue
        rp = "%s/Result_0/linear_99_fc.raw" % od
        nb = int(adb("stat -c %%s %s" % rp).strip())
        if nb != BYTES:
            print("  vtcm=%-2d  ❌ 输出字节数 %d ≠ %d（约束 3）" % (v, nb, BYTES))
            continue
        lp = os.path.join(WORK, "out_v%d.raw" % v)
        subprocess.run([ADB, "-s", DEV, "pull", rp, lp], capture_output=True, timeout=3600)
        y = np.fromfile(lp, np.float32, count=ROWS * N).reshape(ROWS, N).astype(np.float64)
        e = bulk(y_fxp, y)
        res[v] = e
        print("  vtcm=%-2d  E_standalone = %.4f%%   (P0-B 默认档 %.4f%%)" % (v, e, E_REF))

    if len(res) < 2:
        print("  ❌ 有效档位不足，无法判定")
        return
    lo, hi = min(res.values()), max(res.values())
    print("")
    print("=== 主判据 ===")
    print("  跨档位 E 范围 %.4f%% ~ %.4f%%   极差 = %.4f pp" % (lo, hi, hi - lo))
    if hi - lo >= 0.15:
        v = "🔴 **VTCM 分块影响算子级精度** ⇒ #34 在算子级成立，查最优档位"
    elif hi - lo <= 0.05:
        v = ("✅ **无影响** ⇒ #34 在算子级被证否；#88 的 46% 转由 #33（融合）承担。"
             "⚠️ 只否证算子级分量，完整图里的组合作用未测")
    else:
        v = "🔶 **弱影响** ⇒ 记录，不单独推进"
    print("  判定: %s" % v)
    json.dump({str(k): val for k, val in res.items()},
              open(os.path.join(WORK, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
