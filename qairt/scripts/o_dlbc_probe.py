"""#94：`O`（图优化级别）与 `dlbc` 对算子级精度的影响。

方案：scripts/EXP_PLAN_O_DLBC.md（判据事前锁定）

🔴 单变量关键：「带 config」与「不带 config」的产物本身就不同（9ccc464d vs e273d197），
   所以对照臂 CTRL = 带 config 但选项无效果（vtcm_mb:8，#93 已实测 2/4/8 产物相同）。
   所有比较对 CTRL 做，**不是**对历史的 1.5544%。
"""
import os, sys, json, shutil, hashlib, subprocess
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
QDLC = os.path.join(P0B, "standalone_baseline", "fc99_quantized.dlc")
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
WORK = os.path.join(P0B, "odlbc")
GN = "fc99_fp32"
M, K, N, ROWS = 4128, 3840, 3840, 512
BYTES = M * N * 4
E_HIST = 1.5544                      # 历史（无 config），仅作附带记录

ARMS = {
    "CTRL": {"vtcm_mb": 8},
    "O1":   {"vtcm_mb": 8, "O": 1},
    "O3":   {"vtcm_mb": 8, "O": 3},
    "DLBC": {"vtcm_mb": 8, "dlbc": 1},
}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def adb(cmd, timeout=1800):
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("adb rc=%d: %s\n%s" % (r.returncode, cmd[:100], r.stderr))
    return r.stdout


def bulk(a, b):
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return 100.0 * np.linalg.norm(y - x) / np.linalg.norm(x)


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)

    # ---- 建各臂 context ----
    print("=== 建 context 并验 V0 生效门 ===")
    ctx, hs = {}, {}
    for tag, opts in ARMS.items():
        g = {"graph_names": [GN]}
        g.update(opts)
        det = os.path.join(WORK, "d_%s.json" % tag)
        ext = os.path.join(WORK, "e_%s.json" % tag)
        json.dump({"graphs": [g], "devices": [{"soc_model": 69, "dsp_arch": "v79"}]},
                  open(det, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(SDK, "lib", "x86_64-windows-msvc",
                                                "QnnHtpNetRunExtensions.dll"),
            "config_file_path": det}}, open(ext, "w"), indent=1)
        od = os.path.join(WORK, "o_" + tag)
        os.makedirs(od, exist_ok=True)
        r = subprocess.run([os.path.join(SDK, "bin", "x86_64-windows-msvc",
                                         "qnn-context-binary-generator.exe"),
                            "--backend", os.path.join(SDK, "lib", "x86_64-windows-msvc", "QnnHtp.dll"),
                            "--dlc_path", QDLC, "--binary_file", "c_" + tag,
                            "--output_dir", od, "--htp_socs", "sm8750",
                            "--config_file", ext], capture_output=True, text=True, timeout=1800)
        b = [f for f in os.listdir(od) if f.endswith(".bin")]
        if not b:
            err = [l for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l][:1]
            print("  %-6s ❌ 未产出 %s" % (tag, err[0][:90] if err else ""))
            continue
        p = os.path.join(od, b[0])
        ctx[tag] = p
        hs[tag] = md5(p)
        print("  %-6s %s  %7.2f MB  %s" % (tag, hs[tag][:16], os.path.getsize(p) / 1e6, json.dumps(opts)))

    if "CTRL" not in ctx:
        sys.exit("❌ 对照臂未产出，实验无法进行")
    valid = ["CTRL"]
    print("")
    print("  [V0] 各臂与 CTRL 的产物是否不同：")
    for tag in ("O1", "O3", "DLBC"):
        if tag not in hs:
            continue
        diff = hs[tag] != hs["CTRL"]
        print("    %-6s %s" % (tag, "✅ 不同（生效）" if diff else "❌ 与 CTRL 相同 ⇒ 该臂无效果，退出判定"))
        if diff:
            valid.append(tag)

    # ---- 正确定点参考 ----
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
    print("=== 逐臂上机 ===")
    res = {}
    for tag in valid:
        subprocess.run([ADB, "-s", DEV, "push", ctx[tag], "%s/od_%s.bin" % (T, tag)],
                       capture_output=True, timeout=3600)
        od = "%s/od_%s" % (T, tag)
        adb("mkdir -p %s && rm -rf %s/Result_0" % (od, od))
        adb("printf '%%s\\n' 'x:=%s/p0b_x.raw' > %s/list.txt" % (T, od))
        o = adb("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
                "./qnn-net-run --retrieve_context od_%s.bin --backend libQnnHtp.so "
                "--input_list od_%s/list.txt --output_dir %s --log_level error 2>&1 | tail -3"
                % (T, T, T, tag, tag, od))
        if "Finished Executing Graphs" not in o:
            print("  %-6s ❌ 执行失败：%s" % (tag, o.strip()[:150]))
            continue
        rp = "%s/Result_0/linear_99_fc.raw" % od
        nb = int(adb("stat -c %%s %s" % rp).strip())
        if nb != BYTES:
            print("  %-6s ❌ V2 字节数 %d ≠ %d" % (tag, nb, BYTES))
            continue
        lp = os.path.join(WORK, "out_%s.raw" % tag)
        subprocess.run([ADB, "-s", DEV, "pull", rp, lp], capture_output=True, timeout=3600)
        y = np.fromfile(lp, np.float32, count=ROWS * N).reshape(ROWS, N).astype(np.float64)
        res[tag] = bulk(y_fxp, y)
        print("  %-6s E = %.4f%%" % (tag, res[tag]))

    if "CTRL" not in res:
        sys.exit("❌ 对照臂未测出，无法判定")
    ec = res["CTRL"]
    print("")
    print("=== 主判据（一律对 CTRL 比，不对历史值比）===")
    print("  E_CTRL = %.4f%%" % ec)
    verdicts = []
    for tag in ("O1", "O3", "DLBC"):
        if tag not in res:
            continue
        r = res[tag] / ec
        if r <= 0.90:
            v = "🔴 有算子级精度收益"
        elif r >= 1.10:
            v = "⚠️ 有害"
        else:
            v = "✅ 无精度影响"
        verdicts.append(r)
        print("  %-6s E = %.4f%%   E/E_CTRL = %.4f   %s" % (tag, res[tag], r, v))
    if verdicts and all(0.90 < r < 1.10 for r in verdicts):
        print("")
        print("  ⇒ 全部落在 0.90~1.10 ⇒ **`O` 与 `dlbc` 对算子级精度无影响，#94 关闭**")
    print("")
    print("=== 附带记录（非判据）===")
    print("  E_CTRL %.4f%% vs 历史无 config 的 %.4f%%   比值 %.4f" % (ec, E_HIST, ec / E_HIST))
    print("  ⇒ 「仅仅加上 config 文件」本身%s显著影响精度"
          % ("" if abs(ec / E_HIST - 1) > 0.05 else "不"))
    json.dump(res, open(os.path.join(WORK, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
