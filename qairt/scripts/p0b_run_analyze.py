"""P0-B 第 5~6 步：设备上跑单算子重放，并按事前锁定的判据判定。

方案：scripts/EXP_PLAN_P0B_FC_REPLAY.md

判据（不得事后修改）：
  E_standalone >= 1.4%  => 真实数据/encoding 就能触发 => 算子级
  E_standalone <= 0.4%  => 只有完整图才触发     => 图层面（#33/#34 升为首选）
  0.4% ~ 1.4%           => 两者皆有

⚠️ 本脚本自带一道【模拟器已知答案门】：用同一套定点模拟复算 **图内** HTP 的误差，
   必须复现 #39 的 2.72~2.87%。复现不出 => 模拟器或口径不对，standalone 的数字一律不可采信。
"""
import os, sys, json, subprocess, hashlib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
Y_HTP_INGRAPH = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_probe",
                             "htp", "linear_99_fc.raw")
W_FP32 = os.path.join(P0B, "val_1800_fp32.npy")
MODE = (sys.argv[1] if len(sys.argv) > 1 else "baseline").lower()
assert MODE in ("baseline", "symtensor", "perrow")
CTX = os.path.join(P0B, "standalone_" + MODE, "ctx", "fc99_%s_ctx.SM8750.bin" % MODE)
M, K, N = 4128, 3840, 3840
ROWS = 512                       # 与 fxp_sim.py 一致：前 512 行，够统计且省内存
BYTES = M * N * 4
E_INGRAPH_LO, E_INGRAPH_HI = 2.5, 3.1      # #39 实测 2.72~2.87%，留一点余量
GATE_OP, GATE_GRAPH = 1.4, 0.4


def sh(cmd, timeout=1800):
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("adb shell rc=%d: %s\nstderr: %s" % (r.returncode, cmd[:120], r.stderr))
    return r.stdout


def push(l, r):
    x = subprocess.run([ADB, "-s", DEV, "push", l, r], capture_output=True, text=True, timeout=3600)
    if x.returncode != 0:
        raise RuntimeError("push failed %s\n%s" % (l, x.stderr))


def pull(r, l):
    x = subprocess.run([ADB, "-s", DEV, "pull", r, l], capture_output=True, text=True, timeout=3600)
    if x.returncode != 0:
        raise RuntimeError("pull failed %s\n%s" % (r, x.stderr))


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bulk(a, b):
    """主体口径（|a|<=p99）相对 L2 与余弦（约束 7），与 fxp_sim.py 同口径"""
    a = a.ravel(); b = b.ravel()
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    rel = 100.0 * np.linalg.norm(y - x) / np.linalg.norm(x)
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    return rel, cos


def q(v, s, o, bw):
    return np.clip(np.rint(v / s) - o, 0, (1 << bw) - 1)


def main():
    enc = json.load(open(os.path.join(P0B, "linear_99_encodings_baseline.json"), encoding="utf-8"))
    ex, ew, ey = (enc["node_linear_99_pre_reshape"], enc["val_1800"], enc["linear_99_fc"])

    # ---------------- 设备执行 ----------------
    out_dir = "%s/p0b_%s" % (T, MODE)
    print("[V3] 输入 md5 = %s  (与图内那次同一份文件)" % md5(X_RAW))
    print("[推送] context %.1f MB + 输入 %.1f MB ..." % (os.path.getsize(CTX)/1e6,
                                                        os.path.getsize(X_RAW)/1e6), flush=True)
    push(CTX, "%s/fc99_%s.bin" % (T, MODE))
    push(X_RAW, "%s/p0b_x.raw" % T)
    sh("mkdir -p %s && rm -rf %s/Result_0" % (out_dir, out_dir))
    sh("printf '%%s\\n' 'x:=%s/p0b_x.raw' > %s/list.txt" % (T, out_dir))
    print("[执行] qnn-net-run ...", flush=True)
    o = sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
           "./qnn-net-run --retrieve_context fc99_%s.bin --backend libQnnHtp.so "
           "--input_list p0b_%s/list.txt --output_dir %s --log_level error 2>&1 | tail -3"
           % (T, T, T, MODE, MODE, out_dir))
    if "Finished Executing Graphs" not in o:
        raise RuntimeError("V1/V2 FAIL qnn-net-run:\n%s" % o)
    print("  [V2a] Finished Executing Graphs  ✅")
    rp = "%s/Result_0/linear_99_fc.raw" % out_dir
    nb = int(sh("stat -c %%s %s" % rp).strip())
    print("  [V2b] 输出字节数 %d（期望 %d）  %s" % (nb, BYTES, "✅" if nb == BYTES else "❌"))
    if nb != BYTES:
        raise RuntimeError("V2 FAIL：字节数不符（约束 3）")
    local_y = os.path.join(P0B, "standalone_%s_out.raw" % MODE)
    pull(rp, local_y)

    # ---------------- 正确定点参考 ----------------
    print("")
    print("[模拟] 正确定点（前 %d 行）..." % ROWS, flush=True)
    x = np.fromfile(X_RAW, np.float32, count=ROWS * K).reshape(ROWS, K).astype(np.float64)
    W = np.load(W_FP32).astype(np.float64)
    qx = q(x, ex["scale"], ex["offset"], ex["bitwidth"]) + ex["offset"]
    if MODE == "perrow":
        # P0-B3：per-row 对称。激活 encoding 也取自 per-row DLC（与基线略有差异，见方案 §八·3 划界）
        pr = json.load(open(os.path.join(P0B, "linear_99_encodings.json"), encoding="utf-8"))["activations"]
        ex = pr["node_linear_99_pre_reshape"]; ey = pr["linear_99_fc"]
        qx = q(x, ex["scale"], ex["offset"], ex["bitwidth"]) + ex["offset"]
        sc = np.load(os.path.join(P0B, "val_1800_perrow_scales.npy"))
        qw = np.clip(np.rint(W / sc[None, :]), -128, 127)
        w_scale = None
        print("  [配置] perrow：3840 个 scale，offset=0；激活 scale=%.12g（基线 %.12g）"
              % (ex["scale"], enc["node_linear_99_pre_reshape"]["scale"]))
    elif MODE == "symtensor":
        # P0-B2：对称 per-tensor —— 权重 encoding 换成对称，激活保持基线不变（单变量）
        s_sym = float(np.abs(W).max() / 127.0)
        qw = np.clip(np.rint(W / s_sym), -128, 127)
        w_scale = s_sym
        print("  [配置] symtensor：权重 scale=%.12g offset=0" % s_sym)
    else:
        qw = q(W, ew["scale"], ew["offset"], ew["bitwidth"]) + ew["offset"]
        w_scale = ew["scale"]
    # 幅度校验：float64 对整数乘加是否仍精确
    amax = float(np.abs(qx).max() * np.abs(qw).max() * K)
    print("  |acc| 上界 %.3g  < 2^53 = %.3g  %s" % (amax, 2.0**53, "✅ float64 精确" if amax < 2**53 else "❌"))
    acc = qx @ qw
    if MODE == "perrow":
        y_fxp = ex["scale"] * (acc * sc[None, :])      # per-row：每列自己的 scale
    else:
        y_fxp = ex["scale"] * w_scale * acc
    # 再按输出 encoding 重量化（HTP 也会做这一步）
    y_fxp = ey["scale"] * (q(y_fxp, ey["scale"], ey["offset"], ey["bitwidth"]) + ey["offset"])

    y_in = np.fromfile(Y_HTP_INGRAPH, np.float32, count=ROWS * N).reshape(ROWS, N).astype(np.float64)
    y_st = np.fromfile(local_y, np.float32, count=ROWS * N).reshape(ROWS, N).astype(np.float64)

    e_in, c_in = bulk(y_fxp, y_in)
    e_st, c_st = bulk(y_fxp, y_st)

    print("")
    print("=== 模拟器已知答案门（仅 baseline 适用；图内那次就是 baseline 配置）===")
    print("  E_ingraph = %.4f%%（余弦 %.6f）   #39 记录 2.72~2.87%%   %s"
          % (e_in, c_in, "✅ 复现" if E_INGRAPH_LO <= e_in <= E_INGRAPH_HI else "❌ 未复现"))
    if MODE == "baseline" and not (E_INGRAPH_LO <= e_in <= E_INGRAPH_HI):
        print("  ⇒ 模拟器或口径与 #39 不一致，**standalone 的数字一律不可采信**（约束 8）")
        return
    if MODE != "baseline":
        print("  （本臂换了权重 encoding，E_ingraph 不再可比，此门仅供参考）")

    print("")
    print("=== 主判据 ===")
    print("  E_standalone = %.4f%%（余弦 %.6f）" % (e_st, c_st))
    print("  E_ingraph    = %.4f%%" % e_in)
    print("  比值 standalone/ingraph = %.3f" % (e_st / max(e_in, 1e-9)))
    if MODE == "perrow":
        E_B = 0.8310
        print("  [P0-B3 判据] E_B(对称 per-tensor) = %.4f%%   E_C(per-row) = %.4f%%   比值 %.3f"
              % (E_B, e_st, e_st / E_B))
        if e_st <= 0.6 * E_B:
            v = ("🔴 **细化 scale 在消除 offset 之上仍有实质收益** ⇒ 两机制叠加，"
                 "从机制上解释了 #47 为何是唯一有效手段")
        elif e_st >= 0.9 * E_B:
            v = "✅ **细化 scale 无额外收益** ⇒ per-row 的收益主要来自对称化本身"
        else:
            v = "🔶 **部分叠加**"
        print("  判定: %s" % v)
        np.save(os.path.join(P0B, "p0b_%s_yfxp512.npy" % MODE), y_fxp.astype(np.float32))
        return
    if MODE == "symtensor":
        E_A = 1.5544
        print("  [P0-B2 判据] E_A(非对称基线) = %.4f%%   E_B(对称) = %.4f%%   比值 %.3f"
              % (E_A, e_st, e_st / E_A))
        if e_st <= 0.5 * E_A:
            v = "🔴 **offset 项是主要触发因素** ⇒ #29/#40 获强支持"
        elif e_st >= 0.9 * E_A:
            v = "✅ **offset 项不是触发因素** ⇒ #29/#40 在算子级被证否，转查其它性质"
        else:
            v = "🔶 **部分贡献** ⇒ 记录比例，两条线都留"
        print("  判定: %s" % v)
        np.save(os.path.join(P0B, "p0b_%s_yfxp512.npy" % MODE), y_fxp.astype(np.float32))
        return
    if e_st >= GATE_OP:
        v = ("🔴 **真实数据/encoding 就能触发** ⇒ 问题在算子级；#33/#34 降级，"
             "转查这类输入的什么性质触发了它")
    elif e_st <= GATE_GRAPH:
        v = ("🔴 **只有完整图才触发** ⇒ 问题在图层面；**#33/#34（融合、VTCM 分块）升为首选**，"
             "且 #39 不得再解读为「HTP 单算子有问题」")
    else:
        v = "🔶 **两者皆有**（落在 0.4%~1.4% 未定区间）⇒ 两条线都要查，记录比例"
    print("  判定: %s" % v)
    np.save(os.path.join(P0B, "p0b_%s_yfxp512.npy" % MODE), y_fxp.astype(np.float32))
    print("")
    print("[落盘] p0b/standalone_%s_out.raw, p0b_%s_yfxp512.npy" % (MODE, MODE))


if __name__ == "__main__":
    main()
