# -*- coding: utf-8 -*-
"""D2-H1 · C2：真实输入下的三方对照（判据见 `EXP_PLAN_P2_SPEED.md` §6.7，事前锁定）。

三臂：A = 部署 context｜B = 手术 context｜R = 宿主 FP32 ONNX 参考。
输入取真实 step-0（`p0_experiments/L80_chain/s0_part1a/` 的 FP32）。

🔴 口径设计（为什么这样才是单变量）
  · A/B 喂**同一份原生输入**（把 FP32 按契约 encoding 量化）。
  · R 喂**反量化后的同一份输入**（量化→反量化），使「输入量化误差」成为三臂共有项，
    **只剩图内部的差异**。否则 R 与 A/B 之间会混入输入量化这一项。
  · 报口径按约束 7：全量 + 主体(|a|<=p99) + 余弦 + 前 1% 元素占 ‖a‖² 的比例，四个一起报。

子命令:
  inputs   造原生输入与反量化 FP32 输入（纯宿主，秒级）
  fp32     跑 FP32 参考（🔴 约 10 GB 内存，**单独跑，别和建图并行**）
  device   设备两臂（需插线；context 按用户决定**暂存**在设备上，不删）
  compare  三方比对并按事前判据出结论（纯宿主）
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import p2_d1_profile as P  # noqa: E402

SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "L80_chain", "s0_part1a")
OUT = os.path.join(P.REPO, "logs", "p2_20260917", "c2")
NATIVE = os.path.join(OUT, "native_in")
DEQ = os.path.join(OUT, "deq_in")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx",
                    "transformer_part1a_clip_L80.onnx")
NEW_CTX = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_noscat",
                       "ctx_part1a_noscat", "part1a_noscat.SM8750.bin")
ARMS = {"A_deployed": P.HOST_CTX["part1a"], "B_noscat": NEW_CTX}
DEV = "/data/local/tmp/p2ab"        # 用户决定：context 暂存在设备上
TARGET = "add_138"


def spec(name):
    for t in P.contract_graph("part1a")["inputs"]:
        if t["name"] == name:
            return t
    raise SystemExit("🔴 契约里没有输入 %s" % name)


def cmd_inputs():
    os.makedirs(NATIVE, exist_ok=True)
    os.makedirs(DEQ, exist_ok=True)
    print("真实 step-0 输入源：%s" % SRC)
    for t in P.contract_graph("part1a")["inputs"]:
        n = t["name"]
        f32 = np.fromfile(os.path.join(SRC, n + ".raw"), dtype=np.float32)
        if t["physical_dtype"] == "QNN_DATATYPE_BOOL_8":
            q = (f32 > 0.5).astype(np.uint8)
            deq = q.astype(np.float32)
        else:
            s, off = t["quantization"]["scale"], t["quantization"]["offset"]
            q = np.clip(np.rint(f32 / s - off), 0, 65535).astype(np.uint16)   # 与 app 的 quantize() 同式
            deq = (q.astype(np.float32) + off) * s
        q.tofile(os.path.join(NATIVE, n + ".raw"))
        deq.tofile(os.path.join(DEQ, n + ".raw"))
        nb = os.path.getsize(os.path.join(NATIVE, n + ".raw"))
        if nb != t["exact_bytes"]:
            raise SystemExit("🔴 %s 原生字节 %d ≠ 契约 %d" % (n, nb, t["exact_bytes"]))
        err = np.abs(deq - f32)
        print("  %-14s %d 元素｜原生 %d B ✅｜量化往返最大误差 %.6g（%.2f%% 量程）"
              % (n, f32.size, nb, err.max(), 100 * err.max() / (f32.max() - f32.min() + 1e-12)))
    print("⇒ 原生输入 %s\n⇒ 反量化 FP32 输入 %s" % (NATIVE, DEQ))
    return 0


def cmd_fp32():
    import onnxruntime as ort
    t0 = time.time()
    print("🔴 本步约需 10 GB 内存，确认没有别的重活在跑。加载 %s" % os.path.basename(ONNX), flush=True)
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    feed = {}
    for t in P.contract_graph("part1a")["inputs"]:
        n = t["name"]
        v = np.fromfile(os.path.join(DEQ, n + ".raw"), dtype=np.float32)
        if n == "latents":
            v = v.reshape(1, 16, 128, 128)
        elif n == "caption":
            v = v.reshape(1, 80, 2560)
        elif n == "cap_pad_mask":
            v = (v.reshape(1, 80) > 0.5)
        else:
            v = v.reshape(1)
        feed[n] = v
    print("  会话就绪 %.0f s，开始推理…" % (time.time() - t0), flush=True)
    out = sess.run([TARGET], feed)[0].astype(np.float32)
    out.tofile(os.path.join(OUT, "ref_%s_fp32.raw" % TARGET))
    print("  %s FP32 参考：shape %s  std %.4f  |max| %.2f  用时 %.0f s"
          % (TARGET, out.shape, out.std(), np.abs(out).max(), time.time() - t0))
    return 0


def cmd_device():
    import lab_dev
    lab_dev.require_online("C2 device")
    print("设备可用内存 %d MiB，NPU %.1f °C" % (lab_dev.mem_available_mb(), P.npu_c()))
    lab_dev.sh("am force-stop %s" % P.PKG)
    lab_dev.sh("mkdir -p %s/in_real" % DEV)
    names = [t["name"] for t in P.contract_graph("part1a")["inputs"]]
    for n in names:
        h = os.path.join(NATIVE, n + ".raw")
        d = "%s/in_real/%s.raw" % (DEV, n)
        if lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % d).strip() != str(os.path.getsize(h)):
            lab_dev.push(h, d)
    res = []
    for arm, host in ARMS.items():
        d = "%s/%s.bin" % (DEV, arm)
        if lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % d).strip() != str(os.path.getsize(host)):
            print("  push %s（%.2f GiB，之后暂存在设备上）…" % (arm, os.path.getsize(host) / 2**30), flush=True)
            lab_dev.push(host, d)
        got = lab_dev.sh("sha256sum %s" % d).split()[0]
        if got != P.sha256(host):
            raise SystemExit("🔴 %s 设备侧 sha256 与宿主不一致" % arm)
        od = "%s/o_%s_real" % (DEV, arm)
        line = " ".join("%s:=%s/in_real/%s.raw" % (k, DEV, k) for k in names)
        lab_dev.sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' '%s' > %s/list.txt" % (od, od, line, od))
        t0 = time.time()
        o = lab_dev.sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
                       "./qnn-net-run --retrieve_context %s --backend libQnnHtp.so --input_list %s/list.txt "
                       "--output_dir %s --perf_profile burst --use_native_input_files --use_native_output_files "
                       "--profiling_level basic --num_inferences 2 --log_level error; echo RC=$?"
                       % (P.T, P.T, P.T, d, od, od), timeout=1800)
        if "RC=0" not in o:
            raise SystemExit("🔴 %s 失败：%s" % (arm, o[-600:]))
        want = {t["name"]: t["exact_bytes"] for t in P.contract_graph("part1a")["outputs"]}
        ls = lab_dev.sh("for f in %s/Result_0/*.raw; do stat -c '%%n %%s' $f; done; true" % od)
        got_sz = {}
        for l in ls.strip().splitlines():
            p = l.split()
            if len(p) == 2 and p[0].endswith(".raw"):
                got_sz[os.path.basename(p[0])[:-4].replace("_native", "")] = int(p[1])
        bad = [(k, v, got_sz.get(k)) for k, v in want.items() if got_sz.get(k) != v]
        if bad:
            raise SystemExit("🔴 %s 输出字节数与契约不符 ⇒ 作废：%s" % (arm, bad))
        P.pull_robust("%s/Result_0/%s_native.raw" % (od, TARGET), os.path.join(OUT, "%s_%s.raw" % (TARGET, arm)))
        logs = [x for x in lab_dev.sh("ls %s; true" % od).split() if x.startswith("qnn-profiling-data")]
        hd = os.path.join(OUT, arm)
        os.makedirs(hd, exist_ok=True)
        for lg in logs:
            P.pull_robust("%s/%s" % (od, lg), os.path.join(hd, lg))
        accel = P.nums_after("".join(P.view(os.path.join(hd, lg)) for lg in logs), "Accelerator (execute) time")
        print("  ✅ %s 完成 %.1f s｜9 个输出字节数 == 契约｜加速器 %s us" % (arm, time.time() - t0, accel[:2]))
        res.append({"arm": arm, "accel_us": accel})
        lab_dev.sh("rm -rf %s/Result_0; true" % od)
    json.dump(res, open(os.path.join(OUT, "device.json"), "w", encoding="utf-8"), indent=1)
    print("\n设备临时目录 **保留**（用户决定暂存 context）：%s — 可以拔线" % DEV)
    return 0


def metrics(x, r):
    """按约束 7 的四件套：全量相对 L2、主体(|r|<=p99)、余弦、前 1% 元素占 ‖r‖² 比例。"""
    d = x - r
    p99 = np.percentile(np.abs(r), 99)
    m = np.abs(r) <= p99
    top = np.sort(r.ravel() ** 2)[::-1][:max(1, r.size // 100)]
    return {"full_pct": 100 * np.linalg.norm(d) / np.linalg.norm(r),
            "bulk_pct": 100 * np.linalg.norm(d[m]) / np.linalg.norm(r[m]),
            "cos": float((x * r).sum() / (np.linalg.norm(x) * np.linalg.norm(r))),
            "top1_energy_pct": 100 * top.sum() / (r ** 2).sum()}


def cmd_compare():
    ref = np.fromfile(os.path.join(OUT, "ref_%s_fp32.raw" % TARGET), dtype=np.float32)
    enc = [t for t in P.contract_graph("part1a")["outputs"] if t["name"] == TARGET][0]["quantization"]
    s, off = enc["scale"], enc["offset"]
    out = {}
    for arm in ARMS:
        q = np.fromfile(os.path.join(OUT, "%s_%s.raw" % (TARGET, arm)), dtype=np.uint16)
        v = (q.astype(np.float32) + off) * s
        out[arm] = metrics(v, ref)
        out[arm]["_deq"] = v
    print("# C2 三方对照（真实 step-0 输入，目标张量 %s）\n" % TARGET)
    print("| 臂 | 全量相对 L2 | 主体(|r|<=p99) | 余弦 | 前 1% 能量占比 |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        m = out[arm]
        print("| %s | %.4f%% | %.4f%% | %.6f | %.2f%% |" % (arm, m["full_pct"], m["bulk_pct"], m["cos"], m["top1_energy_pct"]))
    ab = metrics(out["B_noscat"]["_deq"], out["A_deployed"]["_deq"])
    print("\nA 与 B 之间：全量 %.4f%%｜主体 %.4f%%｜余弦 %.6f" % (ab["full_pct"], ab["bulk_pct"], ab["cos"]))
    delta = out["B_noscat"]["full_pct"] - out["A_deployed"]["full_pct"]
    print("\n## 判据（§6.7 事前锁定）\n\nΔ = L2(B,R) − L2(A,R) = **%+.4f pp**" % delta)
    if delta <= 0.1 and delta >= -0.5:
        v = "🟢 **B 不差于 A**（Δ ≤ 0.1 pp）⇒ 手术不是画质退化 ⇒ 「是否接受非逐字节一致」交用户决定"
    elif delta < -0.5:
        v = "🔵 **B 明显更好**（A 差 > 0.5 pp）⇒ 部署图在该处本就偏离 FP32，属画质缺陷线索"
    elif delta > 0.5:
        v = "🔴 **B 明显更差** ⇒ 手术有害，H1 当前形态关闭"
    else:
        v = "🟡 落在 0.1~0.5 pp 的中间地带 ⇒ 不下结论，补样本（换 seed / 换 prompt）"
    print(v)
    rec = {k: {kk: float(vv) for kk, vv in m.items() if not kk.startswith("_")} for k, m in out.items()}
    rec.update({"delta_pp": float(delta), "verdict": v,
                "A_vs_B": {k: float(v2) for k, v2 in ab.items() if not k.startswith("_")}})
    json.dump(rec, open(os.path.join(OUT, "compare.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(OUT, exist_ok=True)
    sys.exit({"inputs": cmd_inputs, "fp32": cmd_fp32, "device": cmd_device,
              "compare": cmd_compare}.get(c, lambda: (_ for _ in ()).throw(SystemExit(__doc__)))())
