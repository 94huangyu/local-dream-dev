"""#114 步 1：实测某一段**每个 16-bit 非静态激活**的量程与两种表示的主体误差。

方案：`scripts/EXP_PLAN_MAXFP16.md`（判据事前锁定，本脚本不得偏离）。

口径（与 `seg_dtype_build.py` 逐字一致）：
  候选 = `<seg>_full_enc.csv` 里 **非 STATIC 且 bitwidth==16** 的张量。

每个张量报：`|a|max`（实测）、`median|a|`、`p99`、
  `E_bulk_ufxp16`（用**部署 encoding** 做 quantize-dequantize）、`E_bulk_fp16`、
  **以及全量口径 `E_all_*` 与集中度 `conc1`（前 1% 元素占 ||a||^2 的比例，约束 7 强制）**，
  以及 标定`|max|` 与实测的比值（#110：0.09x~1.90x 不可预测，故必须实测）。

🔴 主体口径（`|a| <= p99`，约束 7）**会把 inf 藏起来** => 溢出必须只看 `|a|max`。

🔴 **一次声明 606 个图输出会把宿主压死**（2026-08-24 实测：RSS 9.18 GB、可用内存降到
   0.3 GB、8 分钟里一个批次都没跑完）。机制：**张量一旦成为 graph output，ORT 的内存
   规划器就不再为它复用缓冲区** —— 与 `run()` 只请求其中一部分无关，整张图的激活会同时
   驻留（part2a 合计 87 GB）。=> 必须**每批单独建 session、只声明该批的输出**。

🔴 DLC 名 != ONNX 名：FullyConnected 的输出在 DLC 里叫 `<x>_fc`，ONNX 里叫 `<x>`。

用法: python maxfp16_stats.py <part1a|part1b|part2a|part2b> [--budget-gb 6]
      python maxfp16_stats.py part1b --only mul_406,mul_431,mul_456   # 只测指定张量
      python maxfp16_stats.py part2a --g0        # 只跑 G0-known（6 个已发表张量）
"""
import os
import re
import io
import gc
import sys
import json
import time
import subprocess

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
FP16MAX = 65504.0
CHUNK = 8 << 20
NMAX = 32          # 每批最多钉住几个输出，见文件头注释

# 四段输入契约：与 seg_dtype_build.py 的 CFG 同源（同一份 step-0 数据）
CFG = {
    "part1a": ("transformer_part1a",
               [("latents", os.path.join(P0, "htp_inloop", "s0", "latents.raw"), (1, 16, 128, 128)),
                ("timestep", os.path.join(P0, "htp_inloop", "s0", "timestep.raw"), (1,)),
                ("caption", os.path.join(P0, "htp_inloop", "const", "caption.raw"), (1, 32, 2560)),
                ("cap_pad_mask", os.path.join(P0, "htp_inloop", "const", "cap_pad_mask.raw"),
                 (1, 32))]),
    "part1b": ("transformer_part1b",
               [(n, os.path.join(P0, "testB", "s0_transformer_part1b", n + ".raw"), s)
                for n, s in [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
                             ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
                             ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))]]),
    "part2a": ("transformer_part2a_fixed",
               [(n, os.path.join(P0, "testB", "s0_transformer_part2", n + ".raw"), s)
                for n, s in [("unified", (1, 4128, 3840)), ("unified_mask", (1, 4128)),
                             ("unified_freqs", (1, 4128, 64, 2)), ("adaln_input", (1, 256))]]),
    "part2b": ("transformer_part2b_fixed",
               [(n, os.path.join(W, "fp32_s0_part2b", n + ".raw"), s)
                for n, s in [("add_92", (1, 4128, 3840)), ("select", (1, 4128, 1, 64)),
                             ("select_1", (1, 4128, 1, 64)), ("split_7_split_2", (1, 1, 3840)),
                             ("split_7_split_3", (1, 1, 3840)), ("val_105", (1, 1, 1, 4128)),
                             ("adaln_input", (1, 256))]]),
}

# --- Tier 2 门（EXP_PLAN_T2_CLIPSET §四 步 3）：L=80 变体 ---
# 输入全部来自 `t2_chain_L80.py emit` dump 的**同一条 FP32 链**，
# 不混用盘上 4128 长的旧产物（那是 CPU 参考链，见 #116）。
L80 = 80
U80 = 4096 + L80                     # 4176
CH80 = os.path.join(P0, "L80_chain")


def _cfg80(seg):
    """把 CFG[seg] 里的形状 32->80 / 4128->4176，路径改到 L80_chain。"""
    onnx_name, ins = CFG[seg]
    out = []
    for n, _p, shape in ins:
        sh = tuple(L80 if d == 32 else U80 if d == 4128 else d for d in shape)
        out.append((n, os.path.join(CH80, "s0_%s" % seg, n + ".raw"), sh))
    return onnx_name + "_L%d" % L80, out


TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[([^\]]*)\]; "
                r"tensor type: ([A-Z]+)\)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")

# G0-known：§15.42.3 已发表的 6 行（|a|max, uFxp16 主体%, FP16 主体%）
G0 = {"mul_4": (10.3, 0.0069, 0.0206), "mul_6": (16.3, 0.0101, 0.0208),
      "mul_19": (21.5, 0.5192, 0.0204), "linear_1": (98.0, 0.0575, 0.0209),
      "add_8": (6206.0, 4.3243, 0.0209), "mul_21": (9818.0, 53.7738, 0.0205)}


def free_gb():
    """宿主可用物理内存（GB）。取不到返回 None，调用方按「未知」处理。"""
    try:
        r = subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        for tok in r.stdout.split():
            if tok.strip().isdigit():
                return int(tok.strip()) / 1048576.0
    except Exception:
        return None
    return None


def parse_enc(seg):
    kind, dim, pt = {}, {}, {}
    with io.open(os.path.join(W, "%s_full_enc.csv" % seg), encoding="utf-8",
                 errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), m.group(4))
                dim.setdefault(m.group(1), m.group(3))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), dict(bw=int(m.group(2)), mn=float(m.group(3)),
                                               mx=float(m.group(4)), sc=float(m.group(5)),
                                               off=float(m.group(6))))
    return kind, dim, pt


def stats(flat, enc):
    """主体口径 |a| <= p99 下的两种表示误差；|a|max 用全量（约束 7：主体会藏 inf）。"""
    n = flat.size
    aa = np.abs(flat)
    amax = float(aa.max())
    k99, k50 = min(int(0.99 * n), n - 1), n // 2
    part = np.partition(aa, [k50, k99])
    med, thr = float(part[k50]), float(part[k99])
    del aa, part

    # 部署 encoding 的可表示区间（QNN：value = scale * (q + offset)，q in [0, 2^bw-1]）
    sc = enc["sc"]
    lo, hi = enc["off"] * sc, (2 ** enc["bw"] - 1 + enc["off"]) * sc
    # 🔴 两个口径都要算（约束 7）：
    #    主体（|a| <= p99）—— 决策用；全量 —— 因为**主体口径按定义排除了离群值**，
    #    而 #112 的机制假设正是「FP16 救主体、毁离群值」。只看主体会系统性地漏掉那种净亏。
    #    另报「前 1% 元素占 ||a||^2 的比例」(conc1)：约束 7 规定用相对 L2 必须同时给这个数。
    num_u = num_f = den = 0.0
    num_ua = num_fa = dena = 0.0
    nb = 0
    for i in range(0, n, CHUNK):
        w = flat[i:i + CHUNK].astype(np.float64)
        dena += float(np.dot(w, w))
        d = np.clip(np.round(w / sc) * sc, lo, hi) - w
        num_ua += float(np.dot(d, d))
        d = w.astype(np.float16).astype(np.float64) - w
        # FP16 溢出会变 inf => 全量口径里 d 可能是 inf/nan，按「已毁」记，不许静默变 0
        dd = np.dot(d, d)
        num_fa += float(dd) if np.isfinite(dd) else float("inf")
        c = w[np.abs(w) <= thr]
        if c.size == 0:
            continue
        nb += c.size
        den += float(np.dot(c, c))
        d = np.clip(np.round(c / sc) * sc, lo, hi) - c
        num_u += float(np.dot(d, d))
        d = c.astype(np.float16).astype(np.float64) - c
        num_f += float(np.dot(d, d))

    def pct(num, d):
        if d <= 0:
            return None
        v = (num / d) ** 0.5
        return 100.0 * v if np.isfinite(v) else float("inf")

    out = dict(amax=amax, med=med, p99=thr, n=int(n), nbulk=int(nb),
               e_ufxp=pct(num_u, den), e_fp16=pct(num_f, den),
               e_ufxp_all=pct(num_ua, dena), e_fp16_all=pct(num_fa, dena),
               conc1=(100.0 * (1.0 - den / dena)) if dena > 0 else None)
    return out


def main():
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort

    seg = sys.argv[1]
    g0only = "--g0" in sys.argv
    budget = 6.0
    if "--budget-gb" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget-gb") + 1])
    l80 = "--L80" in sys.argv
    pure32 = "--pure32" in sys.argv
    if l80 and pure32:
        sys.exit("FAIL --L80 与 --pure32 互斥")
    onnx_name, ins = _cfg80(seg) if l80 else CFG[seg]
    if pure32:
        # #116（方案 scripts/EXP_PLAN_116.md）：**只换段间输入的血统**，其余一字不改。
        # 原 CFG 的段间输入取自 testB（= CPU 跑量化 transformer 的逐段输出，#35/#116），
        # 这里换成 `t2_chain_L80.py dump32` 产出的**纯 FP32 四段链**输出。
        # part1a 无段间输入（它吃 latents/timestep/caption）=> 路径不变。
        _P32 = os.path.join(P0, "L32_pure", "s0_%s" % seg)
        ins = [(n, os.path.join(_P32, n + ".raw"), sh) for n, _p, sh in ins]
        print("[纯 FP32 血统] 输入取自 %s" % _P32, flush=True)
    if l80:
        print("[L=80 模式] 源 %s.onnx，输入取自 %s" % (onnx_name, CH80), flush=True)

    kind, dim, pt = parse_enc(seg)
    cand = sorted(a for a, e in pt.items() if kind.get(a) != "STATIC" and e["bw"] == 16)
    print("[%s] 候选（非 STATIC 且 bw==16）**%d** 个" % (seg, len(cand)), flush=True)

    m = onnx.load(os.path.join(D, onnx_name + ".onnx"), load_external_data=False)
    node_out = {o for n in m.graph.node for o in n.output}
    gin = {i.name for i in m.graph.input}
    gout = {o.name for o in m.graph.output}

    def to_onnx(t):
        if t in node_out:
            return t
        if t.endswith("_fc") and t[:-3] in node_out:
            return t[:-3]
        return None

    # 规则 3/4（EXP_PLAN §3.2）：I/O 边界张量与 ONNX 里映射不到的名字，一律出局
    drop_io = [t for t in cand if t in gin or t in gout]
    pool = [t for t in cand if t not in gin and t not in gout]
    dlc2onnx = {t: to_onnx(t) for t in pool}
    drop_missing = sorted(t for t, v in dlc2onnx.items() if v is None)
    measur = sorted(t for t, v in dlc2onnx.items() if v is not None)
    nfc = sum(1 for t in measur if dlc2onnx[t] != t)
    print("  出局：I/O 边界 %d（%s）｜ONNX 里映射不到 %d ⇒ **可测 %d**（其中 `_fc` 映射 %d）"
          % (len(drop_io), ",".join(sorted(drop_io)[:6]) or "-", len(drop_missing),
             len(measur), nfc), flush=True)
    if drop_missing:
        suf = {}
        for t in drop_missing:
            k = re.sub(r"_\d+", "_N", t)
            suf[k] = suf.get(k, 0) + 1
        print("    映射不到的名字形态（前 8 类）: %s"
              % ", ".join("%s x%d" % (k, v) for k, v in
                          sorted(suf.items(), key=lambda x: -x[1])[:8]), flush=True)

    only = None
    if "--only" in sys.argv:
        only = [x for x in sys.argv[sys.argv.index("--only") + 1].split(",") if x]
        miss = [x for x in only if x not in measur]
        if miss:
            sys.exit("FAIL --only 里这些不在可测集: %s" % miss)
        measur = list(only)
        print("  [--only] 只测 %d 个: %s" % (len(measur), measur), flush=True)

    if g0only:
        measur = [t for t in measur if dlc2onnx[t] in G0]
        print("  [G0-known] 只测 %d 个已发表张量: %s"
              % (len(measur), [dlc2onnx[t] for t in measur]), flush=True)
        if len(measur) != len(G0):
            sys.exit("FAIL G0：已发表张量不在可测集里（%s）"
                     % sorted(set(G0) - {dlc2onnx[t] for t in measur}))

    fetch = sorted({dlc2onnx[t] for t in measur})
    onnx2dlc = {}
    for t in measur:
        onnx2dlc.setdefault(dlc2onnx[t], []).append(t)

    feeds = {}
    for n, path, shape in ins:
        if not os.path.exists(path):
            sys.exit("FAIL 输入不存在: %s" % path)
        feeds[n] = np.fromfile(path, np.float32).reshape(shape)
    for i in m.graph.input:
        if i.type.tensor_type.elem_type == TensorProto.BOOL:
            feeds[i.name] = feeds[i.name].astype(bool)

    def nel(t):
        try:
            v = 1
            for x in dim.get(t, "").split(","):
                v *= int(x.strip())
            return v
        except Exception:
            return 16 << 20

    # 按字节预算分批。峰值 ~= 本批张量字节 + 常驻工作集，故预算要留余量（约束 9 第 7 条）
    batches, cur, cb = [], [], 0
    for t in sorted(fetch, key=lambda x: nel(onnx2dlc[x][0])):
        b = nel(onnx2dlc[t][0]) * 4
        if cur and (cb + b > budget * 1e9 or len(cur) >= NMAX):
            batches.append(cur)
            cur, cb = [], 0
        cur.append(t)
        cb += b
    if cur:
        batches.append(cur)
    print("  分 %d 批（预算 %.1f GB/批；**每批单独建 session**）"
          % (len(batches), budget), flush=True)

    base_out = list(m.graph.output)
    dst = os.path.join(D, "%s_mf16%s.onnx" % (onnx_name, "_g0" if g0only else ""))
    res, t00 = {}, time.time()
    for bi, b in enumerate(batches):
        free = free_gb()
        if free is not None and free < 3.0:
            sys.exit("FAIL 宿主可用内存只剩 %.1f GB，停止（约束 9 第 7 条）" % free)
        del m.graph.output[:]
        m.graph.output.extend(base_out)
        have = {o.name for o in m.graph.output}
        for t in b:
            if t not in have:
                m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
        onnx.save(m, dst)
        t0 = time.time()
        sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        ts = time.time() - t0
        t0 = time.time()
        outs = sess.run(b, feeds)
        tf = time.time() - t0
        t0 = time.time()
        for oname, a in zip(b, outs):
            flat = np.ascontiguousarray(a).ravel()
            for name in onnx2dlc[oname]:      # 同一 ONNX 张量可对应多个 DLC 名
                e = pt[name]
                r = stats(flat, e)
                r["onnx"] = oname
                r["calib_max"] = max(abs(e["mn"]), abs(e["mx"]))
                r["scale"] = e["sc"]
                res[name] = r
            del flat
        del outs, sess
        gc.collect()
        print("  批 %d/%d  %d 张量  session %.0fs  前向 %.0fs  统计 %.0fs  "
              "可用内存 %s  累计 %.1f min"
              % (bi + 1, len(batches), len(b), ts, tf, time.time() - t0,
                 ("%.1f GB" % free) if free is not None else "?",
                 (time.time() - t00) / 60), flush=True)

    if g0only:
        print("")
        print("[G0-known] 对照 §15.42.3 已发表值：")
        print("%-10s %10s %10s | %9s %9s | %9s %9s"
              % ("tensor", "|a|max", "发表", "uFxp16%", "发表", "FP16%", "发表"))
        ok = True
        byonnx = {r["onnx"]: r for r in res.values()}
        for t, (pm, pu, pf) in G0.items():
            r = byonnx[t]
            du = abs(r["e_ufxp"] - pu) / max(pu, 1e-9)
            df = abs(r["e_fp16"] - pf) / max(pf, 1e-9)
            dm = abs(r["amax"] - pm) / max(pm, 1e-9)
            bad = du > 0.05 or df > 0.05 or dm > 0.02
            ok = ok and not bad
            print("%-10s %10.2f %10.1f | %9.4f %9.4f | %9.4f %9.4f  %s"
                  % (t, r["amax"], pm, r["e_ufxp"], pu, r["e_fp16"], pf, "" if not bad else "X"))
        print("")
        print("[G0-known] %s" % ("PASS 统计实现可用" if ok else
                                 "FAIL 实现与已发表值不符，不得用它出白名单"))
        return 0 if ok else 1

    out = os.path.join(W, "%s%s_amax%s.json"
                       % (seg, "_L80" if l80 else ("_pure32" if pure32 else ""),
                          "_only" if only else ""))
    json.dump({"seg": seg, "n_cand": len(cand), "n_drop_io": len(drop_io),
               "drop_io": sorted(drop_io), "n_drop_missing": len(drop_missing),
               "tensors": res}, open(out, "w"), indent=0)
    print("")
    print("⇒ 写出 %s（%d 个张量）" % (out, len(res)), flush=True)

    ov = [t for t, r in res.items() if r["amax"] * 2.0 > FP16MAX]
    fx = [t for t, r in res.items()
          if r["amax"] * 2.0 <= FP16MAX and r["e_ufxp"] is not None
          and r["e_ufxp"] <= r["e_fp16"]]
    white = [t for t, r in res.items()
             if r["amax"] * 2.0 <= FP16MAX and r["e_ufxp"] is not None
             and r["e_ufxp"] > r["e_fp16"]]
    print("  规则①溢出出局 %d ｜规则②定点已更优出局 %d ｜**入选 %d**"
          % (len(ov), len(fx), len(white)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
