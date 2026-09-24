# -*- coding: utf-8 -*-
"""SmoothQuant 步 1：测**逐通道**激活幅度（方案 `scripts/EXP_PLAN_SMOOTHQUANT.md` §四）。

为什么要新测：现有 `<seg>_amax.json` 是**逐张量**的（每张量只有 amax/med/p99/conc1 四个标量），
而 SmoothQuant 的 `s` 是**逐输入通道**的，必须有 `max|X[..., j]|` 这个长度 K 的向量。

口径（与 maxfp16_stats.py 逐字一致，避免两套装置不可比）：
  · 源 ONNX 走 `canonical_sources`（#61/#136/#138/#148：**不得自己拼名字**）
  · 用**当前部署长度** L=80 的那张图
  · 输入取自 `L80_chain`（纯 FP32 血统，#116；不是 testB 的 CPU 参考链）
  · **每批单独建 session、只声明该批输出**（#115：一次声明几百个输出会把宿主压死）

只测「静态权重 MatMul 的激活输入」——注意力的 A@B 两边都是动态张量，
**SmoothQuant 不适用**，不在集合里（这一条要在结论里写明覆盖率）。

用法: python sq_actstats.py [part1b] [--budget-gb 4]
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
from onnx import TensorProto, helper

import canonical_sources as cs

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
CH80 = os.path.join(P0, "L80_chain")
OUT = os.path.join(P0, "smoothquant")

# 段输入契约（形状为 L=80 版；与 maxfp16_stats._cfg80 同源）
INS = {
    "part1b": [("add_138", (1, 4176, 3840)), ("add_131", (1, 1, 3840)),
               ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4176, 1, 64)),
               ("select_46", (1, 4176, 1, 64)), ("adaln_input", (1, 256))],
}

# --sim-src：改到 **sim_qdq.py 实际使用的那张图**（L=32 base，无 Clip）+ 它的 L=32 输入。
# 🔴 为什么必须有这个开关（约束 8）：统计只能在「模拟器实际跑的那张图」上测。
#    默认的 L=80 clip 源与模拟器的 L=32 base 源在**两个轴**上都不同（序列长度、有无 Clip），
#    直接混用会让 s 与重算的 encoding 都对不上。
SIM_ONNX = {"part1a": "transformer_part1a", "part1b": "transformer_part1b",
            "part2a": "transformer_part2a_fixed", "part2b": "transformer_part2b_fixed"}
# 四段的 L=32 输入契约，逐字抄自 maxfp16_stats.CFG（同一份 step-0 数据，口径必须一致）
INS32 = {
    "part1a": [("latents", (1, 16, 128, 128)), ("timestep", (1,)),
               ("caption", (1, 32, 2560)), ("cap_pad_mask", (1, 32))],
    "part1b": [("add_138", (1, 4128, 3840)), ("add_131", (1, 1, 3840)),
               ("tanh_19", (1, 1, 3840)), ("select_45", (1, 4128, 1, 64)),
               ("select_46", (1, 4128, 1, 64)), ("adaln_input", (1, 256))],
    "part2a": [("unified", (1, 4128, 3840)), ("unified_mask", (1, 4128)),
               ("unified_freqs", (1, 4128, 64, 2)), ("adaln_input", (1, 256))],
    "part2b": [("add_92", (1, 4128, 3840)), ("select", (1, 4128, 1, 64)),
               ("select_1", (1, 4128, 1, 64)), ("split_7_split_2", (1, 1, 3840)),
               ("split_7_split_3", (1, 1, 3840)), ("val_105", (1, 1, 1, 4128)),
               ("adaln_input", (1, 256))],
}
# 🔴 四段统一用 `L32_pure` —— **纯 FP32 血统**（#116 单变量实测确认的正确来源）。
#    不用 testB：那是「CPU 跑量化 transformer 的逐段输出」，血统不同，四段混用会不可比。
CH32 = os.path.join(P0, "L32_pure")


def free_gb():
    try:
        import ctypes
        k = ctypes.windll.kernel32

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = S()
        s.dwLength = ctypes.sizeof(S)
        k.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / 1e9
    except Exception:
        return None


def main():
    import onnxruntime as ort

    seg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "part1b"
    budget = 4.0
    if "--budget-gb" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget-gb") + 1])
    os.makedirs(OUT, exist_ok=True)

    simsrc = "--sim-src" in sys.argv
    if simsrc:
        p = os.path.join(os.path.dirname(cs.onnx_for(seg)), SIM_ONNX[seg] + ".onnx")
        print("段 %s   源 %s   (**sim_qdq 用的 L=32 base**)" % (seg, p), flush=True)
    else:
        p = cs.onnx_for(seg, L=cs.DEPLOYED_L)      # 🔴 当前部署的那张图（#148）
        print("段 %s   源 %s   (DEPLOYED_L=%d)" % (seg, p, cs.DEPLOYED_L), flush=True)
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    init_dims = {i.name: list(i.dims) for i in g.initializer}
    node_out = {o for n in g.node for o in n.output}
    gin = {i.name for i in g.input}

    # --- 集合：静态权重 MatMul 的激活输入 ---
    pairs = []          # (act, weight, K, M)
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        a, w = n.input[0], n.input[1]
        if w in init_dims and len(init_dims[w]) == 2:
            pairs.append((a, w, init_dims[w][0], init_dims[w][1]))
    acts = sorted({a for a, _w, _k, _m in pairs})
    print("  静态权重 MatMul %d 个 ⇒ 唯一激活输入 **%d** 个" % (len(pairs), len(acts)),
          flush=True)

    # --- 喂输入 ---
    feeds = {}
    _ins = INS32[seg] if simsrc else INS[seg]
    _dir = os.path.join(CH32 if simsrc else CH80, "s0_%s" % seg)
    for n, sh in _ins:
        f = os.path.join(_dir, n + ".raw")
        want = 4
        for d in sh:
            want *= d
        got = os.path.getsize(f)
        if got != want:
            sys.exit("FAIL 输入 %s 字节 %d != 期望 %d（约束 3：尺寸不符必须停）" % (n, got, want))
        feeds[n] = np.fromfile(f, dtype=np.float32).reshape(sh)
    # 🔴 mask 类输入在图里是 BOOL（cap_pad_mask / unified_mask）。宿主一律按 float32 存盘，
    #    喂给 ORT 前必须按图声明转 dtype —— 照抄 maxfp16_stats.py:270 的做法。
    for i in g.input:
        if i.type.tensor_type.elem_type == TensorProto.BOOL and i.name in feeds:
            feeds[i.name] = feeds[i.name].astype(bool)
    print("  输入 %d 个，字节数全部吻合" % len(feeds), flush=True)

    # --- 分批（#115：每批单独建 session）---
    def nbytes(t):
        for a, _w, k, _m in pairs:
            if a == t:
                # 激活最后一维 = K；总元素数按 unified 长度估
                return ((4128 if simsrc else 4176) * k * 4) if t not in gin else (k * 4)
        return 1 << 24
    batches, cur, cb = [], [], 0
    for t in sorted(acts, key=nbytes):
        b = nbytes(t)
        if cur and (cb + b > budget * 1e9 or len(cur) >= 12):
            batches.append(cur)
            cur, cb = [], 0
        cur.append(t)
        cb += b
    if cur:
        batches.append(cur)
    print("  分 %d 批（预算 %.1f GB/批）" % (len(batches), budget), flush=True)

    base_out = list(g.output)
    dst = os.path.join(os.path.dirname(p), "%s_sqstats.onnx"
                       % os.path.splitext(os.path.basename(p))[0])
    res, t00 = {}, time.time()
    for bi, b in enumerate(batches):
        fr = free_gb()
        if fr is not None and fr < 3.0:
            sys.exit("FAIL 宿主可用内存只剩 %.1f GB，停止（约束 9 清单第 7 条）" % fr)
        del g.output[:]
        g.output.extend(base_out)
        have = {o.name for o in g.output}
        want = [t for t in b if t not in gin]
        for t in want:
            if t not in have:
                g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
        onnx.save(m, dst)
        t0 = time.time()
        sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        outs = sess.run(want, feeds) if want else []
        def chstat(arr):
            """逐通道统计。除 max|·|（给 s 用）外，还要**有符号的 min/max**：
            平滑后 A/s 的全局量程必须精确重算（s>0，除法保号），
            否则 mode S 只能沿用未平滑张量的 encoding —— 那是错的。"""
            v = np.asarray(arr, dtype=np.float32).reshape(-1, np.shape(arr)[-1])
            return (np.abs(v).max(axis=0), v.min(axis=0), v.max(axis=0))

        for name, a in zip(want, outs):
            res[name] = chstat(a)
        for t in b:
            if t in gin:                      # 图输入直接从 feeds 算
                res[t] = chstat(feeds[t])
        del outs, sess
        gc.collect()
        print("    批 %d/%d  %d 张量  用时 %.0fs  可用内存 %.1f GB"
              % (bi + 1, len(batches), len(b), time.time() - t0, free_gb() or -1), flush=True)

    # --- 权重逐行 max（从 ONNX 外部数据按偏移读，一次一个，内存安全）---
    print("  读权重逐行 max（%d 个）..." % len({w for _a, w, _k, _m in pairs}), flush=True)
    ini = {i.name: i for i in g.initializer}
    wmax = {}
    for _a, w, _k, _m in pairs:
        if w in wmax:
            continue
        t = ini[w]
        if t.external_data:
            d = {e.key: e.value for e in t.external_data}
            loc = os.path.join(os.path.dirname(p), d["location"])
            with open(loc, "rb") as f:
                f.seek(int(d.get("offset", 0)))
                raw = f.read(int(d["length"]))
            arr = np.frombuffer(raw, dtype=np.float32).reshape([x for x in t.dims])
        else:
            from onnx import numpy_helper
            arr = numpy_helper.to_array(t)
        wmax[w] = np.abs(arr).max(axis=1)      # 逐输入通道（行）
        del arr

    payload = {
        "seg": seg, "onnx": p, "deployed_L": cs.DEPLOYED_L,
        "n_matmul_static": len(pairs), "n_act": len(acts),
        "pairs": [{"act": a, "w": w, "K": k, "M": mm} for a, w, k, mm in pairs],
        "act_chmax": {k: v[0].tolist() for k, v in res.items()},
        "act_chmin_signed": {k: v[1].tolist() for k, v in res.items()},
        "act_chmax_signed": {k: v[2].tolist() for k, v in res.items()},
        "w_rowmax": {k: v.tolist() for k, v in wmax.items()},
    }
    out = os.path.join(OUT, "%s_%s_chstats.json"
                       % (seg, "sim32" if simsrc else "L%d" % cs.DEPLOYED_L))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("\n写出 %s  (%.1f MB)  总用时 %.0f 分钟"
          % (out, os.path.getsize(out) / 1e6, (time.time() - t00) / 60.0))
    print("  激活逐通道统计 %d 个张量；权重逐行统计 %d 个" % (len(res), len(wmax)))
    miss = [a for a in acts if a not in res]
    if miss:
        print("  🔴 缺 %d 个: %s" % (len(miss), miss[:6]))


if __name__ == "__main__":
    main()
