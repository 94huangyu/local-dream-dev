"""在 FP32 ONNX 上插 QDQ，模拟「算术完全正确的 A16W8」。

方案：`scripts/EXP_PLAN_SIM_CEILING.md`（判据事前锁定）。

为什么要它：**#121 已实测证明 L1 无法换算成 L3**，我们一直在往一个高度未知的天花板爬。
本工具在**宿主**上给出天花板，并顺带回答 #100（「激活 16-bit 量化」vs「HTP 超出正确定点」
从未被分开过，阻碍是 #36 —— 本方案绕开 SNPE，自己插 QDQ）。

模式：
  A  权重 int8 per-channel QDQ + 激活 uFxp_16 QDQ   —— 与部署同配置
  B  权重 int8 QDQ + 激活 FP16                      —— 「最大化 FP16」的上限
  C  权重 int8 QDQ + 激活 FP32                      —— 8-bit 权重的绝对上限
  Z  什么都不插                                     —— 自检：必须与纯 FP32 逐位相同
  F  权重 int8 QDQ + **除溢出风险外全部激活走 FP16**  —— FP16 路线的**上界**
     溢出筛选：优先用实测 `<seg>_amax.json`；没有实测的段退回「标定 |max| x 1.2 > 65504」
     并在输出里报出用了哪种（#110：标定不可靠，必须标注）
  G  权重 int8 QDQ + **全部激活先 Clip(±65504) 再走 FP16**  —— #111 显式钳位 + 全 FP16 的上界
     与 F 的唯一差别：F 把溢出张量留在 uFxp_16，G 把它们钳住后也转 FP16
     => F 与 G 之差 = 「那几十个 massive activation 到底占多大」
  H  = M 的集合 **并上**「溢出张量」，且全部先 Clip(±65504) —— **这是实际能建出来的目标配置**
     （旧机制能给出 M 那样的连续浮点区；Clip 由 ONNX 图手术提供）
  M  权重 int8 QDQ + **指定集合走 FP16、其余 uFxp_16 QDQ**  —— 复现某个真实配置的浮点集合
     集合从 `<seg>_<SIMSET>_enc.csv` 里读「Float_16 且非 STATIC」的张量名，
     环境变量 `SIMSET` 指定（默认 best26）=> `python sim_qdq.py part1b M`

实现要点（都踩过坑）：
  · opset 17，**没有 uint16 的 QuantizeLinear**（需 opset 21）=> 用 Div/Round/Clip/Mul 四件套
  · **不 materialize 权重**：权重 QDQ 也做成图节点，交给 ORT 常量折叠
    （直接载入会吃 9.5 GB/段，本项目已因内存打穿栽过多次）
  · QNN 反量化约定：`value = scale * (q + offset)`，q ∈ [0, 2^bw-1]
    => `x' = scale * clip(round(x/scale), offset, 2^bw-1+offset)`

用法: python sim_qdq.py <part1a|part1b|part2a|part2b> <A|B|C|Z>
"""
import io
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
ONNX = {"part1a": "transformer_part1a", "part1b": "transformer_part1b",
        "part2a": "transformer_part2a_fixed", "part2b": "transformer_part2b_fixed"}

TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: "
                r"\[([^\]]*)\]; tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def parse(seg):
    kind, ch, pt = {}, {}, {}
    with io.open(os.path.join(W, "%s_full_enc.csv" % seg), encoding="utf-8",
                 errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), (m.group(2), m.group(4)))
            for m in CH.finditer(line):
                ch.setdefault(m.group(1), {})[int(m.group(2))] = (
                    int(m.group(3)), float(m.group(6)), float(m.group(7)))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), (int(m.group(2)), float(m.group(5)),
                                           float(m.group(6))))
    return kind, ch, pt


def _smoothquant(seg, alpha, fset, pt, kind, ch):
    """算 SmoothQuant 的逐通道 s，并**重算**受影响张量的 encoding。

    为什么必须重算（不能沿用录得的）：s 改变了张量的数值范围，
    录在 `<seg>_full_enc.csv` 里的 scale/offset 是对**未平滑**张量标定的。

    返回 (s_by_act, wscale_by_weight, aenc_by_act)
      s_by_act    : 激活名 -> np.float32[K]
      wscale_by_weight : 权重名 -> np.float32[M]（per-输出通道对称 int8 的 scale）
      aenc_by_act : 激活名 -> (scale, offset) 非对称 uFxp_16
    """
    import json
    import onnx
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import canonical_sources as cs

    # 🔴 必须用 **本模拟器实际使用的那张图**（ONNX[seg] = L=32 base）上测的统计。
    #    用 L=80 clip 那份会两个轴都对不上（序列长度、有无 Clip）。
    f = os.path.join(os.path.dirname(W), "smoothquant", "%s_sim32_chstats.json" % seg)
    if not os.path.exists(f):
        sys.exit("FAIL mode S 需要 %s（先跑 sq_actstats.py）" % f)
    st = json.load(io.open(f, encoding="utf-8"))
    amx = {k: np.asarray(v, np.float64) for k, v in st["act_chmax"].items()}
    amin = {k: np.asarray(v, np.float64) for k, v in st["act_chmin_signed"].items()}
    amax = {k: np.asarray(v, np.float64) for k, v in st["act_chmax_signed"].items()}
    wrow = {k: np.asarray(v, np.float64) for k, v in st["w_rowmax"].items()}

    # 激活 -> 它喂的所有静态权重（一个 s 必须同时服务全部消费者）
    a2w = {}
    for pr in st["pairs"]:
        a2w.setdefault(pr["act"], []).append(pr["w"])

    # 前置检查（sq_consumers.py 已验）：只保留「只喂 MatMul」的激活。
    p = os.path.join(D, ONNX[seg] + ".onnx")      # 与 main() 的 src 同一张图
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    initn = {i.name for i in g.initializer}
    gout = {o.name for o in g.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    mmof = {}
    for n in g.node:
        if n.op_type in ("MatMul", "Gemm") and n.input[1] in initn:
            mmof.setdefault(n.input[0], []).append(n)

    eps = 1e-8
    s_by, wsc, aenc, w2a = {}, {}, {}, {}
    ini = {i.name: i for i in g.initializer}

    # 🔴 只有「建图时**确实会被缩放**」的权重才算数（2026-08-31 事故）。
    #    建图的权重循环遍历的是 `ch`（DLC 的 per-channel 转储）且要求 bw==8；
    #    而这里是按 ONNX 图枚举权重。两个集合不一致时，
    #    激活被 Div(s) 了、权重却没拿到补偿的 Mul(s) ⇒ **逐通道差一个 s 因子** ⇒ 毁段。
    #    实测：part1a 的 adaln_input 误差 0.24% -> 73.23%、part2b 的 latents 3.60% -> 420%；
    #    part1b 因权重恰好全有 8-bit per-channel encoding 而幸免（3.82% -> 3.59%）。
    scalable = set()
    for dn, per in ch.items():
        on = dn
        if on not in initn and on.endswith("_permute") and on[:-8] in initn:
            on = on[:-8]
        if on in initn and per[0][0] == 8:
            scalable.add(on)
    for a, ws in a2w.items():
        if a not in amx:
            continue
        if a in fset:              # 已走 FP16 的不平滑（它没有量化误差可省）
            continue
        others = [n for n in cons.get(a, []) if n not in mmof.get(a, [])]
        if others or a in gout:    # 有别的消费者 => 整体替换不等价，跳过（实测只有 adaln_input）
            continue
        ax = amx[a]

        def _load(wn):
            t = ini[wn]
            d = {e.key: e.value for e in t.external_data}
            with open(os.path.join(os.path.dirname(p), d["location"]), "rb") as fh:
                fh.seek(int(d.get("offset", 0)))
                raw = fh.read(int(d["length"]))
            arr = np.frombuffer(raw, dtype=np.float32).reshape([x for x in t.dims])
            # 🔴 权重可能以 [M, K] 存放（adaLN 的 [15360,256] 就是）。
            #    预存的 `w_rowmax` 是按 axis=1 求的，对转置位会给出长度 M 而不是 K
            #    => 这里直接从数据判定，不用预存值（alpha=0 时该 bug 被 s=1 掩盖，只在 >0 才炸）。
            if arr.shape[0] != ax.size and arr.shape[1] == ax.size:
                arr = np.ascontiguousarray(arr.T)
            return arr if arr.shape[0] == ax.size else None

        if any(w not in scalable for w in ws):
            continue          # 有消费者权重不会被缩放 => 整条激活不平滑，宁可不做
        # 🔴 建图时的权重 Mul 有守卫 `sv.size == dims[0]`，即**只对按 [K, M] 存放的权重生效**。
        #    adaLN_modulation 那类是转置存放的 [M, K]（如 [15360, 256]）=> 守卫为假 => 不缩放，
        #    而激活仍会被 Div(s) => 逐通道差一个 s 因子 => **静默毁段**。
        #    2026-08-31 实测：四段全部不平衡（part1a 4 个权重 / part1b 7 / part2a 8 / part2b 1），
        #    part1b 因 adaln_input 恰是**图输入**（激活循环跳过 gin）而两边都没做、凑巧平衡。
        #    => 这里与建图守卫用**同一个条件**，把「算了 s」和「会被 Mul」对齐。
        if any(list(ini[w].dims)[0] != ax.size for w in ws):
            continue
        mats = {}
        for w in ws:
            arr = _load(w)
            if arr is None:
                mats = {}
                break
            mats[w] = arr
        if not mats:
            continue
        aw = np.maximum.reduce([np.abs(v).max(axis=1) for v in mats.values()])
        if alpha <= 0:
            s = np.ones_like(ax)
        else:
            s = np.power(np.maximum(ax, eps), alpha) / np.power(np.maximum(aw, eps), 1 - alpha)
        s = np.maximum(s, eps)
        s_by[a] = s.astype(np.float32)
        # 激活：平滑后精确重算非对称 uFxp_16（s>0，除法保号）
        lo = float(np.min(amin[a] / s))
        hi = float(np.max(amax[a] / s))
        sc = max((hi - lo) / 65535.0, eps)
        aenc[a] = (np.float32(sc), float(np.round(lo / sc)))
        # 权重：W*s 之后**逐输出通道**重算对称 int8 scale
        for w, arr in mats.items():
            wsc[w] = np.maximum(np.abs(arr * s[:, None]).max(axis=0) / 127.0,
                                eps).astype(np.float32)
            w2a[w] = a
        del mats
    return s_by, wsc, aenc, w2a


def main():
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    seg, mode = sys.argv[1], sys.argv[2].upper()
    assert mode in ("A", "B", "C", "Z", "M", "F", "G", "H", "S"), \
        "mode 必须是 A/B/C/Z/M/F/G/H/S"
    kind, ch, pt = parse(seg)
    fset = set()
    sq_s, sq_wsc, sq_aenc, sq_w2a = {}, {}, {}, {}   # mode S 用
    sq_done = set()      # 实际被 Mul(s) 的权重：逐激活的 s、重算的权重/激活 encoding
    if mode == "F":
        import json
        ap = os.path.join(W, "%s_amax.json" % seg)
        meas = {}
        if os.path.exists(ap):
            meas = {k: v["amax"] for k, v in
                    json.load(io.open(ap, encoding="utf-8"))["tensors"].items()}
        nm = ns = 0
        for a, (bw, sc, off) in pt.items():
            if bw != 16 or kind.get(a, ("", ""))[1] == "STATIC":
                continue
            if a in meas:                       # 实测优先
                nm += 1
                if meas[a] * 1.2 <= 65504.0:
                    fset.add(a)
            else:                               # 退回标定值（#110：不可靠，仅作上界估算）
                ns += 1
                cm = max(abs(0.0), abs((2 ** bw - 1 + off) * sc), abs(off * sc))
                if cm * 1.2 <= 65504.0:
                    fset.add(a)
        print("  [F] 溢出筛选：实测 %d 个 / 标定 %d 个 ⇒ 入选 FP16 **%d**（③标定部分不可靠）"
              % (nm, ns, len(fset)), flush=True)
    if mode == "H":
        import json
        tag = os.environ.get("SIMSET", "best26")
        # SIMH_BASE=none => 只要溢出张量，不带输入锥。用来分离
        # 「18 个被钳的 massive activation」与「被连带拖进来的 ~400 个锥张量」各值多少。
        # 后者是速度成本主体（每个浮点张量一对 Convert），前者只有十几个算子。
        if os.environ.get("SIMH_BASE", "") != "none":
            fp = os.path.join(W, "%s_%s_enc.csv" % (seg, tag))
            if not os.path.exists(fp):
                sys.exit("FAIL mode H 需要 %s" % fp)
            with io.open(fp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    for mm in TT.finditer(line):
                        if mm.group(2) == "Float_16" and mm.group(4) != "STATIC":
                            fset.add(mm.group(1))
        nbase = len(fset)
        ap = os.path.join(W, "%s_amax.json" % seg)
        meas = {}
        if os.path.exists(ap):
            meas = {k: v["amax"] for k, v in
                    json.load(io.open(ap, encoding="utf-8"))["tensors"].items()}
        nov = 0
        for a, (bw, sc, off) in pt.items():
            if bw != 16 or kind.get(a, ("", ""))[1] == "STATIC" or a in fset:
                continue
            hi = meas[a] if a in meas else max(abs((2 ** bw - 1 + off) * sc), abs(off * sc))
            if hi > 65504.0 / 1.2:              # 溢出或接近溢出 => 钳住后也转 FP16
                fset.add(a)
                nov += 1
        print("  [H] 基础集合 %d + 溢出张量 %d ⇒ FP16 共 %d（全部先 Clip±65504）"
              % (nbase, nov, len(fset)), flush=True)
    if mode == "S":
        # mode S = mode H（部署配置：FP16 集合 + 显式 Clip）**再加 SmoothQuant**。
        # 单变量对照臂：SQ_ALPHA=0 —— 同样重算 encoding，但 s 恒为 1（不迁移）。
        # 这样 S(alpha=0) 与 S(alpha=0.5) 之间唯一的差别就是迁移本身。
        import json
        os.environ.setdefault("SIMSET", "best26")
        _tag = os.environ["SIMSET"]
        fp = os.path.join(W, "%s_%s_enc.csv" % (seg, _tag))
        if os.path.exists(fp):
            with io.open(fp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    for mm in TT.finditer(line):
                        if mm.group(2) == "Float_16" and mm.group(4) != "STATIC":
                            fset.add(mm.group(1))
        ap = os.path.join(W, "%s_amax.json" % seg)
        meas = {}
        if os.path.exists(ap):
            meas = {k: v["amax"] for k, v in
                    json.load(io.open(ap, encoding="utf-8"))["tensors"].items()}
        for a, (bw, sc, off) in pt.items():
            if bw != 16 or kind.get(a, ("", ""))[1] == "STATIC" or a in fset:
                continue
            hi = meas[a] if a in meas else max(abs((2 ** bw - 1 + off) * sc), abs(off * sc))
            if hi > 65504.0 / 1.2:
                fset.add(a)
        alpha = float(os.environ.get("SQ_ALPHA", "0.5"))
        sq_s, sq_wsc, sq_aenc, sq_w2a = _smoothquant(seg, alpha, fset, pt, kind, ch)
        print("  [S] FP16 集合 %d（同 mode H）+ SmoothQuant alpha=%.2f："
              "平滑激活 %d 个、重算权重 encoding %d 个"
              % (len(fset), alpha, len(sq_s), len(sq_wsc)), flush=True)
    if mode == "M":
        tag = os.environ.get("SIMSET", "best26")
        fp = os.path.join(W, "%s_%s_enc.csv" % (seg, tag))
        if not os.path.exists(fp):
            sys.exit("FAIL mode M 需要 %s（某个真实产物的张量类型转储）" % fp)
        with io.open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                for mm in TT.finditer(line):
                    if mm.group(2) == "Float_16" and mm.group(4) != "STATIC":
                        fset.add(mm.group(1))
        print("  [M] 从 %s 读到 Float_16 非静态张量 %d 个" % (os.path.basename(fp), len(fset)),
              flush=True)
    src = os.path.join(D, ONNX[seg] + ".onnx")
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    init = {t.name: t for t in g.initializer}
    nodes = list(g.node)
    new_init, add_nodes = [], []
    uid = [0]

    def const(arr, tag):
        uid[0] += 1
        n = "_simc_%s_%d" % (tag, uid[0])
        t = numpy_helper.from_array(np.ascontiguousarray(arr), n)
        new_init.append(t)
        return n

    def qdq_chain(src_name, dst_name, scale, lo, hi, tag):
        """dst = scale * clip(round(src/scale), lo, hi)；scale 可以是标量或按轴广播的数组"""
        s = const(np.asarray(scale, np.float32), tag + "s")
        a, b, c = ["%s_%s_%d" % (dst_name, x, uid[0]) for x in ("dv", "rd", "cl")]
        lo_n = const(np.asarray(lo, np.float32), tag + "lo")
        hi_n = const(np.asarray(hi, np.float32), tag + "hi")
        add_nodes.extend([
            helper.make_node("Div", [src_name, s], [a]),
            helper.make_node("Round", [a], [b]),
            helper.make_node("Clip", [b, lo_n, hi_n], [c]),
            helper.make_node("Mul", [c, s], [dst_name]),
        ])

    # ---------- 权重：per-channel int8 对称（offset 必须为 0） ----------
    nw = 0
    if mode != "Z":
        prod_in = {}
        for n in nodes:
            for i in n.input:
                prod_in.setdefault(i, []).append(n)
        for name, per in ch.items():
            # DLC 把部分权重转置存放并加 `_permute` 后缀（adaLN）=> 映射回 ONNX 名
            if name not in init and name.endswith("_permute") and name[:-8] in init:
                name = name[:-8]
            if name not in init:
                continue
            bw = per[0][0]
            if bw != 8:                      # bias 是 sFxp_32，不动（#99：gamma/bias 另论）
                continue
            assert all(per[i][2] == 0 for i in per), "%s per-axis offset != 0" % name
            t = init[name]
            dims = list(t.dims)
            nch = len(per)
            # scale 沿哪个轴：长度等于 nch 的那个轴（DLC 的 channel 轴）
            ax = [k for k, d in enumerate(dims) if d == nch]
            if len(ax) != 1:
                # 方阵歧义：取轴 1。依据不是猜——本段 **21 个可唯一判定的样本全部落在轴 1**
                # （[3840,10240]/nch=10240 与 [10240,3840]/nch=3840 都是轴 1），
                # 即 MatMul(X, W) 的输出通道轴。约束 8：操作化先用已知样本验证过。
                if len(dims) == 2 and dims[1] == nch:
                    ax = [1]
                else:
                    print("  跳过 %s：无法确定 channel 轴 dims=%s nch=%d" % (name, dims, nch))
                    continue
            sc = np.array([per[i][1] for i in range(nch)], np.float32)
            shape = [1] * len(dims)
            shape[ax[0]] = nch
            sc = sc.reshape(shape)
            qn = name + "_simq"
            for n in prod_in.get(name, []):
                for k, i in enumerate(n.input):
                    if i == name:
                        n.input[k] = qn
            if mode == "S" and name in sq_wsc:
                # SmoothQuant：先 W*s（s 沿输入通道轴 0 广播），再用**重算的** scale 做 QDQ。
                # 用 Mul 节点而不是 materialize 权重 —— 保持本脚本「不载入权重」的纪律
                # （直接载入会吃 9.5 GB/段，见文件头）。ORT 会常量折叠它。
                # 🔴 s 必须按**权重→激活的显式映射**取，不能按 size 猜：多个激活的 K 相同。
                sv = sq_s.get(sq_w2a.get(name))
                if sv is not None and len(dims) == 2 and sv.size == dims[0]:
                    sq_done.add(name)
                    smul = const(sv.reshape(dims[0], 1).astype(np.float32), "sqs")
                    wsm = name + "_sqw"
                    add_nodes.append(helper.make_node("Mul", [name, smul], [wsm]))
                    nsc = sq_wsc[name].reshape(1, -1)
                    qdq_chain(wsm, qn, nsc, -127.0, 127.0, "w")
                    nw += 1
                    continue
            qdq_chain(name, qn, sc, -128.0, 127.0, "w")
            nw += 1

    # ---------- 激活：16-bit 非静态 ----------
    na = 0
    if mode in ("A", "B", "M", "F", "G", "H", "S"):
        gout = {o.name for o in g.output}
        gin = {i.name for i in g.input}
        prod = {}
        for n in nodes:
            for k, o in enumerate(n.output):
                prod[o] = (n, k)
        cons = {}
        for n in nodes:
            for k, i in enumerate(n.input):
                cons.setdefault(i, []).append((n, k))
        for name, (bw, sc, off) in pt.items():
            if bw != 16 or kind.get(name, ("", ""))[1] == "STATIC":
                continue
            if name in gin:                   # 图输入：改名后在其后插链
                continue
            if name not in prod:
                continue
            pre = name + "_simpre"
            n0, k0 = prod[name]
            n0.output[k0] = pre
            use_fp16 = (mode in ("B", "G")) or (mode in ("M", "F", "H", "S") and name in fset)
            src0, sc_u, off_u = pre, np.float32(sc), off
            if mode == "S" and name in sq_s and not use_fp16:
                # A -> A/s。真实部署里这个除法折进上游 RmsNorm 的乘法常量 => 运行时零代价。
                # 前置已验：这些激活**只喂 MatMul**，整体替换等价（sq_consumers.py）。
                sdiv = const(sq_s[name].astype(np.float32), "sqa")
                sqd = pre + "_sqdiv"
                add_nodes.append(helper.make_node("Div", [pre, sdiv], [sqd]))
                src0 = sqd
                sc_u, off_u = sq_aenc[name]      # 平滑后重算的 encoding
            if not use_fp16:
                qdq_chain(src0, name, np.float32(sc_u), off_u,
                          (1 << bw) - 1 + off_u, "a")
            else:                             # FP16 往返
                src_t = src0
                if mode in ("G", "H", "S"):
                    # 🔴 #111：**显式**钳位到 ±65504。危险的是隐式溢出（变 inf），
                    #    显式钳位后超出部分变成 65504 而非 inf ⇒ 1.0 倍余量是安全的。
                    #    阈值锁定 65504，不得为保险多钳（钳到 32752 全量误差跳到 15.49%）。
                    cl = pre + "_clip"
                    lo_n = const(np.asarray(-65504.0, np.float32), "gl")
                    hi_n = const(np.asarray(65504.0, np.float32), "gh")
                    add_nodes.append(helper.make_node("Clip", [src_t, lo_n, hi_n], [cl]))
                    src_t = cl
                h = pre + "_h16"
                add_nodes.append(helper.make_node("Cast", [src_t], [h], to=TensorProto.FLOAT16))
                add_nodes.append(helper.make_node("Cast", [h], [name], to=TensorProto.FLOAT))
            na += 1

    if mode == "S":
        # 🔴 一致性断言（2026-08-31 事故）：SmoothQuant 是 Y=(X/s)(sW)，
        #    两边必须同时生效。只要有一个权重没拿到 Mul(s)，该 MatMul 就逐通道差一个 s 因子，
        #    结果是**静默毁段**（实测 part2b 的 latents 误差 3.60% -> 420%）。
        #    => 宁可建图失败，也不许放一个不平衡的图出去。
        miss = [w for w in sq_wsc if w not in sq_done]
        if miss:
            bad = sorted({sq_w2a[w] for w in miss if w in sq_w2a})
            sys.exit("FAIL mode S 不平衡：%d 个权重算了 s 却没被 Mul(s)（涉及激活 %d 个）"
                     "  权重例: %s  激活例: %s"
                     "  => 这些激活会被 Div(s) 却无补偿，必须先修再建图"
                     % (len(miss), len(bad), miss[:4], bad[:4]))
        print("  [S] 一致性断言通过：%d 个权重全部拿到 Mul(s)" % len(sq_done), flush=True)
    g.node.extend(add_nodes)
    g.initializer.extend(new_init)
    # 拓扑排序：新节点必须排在其输入之后
    order, ready, pend = [], {t.name for t in g.initializer} | {i.name for i in g.input}, list(g.node)
    guard = 0
    while pend and guard < 200000:
        guard += 1
        nxt = []
        for n in pend:
            if all((i in ready) or (i == "") for i in n.input):
                order.append(n)
                ready.update(n.output)
            else:
                nxt.append(n)
        if len(nxt) == len(pend):
            sys.exit("FAIL 拓扑排序卡住，剩 %d 个节点，例: %s" % (len(nxt), nxt[0].name or nxt[0].op_type))
        pend = nxt
    del g.node[:]
    g.node.extend(order)

    dst = os.path.join(D, "%s_sim%s.onnx" % (ONNX[seg], mode))
    onnx.save(m, dst)
    print("[%s/%s] 权重 QDQ %d 个｜激活 %s %d 个｜新增节点 %d ⇒ %s"
          % (seg, mode, nw, {"A": "uFxp16 QDQ", "B": "FP16 往返",
                             "M": "混合(FP16 %d / 其余 uFxp16)" % len(fset),
                             "F": "上界(FP16 %d / 其余 uFxp16)" % len(fset),
                             "G": "全 FP16 + 显式 Clip(±65504)",
                             "H": "目标配置(FP16 %d + Clip)" % len(fset),
                             "S": "目标配置 + SmoothQuant(alpha=%s, 平滑 %d 个)"
                                  % (os.environ.get("SQ_ALPHA", "0.5"), len(sq_s))}.get(mode, "不动"),
             na, len(add_nodes), os.path.basename(dst)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
