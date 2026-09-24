# -*- coding: utf-8 -*-
"""P2 生图速度 —— 零设备证据复算（EXP_PLAN_P2_SPEED §1 的每个数都从这里导出）。

只读盘上已有产物，不碰设备：
  · app logcat 里的 `[segtime]` 与后端单调时钟（scratch_runs/oneshot/evidence_20260917/）
  · 四段 ONNX 结构（不加载外部权重）→ MatMul/Gemm 乘法次数
  · mg context 元数据 info.json（qnn-context-binary-utility 转储）
  · 部署 DLC 的 snpe-dlc-info 转储（logs/p2_20260917/dlcinfo_*.txt）
  · mg 契约（段间张量的 scale/offset）
  · 宿主探针记录 logs/p2_20260917/probe_results.json

🔴 口径纪律
  · `[segtime]` 是**段级墙钟**，含输入重量化与输出拷贝，不是纯算子时间。
  · 「part1a 超出量」是**模型推算**（按其余三段同比例的 ms/G 外推），不是测量 ⇒ 只能写 ②/③。
  · MatMul 乘法次数只数矩阵乘，不含 Softmax/RMSNorm/逐元素算子。

用法: python scripts/p2_zero_device_evidence.py > logs/p2_20260917/evidence.md
"""
import collections
import io
import json
import os
import re
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join("D:", os.sep, "LocalDreamZImage")
EV = os.path.join(REPO, "scratch_runs", "oneshot", "evidence_20260917")
LOGD = os.path.join(REPO, "logs", "p2_20260917")
ONNX = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
MG = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg")
CONTRACT = os.path.join(REPO, "logs", "contracts_20260906", "final_qnn_contract.mg.json")
SEGS = ("part1a", "part1b", "part2a", "part2b")
# app 请求尺寸 -> (ONNX 后缀, mg 图名里的比例标签)
ASPECTS = collections.OrderedDict([
    ("1024x1024", ("", "fp16_L80")), ("1184x896", ("_1184x896", "1184x896")),
    ("896x1184", ("_896x1184", "896x1184")), ("1280x720", ("_1280x720", "1280x720")),
    ("720x1280", ("_720x1280", "720x1280"))])
SEGT = re.compile(r"\[segtime\]\s+transformer_(\S+)\s+(\d+)\s*ms")
CLK = re.compile(r"Backend:\s+([0-9]+\.[0-9]+)ms")


def segtimes(path):
    per = collections.defaultdict(list)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SEGT.search(line)
            if m:
                per[m.group(1)].append(int(m.group(2)))
    return {k: statistics.median(v) for k, v in per.items()}, {k: len(v) for k, v in per.items()}


def onnx_name(seg, sfx):
    return ("transformer_%s_clip_L80%s.onnx" % (seg, sfx) if seg in ("part1a", "part1b")
            else "transformer_%s_fixed_clip_L80%s.onnx" % (seg, sfx))


def matmul_flops(path):
    import onnx
    from onnx import shape_inference
    m = onnx.load(path, load_external_data=False)
    m = shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=True)
    sh = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            sh[vi.name] = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]
    for i in m.graph.initializer:
        sh[i.name] = list(i.dims)
    fl, unk = 0, 0
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        sa, so = sh.get(n.input[0]), sh.get(n.output[0])
        if not sa or not so or None in sa or None in so:
            unk += 1
            continue
        p = 1
        for v in so:
            p *= v
        fl += p * (sa[-1] if len(sa) >= 2 else sa[0])
    ops = collections.Counter(n.op_type for n in m.graph.node)
    return {"flops": fl, "unknown": unk, "nodes": len(m.graph.node),
            "softmax": ops["Softmax"], "scatter_elements": ops["ScatterElements"]}


def flops_table():
    cache = os.path.join(LOGD, "flops.json")
    if os.path.isfile(cache):
        return json.load(open(cache, encoding="utf-8"))
    out = {}
    for a, (sfx, _) in ASPECTS.items():
        for s in SEGS:
            out["%s/%s" % (a, s)] = matmul_flops(os.path.join(ONNX, onnx_name(s, sfx)))
    json.dump(out, open(cache, "w", encoding="utf-8"), indent=1)
    return out


def mg_meta():
    out = {}
    for s in SEGS:
        j = json.load(open(os.path.join(MG, s, "info.json"), encoding="utf-8"))
        for g in j["info"]["graphs"]:
            gi = g["info"]
            for a, (_, tag) in ASPECTS.items():
                if gi["graphName"] == "%s_%s_fp32" % (s, tag):
                    v2 = gi["graphBlobInfoV2"]
                    out["%s/%s" % (a, s)] = {
                        "sharedWeightsMiB": v2["sharedWeightsSize"] / 2**20,
                        "constMiB": v2["constSize"] / 2**20, "opDataMiB": v2["opDataSize"] / 2**20,
                        "ioMiB": v2["ioTensorSize"] / 2**20,
                        "spillFillMiB": gi["graphBlobInfo"]["info"]["spillFillBufferSize"] / 2**20,
                        "vtcm": gi["graphBlobInfo"]["info"]["vtcmSize"],
                        "numHvxThreads": gi["graphBlobInfo"]["info"]["numHvxThreads"]}
    return out


def budget(path, sse_path):
    """用后端单调时钟切阶段。标记：TE/transformer 各段 Initializing/Initialized、spill-fill 行、step 标记。"""
    ev = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CLK.search(line)
            if m:
                ev.append((float(m.group(1)), line[m.end():].strip()))
    sse = io.open(sse_path, encoding="utf-8", errors="replace").read()
    gen = int(re.search(r"generation_time_ms\D{0,4}(\d+)", sse).group(1))
    first = int(re.search(r"first_step_time_ms\D{0,4}(\d+)", sse).group(1))
    t_first, t_last = ev[0][0], ev[-1][0]
    init_a = [(t, re.search(r"Buffer: (\S+)", s).group(1)) for t, s in ev if "Initializing QNN App from Buffer" in s]
    init_b = {re.search(r"Buffer: (\S+)", s).group(1): t for t, s in ev if "QNN App Initialized from Buffer" in s}
    steps = {int(re.search(r"step (\d+):", s).group(1)): t for t, s in ev if "latents after step" in s}
    t_sf = [t for t, s in ev if "[spill-fill] group sharing enabled" in s][0]
    loop_start = steps[0] - first
    tr = [(t, n) for t, n in init_a if n.startswith("transformer_")]
    rows = []
    prev_end = t_sf
    for t, n in tr:
        rows.append((n, (t - prev_end) / 1000.0, (init_b[n] - t) / 1000.0))
        prev_end = init_b[n]
    vae = [t for t, n in init_a if n == "vae_decoder"][0]
    return {"generation_s": gen / 1000.0, "backend_span_s": (t_last - t_first) / 1000.0,
            "te_phase_s": (t_sf - t_first) / 1000.0, "transformer_load_s": (loop_start - t_sf) / 1000.0,
            "loop_s": (steps[max(steps)] - loop_start) / 1000.0, "vae_tail_s": (t_last - steps[max(steps)]) / 1000.0,
            "per_load": rows, "after_last_load_s": (loop_start - prev_end) / 1000.0}


def dlc_scatter(dump):
    txt = io.open(dump, encoding="utf-8", errors="replace").read()
    ops = []
    for m in re.finditer(r"\|\s*(\d+)\s*\|\s*(node_select_scatter\w*)\s*\|\s*ScatterElements\s*\|([^\n]*)\n((?:\|\s*\|[^\n]*\n){0,3})", txt):
        block = m.group(3) + m.group(4)
        tens = re.findall(r"(\w+) \(data type: (\w+); tensor dimension: \[([\d,]+)\]; tensor type: (\w+)\)", block)
        ops.append((m.group(2), tens))
    enc = dict(re.findall(r"(\w+) encoding : (bitwidth 16, min [-\d.e]+, max [-\d.e]+, scale [-\d.e]+, offset [-\d.]+)", txt))
    return ops, enc


def main():
    print("# P2 零设备证据（`scripts/p2_zero_device_evidence.py` 复算输出）\n")

    # --- 1. 逐段耗时 & MatMul 乘法次数 ---
    fl = flops_table()
    meta = mg_meta()
    print("## 1. 逐段 `[segtime]` 中位 ÷ MatMul/Gemm 乘法次数（mg 形态，每格 n=8 步）\n")
    print("| 比例 | 段 | 中位 ms | MatMul G | ms/G | 节点数 | Softmax | ScatterElements |")
    print("|---|---|---|---|---|---|---|---|")
    excess = {}
    for a in ASPECTS:
        med, n = segtimes(os.path.join(EV, "os_mg_%s_logcat.txt" % a))
        ref = statistics.median([med[s] / (fl["%s/%s" % (a, s)]["flops"] / 1e9) for s in ("part1b", "part2a", "part2b")])
        for s in SEGS:
            f = fl["%s/%s" % (a, s)]
            g = f["flops"] / 1e9
            print("| %s | %s | %d | %.1f | %.3f | %d | %d | %d |" % (a, s, med[s], g, med[s] / g, f["nodes"], f["softmax"], f["scatter_elements"]))
        g1a = fl["%s/part1a" % a]["flops"] / 1e9
        excess[a] = (med["part1a"], ref, med["part1a"] - ref * g1a)
    print("\n**part1a 超出量（②模型推算）** = 实测 − 其余三段 ms/G 中位 × part1a 的 G：\n")
    for a, (t, ref, ex) in excess.items():
        print("- %s：实测 %d ms，参照 %.3f ms/G ⇒ 超出 **%.0f ms/步**" % (a, t, ref, ex))

    print("\n### single 形态与 mg 形态 1:1 对照（形态是否影响逐段耗时）\n")
    for tag in ("os_sf_off", "os_sf_on", "os_mg_1024x1024", "smoke_os_mg_1024x1024", "os_post_cleanup"):
        med, n = segtimes(os.path.join(EV, tag + "_logcat.txt"))
        print("- `%s`：%s" % (tag, "  ".join("%s %d" % (s, med[s]) for s in SEGS)))

    # --- 2. mg 元数据 ---
    print("\n## 2. mg context 元数据（1:1 图）\n")
    print("| 段 | sharedWeights MiB | const MiB | opData MiB | io MiB | spillFill MiB | vtcm | numHvxThreads |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SEGS:
        m = meta["1024x1024/%s" % s]
        print("| %s | %.0f | %.1f | %.1f | %.1f | %.1f | %d | %d |" % (s, m["sharedWeightsMiB"], m["constMiB"], m["opDataMiB"], m["ioMiB"], m["spillFillMiB"], m["vtcm"], m["numHvxThreads"]))
    print("\npart1a 各比例图的 constSize：%s" % "，".join("%s %.2f MiB" % (a, meta["%s/part1a" % a]["constMiB"]) for a in ASPECTS))

    # --- 3. 时间预算 ---
    b = budget(os.path.join(EV, "os_post_cleanup_logcat.txt"), os.path.join(EV, "os_post_cleanup.sse"))
    print("\n## 3. 单张时间预算（`os_post_cleanup`，后端单调时钟）\n")
    g = b["generation_s"]
    other = g - b["te_phase_s"] - b["transformer_load_s"] - b["loop_s"] - b["vae_tail_s"]
    print("| 阶段 | 秒 | 占比 |\n|---|---|---|")
    for k, v in (("文本编码阶段（首条后端日志 → spill-fill 行）", b["te_phase_s"]),
                 ("transformer 四段装载（spill-fill 行 → 步循环起点）", b["transformer_load_s"]),
                 ("8 步循环", b["loop_s"]), ("VAE 至末条后端日志", b["vae_tail_s"]),
                 ("后端日志时间跨度之外", other)):
        print("| %s | %.1f | %.1f%% |" % (k, v, 100 * v / g))
    print("| **合计（SSE generation_time）** | **%.1f** | |" % g)
    print("\n四段装载拆分（「标记前空白」= 上一段 Initialized 到本段 Initializing，**含本段整文件读入 vector**）：\n")
    for n, gap, init in b["per_load"]:
        print("- `%s`：标记前空白 %.2f s ＋ createFromBuffer %.2f s" % (n, gap, init))
    print("- 合计：空白 %.2f s ＋ createFromBuffer %.2f s ＋ 末段到步循环 %.2f s" % (
        sum(x[1] for x in b["per_load"]), sum(x[2] for x in b["per_load"]), b["after_last_load_s"]))

    # --- 4. ScatterElements ---
    print("\n## 4. part1a 的 ScatterElements（部署 DLC 转储）\n")
    for tag, fn in (("1:1", "dlcinfo_part1a_fp16_L80.txt"), ("4:3", "dlcinfo_part1a_1184x896.txt")):
        ops, enc = dlc_scatter(os.path.join(LOGD, fn))
        print("**%s**（`%s`）：" % (tag, fn))
        for name, tens in ops:
            print("- `%s`：%s" % (name, "；".join("%s %s [%s] %s" % t for t in tens)))
        if tag == "1:1":
            for out, upd in (("select_scatter", "val_225"), ("select_scatter_4", "val_851")):
                print("- 编码核对 `%s` vs updates `%s`：**%s**（%s）" % (out, upd, "相同" if enc.get(out) == enc.get(upd) else "不同", enc.get(out)))

    # --- 5. 段间 CPU 侧重量化 ---
    c = json.load(open(CONTRACT, encoding="utf-8"))
    mm = {e["internal_graph_name"]: e for e in c["models"]}
    print("\n## 5. 段间大张量是否触发 CPU 侧逐元素重量化（mg 契约 scale/offset 比较）\n")
    for a_, t, b_ in (("transformer_part1a", "add_138", "transformer_part1b"), ("transformer_part1b", "unified", "transformer_part2a"),
                      ("transformer_part2a", "add_92", "transformer_part2b"), ("transformer_part1a", "unified_freqs", "transformer_part2a")):
        qo = [x for x in mm[a_]["outputs"] if x["name"] == t][0]
        qi = [x for x in mm[b_]["inputs"] if x["name"] == t][0]
        same = qo["quantization"] == qi["quantization"]
        print("- `%s` %s → %s：%d 元素，%s" % (t, a_[12:], b_[12:], qi["exact_bytes"] // 2, "编码相同（直拷）" if same else "**编码不同 ⇒ 每步 CPU 逐元素重量化**"))

    # --- 6. 探针 ---
    p = json.load(open(os.path.join(LOGD, "probe_results.json"), encoding="utf-8"))
    print("\n## 6. 宿主建图探针（fc99 单算子，SDK 2.48 Windows，soc_model 69）\n")
    print("| 臂 | 图名生效 | 配置 graphs 段 | dspArch | vtcmSize | numHvxThreads | md5 | Performance Estimates |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in p.items():
        if k.startswith("_") or not v.get("graphName"):
            continue
        cfg = (v.get("config") or {}).get("graphs", [{}])[0]
        ok = cfg.get("graph_names") == [v["graphName"]]
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (k, "✅" if ok else "❌", {x: y for x, y in cfg.items() if x != "graph_names"},
              v.get("dspArch"), v.get("vtcmSize"), v.get("numHvxThreads"), v["bin_md5"][:10], v.get("profile_has_performance_estimates", "—")))


if __name__ == "__main__":
    main()
