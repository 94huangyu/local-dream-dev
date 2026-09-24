# -*- coding: utf-8 -*-
"""表一 · 时间拆解 —— 单张 ~160 s 花在哪（`EXP_PLAN_MEMSPEED.md` §5）。

## 为什么必须先有它
方案 E 要用「32 次图切换」换内存。**没有这张表就无法判断 +32×t_switch 是可忽略还是 20% 劣化。**

## 数据来源（零设备：用已经落盘的 logcat）
后端自己每行都打了一个单调毫秒钟（`Backend: 171294.0ms [ INFO ] …`），
比 logcat 的墙钟时间戳可靠（后者会被批量刷新打乱）。用到四类标记：
  · `Initializing QNN App from Buffer: <graph> (size: N bytes)` / `QNN App Initialized from Buffer: <graph>`
    ⇒ 每个 context 的 **createFromBinary 耗时**
  · `[diagnostic] latents after step N: …`  ⇒ **每一步的结束时刻**
  · SSE 里的 `generation_time_ms` / `first_step_time_ms` ⇒ 与上面互校

## 🔴 这张表能说什么、不能说什么
✅ 能说：各阶段的**实测占比**，以及「+X 秒落在总时长的什么量级」。
❌ 不能说：它**不是当前 APK 的测量**——默认样本是 2026-08-30 F 验收那批（M1/g4_*），
   与现网的差别是 #86 加的惰性校验与多比例契约切换，**步循环结构未变**。
   要给现网数字，需在当前 APK 上重跑一次 `bash scripts/app_generate.sh`。

用法:
    python scripts/timing_breakdown.py                       # 默认 M1 + g4_01..03
    python scripts/timing_breakdown.py <logcat> [<logcat>…]
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RUNS = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
DEFAULT = ["M1", "g4_01", "g4_02", "g4_03"]

CLK = re.compile(r"Backend:\s+([0-9]+\.[0-9]+)ms")
# 逐段计时（2026-09-08 起 app 内埋点，见 PipelineZImage.hpp::runSeg）。
# 🔴 在此之前**没有任何 app 内的逐段数据**：#147 那些是 qnn-net-run 离线单段，
#    它自己标了「不等于 app 内每步耗时」。表一只能到「每步 16.22 s」为止。
SEGT = re.compile(r"\[segtime\]\s+(\S+)\s+(\d+)\s*ms(.*)$")
INIT_A = re.compile(r"Initializing QNN App from Buffer:\s+(\S+)\s+\(size:\s+(\d+)")
INIT_B = re.compile(r"QNN App Initialized from Buffer:\s+(\S+)")
STEP = re.compile(r"\[diagnostic\] latents after step (\d+):")
SEG = ("transformer_part1a", "transformer_part1b", "transformer_part2a", "transformer_part2b")
TE = ("text_encoder_part1", "text_encoder_part2", "text_encoder_part3", "text_encoder_part4")
# H5（2026-09-19 起的 APK）：段内五段拆时；H2：装载路径与该段装载起点。
SPLIT = re.compile(r"\[segsplit\]\s+(\S+)\s+n=(\d+)\s+prep\s+([\d.]+)\s+\|\s+in\s+([\d.]+)"
                   r"\s+\|\s+exec\s+([\d.]+)\s+\|\s+out\s+([\d.]+)\s+\|\s+post\s+([\d.]+)")
LOADPATH = re.compile(r"\[loadpath\]\s+(mmap|read-into-vector)\s+(\S+)\s+\((\d+) bytes\)")


def parse(path):
    """只扫一遍，取四类标记。**遇不到就报缺失，不填 0**（假数据比没数据更贵）。"""
    loads, steps, last = {}, {}, None
    open_at = {}
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CLK.search(line)
            if not m:
                continue
            t = float(m.group(1))
            last = t
            a = INIT_A.search(line)
            if a:
                open_at[a.group(1)] = (t, int(a.group(2)))
                continue
            b = INIT_B.search(line)
            if b and b.group(1) in open_at:
                t0, nbytes = open_at.pop(b.group(1))
                loads[b.group(1)] = (t0, t, t - t0, nbytes)
                continue
            s = STEP.search(line)
            if s:
                steps[int(s.group(1))] = t
    return loads, steps, last


def sse_numbers(path):
    if not os.path.isfile(path):
        return {}
    t = io.open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for k in ("generation_time_ms", "first_step_time_ms"):
        m = re.search(k + r"\D{0,4}(\d+)", t)
        if m:
            out[k] = int(m.group(1))
    return out


def one(tag, path):
    loads, steps, last = parse(path)
    sse = sse_numbers(os.path.join(RUNS, tag + ".sse"))
    if not steps:
        print("  🔴 %s：没有 step 标记，跳过（该 logcat 可能不是一次完整生成）" % tag)
        return None
    n_steps = max(steps) + 1
    t_step_end = steps[max(steps)]
    t_step0 = steps[0]
    seg_loads = {k: loads[k] for k in SEG if k in loads}
    te_loads = {k: loads[k] for k in TE if k in loads}
    r = {"tag": tag, "n_steps": n_steps}
    r["te_load_s"] = sum(v[2] for v in te_loads.values()) / 1000.0
    r["seg_load_s"] = sum(v[2] for v in seg_loads.values()) / 1000.0
    # 步循环：step0 结束 − first_step_ms = 循环起点（SSE 给的 first_step 是步 0 的墙钟耗时）
    if "first_step_time_ms" in sse:
        loop_start = t_step0 - sse["first_step_time_ms"]
    else:
        loop_start = t_step0 - (t_step_end - t_step0) / max(1, (n_steps - 1))
    r["loop_s"] = (t_step_end - loop_start) / 1000.0
    r["per_step_s"] = r["loop_s"] / n_steps
    vae = loads.get("vae_decoder")
    r["vae_load_s"] = vae[2] / 1000.0 if vae else None
    r["vae_tail_s"] = (last - vae[0]) / 1000.0 if vae and last else None
    r["total_s"] = sse.get("generation_time_ms", 0) / 1000.0 or None
    r["seg_detail"] = {k: v[2] / 1000.0 for k, v in seg_loads.items()}
    return r


def seg_report(path):
    """逐段耗时表（需要 2026-09-08 之后的 APK；旧 logcat 里没有 [segtime]）。"""
    per = {}
    order = []
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SEGT.search(line)
            if not m:
                continue
            name, ms, note = m.group(1), int(m.group(2)), m.group(3).strip()
            if name not in per:
                per[name] = []
                order.append(name)
            per[name].append((ms, note))
    if not per:
        print("  （这份 logcat 里没有 [segtime] —— 说明是 2026-09-08 之前的 APK）")
        return False
    print("\n## 逐段耗时（app 内实测，n = 每段的步数）\n")
    print("| 段 | 次数 | 中位 ms | 最小 | 最大 | 合计 ms | 占步循环 |")
    print("|---|---|---|---|---|---|---|")
    tot = sum(sum(x for x, _ in v) for v in per.values())
    for n in order:
        v = sorted(x for x, _ in per[n])
        s = sum(v)
        print("| `%s` | %d | **%d** | %d | %d | %d | %.1f%% |"
              % (n, len(v), v[len(v) // 2], v[0], v[-1], s, 100.0 * s / max(tot, 1)))
    print("| **合计** | | | | | **%d** | 100%% |" % tot)
    notes = sorted(set(nt for v in per.values() for _, nt in v if nt))
    if notes:
        print("\n⊕ 标注：%s" % "；".join(notes))
    print("\n🔴 这是**段级墙钟**，含该段的输入搬运与输出拷贝，不是纯计算时间。")
    return True


def split_report(path):
    """H5：段内五段拆时（prep/in/exec/out/post）+ H2：装载路径与逐段装载墙钟。

    🔴 自证（EXP_PLAN_P2_SPEED §7.2 G2，事前锁定）：各段 exec 之和必须与同一份
       logcat 的 `[segtime]` 之和相差 < 5%，否则埋点口径有误 ⇒ **本表作废**。
       （约束 7：标尺没被验证过就不许用它下结论。）
    """
    rows, paths, clk, load = [], [], {}, {}
    segt_total = 0
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            mc = CLK.search(line)
            t = float(mc.group(1)) if mc else None
            m = SPLIT.search(line)
            if m:
                rows.append((m.group(1), int(m.group(2))) +
                            tuple(float(m.group(i)) for i in range(3, 8)))
            mp = LOADPATH.search(line)
            if mp:
                paths.append((mp.group(1), mp.group(2), int(mp.group(3))))
                if t is not None:
                    clk[mp.group(2)] = t          # 该段装载的起点（后端单调钟）
            mb = INIT_B.search(line)
            if mb and t is not None and mb.group(1) in clk:
                load[mb.group(1)] = t - clk.pop(mb.group(1))
            ms = SEGT.search(line)
            if ms:
                segt_total += int(ms.group(2))
    if not rows and not paths:
        print("  （这份 logcat 里没有 [segsplit]/[loadpath] —— 说明是 2026-09-19 之前的 APK）")
        return False

    if paths:
        kinds = sorted(set(k for k, _, _ in paths))
        print("\n## 装载路径（H2）\n")
        print("  路径：%s%s" % ("、".join(kinds),
                                "  🔴 同一次运行里混用了两条路径，A/B 失效" if len(kinds) > 1 else ""))
        print("\n| 段 | 路径 | 文件 MB | 装载墙钟 ms |")
        print("|---|---|---|---|")
        tot = 0.0
        for kind, name, nbytes in paths:
            ms = load.get(name)
            tot += ms or 0.0
            print("| `%s` | %s | %.0f | %s |" % (name, kind, nbytes / 1e6,
                                                 ("%.0f" % ms) if ms is not None else "—（缺配对行）"))
        print("| **合计** | | | **%.0f** |" % tot)
        print("\n⊕ 口径：`[loadpath]` 行 → 同段 `QNN App Initialized from Buffer` 行，"
              "后端自打的单调钟；含 createFromBuffer 全过程。")

    if rows:
        agg = {}
        for name, n, prep, tin, ex, out, post in rows:
            a = agg.setdefault(name, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            a[0] += n
            for i, v in enumerate((prep, tin, ex, out, post)):
                a[i + 1] += v
        print("\n## 段内拆时（H5，app 内实测）\n")
        print("| 段 | 次数 | prep | 拷入 | **graphExecute** | 拷出 | 落库 | 合计 ms | 非算力 |")
        print("|---|---|---|---|---|---|---|---|---|")
        ex_sum = 0.0
        all_sum = 0.0
        for name in sorted(agg):
            n, prep, tin, ex, out, post = agg[name]
            s = prep + tin + ex + out + post
            ex_sum += ex
            all_sum += s
            print("| `%s` | %d | %.0f | %.0f | **%.0f** | %.0f | %.0f | %.0f | %.1f%% |"
                  % (name, n, prep, tin, ex, out, post, s, 100.0 * (s - ex) / max(s, 1e-9)))
        print("| **合计** | | | | **%.0f** | | | **%.0f** | **%.1f%%** |"
              % (ex_sum, all_sum, 100.0 * (all_sum - ex_sum) / max(all_sum, 1e-9)))
        if segt_total:
            d = abs(all_sum - segt_total) / max(segt_total, 1)
            print("\n**G2 自证**：拆时合计 %.0f ms vs `[segtime]` 合计 %d ms ⇒ 相差 %.1f%% %s"
                  % (all_sum, segt_total, 100 * d,
                     "✅ < 5%，本表可用" if d < 0.05 else "🔴 ≥ 5%，**本表作废**（埋点口径有误）"))
        else:
            print("\n⚠️ 同一份 logcat 里没有 `[segtime]`，G2 自证做不了 ⇒ 本表**只能参考**。")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--split" in sys.argv:
        p = args[0] if args else os.path.join(RUNS, "os_g0_live_logcat.txt")
        if not os.path.isfile(p):
            raise SystemExit("🔴 找不到 %s" % p)
        print("# 段内拆时 / 装载路径 —— %s" % os.path.basename(p))
        return 0 if split_report(p) else 1
    if "--seg" in sys.argv:
        p = args[0] if args else os.path.join(RUNS, "os_g0_live_logcat.txt")
        if not os.path.isfile(p):
            raise SystemExit("🔴 找不到 %s" % p)
        print("# 逐段耗时 —— %s" % os.path.basename(p))
        return 0 if seg_report(p) else 1
    tags = args or DEFAULT
    rows = []
    for tag in tags:
        p = tag if os.path.isfile(tag) else os.path.join(RUNS, tag + "_logcat.txt")
        if not os.path.isfile(p):
            print("  ⚠️ 找不到 %s，跳过" % p)
            continue
        r = one(os.path.basename(p).replace("_logcat.txt", ""), p)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit("🔴 没有可用样本")

    print("# 表一 · 时间拆解（后端自打的单调毫秒钟；样本 = 已落盘 logcat，零设备）\n")
    print("| 样本 | 总时长 s | 文本编码装载 | 四段装载 | 8 步循环 | 每步 | VAE 段 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %.2f | %.2f | **%.2f** | %.2f | %s |" % (
            r["tag"], ("%.2f" % r["total_s"]) if r["total_s"] else "—",
            r["te_load_s"], r["seg_load_s"], r["loop_s"], r["per_step_s"],
            ("%.2f" % r["vae_tail_s"]) if r["vae_tail_s"] else "—"))

    ref = rows[0]
    tot = ref["total_s"] or (ref["te_load_s"] + ref["seg_load_s"] + ref["loop_s"])
    print("\n## 以 %s 为准的占比\n" % ref["tag"])
    print("| 阶段 | 秒 | 占总时长 |")
    print("|---|---|---|")
    for name, v in (("文本编码四段装载", ref["te_load_s"]),
                    ("transformer 四段装载", ref["seg_load_s"]),
                    ("**8 步去噪循环**", ref["loop_s"])):
        print("| %s | %.2f | %.1f%% |" % (name, v, 100.0 * v / tot))
    other = tot - ref["te_load_s"] - ref["seg_load_s"] - ref["loop_s"]
    print("| 其余（分词/文本编码执行/VAE/像素重排） | %.2f | %.1f%% |" % (other, 100.0 * other / tot))

    print("\n## 四段各自的 createFromBinary 耗时（%s）\n" % ref["tag"])
    for k, v in ref["seg_detail"].items():
        print("- `%s` **%.2f s**" % (k, v))

    print("\n## 对 E/D 决策的直接含义\n")
    print("- 步循环占 **%.1f%%**，每步 **%.2f s** ⇒ 一次图切换要摊进每步的 %.2f s 里。"
          % (100.0 * ref["loop_s"] / tot, ref["per_step_s"], ref["per_step_s"]))
    for ts in (1.0, 2.0, 4.0):
        print("  · t_switch = %.1f s ⇒ 32 次 = +%.0f s，总时长 %.0f → %.0f s（**+%.1f%%**）"
              % (ts, 32 * ts, tot, tot + 32 * ts, 100.0 * 32 * ts / tot))
    print("- 参照：D（回退 F、part2 每步装卸）**实测 +43.4 s**（#145/F）"
          "⇒ 总时长 %.0f → %.0f s（**+%.1f%%**）。E 要赢 D 需 t_switch < %.2f s。"
          % (tot, tot + 43.4, 100.0 * 43.4 / tot, 43.4 / 32))
    print("\n🔴 样本口径：默认样本来自 2026-08-30 的 F 验收批，**不是当前 APK**。"
          "现网数字需在当前 APK 上重跑一次 `bash scripts/app_generate.sh` 再解析。")


if __name__ == "__main__":
    main()
