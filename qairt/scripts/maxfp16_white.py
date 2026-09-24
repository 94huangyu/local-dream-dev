"""#114 步 2：把四段 `<seg>_amax.json` 按 EXP_PLAN_MAXFP16 §3.2 的规则出白名单。

规则（事前锁定，本脚本不得偏离）：
  1) 不溢出：**实测** |a|max * 2.0 <= 65504
  2) 定点没有已经更优：E_bulk(uFxp_16, 部署 encoding) > E_bulk(FP16)
  3) 不是本段图输入/图输出   -- 已在步 1 出局
  4) 名字要能映射到 ONNX     -- 已在步 1 出局

用法: python maxfp16_white.py [--apply] [--top N]
  默认只打印；--apply 才写 allseg_white.json（旧的备份为 allseg_white_26.json）
  --top N: 只取 E_bulk(uFxp16) 最大的 N 个（G2 撞 PD 红线时的降级方案，EXP_PLAN §五）
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
SEGS = ["part1a", "part1b", "part2a", "part2b"]
FP16MAX = 65504.0
MARGIN = 2.0


def main():
    top = None
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    old = json.load(open(os.path.join(W, "allseg_white.json"), encoding="utf-8"))
    new, report = {}, []
    for seg in SEGS:
        p = os.path.join(W, "%s_amax.json" % seg)
        if not os.path.exists(p):
            sys.exit("FAIL 缺 %s -- 步 1 未完成" % p)
        d = json.load(open(p, encoding="utf-8"))
        t = d["tensors"]
        ov = [k for k, r in t.items() if r["amax"] * MARGIN > FP16MAX]
        fx = [k for k, r in t.items()
              if r["amax"] * MARGIN <= FP16MAX and r["e_ufxp"] is not None
              and r["e_ufxp"] <= r["e_fp16"]]
        wl = [k for k, r in t.items()
              if r["amax"] * MARGIN <= FP16MAX and r["e_ufxp"] is not None
              and r["e_ufxp"] > r["e_fp16"]]
        wl.sort(key=lambda k: -t[k]["e_ufxp"])
        if top is not None:
            wl = wl[:top]
        # 旧白名单里有而新规则漏掉的 -- 必须显式报出来（不得静默丢失当前最优的成分）
        lost = [k for k in old.get(seg, []) if k not in wl]
        new[seg] = sorted(wl)
        report.append((seg, d["n_cand"], len(t), len(ov), len(fx), len(wl), lost))

    print("%-8s %7s %7s %8s %9s %8s %s" %
          ("段", "候选", "可测", "①溢出", "②定点优", "**入选**", "旧名单丢失"))
    for seg, nc, nm, no, nf, nw, lost in report:
        print("%-8s %7d %7d %8d %9d %8d  %s" %
              (seg, nc, nm, no, nf, nw, ",".join(lost) if lost else "-"))
    tot = sum(r[5] for r in report)
    print("\n四段合计入选 **%d**（当前部署白名单 %d）" %
          (tot, sum(len(v) for v in old.values())))

    # 入选张量里 uFxp16 误差最大的 10 个（说明收益主要来自哪里）
    allt = []
    for seg in SEGS:
        d = json.load(open(os.path.join(W, "%s_amax.json" % seg), encoding="utf-8"))["tensors"]
        for k in new[seg]:
            allt.append((d[k]["e_ufxp"], d[k]["e_fp16"], d[k]["amax"], seg, k))
    allt.sort(reverse=True)
    print("\n入选里 uFxp16 主体误差最大的 10 个：")
    print("%-8s %-22s %10s %10s %10s" % ("段", "张量", "|a|max", "uFxp16%", "FP16%"))
    for eu, ef, am, seg, k in allt[:10]:
        print("%-8s %-22s %10.1f %10.4f %10.4f" % (seg, k, am, eu, ef))

    if "--apply" in sys.argv:
        bak = os.path.join(W, "allseg_white_26.json")
        if not os.path.exists(bak):
            json.dump(old, open(bak, "w"), indent=1)
            print("\n旧白名单已备份 -> %s" % bak)
        json.dump(new, open(os.path.join(W, "allseg_white.json"), "w"), indent=1)
        print("已写入 allseg_white.json（%d 个张量）" % tot)
    else:
        print("\n（只读预览。加 --apply 才写盘）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
