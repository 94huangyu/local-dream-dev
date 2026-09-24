"""#114 的 **G0-contra 门**：白名单规则必须与已有的端到端反例一致。

背景（EXP_PLAN_MAXFP16 §3.3）：
  #112 是**端到端实测**：part1b 白名单从 2 个扩到 5 个（新增 `mul_406`/`mul_431`/`mul_456`），
  L3 PSNR 从 **15.53 dB 掉到 14.32 dB**。
  而 #114 的选择规则来自**宿主表示误差**（交叉点分析），
  按 #52 实测，表示误差**不是** HTP 误差的预测量（37.6 倍偏差）。

  => 若本规则把这 3 个张量判为**入选**，就等于规则与唯一的端到端反例直接冲突,
     该规则作为预测量已被证否，**不得**据它投入步 3 的 7.3 小时。

判据（事前锁定，不得事后修改）：
  PASS  = 3 个全部**出局**（规则解释了 #112）
  FAIL  = 3 个全部**入选**（规则被证否）
  MIXED = 部分入选  => 规则有部分解释力但不完整，**按 FAIL 处理**（不得选择性采信）

用法: python maxfp16_contra.py
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
FP16MAX = 65504.0
MARGIN = 2.0
CONTRA = ["mul_406", "mul_431", "mul_456"]     # #112 实测使 L3 −1.21 dB 的三个张量
BASE = ["mul_306", "mul_356"]                  # 当前最优配置里 part1b 的 2 个（对照）


def verdict(r):
    if r["amax"] * MARGIN > FP16MAX:
        return "出局", "规则①溢出（|a|max*2=%.0f > 65504）" % (r["amax"] * MARGIN)
    if r["e_ufxp"] is None:
        return "出局", "无有效主体统计"
    if r["e_ufxp"] <= r["e_fp16"]:
        return "出局", "规则②定点已更优（uFxp16 %.4f%% <= FP16 %.4f%%）" % (r["e_ufxp"], r["e_fp16"])
    return "入选", "uFxp16 %.4f%% > FP16 %.4f%%（FP16 好 %.1f 倍）" % (
        r["e_ufxp"], r["e_fp16"], r["e_ufxp"] / max(r["e_fp16"], 1e-9))


def main():
    full = os.path.join(W, "part1b_amax.json")
    part = os.path.join(W, "part1b_amax_only.json")
    if os.path.exists(full):
        p, tag = full, "全量步 1"
    elif os.path.exists(part):
        p, tag = part, "**--only 快检**（只测了本门需要的 5 个张量）"
    else:
        sys.exit("FAIL 既没有 %s 也没有 %s —— part1b 步 1 未跑" % (full, part))
    t = json.load(open(p, encoding="utf-8"))["tensors"]
    print("数据来源: %s  [%s]" % (os.path.basename(p), tag))

    print("G0-contra 门：#112 的三个反例张量按 §3.2 规则的判定")
    print("（#112 实测：把它们加进白名单 => L3 15.53 -> 14.32 dB，**倒退 1.21 dB**）")
    print("")
    print("%-10s %11s %11s %9s %9s %9s %9s %7s %-6s"
          % ("张量", "标定|max|", "实测|a|max", "主体uFxp", "主体FP16",
             "全量uFxp", "全量FP16", "集中度", "判定"))
    print("-" * 108)
    res = {}
    for n in CONTRA + BASE:
        r = t.get(n)
        if r is None:
            print("%-10s  <不在可测集里>" % n)
            continue
        v, why = verdict(r)
        res[n] = v

        def f(x):
            return "  inf" if x is None or x != x or x == float("inf") else "%9.4f" % x
        print("%-10s %11.1f %11.1f %s %s %s %s %6.2f%% %-6s"
              % (n, r["calib_max"], r["amax"], f(r["e_ufxp"]), f(r["e_fp16"]),
                 f(r.get("e_ufxp_all")), f(r.get("e_fp16_all")),
                 r.get("conc1") or 0.0, v))
        print("           理由: %s" % why)
    print("")
    print("🔴 看全量列：若某张量**主体口径 FP16 更优、但全量口径 FP16 反而更差**，")
    print("   就是 #112 所说的「FP16 救主体、毁离群值」—— 而规则②只看主体，会漏掉它。")

    hit = [n for n in CONTRA if res.get(n) == "入选"]
    print("")
    print("对照：当前最优配置里 part1b 的 2 个张量 %s 应当**入选**（否则规则连已知有效的都筛掉了）"
          % BASE)
    base_ok = all(res.get(n) == "入选" for n in BASE if n in res)
    print("  => 对照 %s" % ("PASS（2 个都入选）" if base_ok else
                            "🔴 FAIL：规则把已验证有效的张量也筛掉了，规则过严"))
    print("")
    if not hit:
        print("[G0-contra] **PASS** —— 3 个反例张量全部出局，规则解释了 #112。")
        print("            => #112 与 #114 的冲突消解，可以投入步 3。")
        return 0 if base_ok else 1
    print("[G0-contra] 🔴 **FAIL** —— %d/3 个反例张量被判为入选：%s" % (len(hit), hit))
    print("            => 规则与唯一的端到端反例冲突，**作为预测量已被证否**。")
    print("            => 不得投入步 3 的 7.3 小时。必须先解释 #112 或换判据。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
