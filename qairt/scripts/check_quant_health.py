#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化健康检查 —— 卡 C 门 C5（指南 §二十八、§二十九）

转任何新模型，量化后**第一件事**跑这个。

唯一需要的指标：**量化级/元素**
--------------------------------
    levels = median(|a|) / scale

它回答"一个典型元素能用到几个量化级"。`< 10` 就是分辨率被摧毁了。
§28.2 说明了它为什么比相对 L2 好用：相对 L2 被离群值主导（本项目低估达 48 倍，约束 7）。

🔴 两条必须遵守的口径（否则筛出来的名单是错的）
-----------------------------------------------
1. **`|a|max` 必须用实测值，不能用标定值**（§28.6、§28.10）。
   本项目实测：53 个张量里 **26 个**的真实 amax 超过标定 calib_max，
   最大低估 **1.9512 倍** —— 照标定值筛会把超范围的张量放进 FP16 白名单，
   设备上直接出 inf（§29.4 门 G4 就拦下过 38% inf）。
   没有实测值时，§29.2 规则 2 要求用「标定 × 4.0」作保守代理。
2. **stats 怎么来**：ONNX 加 `graph.output` 跑一次 FP32 前向，
   **只算 `median(|a|)` 与 `|a|max` 两个标量、不落盘**。
   ⚠️ §31.5：**一次声明几百个图输出会把机器压死** —— 必须分批。

用法
----
    python scripts/check_quant_health.py <stats.json>
    python scripts/check_quant_health.py <stats.json> --fp16-candidates
    python scripts/check_quant_health.py --selftest

stats.json 格式：{"<张量名>": {"scale": s, "med": m, "amax": a, "calib_max": c}, ...}
（`med`/`amax` 来自实测前向；`scale`/`calib_max` 来自 `snpe-dlc-info`）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LEVELS_THRESHOLD = 10.0        # §28.1 / §29.2 规则 1
FP16_MAX = 65504.0             # float16 上限
SAFETY_MEASURED = 2.0          # §29.2 规则 2：实测值留 2 倍余量
SAFETY_CALIB = 4.0             # 没有实测值时的保守代理

# 已知样本（约束 8）。期望值 2026-09-20 **实测**锁定，不是从 §29.1 抄的。
# 这份 stats 是 **L80 版**，与当前交付态一致（谱系确认交付用的是 part*_fp16_L80_quantized.dlc）。
SELFTEST = {
    "path": Path(r"D:\ZImage_Work\p0_experiments\p2attr\risky_stats.json"),
    "n_total": 53,
    "n_risky": 29,                    # levels < 10
    "lowest_name": "mul_71",
    "lowest_levels": 0.0625,
    "n_underestimated": 26,           # 实测 amax > 标定 calib_max
    "max_underestimate": 1.9512,      # §28.10 记的「1.83~1.95 倍」
}


def load_stats(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "scale" in v:
            out[k] = v
    return out


def analyze(stats: dict):
    rows = []
    for name, v in stats.items():
        scale, med = v.get("scale"), v.get("med")
        if not scale or med is None:
            continue
        levels = med / scale
        amax, calib = v.get("amax"), v.get("calib_max")
        # §29.2 规则 2：有实测用 ×2，没有用标定 ×4
        headroom_ok = None
        if amax is not None:
            headroom_ok = amax * SAFETY_MEASURED <= FP16_MAX
            basis = f"实测 amax {amax:.6g} × {SAFETY_MEASURED}"
        elif calib is not None:
            headroom_ok = calib * SAFETY_CALIB <= FP16_MAX
            basis = f"标定 calib_max {calib:.6g} × {SAFETY_CALIB}（无实测，保守代理）"
        else:
            basis = "🔴 既无实测也无标定量程 —— 无法判 FP16 可行性"
        rows.append({"name": name, "levels": levels, "med": med, "scale": scale,
                     "amax": amax, "calib_max": calib,
                     "headroom_ok": headroom_ok, "basis": basis,
                     "under": (amax / calib) if (amax and calib and calib > 0) else None})
    rows.sort(key=lambda r: r["levels"])
    return rows


def report(rows, fp16=False):
    risky = [r for r in rows if r["levels"] < LEVELS_THRESHOLD]
    print(f"\n共 {len(rows)} 个张量；**量化级/元素 < {LEVELS_THRESHOLD:g} 的有 {len(risky)} 个**")

    print(f"\n--- 分辨率最差的（手术名单，§29.1）---")
    for r in risky[:15]:
        print(f"  {r['name']:<24} levels={r['levels']:9.4f}  med={r['med']:.4g}  scale={r['scale']:.4g}")
    if len(risky) > 15:
        print(f"  … 另 {len(risky) - 15} 个")

    under = [r for r in rows if r["under"] and r["under"] > 1.0]
    if under:
        mx = max(r["under"] for r in under)
        n_have = len([r for r in rows if r["under"]])
        print(f"\n--- 🔴 标定量程低估（§28.6/§28.10）---")
        print(f"  {len(under)}/{n_have} 个张量的**实测 amax 超过标定 calib_max**，最大低估 {mx:.4f} 倍")
        print(f"  ⇒ **筛 FP16 必须按实测量程**，按标定值筛会漏掉超范围的张量 ⇒ 设备上出 inf")

    if fp16:
        print(f"\n--- FP16 候选（§29.2 三条规则，缺一不可）---")
        print(f"  规则 1 levels < {LEVELS_THRESHOLD:g}；规则 2 量程留余量 ≤ {FP16_MAX:g}；"
              f"规则 3 🔴 **残差流张量不得入选（本工具判不了，要你自己排除）**")
        ok = [r for r in risky if r["headroom_ok"]]
        no = [r for r in risky if r["headroom_ok"] is False]
        for r in ok[:15]:
            print(f"  ✅ {r['name']:<24} levels={r['levels']:8.4f}  {r['basis']}")
        for r in no[:8]:
            print(f"  ❌ {r['name']:<24} 量程不够：{r['basis']} > {FP16_MAX:g}")
        print(f"\n  候选 {len(ok)} 个，量程否决 {len(no)} 个")
        print(f"  🔴 规则 3 必须人工过一遍：残差流的输入锥 = 它之前的整个网络，"
              f"会把浮点铺满全图\n     （本项目含 15 个残差 Add 时 Float_16 达 183 个，去掉后 153 个才建得出图）")
        print(f"  🔴 §29.7：**转更多张量到 FP16 不是单调更好的，存在最优点**")
    return risky


def selftest():
    s = SELFTEST
    print("=== 判据自检（已知样本）===")
    if not s["path"].exists():
        print(f"  ⚠️  样本不在：{s['path']}")
        return 1
    rows = analyze(load_stats(s["path"]))
    ok = True

    def chk(label, got, want, tol=0.0):
        good = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
        print(f"  {'✅' if good else '❌'} {label}: {got}（期望 {want}）")
        return good

    ok &= chk("张量总数", len(rows), s["n_total"])
    ok &= chk(f"levels < {LEVELS_THRESHOLD:g} 的个数",
              len([r for r in rows if r["levels"] < LEVELS_THRESHOLD]), s["n_risky"])
    ok &= chk("最低 levels 的张量名", rows[0]["name"], s["lowest_name"])
    ok &= chk("最低 levels 值", round(rows[0]["levels"], 4), s["lowest_levels"], 1e-4)
    und = [r for r in rows if r["under"] and r["under"] > 1.0]
    ok &= chk("实测 amax 超过标定的个数", len(und), s["n_underestimated"])
    ok &= chk("最大低估倍数", round(max(r["under"] for r in und), 4),
              s["max_underestimate"], 1e-4)

    # 公式自证：levels 必须恒等于 med/scale（防解析/单位错）
    bad = [r for r in rows if abs(r["levels"] - r["med"] / r["scale"]) > 1e-9 * max(1.0, r["levels"])]
    ok &= chk("levels == med/scale 不成立的个数", len(bad), 0)

    print("\n判据自检：" + ("✅ 通过" if ok else "❌ 未通过 —— 结论不可用"))
    print("⊕ 这份样本是 **L80 版**，与当前交付态一致（谱系：part*_fp16_L80_quantized.dlc）。")
    print("🔴 但它只覆盖 part2a 的 53 个高危候选，**不是全图健康检查**；"
          "新模型要自己跑前向拿 stats（注意 §31.5 分批，别压死宿主）。")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="量化健康检查：量化级/元素")
    ap.add_argument("stats", nargs="?", help="stats JSON")
    ap.add_argument("--fp16-candidates", action="store_true", help="按 §29.2 筛 FP16 候选")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.stats:
        ap.error("需要 stats JSON，或用 --selftest")

    p = Path(a.stats)
    if not p.exists():
        sys.exit(f"不存在：{p}")
    rows = analyze(load_stats(p))
    if not rows:
        sys.exit("没有可用条目（需要每项含 scale 与 med）")
    report(rows, a.fp16_candidates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
