# -*- coding: utf-8 -*-
"""S7：补验台账 #177 —— noscatA 交付后，四个**非 1:1 比例**的首次出图验证。

## 为什么必须单独验
交付的是五比例 mg context，但 2026-09-19 只有 **1:1 出过图**。其余四张图在同一个文件里，
**一次都没跑过**。不得由 1:1 外推 —— §47.3 实测过「同型结构的两个实例结论可以相反」
（`node_select_scatter` 与 `node_select_scatter_4` 结构完全相同，一个安全一个毁图），
这里是同一处手术施加在五个不同形状的图上，情形同类。

## 判据（事前锁定，同步在 `EXP_PLAN_P2_SPEED.md` §8，执行后不得改）
| 门 | 内容 | 不过怎么办 |
|---|---|---|
| G1 不崩 | rc=0 且产出 PNG | 记录，继续跑其余比例 |
| G2 尺寸 | PNG 宽高 == 请求（防 `RequestParser` 对未交付尺寸**静默回落到 1024**） | 同上 |
| G3 用的是新件 | 该次 logcat 里 part1a 的 `[loadpath]` 字节数 == **2,877,374,464** | 同上 |
| G4 肉眼 | 人工打开图看（约束 1：不得转述形容词，必须自己看） | 脚本不判，只列路径 |

⊕ ION 峰值 / MemAvail 最低点**只记录、不设门**：非 1:1 没有同装置对照
（与 #172 是同一个缺口——非 1:1 从未存过参照 sha）。

🔴 **不自动回滚**：设备当前是交付态，回滚要推 3.15 GB。某个比例坏 ≠ 必须回滚
（用户可以避开那个比例），**全坏才必须回滚** —— 那是用户的成本取舍，拿到数据再由他定。

用法: python scripts/p2_s7_aspects.py --run
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D
import oneshot_mg_session as S
import p2_s6_session as P

OUT = os.path.join(D.RUNS, "p2s7")
STATE = os.path.join(OUT, "state.json")
SIZES = [(1184, 896), (896, 1184), (1280, 720), (720, 1280)]
WANT_BYTES = 2877374464          # noscatA 的字节数（G3 用它证明装载的是新件）


def main():
    if "--run" not in sys.argv:
        print(__doc__)
        return 0
    import lab_dev as dev
    os.makedirs(OUT, exist_ok=True)
    dev.require_online("p2s7")
    print("\n⚠️ 本脚本全程占用设备（4 次出图 + 冷却，约 25~30 分钟）。**结束前请勿拔线。**", flush=True)
    dev.sh("am force-stop %s || true" % S.PKG)
    time.sleep(5)
    base_c = D.npu_c(dev)
    print("热基线 NPU %.1f C" % base_c, flush=True)
    # 交付态自检：marker 必须在（H2 的交付形态）、part1a 必须是新件
    mk = dev.sh("run-as %s sh -c 'ls %s 2>/dev/null | wc -l'" % (S.PKG, P.MMAP_MARKER)).strip()
    nb = dev.sh("run-as %s stat -c %%s %s" % (S.PKG, P.DEV_PART1A)).strip()
    print("  交付态：LOAD_MMAP %s｜part1a %s 字节 %s"
          % ("在" if mk.endswith("1") else "**不在**", nb,
             "✅" if nb == str(WANT_BYTES) else "🔴 不是 noscatA"), flush=True)
    if nb != str(WANT_BYTES):
        raise SystemExit("🔴 设备上不是 noscatA ⇒ 本次补验没有意义，停")

    rows = []
    for w, h in SIZES:
        tag = "s7_%dx%d" % (w, h)
        try:
            info = S.gen(dev, tag, w=w, h=h, base_c=base_c)
        except SystemExit as exc:
            print("  🔴 %s 出图失败：%s" % (tag, exc), flush=True)
            rows.append({"size": "%dx%d" % (w, h), "g1": False, "g2": None, "g3": None})
            continue
        sp = P.split_of(tag)
        g3 = sp.get("part1a_bytes") == WANT_BYTES
        rows.append({"size": "%dx%d" % (w, h), "g1": True, "g2": bool(info["size_ok"]),
                     "g3": g3, "png_wh": info["png_wh"], "img": info["img"],
                     "sha256": info["sha256"], "gen_s": info.get("generation_s"),
                     "ion_peak": info["ion_peak"], "avail_low": info["avail_low"],
                     "part1a_bytes": sp.get("part1a_bytes"), "paths": sp.get("paths")})
        print("     G2 尺寸 %s｜G3 装载新件 %s（%s 字节）"
              % ("✅" if info["size_ok"] else "🔴", "✅" if g3 else "🔴",
                 sp.get("part1a_bytes")), flush=True)
        json.dump(rows, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n== #177 补验结果 ==")
    print("| 比例 | G1 不崩 | G2 尺寸 | G3 新件 | 单张 s | ION 峰值 | MemAvail 最低 | 图 |")
    print("|---|---|---|---|---|---|---|---|")
    allok = True
    for r in rows:
        ok = r["g1"] and r.get("g2") and r.get("g3")
        allok &= bool(ok)
        print("| %s | %s | %s | %s | %s | %s | %s | %s |"
              % (r["size"], "✅" if r["g1"] else "🔴",
                 "✅" if r.get("g2") else "🔴", "✅" if r.get("g3") else "🔴",
                 ("%.1f" % r["gen_s"]) if r.get("gen_s") else "—",
                 r.get("ion_peak", "—"), r.get("avail_low", "—"),
                 os.path.basename(r.get("img", "—"))))
    print("\n%s" % ("✅ 四个比例结构判据全过 —— **仍需人工看图**（G4，约束 1）" if allok else
                    "🔴 有比例未过，**不得声称五比例可用**；是否回滚由用户定（推 3.15 GB，约 5 分钟）"))
    print("图在 %s，我会逐张打开看。" % D.RUNS)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
