# -*- coding: utf-8 -*-
"""表二 · 内存拆解 —— 峰值常驻 ION 由什么构成（`EXP_PLAN_MEMSPEED.md` §5）。

## 为什么要有它
方案 C~F 的收益上限完全取决于「省掉的是哪一项」：
图切换省的是**未加载图的全部项**，spill-fill 共享只省 `spillFillBufferSize`。
**没有这张表，任何「能省多少」都是猜。**

## 口径（官方公式，出处必须连同适用范围一起引）
`docs/QAIRT-Docs/QNN/general/tools.html` → *Memory Usage Scenarios*：
  Use Case 1（单模型推理）:
    Total RAM = OpDataSize + constSize + DDRTensorSize
              + spillFillBufferSize + graphIOTensorSize + vtcmSize
  Use Case 2（LLM 权重共享）: 各图上式之和 **+ Shared Weights 一次**
⇒ 本项目 single 形态属 Use Case 1，四段各算一次再相加。
字段单位：vtcmSize 是 **MB**，其余是**字节**（`check_mg_equivalence.mem()` 已踩过这个坑）。

## 这张表能说什么、不能说什么
✅ 能说：**各项的相对占比**，以及「某个方案最多能省掉哪几项」。
❌ 不能说：它**不是 ION 实测值**。公式是官方的静态估算，实测 ION 另有开销
   （#145 实测 DSP/ION ≈ context 文件大小的 1.42×）。
   ⇒ 报告里必须与实测 ION 并列，标明哪一列是实测、哪一列是公式。

用法:
    python scripts/mem_breakdown.py                 # 默认跑现网 1:1 四段
    python scripts/mem_breakdown.py <bin> [<bin>…]  # 指定文件
    python scripts/mem_breakdown.py --md            # 输出 markdown 表格
"""
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
P2ATTR = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")

# 现网 1:1（L=80，Clip+FP16）四段 —— 文件大小已与交付契约 size_bytes 逐字节核对
DEPLOYED_1x1 = [
    ("part1a", os.path.join(P2ATTR, "ctx_part1a_fp16_L80", "part1a_fp16_L80.SM8750.bin")),
    ("part1b", os.path.join(P2ATTR, "ctx_part1b_fp16_L80", "part1b_fp16_L80.SM8750.bin")),
    ("part2a", os.path.join(P2ATTR, "ctx_part2a_fp16_L80", "part2a_fp16_L80.SM8750.bin")),
    ("part2b", os.path.join(P2ATTR, "ctx_part2b_fp16_L80", "part2b_fp16_L80.SM8750.bin")),
]

# 实测 ION 增量（MiB）。来源必须可追：
#   probe = 台账 #145，scripts/quadctx_probe 独立 ARM64 探针，四段依次加载并保持存活
#   app   = 台账 #145，真实 app 内测得
MEASURED_ION_MIB = {
    "part1a": {"probe": 2928, "app": 2967},
    "part1b": {"probe": 1989, "app": 1986},
    "part2a": {"probe": 1954, "app": 1990},
    "part2b": {"probe": 1812, "app": 1936},
}

MIB = 1024.0 * 1024.0
FIELDS = ["opDataSize", "constSize", "ddrTensorSize",
          "spillFillBufferSize", "ioTensorSize", "vtcmSize"]


def dump(binpath):
    """读 context 元数据。🔴 透传 stderr 与返回码（约束 11·再补：报错会说谎）。"""
    if not os.path.isfile(binpath):
        raise SystemExit(f"🔴 文件不存在: {binpath}")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "m.json")
        r = subprocess.run([UTIL, "--context_binary", binpath, "--json_file", out],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if not os.path.isfile(out):
            sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
            raise SystemExit(f"🔴 qnn-context-binary-utility 未产出 json (rc={r.returncode})")
        with open(out, encoding="utf-8") as f:
            return json.load(f)


def graph_mem(ginfo):
    """单个图的六项。字段分两层且单位不统一 —— vtcmSize 是 MB。"""
    b1 = (ginfo.get("graphBlobInfo") or {}).get("info") or {}
    b2 = ginfo.get("graphBlobInfoV2") or {}
    if not b1 and not b2:
        raise SystemExit("🔴 元数据里没有 graphBlobInfo/graphBlobInfoV2 —— "
                         "字段路径变了，先修脚本，不得让本表变成一排 0")
    m = {
        "opDataSize": int(b2.get("opDataSize", 0) or 0),
        "constSize": int(b2.get("constSize", 0) or 0),
        "ddrTensorSize": int(b2.get("ddrTensorSize", 0) or 0),
        "spillFillBufferSize": int(b1.get("spillFillBufferSize", 0) or 0),
        "ioTensorSize": int(b2.get("ioTensorSize", 0) or 0),
        "vtcmSize": int(b1.get("vtcmSize", 0) or 0) * 1024 * 1024,
    }
    if sum(m.values()) == 0:
        raise SystemExit("🔴 六项全为 0 —— 拒绝用假数据出表")
    return m, int(b2.get("sharedWeightsSize", 0) or 0)


def collect(pairs):
    rows = []
    for tag, path in pairs:
        info = dump(path)["info"]
        for g in info["graphs"]:
            gi = g["info"]
            m, shared = graph_mem(gi)
            rows.append({
                "tag": tag,
                "graph": gi["graphName"],
                "file_bytes": os.path.getsize(path),
                "mem": m,
                "shared": shared,
                "total": sum(m.values()),
            })
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_md = "--md" in sys.argv
    pairs = DEPLOYED_1x1 if not args else [(os.path.basename(p), p) for p in args]
    rows = collect(pairs)

    grand = sum(r["total"] for r in rows)
    print("# 表二 · 内存拆解（官方公式 tools.html / Memory Usage Scenarios · Use Case 1）\n")
    hdr = ["段", "图名", "文件 MiB", "constSize", "opData", "ddrTensor",
           "spillFill", "ioTensor", "vtcm", "公式合计 MiB", "实测 ION MiB(app)"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        m = r["mem"]
        meas = MEASURED_ION_MIB.get(r["tag"], {}).get("app")
        print("| {} | `{}` | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | **{:.0f}** | {} |".format(
            r["tag"], r["graph"], r["file_bytes"] / MIB,
            m["constSize"] / MIB, m["opDataSize"] / MIB, m["ddrTensorSize"] / MIB,
            m["spillFillBufferSize"] / MIB, m["ioTensorSize"] / MIB, m["vtcmSize"] / MIB,
            r["total"] / MIB,
            f"**{meas}**" if meas else "—"))

    print("\n## 各项在四段合计中的占比\n")
    print("| 项 | MiB | 占公式合计 | 能被什么手段省掉 |")
    print("|---|---|---|---|")
    lever = {
        "constSize": "图切换（未加载图的权重不驻留）／跨比例权重共享",
        "opDataSize": "图切换",
        "ddrTensorSize": "图切换",
        "spillFillBufferSize": "**跨 context 共享 spill-fill（#146）**：四段共用最大者一份",
        "ioTensorSize": "图切换（幅度小）",
        "vtcmSize": "无（片上 VTCM，8 MB/图，可忽略）",
    }
    for f in FIELDS:
        v = sum(r["mem"][f] for r in rows)
        print(f"| `{f}` | {v / MIB:.0f} | {100.0 * v / grand:.2f}% | {lever[f]} |")
    print(f"| **合计** | **{grand / MIB:.0f}** | 100% | |")

    # spill-fill 共享的收益上限：四段共用最大者一份
    sf = [r["mem"]["spillFillBufferSize"] for r in rows]
    if len(sf) > 1:
        save = (sum(sf) - max(sf)) / MIB
        print(f"\n⊕ **#146 跨 context 共享 spill-fill 的收益上限** = "
              f"{sum(sf) / MIB:.0f} − {max(sf) / MIB:.0f}（最大者）= **{save:.0f} MiB**"
              f"（官方教程：共用一份，尺寸取所有 buffer 中的最大者）")

    # 图切换的收益上限：只留最大的一段
    tot = [r["total"] for r in rows]
    if len(tot) > 1:
        print(f"⊕ **图切换（只驻留一段）的公式收益上限** = "
              f"{grand / MIB:.0f} − {max(tot) / MIB:.0f}（最大段）= "
              f"**{(grand - max(tot)) / MIB:.0f} MiB**")

    meas_sum = sum(MEASURED_ION_MIB[r["tag"]]["app"] for r in rows
                   if r["tag"] in MEASURED_ION_MIB)
    if meas_sum:
        print(f"\n🔴 **公式 vs 实测**：公式合计 {grand / MIB:.0f} MiB，"
              f"实测 ION 增量合计 {meas_sum} MiB（#145，真实 app），"
              f"比值 **{meas_sum * MIB / grand:.3f}**。"
              f"⇒ 公式**低估**，差额未归因；**收益上限一律按比例换算后再报**，"
              f"不得直接把公式数字当 ION 数字用。")


if __name__ == "__main__":
    main()
