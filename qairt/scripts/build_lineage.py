#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付谱系重建 —— 从**产物自身**反推「设备上这个文件是怎么来的」

为什么需要它
------------
`D:\\ZImage_Work` 有约 198 GB、120 个 `.dlc`、29 个 context `.bin`，光 part1a 的 ONNX 就 38 个。
其中绝大多数是历史实验产物，**只有一小部分是当前交付态**，而两者混在同一批目录里。

2026-09-20 实际代价：给新写的检查工具做已知样本自检时，随手取了 08-14 的
`part2_dlcinfo.txt` —— 那是 **part2 拆分前**的转储，当前交付里**根本没有这个段**
（现在是 part2a/part2b）。⇒ 判据"通过"了，但验的是一个已经不存在的东西。

谱系其实是完整的，只是**没有被汇总到一处**：
  设备 sha256（DELIVERY_FROZEN）→ 宿主 .bin → 同目录 d.json（图名/soc/vtcm/权重共享）
  → 图名主干 = fp32 DLC 名（§24.2）→ 量化 DLC → **内嵌的 Converter/Quantizer 命令**（§二）

⇒ 本脚本把这条链走一遍，产出「当前交付态白名单」。**不在白名单里的才是历史产物**，
   删之前再按再生成本判（`DISK_INVENTORY.md` §1.2）。

🔴 三条必须知道的解释规则
-------------------------
1. **锚点只认 sha256，不认文件名、不认字节数**。实测过两个内容不同的 APK 字节数完全相同（§47.2）。
2. **内嵌命令里的参数不一定是生效的参数**。实测：当前交付的 part2a 内嵌
   `act_bitwidth=8`，但真实激活是 **16-bit** —— 因为用了 `--quantization_overrides`，
   激活 encoding 是注入的，`act_bitwidth` 停在默认值。**照字面读会误判成 A8W8。**
3. 本脚本报告的是**来源链**，不是**正确性**。链对得上 ≠ 数值对。

用法
----
    python scripts/build_lineage.py                 # 骨架（秒级）：sha256 锚定 + 图名 + DLC 定位
    python scripts/build_lineage.py --deep          # 加提取内嵌命令（慢，每个 DLC 数十秒）
    python scripts/build_lineage.py --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "docs" / "DELIVERY_FROZEN.md"
SEARCH_ROOTS = [Path(r"D:\ZImage_Work")]
QAIRT_TOOL = REPO / "scripts" / "qairt_tool.py"

FROZEN_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`\s*\|")


def read_anchors():
    """从 DELIVERY_FROZEN.md 读设备侧现态表：[(名, 字节数, sha256)]。"""
    if not FROZEN.exists():
        sys.exit(f"找不到锚点文件：{FROZEN}\n先跑 python scripts/freeze_delivery.py")
    out = []
    for line in FROZEN.read_text(encoding="utf-8").splitlines():
        m = FROZEN_ROW.match(line.strip())
        if m:
            out.append({"device_path": m.group(1), "size": int(m.group(2)), "sha256": m.group(3)})
    return out


def sha256_of(path: Path, buf=4 << 20):
    """4 MiB 缓冲 —— 64 KiB 吃不满 I/O（§47.9.2 实测）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def index_by_size(sizes):
    """先按字节数建索引（廉价），后面只对候选算 sha256。"""
    idx = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.stat().st_size in sizes:
                    idx.setdefault(p.stat().st_size, []).append(p)
            except OSError:
                continue
    return idx


def read_build_config(bin_path: Path):
    """交付件同目录的建图配置：图名 / soc / vtcm / 权重共享。"""
    info = {"graph_names": [], "soc_model": None, "dsp_arch": None,
            "vtcm_mb": None, "weight_sharing": None, "config_file": None}
    for cand in sorted(bin_path.parent.glob("*.json")):
        try:
            j = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "graphs" in j or "devices" in j:
            info["config_file"] = str(cand)
            for g in j.get("graphs", []):
                info["graph_names"].extend(g.get("graph_names", []))
                if g.get("vtcm_mb") is not None:
                    info["vtcm_mb"] = g["vtcm_mb"]
            for d in j.get("devices", []):
                info["soc_model"] = d.get("soc_model", info["soc_model"])
                info["dsp_arch"] = d.get("dsp_arch", info["dsp_arch"])
            ctx = j.get("context", {})
            if "weight_sharing_enabled" in ctx:
                info["weight_sharing"] = ctx["weight_sharing_enabled"]
    return info


def find_dlc_for_graph(graph_name: str, dlc_index):
    """图名主干 = converter 产出的 fp32 DLC 文件名主干（§24.2）。
    量化产物通常叫 <主干去掉 _fp32>_quantized.dlc，也可能就是 <主干>.dlc。"""
    stem = graph_name[:-5] if graph_name.endswith("_fp32") else graph_name
    cands = []
    for name, paths in dlc_index.items():
        if name in (f"{stem}_quantized.dlc", f"{stem}.dlc", f"{graph_name}.dlc"):
            cands.extend(paths)
    if not cands:                       # 放宽：主干前缀匹配
        for name, paths in dlc_index.items():
            if name.startswith(stem) and name.endswith(".dlc"):
                cands.extend(paths)
    return [str(c) for c in cands]


def build_dlc_index():
    idx = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.dlc"):
            idx.setdefault(p.name, []).append(p)
    return idx


def extract_commands(dlc: Path):
    """从量化 DLC 里读内嵌的 Converter / Quantizer 命令（§二：不必凭猜重建）。"""
    try:
        r = subprocess.run([sys.executable, str(QAIRT_TOOL), "snpe-dlc-info", "-i", str(dlc)],
                           capture_output=True, text=True, timeout=1800,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"error": "snpe-dlc-info 超时"}
    if r.returncode != 0:
        return {"error": f"snpe-dlc-info rc={r.returncode}: {(r.stderr or '')[:200]}"}
    out = {}
    for line in (r.stdout or "").splitlines():
        low = line.lower()
        if low.startswith("converter command:"):
            out["converter"] = line.split(":", 1)[1].strip()
        elif low.startswith("quantizer command:"):
            out["quantizer"] = line.split(":", 1)[1].strip()
    # 摘出几个决定配方的开关
    q = out.get("quantizer", "")
    keys = ("use_per_row_quantization", "use_per_channel_quantization", "weights_bitwidth",
            "act_bitwidth", "float_bitwidth", "float_fallback", "input_list",
            "keep_weights_quantized", "act_quantizer_calibration")
    out["recipe"] = {k: v for part in q.split(";")
                     if "=" in part
                     for k, v in [tuple(x.strip() for x in part.split("=", 1))]
                     if k in keys}
    return out


def main():
    ap = argparse.ArgumentParser(description="从产物自身重建当前交付态的谱系")
    ap.add_argument("--deep", action="store_true", help="提取内嵌 Converter/Quantizer 命令（慢）")
    ap.add_argument("--json", help="谱系写到这个 JSON")
    a = ap.parse_args()

    anchors = read_anchors()
    print(f"锚点：{FROZEN.name} 的设备侧现态表，共 {len(anchors)} 个文件\n")

    size_idx = index_by_size({x["size"] for x in anchors})
    dlc_idx = build_dlc_index()
    print(f"宿主扫描完成：候选文件按字节数命中 {sum(len(v) for v in size_idx.values())} 个，"
          f"DLC 索引 {sum(len(v) for v in dlc_idx.values())} 个\n")

    lineage, unmatched = [], []
    for anc in anchors:
        rec = dict(anc, host_path=None, verified=False, build=None, dlcs={})
        for cand in size_idx.get(anc["size"], []):
            if sha256_of(cand) == anc["sha256"]:     # 🔴 只认 sha256
                rec["host_path"] = str(cand)
                rec["verified"] = True
                break
        if not rec["verified"]:
            unmatched.append(rec)
            lineage.append(rec)
            continue

        hp = Path(rec["host_path"])
        if hp.suffix == ".bin":
            rec["build"] = read_build_config(hp)
            for g in rec["build"]["graph_names"]:
                rec["dlcs"][g] = find_dlc_for_graph(g, dlc_idx)
        lineage.append(rec)

    if a.deep:
        print("--deep：提取内嵌命令（每个 DLC 数十秒）…\n")
        seen = {}
        for rec in lineage:
            for g, paths in rec.get("dlcs", {}).items():
                if not paths:
                    continue
                p = paths[0]
                if p not in seen:
                    seen[p] = extract_commands(Path(p))
                rec.setdefault("commands", {})[g] = seen[p]

    # ── 报告 ──
    print("=" * 78)
    for rec in lineage:
        mark = "✅" if rec["verified"] else "❌"
        print(f"\n{mark} {rec['device_path']}  ({rec['size']:,} B)")
        if not rec["verified"]:
            print(f"   🔴 宿主上找不到 sha256 相同的文件 —— **没有回滚源**")
            continue
        print(f"   宿主: {rec['host_path']}")
        b = rec.get("build") or {}
        if b.get("graph_names"):
            ws = {True: "开", False: "关", None: "未写"}[b.get("weight_sharing")]
            print(f"   建图: soc={b['soc_model']} arch={b['dsp_arch']} "
                  f"vtcm={b['vtcm_mb']}MB 权重共享={ws}  图 {len(b['graph_names'])} 个")
            for g in b["graph_names"]:
                d = rec["dlcs"].get(g) or []
                print(f"     · {g}  ->  {d[0] if d else '🔴 找不到对应 DLC'}"
                      f"{'' if len(d) < 2 else f'  ⚠️ {len(d)} 个同名候选'}")
                cmd = (rec.get("commands") or {}).get(g)
                if cmd and cmd.get("recipe"):
                    print(f"       配方: {cmd['recipe']}")
        elif b:
            print("   ⚠️ 同目录没有建图配置 json —— 谱系断在这里")

    ok = sum(1 for r in lineage if r["verified"])
    print("\n" + "=" * 78)
    print(f"锚定成功 {ok}/{len(lineage)}")
    if unmatched:
        print(f"🔴 {len(unmatched)} 个交付件在宿主上没有同 sha256 的副本 ⇒ 没有回滚源（约束 11 第 2 条）")
    print("\n🔴 解释规则：①只认 sha256；②内嵌命令里的参数不一定生效"
          "（overrides 会覆盖，实测 act_bitwidth=8 而真实激活是 16-bit）；③链对得上 ≠ 数值对。")

    if a.json:
        Path(a.json).write_text(json.dumps(lineage, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n谱系 -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
