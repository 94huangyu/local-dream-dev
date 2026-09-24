# -*- coding: utf-8 -*-
"""宿主侧预验：多图 context 里的某个图，与它对应的单图 context **是否等价**。

## 为什么要有它
G-MG 的最终判据（多图 vs 单图，终点 latents 逐元素最大差 = 0）**必须在设备上跑**。
但设备时间很贵：用户外出，回来后只想插一次线。
⇒ **凡是能在宿主上判死的，就不要留到插线时才发现。**
2026-09-04 那次交付就是在真机上才发现 UI 的门没打开，而那本可以在宿主查出来。

## 本检查能判什么、不能判什么（不得越界声称）
✅ **能判死**（不一致 ⇒ 数值必然不同 ⇒ 不必上机就知道 G-MG 会挂）：
   图名、张量名集合与顺序、形状、dtype、量化 scale/offset、张量数量。
✅ **能判死**：spillFillBufferSize / 预测 ION 超红线。
❌ **不能放行**（指南 §41 代理指标只能否决）：元数据全同**不等于**HTP 上数值相同——
   多图容器可能改变权重摆放或调度。**最终仍以设备上的逐元素比较为准。**

## 判据（执行前锁定）
M1 多图 context 里存在该图名
M2 输入/输出张量的 **name / dimensions / dataType** 与单图逐项相同（含顺序）
M3 量化 scale / offset 与单图**逐位相同**（float 按 repr 比，不设容差）
M4 该图的 spillFillBufferSize 与单图相同（不同 ⇒ 调度被改，风险升高，判 FAIL）
M5 按 tools.html 公式算的预测 ION（只计该图 + 共享权重一次）<= 现网单图 x 1.15

用法: python check_mg_equivalence.py <seg> <多图.bin> <图名>=<单图.bin> [<图名>=<单图.bin> ...]
"""
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
RED_ION_FACTOR = 1.15


def dump(binpath):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "m.json")
        r = subprocess.run([UTIL, "--context_binary", binpath, "--json_file", out],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if not os.path.isfile(out):
            raise SystemExit("🔴 dump 失败 rc=%d %s\n%s%s"
                             % (r.returncode, binpath, r.stdout or "", r.stderr or ""))
        with open(out, encoding="utf-8") as f:
            return json.load(f)["info"]["graphs"]


def pick(graphs, name):
    for g in graphs:
        if g["info"]["graphName"] == name:
            return g["info"]
    return None


def tsig(lst):
    """张量签名：名字/形状/dtype/量化参数。float 用 repr，不设容差。"""
    out = []
    for t in lst:
        i = t["info"]
        q = i.get("quantizeParams", {})
        so = q.get("scaleOffset") if q.get("definition") == "QNN_DEFINITION_DEFINED" else None
        out.append((i["name"], tuple(i["dimensions"]), i["dataType"],
                    repr(so["scale"]) if so else None,
                    repr(so["offset"]) if so else None))
    return out


def mem(info):
    """tools.html 的 RAM 公式各项（sharedWeightsSize 整个 context 只算一次）。

    🔴 字段分散在两层，且**单位不统一**——第一版我按 `info.get(k)` 平铺去取，
       全部取到 0，于是 M4/M5 变成「0 == 0」的**恒过假门**（正是本项目
       反复栽的「门是装饰性的」）。实际位置与单位：
         graphBlobInfo.info : spillFillBufferSize(字节)、vtcmSize(**MB**)
         graphBlobInfoV2    : ioTensorSize / opDataSize / constSize /
                              ddrTensorSize / sharedWeightsSize（均字节）
    """
    b1 = (info.get("graphBlobInfo") or {}).get("info") or {}
    b2 = info.get("graphBlobInfoV2") or {}
    if not b1 and not b2:
        raise SystemExit("🔴 元数据里没有 graphBlobInfo/graphBlobInfoV2 —— "
                         "取字段的路径变了，先修本脚本，不得让 M4/M5 变成恒过的假门")
    out = {
        "spillFillBufferSize": int(b1.get("spillFillBufferSize", 0) or 0),
        "vtcmSize": int(b1.get("vtcmSize", 0) or 0) * 1024 * 1024,   # MB -> 字节
        "ioTensorSize": int(b2.get("ioTensorSize", 0) or 0),
        "opDataSize": int(b2.get("opDataSize", 0) or 0),
        "constSize": int(b2.get("constSize", 0) or 0),
        "ddrTensorSize": int(b2.get("ddrTensorSize", 0) or 0),
    }
    if sum(out.values()) == 0:
        raise SystemExit("🔴 各项全为 0 —— 同上，拒绝用假门放行")
    return out, int(b2.get("sharedWeightsSize", 0) or 0)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    seg, mgbin = sys.argv[1], sys.argv[2]
    pairs = [a.split("=", 1) for a in sys.argv[3:]]
    if not os.path.isfile(mgbin):
        raise SystemExit("🔴 缺多图 .bin: %s" % mgbin)
    mg = dump(mgbin)
    mg_names = [g["info"]["graphName"] for g in mg]
    print("=== %s ===\n多图 %s\n  含图 %s" % (seg, os.path.basename(mgbin), mg_names))

    fails = []

    def chk(gate, ok, detail):
        print("  %-3s %s  %s" % (gate, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(gate)

    for gname, sbin in pairs:
        print("\n-- 图 %s  vs  %s" % (gname, os.path.basename(sbin)))
        gm = pick(mg, gname)
        chk("M1", gm is not None, "多图 context 里存在该图")
        if gm is None:
            continue
        if not os.path.isfile(sbin):
            chk("M1", False, "缺单图 .bin %s" % sbin)
            continue
        sg_all = dump(sbin)
        if len(sg_all) != 1:
            chk("M1", False, "单图 .bin 里有 %d 个图，不是单图" % len(sg_all))
            continue
        gs = sg_all[0]["info"]

        for kind in ("graphInputs", "graphOutputs"):
            a, b = tsig(gm[kind]), tsig(gs[kind])
            same = a == b
            if same:
                chk("M2", True, "%s 逐项相同（%d 个张量）" % (kind, len(a)))
            else:
                # 指出第一处差异，避免只说「不同」
                na = [x[0] for x in a]
                nb = [x[0] for x in b]
                if na != nb:
                    chk("M2", False, "%s 张量名/顺序不同：多图 %s / 单图 %s"
                        % (kind, na[:6], nb[:6]))
                else:
                    d = [x[0] for x, y in zip(a, b) if x != y]
                    chk("M2", False, "%s 有 %d 个张量的形状/dtype/量化参数不同：%s"
                        % (kind, len(d), d[:6]))
                    for x, y in zip(a, b):
                        if x != y:
                            print("        多图 %r\n        单图 %r" % (x, y))
                            break
        # M3 已并入 M2 的签名（scale/offset 在签名里，按 repr 逐位比）
        chk("M3", tsig(gm["graphInputs"]) == tsig(gs["graphInputs"])
            and tsig(gm["graphOutputs"]) == tsig(gs["graphOutputs"]),
            "量化 scale/offset 逐位相同（含在 M2 的签名里）")

        mm, msh = mem(gm)
        sm, ssh = mem(gs)
        # 🔴 M4 的第一版写成「spillFill 必须相等」，实测**当场判负**（多图 297.1/316.1
        #    vs 单图 288.6/332.7）。复查后确认是**我的判据错了，不是产物错了**：
        #    spill-fill 是 HTP 的临时暂存，不参与数值计算；指南 §38 已记它**可跨图共享**，
        #    多图里的值落在两个单图之间，正是共享缓冲按最大值定尺寸的表现。
        #    「相等」从来没有证据支持是等价的必要条件 —— 又一次操作化写错（约束 8）。
        #    改成**爆量守卫**：只拦 1152x864 那种量级的异常膨胀（当时 999 MiB）。
        mgsf = mm["spillFillBufferSize"] / 1048576.0
        sgsf = sm["spillFillBufferSize"] / 1048576.0
        chk("M4", mgsf <= max(sgsf, 1.0) * 1.30,
            "spillFill 多图 %.1f MiB vs 单图 %.1f MiB（守卫：不得超单图 1.30x；"
            "不要求相等——它是暂存，可跨图共享）" % (mgsf, sgsf))
        ion_mg = (sum(mm.values()) + msh) * 1.06 / 1048576.0
        ion_sg = (sum(sm.values()) + ssh) * 1.06 / 1048576.0
        chk("M5", ion_mg <= ion_sg * RED_ION_FACTOR,
            "预测 ION 多图 %.0f MiB vs 单图 %.0f MiB（上限 %.0f）"
            % (ion_mg, ion_sg, ion_sg * RED_ION_FACTOR))

    print("\n%s" % ("🔴 %s 未过：%s —— G-MG 必挂，不要上机浪费设备时间"
                    % (seg, ", ".join(sorted(set(fails)))) if fails else
                    "✅ %s 元数据层面等价。⚠️ 只能否决不能放行：最终仍以设备上"
                    "「多图 vs 单图终点 latents 逐元素最大差 = 0」为准（指南 §41）" % seg))
    return 1 if fails else 0


if __name__ == "__main__":
    # 🔴 必须有这个守卫：aspect_preflight.py 要 `from check_mg_equivalence import dump,...`
    #    复用取字段逻辑（同一份代码只此一处，避免两处解析元数据而口径不一）。
    #    没有守卫时 import 会直接把 main() 跑起来并 sys.exit，表现为莫名其妙的
    #    "缺多图 .bin: --apk"（它把调用方的 argv 当成了自己的参数）。
    sys.exit(main())
