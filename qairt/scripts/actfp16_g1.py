"""EXP_PLAN_ACTFP16 G1 注入门：读回产物数据类型（#96：读元数据，不看命令行）。

期望：权重仍 sFxp_8 / per-row；6 个溢出张量仍 uFxp_16；其余 16-bit 激活变 Float_16。
"""
import re, io, os, sys, subprocess
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "actfp16")
DLC = os.path.join(W, "part1b_actfp16_quantized.dlc")
CSV = os.path.join(W, "actfp16_enc.csv")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
OVERFLOW = ["linear_%d" % i for i in (105, 121, 129, 137, 145, 153)]
TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[[^\]]*\]; "
                r"tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+):")


def main():
    if not os.path.exists(CSV):
        print("[dump] snpe-dlc-info -d -s ...", flush=True)
        subprocess.run([sys.executable, TOOL, "snpe-dlc-info", "-i", DLC, "-d", "-s", CSV],
                       capture_output=True, text=True, timeout=7200)
    kind, nch = {}, {}
    with io.open(CSV, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), (m.group(2), m.group(3)))
            for m in CH.finditer(line):
                nch[m.group(1)] = nch.get(m.group(1), 0) + 1
    static = {k: v for k, v in kind.items() if v[1] == "STATIC"}
    act = {k: v for k, v in kind.items() if v[1] != "STATIC"}
    print("张量总数 %d  (STATIC %d / 非 STATIC %d)" % (len(kind), len(static), len(act)))
    print("\n[权重] STATIC 数据类型分布:", dict(Counter(v[0] for v in static.values())))
    print("       per-channel encoding 张量数 %d，scale 合计 %d（部署 per-row 为 66 / 517120）"
          % (len(nch), sum(nch.values())))
    print("\n[激活] 非 STATIC 数据类型分布:", dict(Counter(v[0] for v in act.values())))
    print("\n[6 个溢出张量] 期望仍为定点 uFxp_16:")
    ok_ov = True
    for t in OVERFLOW:
        for n in (t, t + "_fc"):
            if n in kind:
                good = kind[n][0].startswith("uFxp") or kind[n][0].startswith("sFxp")
                ok_ov &= good
                print("   %-18s %-10s %s" % (n, kind[n][0], "OK" if good else "FAIL 已变浮点"))
    nfp16 = sum(1 for v in act.values() if v[0] == "Float_16")
    print("\n=== G1 注入门 ===")
    print("  Float_16 激活张量 %d 个（期望约 641）" % nfp16)
    print("  权重 sFxp_8 %d 个（部署为 59）" % sum(1 for v in static.values() if v[0] == "sFxp_8"))
    print("  6 个溢出张量保持定点: %s" % ("PASS" if ok_ov else "FAIL"))


if __name__ == "__main__":
    main()
