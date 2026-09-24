"""G1 注入门：读回产物数据类型，验白名单落 Float_16、其余仍 uFxp_16、权重仍 sFxp_8。"""
import os, re, io, sys, json, subprocess
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
DLC, CSV = os.path.join(W, "part2a_fp16_quantized.dlc"), os.path.join(W, "p2a_fp16_enc.csv")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[[^\]]*\]; "
                r"tensor type: ([A-Z]+)\)")
if not os.path.exists(CSV):
    subprocess.run([sys.executable, TOOL, "snpe-dlc-info", "-i", DLC, "-d", "-s", CSV],
                   capture_output=True, text=True, timeout=7200)
kind = {}
with io.open(CSV, encoding="utf-8", errors="replace") as f:
    for line in f:
        for m in TT.finditer(line):
            kind.setdefault(m.group(1), (m.group(2), m.group(3)))
stats = json.load(open(os.path.join(W, "risky_stats.json"), encoding="utf-8"))
# 与生成器保持同一规则（含第 4 条：排除残差流张量）
WHITE = sorted(k for k, v in stats.items()
               if v["levels"] < 10 and v["fp16ok"] and not k.startswith("add_"))
act = {k: v[0] for k, v in kind.items() if v[1] != "STATIC"}
st = {k: v[0] for k, v in kind.items() if v[1] == "STATIC"}
print("张量总数 %d（STATIC %d / 激活 %d）" % (len(kind), len(st), len(act)))
print("激活类型分布:", dict(Counter(act.values())))
print("权重类型分布:", dict(Counter(st.values())))
print("\n[G1] 白名单张量应为 Float_16：")
ok = True
for t in WHITE:
    d = act.get(t, "(不存在)")
    good = d == "Float_16"
    ok &= good
    print("   %-12s %-10s %s" % (t, d, "✅" if good else "🔴"))
nf = sum(1 for v in act.values() if v == "Float_16")
print("\n  Float_16 激活总数 %d（期望接近 %d + 少量被带动的）" % (nf, len(WHITE)))
print("  权重 sFxp_8 %d 个" % sum(1 for v in st.values() if v == "sFxp_8"))
print("  G1 = %s" % ("PASS" if ok else "FAIL"))
