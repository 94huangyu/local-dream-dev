"""台账数字溯源：MAINLINE 里引用的实测数字，必须能在盘上的产物里重新找到。

存在理由（2026-08-24 事故）：我改坏了台账表格结构，随后**凭记忆重打**把内容"补回去"。
用户的质疑是对的 —— **重打无法自证**。两条能自证的路，本脚本管第二条：
  ① 原文块：从快照按**字节切片**拷贝（见当轮 restore2.py），不由人产生一个字；
  ② 数字：**从原始产物重新导出**，与台账里写的逐个比对 —— 就是本脚本。

用法: python ledger_provenance.py [--strict]

⚠️ 本脚本只覆盖"有盘上产物可查"的数字。没有产物的数字**不在本脚本的保证范围内**，
   不得因为本脚本 PASS 就认为台账全部数字都已溯源。
"""
import io
import os
import re
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MAIN = os.path.join(ROOT, "MAINLINE.md")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")

# (台账里写的值, 说明, 取数函数)  —— 取数函数返回 None 表示"产物还不在盘上"
CLAIMS = []


def amax_of(seg, tensor, field="amax"):
    """从 <seg>_amax.json / <seg>_amax_only.json 里取某张量的实测值。"""
    for fn in ("%s_amax.json" % seg, "%s_amax_only.json" % seg):
        p = os.path.join(W, fn)
        if not os.path.exists(p):
            continue
        t = json.load(open(p, encoding="utf-8"))["tensors"]
        if tensor in t and t[tensor].get(field) is not None:
            return t[tensor][field], fn
    return None, None


def f16_count(csv):
    p = os.path.join(W, csv)
    if not os.path.exists(p):
        return None, None
    # 🔴 必须**按张量名去重**：snpe-dlc-info 的转储里同一个张量会出现两次
    #    （2026-08-24 实测：按出现次数数得 146/176，正好是去重后 73/88 的两倍）。
    #    台账写的是"张量个数"，所以这里必须用集合。
    seen = set()
    pat = re.compile(r"([A-Za-z0-9_./-]+) \(data type: Float_16; tensor dimension: "
                     r"\[[^\]]*\]; tensor type: ([A-Z]+)\)")
    with io.open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in pat.finditer(line):
                if m.group(2) != "STATIC":
                    seen.add(m.group(1))
    return len(seen), csv


def main():
    rows = {}
    for ln in io.open(MAIN, encoding="utf-8").read().split("\n"):
        m = re.match(r"^\| (\d+) \|", ln)
        if m:
            rows[m.group(1)] = ln
    txt = "\n".join(rows.values())

    checks, miss, bad = [], 0, 0

    # A. 张量实测量程（#117 / #114 引用）
    for seg, ten, claim in [("part1b", "linear_129_fc", 121291.7),
                            ("part1b", "linear_145_fc", 110337.2),
                            ("part1b", "linear_137_fc", 94472.5),
                            ("part1b", "linear_113_fc", 47719.5),
                            ("part1b", "linear_97_fc", 26593.9)]:
        got, src = amax_of(seg, ten)
        checks.append(("|a|max %s" % ten, claim, got, src))

    # B. 主体误差（#114 ③ 的五个张量）
    for ten, claim in [("mul_306", 17.94), ("mul_356", 29.53), ("mul_406", 84.71),
                       ("mul_431", 79.91), ("mul_456", 97.97)]:
        got, src = amax_of("part1b", ten, "e_ufxp")
        checks.append(("主体 uFxp %s" % ten, claim, got, src))

    # C. 浮点张量计数（#117 的 73 / 88）
    for csv, claim, label in [("part1b_best26_enc.csv", 73, "part1b 白名单2 Float_16"),
                              ("part1b_fp16_now.csv", 88, "part1b 白名单5 Float_16")]:
        got, src = f16_count(csv)
        checks.append((label, claim, got, src))

    print("台账数字溯源：逐个从盘上产物重新导出")
    print("%-30s %14s %14s  %s" % ("台账里写的", "台账值", "重新导出", "产物"))
    print("-" * 92)
    for label, claim, got, src in checks:
        if got is None:
            print("%-30s %14.4g %14s  %s" % (label, claim, "产物不在盘上", "—"))
            miss += 1
            continue
        ok = abs(got - claim) <= max(abs(claim) * 0.005, 0.01)
        if not ok:
            bad += 1
        print("%-30s %14.4g %14.4g  %s %s"
              % (label, claim, got, src, "" if ok else "<<< 对不上"))

    print("")
    if bad:
        print("🔴 FAIL：%d 个数字与产物对不上 —— 台账里有**无法溯源或写错**的数" % bad)
    elif miss:
        print("⚠️  %d 个数字的产物还不在盘上（跑完再查），其余全部对上" % miss)
    else:
        print("✅ 全部对上")
    print("⚠️ 本脚本只覆盖有产物可查的数字，PASS 不等于台账全部数字已溯源")
    return 1 if bad else 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if "--strict" in sys.argv else 0)
