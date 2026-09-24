"""统一的 `unified` 张量误差度量（约束 7 的强制口径）。

任何对 part1b 输出 `unified` 的比较都必须走这里，保证口径一致、且每次都自动
报告"前 1% 元素占 ||a||^2 的比例"。

口径定义（事前定稿，不得事后修改）：
  · 全量口径  E_all  = ||b-a|| / ||a||                （仅作记录，不用于下结论）
  · 主体口径  E_bulk = ||b-a||_M / ||a||_M            M = {|a| <= p99(|a|)}
  · 主体余弦  cos_bulk = <a_M, b_M> / (||a_M||*||b_M||)
  · 能量集中度 conc = 最大 1% 元素的平方和 / ||a||^2

判读规则（CLAUDE.md 约束 7）：
  · conc > 50%  ⇒ 不得用 E_all 描述张量整体，必须用 E_bulk
  · cos_bulk < 0.3 ⇒ 该张量已与真值去相关，退出定量比较，只能标记"已毁"

用法: python metrics_unified.py <ref.raw> <test.raw> [<test2.raw> ...]
"""
import sys

import numpy as np


def load(p):
    return np.fromfile(p, dtype=np.float32)


def concentration(a):
    """前 1% 最大 |a| 元素占 ||a||^2 的比例"""
    sq = a.astype(np.float64) ** 2
    k = max(1, int(round(len(a) * 0.01)))
    top = np.partition(sq, -k)[-k:]
    return float(top.sum() / sq.sum())


def metrics(a, b):
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    out = {}
    out["E_all"] = float(np.linalg.norm(b64 - a64) / np.linalg.norm(a64))
    out["cos_all"] = float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))
    thr = np.percentile(np.abs(a64), 99.0)
    m = np.abs(a64) <= thr
    am, bm = a64[m], b64[m]
    out["p99"] = float(thr)
    out["n_bulk"] = int(m.sum())
    out["E_bulk"] = float(np.linalg.norm(bm - am) / np.linalg.norm(am))
    out["cos_bulk"] = float(am @ bm / (np.linalg.norm(am) * np.linalg.norm(bm)))
    # 系统性增益分解（HANDOVER 15.13.1 的口径，主体上做）
    k = float(am @ bm / (am @ am))
    out["slope_bulk"] = k
    out["resid_bulk"] = float(np.linalg.norm(bm - k * am) / np.linalg.norm(am))
    out["std_ref_bulk"] = float(am.std())
    out["std_test_bulk"] = float(bm.std())
    return out


def main():
    ref_p, tests = sys.argv[1], sys.argv[2:]
    a = load(ref_p)
    conc = concentration(a)
    print(f"参考: {ref_p}  n={a.size}")
    print(f"能量集中度: 前 1% 元素占 ||a||^2 的 {conc*100:.2f}%"
          f"  ⇒ {'必须用主体口径' if conc > 0.5 else '两口径均可'}")
    print()
    hdr = (f"{'张量':<26} {'E_all':>9} {'E_bulk':>9} {'cos_bulk':>9} "
           f"{'slope':>8} {'resid':>9} {'std_test':>9}")
    print(hdr)
    print("-" * len(hdr))
    for t in tests:
        b = load(t)
        if b.size != a.size:
            print(f"{t}: 尺寸不符 {b.size} != {a.size}  ← 拒绝比较")
            continue
        m = metrics(a, b)
        name = t.replace("\\", "/").split("/")[-2] + "/" + t.replace("\\", "/").split("/")[-1]
        print(f"{name[:26]:<26} {m['E_all']*100:8.4f}% {m['E_bulk']*100:8.4f}% "
              f"{m['cos_bulk']:9.4f} {m['slope_bulk']:8.4f} {m['resid_bulk']*100:8.4f}% "
              f"{m['std_test_bulk']:9.4f}")
    a64 = a.astype(np.float64)
    thr = np.percentile(np.abs(a64), 99.0)
    print(f"\n参考主体 std = {a64[np.abs(a64) <= thr].std():.4f}  p99(|a|) = {thr:.4f}")


if __name__ == "__main__":
    main()
