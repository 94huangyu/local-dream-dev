"""#66 代价门预筛：`--param_quantizer_calibration mse/sqnr` 对权重还有没有空间？

方案 `scripts/EXP_PLAN_PARAMCAL.md`（执行前定稿，判据不得改）。全程宿主，**不占设备**。

问的是：**在 per-row min-max 已经拿到的基础上**，换 mse/sqnr 校准还有没有额外空间？
（不是问 mse vs per-tensor —— per-row 已经在用了，那个比较没有决策意义。）

A 组 = 当前配置：per-row min-max，`s = (max_row - min_row)/255`
B 组 = mse **可达上界**：在裁剪比例 α 上网格搜索，逐行取使 ||W - Q(W)||^2 最小的那组
       （sqnr 与 mse 在纯表示误差上同解，故只报一次）

⚠️ B 组是**上界**：SDK 的 mse 实现细节未知，真实收益 <= 本组。
   ⇒ **上界都没有收益 ⇒ 真实实现更不可能有**，这是一个**单向否决**门。
   ⇒ 反向不成立：上界有收益 **不等于** HTP 上有收益（#52：表示误差不是 HTP 误差的预测量）。

用法: python paramcal_costgate.py [张量数上限] [part1b|part2]
"""
import sys
import time

import numpy as np
import onnx

import map_opid as M
from perrow_feasibility import PARTS, fc_weights
from repr_cost_sym import read_external

# 裁剪比例网格（1.0 = 不裁剪 = min-max）。**近 1 处必须加密**：
# 已验证的已知样本——高斯 N(0,1) 的 MSE 最优 α = 0.976（收益 2.5%），
# 粗网格会整个跳过这个区间，等于**没给候选公平机会**。
ALPHAS = np.unique(np.concatenate([np.linspace(0.40, 0.90, 11),
                                   np.linspace(0.90, 1.00, 41)]))


def perrow_minmax(w):
    """A 组：逐行 min-max（轴 0 为行，与 DLC 里 axis-quant 实测的轴一致）。"""
    mn = w.min(axis=1, keepdims=True)
    mx = w.max(axis=1, keepdims=True)
    s = np.maximum((mx - mn) / 255.0, 1e-12)
    off = np.rint(mn / s)
    q = np.clip(np.rint(w / s) - off, 0, 255)
    return s * (q + off), s, mn, mx


def perrow_mse(w, mn, mx):
    """B 组：逐行在 α 网格上取 MSE 最优（可达上界）。返回反量化结果与被裁剪掩码。"""
    best_err = np.full((w.shape[0], 1), np.inf)
    best_wd = np.zeros_like(w)
    best_a = np.ones((w.shape[0], 1))
    for a in ALPHAS:
        lo, hi = mn * a, mx * a
        s = np.maximum((hi - lo) / 255.0, 1e-12)
        off = np.rint(lo / s)
        q = np.clip(np.rint(w / s) - off, 0, 255)
        wd = s * (q + off)
        err = ((wd - w) ** 2).sum(axis=1, keepdims=True)
        m = err < best_err
        best_err = np.where(m, err, best_err)
        best_wd = np.where(m, wd, best_wd)
        best_a = np.where(m, a, best_a)
    return best_wd, best_a


def rel2(a, b):
    """全量与主体（|a| <= p99）两个口径的相对 L2（约束 7）。"""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    thr = np.percentile(np.abs(a), 99.0)
    m = np.abs(a) <= thr
    return (np.linalg.norm(b - a) / np.linalg.norm(a),
            np.linalg.norm(b[m] - a[m]) / np.linalg.norm(a[m]))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    part = sys.argv[2] if len(sys.argv) > 2 else "part2"
    onnx_path, dlcinfo = PARTS[part]
    print(f"=== #66 代价门预筛 / 分段 {part} ===")
    print(f"网格 α ∈ [{ALPHAS[0]:.2f}, {ALPHAS[-1]:.2f}]，{len(ALPHAS)} 档\n", flush=True)

    base = fc_weights(dlcinfo)
    model = onnx.load(onnx_path, load_external_data=False)
    inits = {i.name: i for i in model.graph.initializer}
    import os
    base_dir = os.path.dirname(onnx_path)

    A_all, A_bulk, B_all, B_bulk = [], [], [], []
    clip_frac, clip_energy, rng_keep = [], [], []
    t0 = time.time()
    n = 0
    for name in sorted(base):
        if n >= limit:
            break
        _, dims = base[name]
        oname = name[:-len("_permute")] if name.endswith("_permute") else name
        if oname not in inits:
            continue
        w = read_external(inits[oname], base_dir).reshape(dims).astype(np.float32)
        wa, _, mn, mx = perrow_minmax(w)
        wb, alpha = perrow_mse(w, mn, mx)

        aa, ab = rel2(w, wa)
        ba, bb = rel2(w, wb)
        A_all.append(aa); A_bulk.append(ab); B_all.append(ba); B_bulk.append(bb)

        lo, hi = mn * alpha, mx * alpha
        clipped = (w < lo) | (w > hi)
        clip_frac.append(clipped.mean())
        e = (w.astype(np.float64) ** 2)
        clip_energy.append(e[clipped].sum() / e.sum())
        rng_keep.append(float(np.median(alpha)))

        n += 1
        if n <= 6:
            print(f"  {name[:30]:30} A(主体) {ab*100:6.3f}%  B(主体) {bb*100:6.3f}%  "
                  f"α中位 {np.median(alpha):.3f}  裁剪 {clipped.mean()*100:.4f}%", flush=True)

    A_all, A_bulk = np.array(A_all)*100, np.array(A_bulk)*100
    B_all, B_bulk = np.array(B_all)*100, np.array(B_bulk)*100
    eA, eB = float(np.median(A_bulk)), float(np.median(B_bulk))

    print(f"\n张量数 {n}   耗时 {time.time()-t0:.0f}s")
    print("=" * 74)
    print(f"{'组':<28}{'全量中位':>12}{'主体中位':>12}{'主体最大':>12}")
    print("-" * 74)
    print(f"{'A per-row min-max (当前)':<28}{np.median(A_all):>11.4f}%"
          f"{eA:>11.4f}%{A_bulk.max():>11.4f}%")
    print(f"{'B per-row mse (可达上界)':<28}{np.median(B_all):>11.4f}%"
          f"{eB:>11.4f}%{B_bulk.max():>11.4f}%")

    print("\n" + "=" * 74)
    print("附带必报项（方案第 3 节，事前要求）")
    print(f"  被裁剪元素占比       中位 {np.median(clip_frac)*100:.4f}%")
    print(f"  被裁剪元素占 ||W||^2 中位 {np.median(clip_energy)*100:.4f}%")
    print(f"  动态范围保留比 α     中位 {np.median(rng_keep):.4f}")

    print("\n" + "=" * 74)
    print("判据（事前定稿，不得修改）")
    print(f"  E_A = {eA:.4f}%   E_B = {eB:.4f}%   E_B/E_A = {eB/eA:.4f}")
    if eB >= 0.95 * eA:
        print("  ⇒ **P-否决**：mse/sqnr 在【可达上界】上都拿不到 5% 以上改善")
        print("     ⇒ 该候选出局，**不许上机**，从队列删除并写明理由")
    elif eB >= 0.80 * eA:
        print("  ⇒ **P-无价值**：改善 <20%。对照 per-row 自己是 0.2781 倍（改善 72%）")
        print("     ⇒ 量级差一个数量级，不值得 85 分钟，降为最低优先级")
    else:
        print("  ⇒ **P-值得**：有实质空间，可进入正式实验")
        print("     ⇒ 届时另写方案，并必须先过 G0-cost 全量报数")

    print("\n⚠️ 有效性边界（方案第 4 节，必须与结论一起写）：")
    print("   表示误差**不是** HTP 误差的预测量（#52：同一改动 CPU Δcos −0.0014、"
          "HTP −0.0526，差 37.6 倍）。")
    print("   ⇒ 本实验**只能否决，不能肯定**：E_B < E_A 只说明表示层面有空间，"
          "**不得**据此宣称 HTP 上会有收益。")


if __name__ == "__main__":
    main()
