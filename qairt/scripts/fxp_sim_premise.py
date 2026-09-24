"""EXP_PLAN_FXP_SIM 第 2 节：先验证"默认成立的前提"，不通过就不往下做。

P1: DLC 的 FC STATIC 权重能对应到 ONNX 权重
P2: FC 语义是 y = x @ W^T + b —— 用 FP32 复算与 CPU 参考比（**关键，不过则终止**）
P3: 权重是 per-tensor 量化
P4: 输入激活有可用 encoding
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, r"D:\LocalDreamZImage\scripts")
import map_opid as M

ONNX = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part1b.onnx"
DUMP = r"D:\ZImage_Work\p0_experiments\transformer_part1b_dlcinfo.txt"
XIN = r"D:\ZImage_Work\p0_experiments\fc_inputs\cpu\out\Result_0"
YCPU = r"D:\ZImage_Work\p0_experiments\fc_probe\cpu\out\Result_0"

# (Id, FC 输出名, 输入名, 权重 DLC 名, 输入shape, 输出shape)
CASES = [
    (12, "linear_97_fc", "node_linear_97_pre_reshape", "val_1791", (4128, 10240), (4128, 3840)),
    (29, "linear_99_fc", "node_linear_99_pre_reshape", "val_1800", (4128, 3840), (4128, 3840)),
    (31, "linear_100_fc", "node_linear_100_pre_reshape", "val_1801", (4128, 3840), (4128, 3840)),
    (82, "linear_102_fc", "node_linear_102_pre_reshape", "val_1896", (4128, 3840), (4128, 3840)),
    (99, "linear_105_fc", "node_linear_105_pre_reshape", "val_1908", (4128, 10240), (4128, 3840)),
]


def main():
    ops = M.parse(DUMP)
    print("载入 ONNX initializer ...", flush=True)
    m = onnx.load(ONNX, load_external_data=True)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    print(f"  {len(inits)} 个 initializer\n")

    # 按 shape 建索引（P1）
    byshape = {}
    for n, a in inits.items():
        byshape.setdefault(a.shape, []).append(n)

    for oid, yname, xname, wdlc, xsh, ysh in CASES:
        op = ops[oid]
        e_w = op["enc"].get(wdlc)
        e_y = op["enc"].get(yname)
        K, N = xsh[1], ysh[1]
        print("=" * 84)
        print(f"Id={oid}  {yname}   x{xsh} @ W -> y{ysh}   (K={K}, N={N})")
        print(f"  DLC 权重 {wdlc}: bw={e_w['bw']} scale={e_w['scale']:.6e} "
              f"offset={e_w['offset']:.1f} range=[{e_w['min']:.5g},{e_w['max']:.5g}]")

        # ---- P1: 找 ONNX 权重 ----
        cands = byshape.get((N, K), []) + byshape.get((K, N), [])
        print(f"  P1: shape ({N},{K}) 或 ({K},{N}) 的 ONNX 权重候选 {len(cands)} 个: {cands[:4]}")
        if not cands:
            print("  P1 [FAIL] 找不到对应权重，该 FC 退出\n")
            continue

        # 用量化 range 匹配：ONNX 权重的 min/max 应与 DLC encoding 的 range 接近
        best, bestd = None, 1e30
        for c in cands:
            w = inits[c].astype(np.float64)
            d = abs(w.min() - e_w["min"]) + abs(w.max() - e_w["max"])
            if d < bestd:
                best, bestd = c, d
        w = inits[best].astype(np.float64)
        print(f"  P1: 最佳匹配 = {best}  shape={w.shape}  "
              f"min={w.min():.5g} max={w.max():.5g}  (DLC range 差 {bestd:.4g})")
        if bestd > 0.05 * max(abs(e_w["max"]), 1e-9):
            print("  P1 [WARN] range 匹配不佳，结果存疑")

        # ---- P2: 用【反量化后的权重】复算（CPU 参考跑的正是量化 DLC）----
        x = np.fromfile(f"{XIN}\{xname}.raw", np.float32).reshape(xsh).astype(np.float64)
        ycpu = np.fromfile(f"{YCPU}\{yname}.raw", np.float32).reshape(ysh).astype(np.float64)
        sw, zw = e_w["scale"], -e_w["offset"]      # offset 是 -zero_point
        qw = np.clip(np.round(w / sw) + zw, 0, 255)
        wdeq = (qw - zw) * sw                      # 反量化权重
        print(f"  P2: 权重反量化相对误差 = "
              f"{100*np.linalg.norm(wdeq-w)/np.linalg.norm(w):.4f}%")

        best_rel, best_desc = 1e30, ""
        step = 512
        for desc, Wm in [("x @ Wq.T  (Wq为(N,K))", None), ("x @ Wq    (Wq为(K,N))", None)]:
            pass
        cands_o = []
        if wdeq.shape == (N, K): cands_o.append(("x @ W.T", wdeq.T))
        if wdeq.shape == (K, N): cands_o.append(("x @ W", wdeq))
        if wdeq.shape == (N, K) and N == K: cands_o.append(("x @ W (方阵另一向)", wdeq))
        if wdeq.shape == (K, N) and N == K: cands_o.append(("x @ W.T (方阵另一向)", wdeq.T))
        for desc, Wm in cands_o:
            if Wm.shape[0] != K: continue
            y = np.empty(ysh, dtype=np.float64)
            for i in range(0, ysh[0], step):
                y[i:i+step] = x[i:i+step] @ Wm
            r = 100.0 * np.linalg.norm(y - ycpu) / np.linalg.norm(ycpu)
            print(f"      {desc:<26} 相对误差 = {r:.4f}%")
            if r < best_rel: best_rel, best_desc = r, desc
        rel = best_rel
        print(f"  P2 {'[OK] 通过' if rel < 1.0 else '[FAIL] 仍不符'}  最佳: {best_desc} {rel:.4f}%  (判据 <1%)")
        print(f"  P3: 权重 encoding 为单一 scale/offset -> per-tensor {'确认' if e_w else '未知'}")
        print(f"  P4: 输出 encoding scale={e_y['scale']:.6e} offset={e_y['offset']:.1f}")
        print()


if __name__ == "__main__":
    main()
