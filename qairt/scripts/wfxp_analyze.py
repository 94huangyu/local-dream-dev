"""EXP_PLAN_WFXP_ACTFP 阶段 2 判据落档。

判据逐字取自方案 §四（**事前锁定**）：
  E_wfxp <  0.5 * E_perrow            -> 🟢 显著更好
  0.5*E_perrow <= E_wfxp < E_perrow   -> 🟡 有改善但不显著
  E_wfxp >= E_perrow                  -> ❌ 不更好，关闭该形态
并行硬约束：t_wfxp > 3 * t_perrow -> 🔴 产品级不可行（依据 #16）

真值 = 用**原始 fp32 权重**做的精确 float64 matmul，三臂共用。
度量按约束 7：全量相对 L2 + 主体相对 L2(|a|<=p99) + 主体余弦 + 前 1% 能量占比。
"""
import os, sys, json
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
W_FP32 = os.path.join(P0B, "val_1800_fp32.npy")
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
M, K, N = 4128, 3840, 3840


def metrics(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    full = 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    bulk = 100.0 * np.linalg.norm(y - x) / np.linalg.norm(x)
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    e = np.sort(a * a)[::-1]
    top1 = 100.0 * e[:max(1, e.size // 100)].sum() / e.sum()
    return full, bulk, cos, top1


def main():
    dev = json.load(open(os.path.join(P0B, "wfxp", "device.json"), encoding="utf-8"))
    print("[真值] 精确 float64 matmul（原始 fp32 权重）...", flush=True)
    X = np.fromfile(X_RAW, np.float32).reshape(M, K)
    W = np.load(W_FP32)
    Y = np.empty((M, N), np.float64)
    for i in range(0, M, 512):                      # 分块，控内存（宿主 25.5 GB，被压死过）
        Y[i:i+512] = X[i:i+512].astype(np.float64) @ W.astype(np.float64)
    print("      truth std=%.6f" % Y.std())

    rows = {}
    for tag in ("perrow", "wfxp"):
        d = dev.get(tag, {})
        if "local" not in d:
            print("  %-7s ❌ 无产物: %s" % (tag, d.get("error", "?"))); continue
        dt = np.float32 if d["dtype"] == "float32" else np.float16
        a = np.fromfile(d["local"], dt).reshape(M, N)
        full, bulk, cos, top1 = metrics(Y, a)
        rows[tag] = dict(full=full, bulk=bulk, cos=cos, top1=top1, wall=d["wall"], dtype=d["dtype"])
        print("  %-7s dtype=%-8s 全量 %8.4f%%  主体 %8.4f%%  主体余弦 %.6f  wall %.1fs"
              % (tag, d["dtype"], full, bulk, cos, d["wall"]))
    if len(rows) < 2:
        print("\n两臂未齐，不落判据"); return
    print("\n[约束 7] 真值前 1%% 元素占 ||a||^2 = %.2f%%  ⇒ %s"
          % (rows["perrow"]["top1"], "全量口径合法" if rows["perrow"]["top1"] <= 50 else "全量口径失效，以主体为准"))

    ew, ep = rows["wfxp"]["bulk"], rows["perrow"]["bulk"]
    tw, tp = rows["wfxp"]["wall"], rows["perrow"]["wall"]
    print("\n=== 判据（方案 §四，事前锁定）===")
    print("  E_wfxp = %.4f%%   E_perrow = %.4f%%   比值 = %.3f" % (ew, ep, ew / ep))
    if ew < 0.5 * ep:   v = "🟢 显著更好 ⇒ 值得进全模型验证（须先测 PD 估算）"
    elif ew < ep:       v = "🟡 有改善但不显著 ⇒ 登记，暂不投入全模型"
    else:               v = "❌ 不更好 ⇒ 关闭 #48 的这条形态"
    print("  判定：%s" % v)
    print("\n  速度：t_wfxp=%.1fs  t_perrow=%.1fs  比值 %.2f  ⇒ %s"
          % (tw, tp, tw / tp, "🔴 产品级不可行（>3×，依据 #16）" if tw > 3 * tp else "在 3× 内"))
    print("\n🔴 划界（方案 §六）：单算子、part1b 的 linear_99、良态输入。")
    print("   不得外推到 part2，不得外推到端到端，不回答全模型 PD 是否超红线。")
    json.dump(rows, open(os.path.join(P0B, "wfxp", "verdict.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
