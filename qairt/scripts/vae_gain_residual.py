"""残差性质判别：VAE 导出差异里的非增益部分，是逐点非线性还是空间结构？

方案：scripts/EXP_PLAN_VAE_GAIN.md §十（判据先锁后看）
输入：scratch_runs/vae_gain_px[AB].npy（vae_gain_probe.py 落盘）
"""
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join("D:", os.sep, "LocalDreamZImage", "scratch_runs")
NBIN = 64


def to_img(px):
    return np.clip((px[0] + 1.0) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def d(a, b):
    return float(np.abs(to_img(a).astype(np.float64) - to_img(b).astype(np.float64)).mean())


def box3(x):
    p = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    s = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            s += p[:, :, dy:dy + x.shape[2], dx:dx + x.shape[3]]
    return s / 9.0


def pointwise_fit(A, B, nbin=NBIN):
    """用 A 的分位桶拟合逐点单调映射 f，返回 f(A)"""
    a, b = A.ravel(), B.ravel()
    edges = np.quantile(a, np.linspace(0, 1, nbin + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, nbin - 1)
    cnt = np.bincount(idx, minlength=nbin)
    sa = np.bincount(idx, weights=a, minlength=nbin)
    sb = np.bincount(idx, weights=b, minlength=nbin)
    ca = sa / np.maximum(cnt, 1)
    cb = sb / np.maximum(cnt, 1)
    ok = cnt > 0
    return np.interp(a, ca[ok], cb[ok]).reshape(A.shape), ca[ok], cb[ok]


def main():
    A = np.load(os.path.join(OUT, "vae_gain_pxA.npy")).astype(np.float64)
    B = np.load(os.path.join(OUT, "vae_gain_pxB.npy")).astype(np.float64)
    d_raw = d(A, B)
    print(f"[输入] shape={A.shape}  d_raw={d_raw:.3f}")

    def ef_pw(ref, cmp_):
        dr = d(ref, cmp_)
        fit, _, _ = pointwise_fit(ref, cmp_)
        dc = d(fit, cmp_)
        return dr, dc, 1.0 - dc / max(dr, 1e-12)

    blur = box3(A)
    ctrl = [("C1 纯增益 1.2815x+0.05", 1.2815 * A + 0.05, "EF_pw>=0.90", lambda e: e >= 0.90),
            ("C2 3x3 模糊(结构性)", blur, "EF_pw<=0.50", lambda e: e <= 0.50),
            ("C3 锐化(竞争假设本身)", A + 0.5 * (A - blur), "EF_pw<=0.50", lambda e: e <= 0.50)]
    print()
    print("=== G-check（三组全过才采信）===")
    print(f"  {'控制组':<24}{'d_raw':>8}{'d_pw':>8}{'EF_pw':>9}   要求")
    ok_all = True
    for name, arr, req, fn in ctrl:
        dr, dc, e = ef_pw(A, arr)
        ok = fn(e); ok_all &= ok
        print(f"  {name:<24}{dr:>8.3f}{dc:>8.3f}{e:>9.3f}   {req}  {'✅' if ok else '❌'}")
    if not ok_all:
        sys.exit("❌ G-check 不过：该操作化有漏洞，结论作废")

    fit, ca, cb = pointwise_fit(A, B)
    d_pw = d(fit, B)
    EF_pw = 1.0 - d_pw / d_raw
    verdict = ("🔴 纯逐点，无空间结构 ⇒ #71 用「逐层 std 之比 + 传递曲线」" if EF_pw >= 0.90
               else "✅ 存在实质空间结构 ⇒ #71 必须用形状敏感量" if EF_pw <= 0.50
               else "🔶 兼有 ⇒ #71 两者都记")
    print()
    print("=== 主判据 ===")
    print(f"  d_raw={d_raw:.3f}  d_pointwise={d_pw:.3f}  EF_pw={EF_pw:.4f}")
    print(f"  判定: {verdict}")

    print()
    print("=== 附带记录（非判据）===")
    res = B - fit
    g = np.gradient(A[0].mean(axis=0))
    gm = np.sqrt(g[0] ** 2 + g[1] ** 2).ravel()
    rm = np.abs(res[0]).mean(axis=0).ravel()
    r = float(np.corrcoef(gm, rm)[0, 1])
    print(f"  残差幅值 vs 局部梯度幅值 相关系数 = {r:+.4f}")
    lin = 1.3324 * A + 0.00222
    print(f"  逐点残差相对L2 = {np.linalg.norm(res)/np.linalg.norm(B-B.mean())*100:.4f}%"
          f"   (线性仿射残差 {np.linalg.norm(B-lin)/np.linalg.norm(B-B.mean())*100:.4f}%)")
    print(f"  传递曲线（A 分位桶 -> B 均值），每 8 桶取一个：")
    for i in range(0, len(ca), 8):
        print(f"    A={ca[i]:+.4f} -> B={cb[i]:+.4f}   斜率(局部)="
              f"{(cb[min(i+1,len(cb)-1)]-cb[i])/max(ca[min(i+1,len(ca)-1)]-ca[i],1e-9):.4f}")


if __name__ == "__main__":
    main()
