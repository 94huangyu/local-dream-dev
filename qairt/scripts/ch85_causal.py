"""EXP_PLAN_CH85_CAUSAL：part1b 的 19.47% 里，是"12 个极端点"还是"弥散本底"在毁图？

四个变体全部用 SNPE CPU 参考跑 part2（走 snpe_runner.py，约束 3），
唯一变量是喂进去的 unified 里哪些元素被换成了 HTP 的值。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snpe_runner

TB = r"D:\ZImage_Work\p0_experiments\testB"
HV = r"D:\ZImage_Work\p0_experiments\htp_vs_cpu"
WORK = r"D:\ZImage_Work\p0_experiments\ch85_causal"
K = 16  # 取 |差| 最大的前 K 个元素作为"极端点"集合 M

S_UNIFIED = (1, 4128, 3840)
S_LAT = (1, 16, 128, 128)


def rel(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)


def main():
    os.makedirs(WORK, exist_ok=True)

    u_cpu = np.fromfile(f"{TB}\\s0_transformer_part1b\\out\\Result_0\\unified.raw", np.float32)
    u_htp = np.fromfile(f"{HV}\\A_part1b\\unified.raw", np.float32)
    assert u_cpu.size == u_htp.size == 15851520

    d = np.abs(u_htp.astype(np.float64) - u_cpu.astype(np.float64))
    M = np.argsort(d)[-K:]                       # 极端点集合

    # ---- 判据 3：能量占比复核 ----
    e_tot = float((d ** 2).sum())
    e_M = float((d[M] ** 2).sum())
    print(f"判据 3（能量占比复核）：前 {K} 个极端点占总误差能量的 {100*e_M/e_tot:.2f}%")
    print(f"  （第 0 节的②推论说 ~81%）")
    for i in M[::-1][:6]:
        r, c = divmod(int(i), 3840)
        seg = "caption" if r >= 4096 else "image"
        print(f"    token {r:<5}({seg:7}) ch {c:<5} CPU={u_cpu[i]:>11.3f} HTP={u_htp[i]:>11.3f}")
    print()

    # ---- 四个变体 ----
    v_only_out = u_cpu.copy(); v_only_out[M] = u_htp[M]      # 仅极端点
    v_only_base = u_htp.copy(); v_only_base[M] = u_cpu[M]    # 仅本底

    variants = [("基线(U_cpu)", u_cpu),
                ("全HTP(U_htp)", u_htp),
                ("仅极端点", v_only_out),
                ("仅本底", v_only_base)]

    # 其余输入四次相同，取 CPU 参考
    common = {}
    for nm, shp in [("adaln_input", (1, 256)), ("unified_freqs", (1, 4128, 64, 2)),
                    ("unified_mask", (1, 4128))]:
        common[nm] = np.fromfile(
            f"{TB}\\s0_transformer_part2\\{nm}.raw", np.float32).reshape(shp)

    outs = {}
    for tag, u in variants:
        print(f"--- 跑 part2 (CPU 参考): {tag} ---", flush=True)
        feeds = dict(common)
        feeds["unified"] = u.reshape(S_UNIFIED)
        safe = tag.replace("(", "_").replace(")", "").replace("/", "_")
        r = snpe_runner.run("transformer_part2", feeds, ["latents"],
                            os.path.join(WORK, safe))
        outs[tag] = r["latents"].reshape(-1).astype(np.float64)
        print(f"    latents std={outs[tag].std():.4f}")

    base = outs["基线(U_cpu)"]
    R_all = rel(base, outs["全HTP(U_htp)"])
    R_out = rel(base, outs["仅极端点"])
    R_base = rel(base, outs["仅本底"])

    print()
    print("=" * 72)
    print("判据 1（归因）—— part2 输出相对基线的相对 L2")
    print("=" * 72)
    print(f"  R_all (全 HTP)   = {R_all:.4f}%")
    print(f"  R_out (仅极端点) = {R_out:.4f}%   占比 R_out/R_all = {R_out/R_all:.3f}")
    print(f"  R_base(仅本底)   = {R_base:.4f}%   占比 R_base/R_all = {R_base/R_all:.3f}")
    print()
    if R_out / R_all > 0.7 and R_base / R_all < 0.3:
        print("  => 【极端点是主因】H-outlier 成立。")
        print("     下一步聚焦：HTP 为何把量程 98.6% 处的值算成 55.8%。")
    elif R_base / R_all > 0.7 and R_out / R_all < 0.3:
        print("  => 【弥散本底是主因】H-outlier 被否定。")
        print("     下一步转向全局精度配置，不要再纠结 ch85。")
    else:
        print("  => 两者都有实质贡献，按比例报告，不得只归因一边。")

    print()
    print("判据 2（自洽性）：误差可加性检查")
    lhs = (R_out ** 2 + R_base ** 2) ** 0.5
    print(f"  sqrt(R_out^2 + R_base^2) = {lhs:.4f}%   vs   R_all = {R_all:.4f}%")
    print(f"  比值 = {lhs/R_all:.3f}  " +
          ("(近似可加，线性)" if 0.8 <= lhs / R_all <= 1.25 else "(偏离较大，part2 对该扰动非线性，结论中须注明)"))


if __name__ == "__main__":
    main()
