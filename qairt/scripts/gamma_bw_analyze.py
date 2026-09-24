"""EXP_PLAN_GAMMA_BW 第 3 步：落判据。

判据逐字取自 scripts/EXP_PLAN_GAMMA_BW.md §四（**事前锁定，不得事后修改**）。
本脚本只负责算数和落档，不做任何解释性判断。
"""
import os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
G = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "gamma_bw")
REF = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "vs_fp32", "latents_fp32_s0.raw")

FLOOR = 15.99      # CPU 参考（全部权重 8-bit，激活浮点）vs FP32
CUR   = 44.80      # 当前四段 per-row 端到端
GATE_V0 = 0.01     # A 臂必须复现原点，相对 L2 < 0.01%
HI, LO  = 7.15, 2.26   # 判据阈值：share=(E/15.99)^2 的 20% / 2%


def rel_l2(a, b):
    return 100.0 * np.linalg.norm(b - a) / np.linalg.norm(a)


def top1_share(a):
    """约束 7 强制：前 1% 元素占 ||a||^2 的比例"""
    e = np.sort(a.astype(np.float64) ** 2)[::-1]
    return 100.0 * e[:max(1, e.size // 100)].sum() / e.sum()


def main():
    ref = np.fromfile(REF, np.float32).astype(np.float64)
    A = np.fromfile(os.path.join(G, "latents_A_s0.raw"), np.float32).astype(np.float64)
    B = np.fromfile(os.path.join(G, "latents_B_s0.raw"), np.float32).astype(np.float64)
    assert ref.size == A.size == B.size, (ref.size, A.size, B.size)

    # ---- V3 口径门 ----
    sh = top1_share(ref)
    v3 = sh <= 50.0
    print("[V3 口径门] FP32 参考前 1%% 元素占 ||a||^2 = %.2f%%（>50%% 则全量口径失效）  %s"
          % (sh, "PASS" if v3 else "FAIL"))
    print("            §3.1 记录值为 8.15%%，本次复算用于确认原点未被换过")

    # ---- V0 装置门 ----
    e_a = rel_l2(ref, A)
    v0 = e_a < GATE_V0
    print("[V0 装置门] A 臂 vs 已存原点 latents_fp32_s0.raw = %.6f%%（要求 <%.2f%%）  %s"
          % (e_a, GATE_V0, "PASS" if v0 else "FAIL"))
    if not v0:
        print("\n🔴 V0 未过 ⇒ 我的执行环境与产生原点那次不同，**整个实验作废**，不得解读 B 臂数值。")
        return

    # ---- 结果 ----
    e_g = rel_l2(ref, B)
    share = (e_g / FLOOR) ** 2 * 100.0
    print("\n=== 结果 ===")
    print("E_gamma（B 臂 vs FP32，全量相对 L2） = %.4f%%" % e_g)
    print("参照：权重量化地板 %.2f%%  |  当前端到端 %.2f%%" % (FLOOR, CUR))
    print("share = (E_gamma / %.2f)^2 = %.2f%%  （gamma 占权重量化地板的能量比例）" % (FLOOR, share))
    print("若与端到端正交叠加，去掉 gamma 后 E_all ≈ %.2f%%（改善 %.2f pp）"
          % (np.sqrt(max(CUR ** 2 - e_g ** 2, 0)), CUR - np.sqrt(max(CUR ** 2 - e_g ** 2, 0))))

    if e_g >= HI:
        v = "🟢 主要成分 ⇒ 值得专门跑一轮完整验证（重量化+建图+上机）"
    elif e_g >= LO:
        v = "🟡 次要但非零 ⇒ 捆绑进下一次为其他 P1 而做的重量化（边际成本≈0），不单独立项"
    else:
        v = "🔶 低优先 ⇒ 登记【未查·低优先】，本轮不投入"
    print("\n判定：%s" % v)
    print("\n🔴 无论落哪一档均**不得**写「已排除/已证明无效」——#52 实测 HTP 对权重量化步长的")
    print("   敏感度 = CPU 的 37.6 倍，本实验是宿主浮点测量，**只是 HTP 影响的下界**。")
    print("🔴 止损：本实验之后的下一个动作一律是 P1 三条之一（方案 §七）。")


if __name__ == "__main__":
    main()
