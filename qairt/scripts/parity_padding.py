# -*- coding: utf-8 -*-
"""padding 槽是否漏进结果 —— 单变量，纯宿主（`EXP_PLAN_OFFICIAL_PARITY.md` 派生）。

## 问题从哪来
G-E（`parity_te.py`）实测：官方只喂 **22 个真实 token 的变长 list**，
我们喂 **定长 80 槽**，而第 22~79 槽**不是零** —— std 7.90、|max| 305.67、
占 caption 全张量能量 **4.08%**。
`cap_pad_mask` 的极性已实测为「**1 = padding**」（前 22 槽为 0，后 58 槽为 1）。

⇒ 如果图内正确使用了这个 mask，padding 里装什么都不影响结果；
   如果没有，那是一条**与官方实现的实质分歧**，而且一直带在现网里。

## 单变量
两臂唯一差别 = caption 的第 22~79 行是否置零。
latents、mask、timestep、模型、执行器全部相同。

## 判据（事前锁定）
| 结果 | 判读 |
|---|---|
| 两臂 step-0 输出 **逐字节相同** | 🟢 mask 生效，padding 内容无害，本条关闭 |
| 相对 L2 < 1e-6 | 🟢 等价（浮点非确定性量级） |
| 相对 L2 >= 1e-6 | 🔴 **padding 泄漏** ⇒ 与官方的实质分歧，且是现网缺陷 |

用法: python scripts/parity_padding.py
"""
import hashlib
import os
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
WORK = os.path.join(P0, "parity")
LAT = os.path.join(P0, "latents_cxx_seed42.raw")
CAP = os.path.join(P0, "cap_r4x3.raw")
MASK = os.path.join(P0, "mask_r4x3.raw")
L, N_REAL = 80, 22


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_step(cap_path, tag):
    out = os.path.join(WORK, "s0_%s.raw" % tag)
    if os.path.isfile(out):
        os.remove(out)
    cmd = [sys.executable, "-u", os.path.join(HERE, "t2_fp32_ref.py"), "step",
           str(L), "0", LAT, out, cap_path, MASK]
    print("  跑 %s ..." % tag, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=7200)
    # 🔴 透传：报错会说谎，先把子进程说了什么打出来（约束 11·再补）
    for ln in (r.stdout or "").strip().splitlines()[-4:]:
        print("     " + ln)
    if r.returncode != 0 or not os.path.isfile(out):
        print((r.stdout or "")[-1500:])
        print((r.stderr or "")[-1500:])
        raise SystemExit("🔴 %s 臂失败 rc=%d" % (tag, r.returncode))
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    cap = np.fromfile(CAP, np.float32).reshape(L, 2560)
    pad_energy = float((cap[N_REAL:].astype(np.float64) ** 2).sum())
    print("原始 caption padding 区: |max|=%.4f  能量占比 %.2f%%"
          % (np.abs(cap[N_REAL:]).max(),
             100.0 * pad_energy / float((cap.astype(np.float64) ** 2).sum())))

    zeroed = cap.copy()
    zeroed[N_REAL:] = 0.0
    cap_zero = os.path.join(WORK, "cap_r4x3_padzero.raw")
    zeroed.astype(np.float32).tofile(cap_zero)
    print("已写出置零版 caption -> %s" % cap_zero)
    print("两臂唯一差别 = 第 %d~%d 行是否置零\n" % (N_REAL, L - 1))

    a = run_step(CAP, "orig")
    b = run_step(cap_zero, "padzero")

    print("\n=== 判据（事前锁定）===")
    sa, sb = sha(a), sha(b)
    print("  sha256 orig    = %s" % sa[:32])
    print("  sha256 padzero = %s" % sb[:32])
    if sa == sb:
        print("  ⇒ 🟢 **逐字节相同** ⇒ mask 生效，padding 内容对结果无影响")
        return 0
    x = np.fromfile(a, np.float32).astype(np.float64)
    y = np.fromfile(b, np.float32).astype(np.float64)
    rel = np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30)
    cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
    print("  相对 L2 = %.6e   余弦 = %.9f   最大逐元素绝对差 = %.6e"
          % (rel, cos, np.abs(y - x).max()))
    if rel < 1e-6:
        print("  ⇒ 🟢 等价（差异在浮点非确定性量级）")
        return 0
    print("  ⇒ 🔴🔴 **padding 泄漏**：padding 槽里的非零内容进入了结果。")
    print("     这是与官方实现（只喂真实 token、无 padding 槽）的**实质分歧**，")
    print("     且一直带在现网里。必须登记台账并定位到具体算子。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
