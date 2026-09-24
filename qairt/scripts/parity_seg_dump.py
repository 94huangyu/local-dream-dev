# -*- coding: utf-8 -*-
"""在**我们自己的链内部**逐段定位 L=32 与 L=80 的分歧（主线 A / 台账 #167）。

## 为什么在自己链内比，而不是继续跟官方比
已实测：我们 L=32 对官方 **1.79%**、L=80 对官方 **15.52%**、两者互比 **15.80%**
⇒ **分歧完全在我们自己的两个图之间**，不需要再花 20 分钟跑官方。
两臂都是我们的 ONNX，各约 5 分钟。

## 已排除的候选（都有实测）
| 候选 | 状态 |
|---|---|
| 图像位置常数 `stack_1`=33 | ❌ 排除：改成 81 反而更差（15.52% -> 20.22%）|
| caption padding **内容**泄漏 | ❌ 排除（#164：置零后输出逐字节相同）|
| caption padding **位置** | 🟡 弱证据排除：L=32 图给 padding 槽的位置是 23..32（官方给 0），仍能对到 1.79% |

⇒ 剩下的嫌疑在手术替换的 **19 个形状常量**（`32->80`、`4128->4176`）。

## 做法
把四段依次跑完，**每一段的每一个输出都落盘**。两个 L 各跑一次。
比较时只取**图像 token 那一段**（最后 4096 行），因为 caption 部分的槽数本来就不同。

## 判据（事前锁定）
逐段看「第一次出现显著分歧」的位置：

| 观察 | 判读 |
|---|---|
| part1a 的输出就已分歧 > 1% | 分歧起自 part1a（手术改动最多的一段）|
| part1a 一致、part1b 起分歧 | 起自 part1b |
| 各段都一致、只有最终输出分歧 | 说明是拼接/切片口径问题 |

🔴 只报「第一次超过 1% 的段」，不对更下游的段做因果解读（下游必然继承上游误差）。

用法: python scripts/parity_seg_dump.py <L>        # L = 32 或 80
      python scripts/parity_seg_dump.py compare    # 比较两次的落盘结果
"""
import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
W = os.path.join(P0, "parity", "segdump")
IMG_TOK = 4096


def run(L):
    import onnxruntime as ort
    import onnx
    from onnx import TensorProto
    import t2_fp32_ref as T

    out = os.path.join(W, "L%d" % L)
    os.makedirs(out, exist_ok=True)
    cap_p = os.path.join(P0, "cap_r4x3.raw") if L == 80 else os.path.join(P0, "parity", "cap_L32.raw")
    msk_p = os.path.join(P0, "mask_r4x3.raw") if L == 80 else os.path.join(P0, "parity", "mask_L32.raw")

    latents = np.fromfile(os.path.join(P0, "latents_cxx_seed42.raw"),
                          np.float32).reshape(1, 16, 128, 128)
    cap = np.fromfile(cap_p, np.float32).reshape(1, L, 2560)
    mask = np.fromfile(msk_p, np.float32).reshape(1, L)
    ts = T.sigmas(T.STEPS)[:-1] * 1000.0
    timestep = np.array([1.0 - ts[0] / 1000.0], dtype=np.float32)
    t = {"latents": latents, "timestep": timestep, "caption": cap, "cap_pad_mask": mask}

    print("L=%d  caption=%s  mask 真实槽=%d" % (L, cap_p, int((mask == 0).sum())))
    for seg in ("part1a", "part1b", "part2a", "part2b"):
        path = T.seg_path(seg, L)
        m = onnx.load(path, load_external_data=False)
        need = [(i.name, i.type.tensor_type.elem_type) for i in m.graph.input]
        miss = [n for n, _ in need if n not in t]
        if miss:
            raise KeyError("%s 缺输入 %s" % (seg, miss))
        se = T._sess(path)
        feed = {n: (t[n].astype(bool) if e == TensorProto.BOOL else t[n]) for n, e in need}
        outs = se.run(None, feed)
        names = [o.name for o in se.get_outputs()]
        for nm, v in zip(names, outs):
            v = np.ascontiguousarray(v, np.float32) if v.dtype != np.bool_ else v.astype(np.float32)
            t[nm] = v
            v.tofile(os.path.join(out, "%s__%s.raw" % (seg, nm.replace("/", "_"))))
        print("  %-8s -> %s" % (seg, ", ".join("%s%s" % (n, tuple(t[n].shape)) for n in names)),
              flush=True)
        del se, m
        gc.collect()
    print("落盘 -> %s" % out)
    return 0


# 🔴 布局已实测（不是假设）：`add_138` 的行范数在 L=32 下——
#    0..4095 均值 **193.9**（均匀 ~200），4096..4127 均值 **754.6**（末尾冲到 1514）
#    ⇒ **图像在前 [0..4095]，caption 在后 [4096..4096+L)**。
#    第一版按「图像在最后 4096 行」切，在 L=32 下会切成「图像的一部分 + 整个 caption」，
#    完全错位。⇒ 切片口径必须先用已知样本验（约束 8）。
IMG_FIRST = True


def compare():
    a_dir, b_dir = os.path.join(W, "L32"), os.path.join(W, "L80")
    for d in (a_dir, b_dir):
        if not os.path.isdir(d):
            raise SystemExit("🔴 缺 %s —— 先跑 parity_seg_dump.py 32 / 80" % d)
    names = sorted(set(os.listdir(a_dir)) & set(os.listdir(b_dir)))
    print("两边共有的张量 %d 个\n" % len(names))
    print("%-42s %-22s %-22s %s" % ("张量", "L32 形状", "L80 形状", "图像段相对 L2"))
    print("-" * 110)
    first = None
    for n in names:
        a = np.fromfile(os.path.join(a_dir, n), np.float32)
        b = np.fromfile(os.path.join(b_dir, n), np.float32)
        # 形状要从元素数反推：只处理能对齐图像段的情形
        if a.size == b.size:
            ra, rb = a.astype(np.float64), b.astype(np.float64)
            rel = np.linalg.norm(rb - ra) / max(np.linalg.norm(ra), 1e-30)
            tag = "整张量 %.4f%%" % (rel * 100)
        else:
            # 序列维不同：按 (seq, C) 还原后切**前** 4096 行（图像在前，已实测）
            sa, sb = 32 + IMG_TOK, 80 + IMG_TOK
            if a.size % sa == 0 and b.size % sb == 0 and a.size // sa == b.size // sb:
                C = a.size // sa
                ra = a.reshape(sa, C)[:IMG_TOK].astype(np.float64)
                rb = b.reshape(sb, C)[:IMG_TOK].astype(np.float64)
                rel = np.linalg.norm(rb - ra) / max(np.linalg.norm(ra), 1e-30)
                tag = "图像段(前4096) %.4f%%" % (rel * 100)
            else:
                tag = "形状不可比（%d vs %d）" % (a.size, b.size)
                rel = None
        print("%-42s %-22s %-22s %s" % (n[:42], a.size, b.size, tag))
        if rel is not None and rel > 0.01 and first is None:
            first = (n, rel)
    print()
    if first:
        print("🔴 **第一个相对 L2 > 1%% 的张量：%s（%.4f%%）**" % (first[0], first[1] * 100))
        print("   ⇒ 分歧起自这里。**不对更下游的张量做因果解读**（下游必然继承上游）。")
    else:
        print("🟢 所有可比张量的相对 L2 都 <= 1%")
    return 0


if __name__ == "__main__":
    os.makedirs(W, exist_ok=True)
    c = sys.argv[1] if len(sys.argv) > 1 else "compare"
    sys.exit(compare() if c == "compare" else run(int(c)))
