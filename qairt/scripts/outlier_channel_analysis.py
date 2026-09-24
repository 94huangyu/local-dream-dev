"""离群值是否集中在【少数固定通道】？—— 决定"激活离群迁移"这条路是否成立。

已实测的病态：`mul_481` 99.97% 的能量集中在 0.01% 的元素；
`unified`/`add_138` 的极端离群全部落在通道 85。

若离群在通道维上固定，则属 transformer 的 "massive activation" 现象，
可用逐通道缩放把幅值从激活迁移到权重（激活变平滑、权重仍 8-bit，
**模型大小与推理速度均不变**）。

若离群随机散布，则该路不成立，须另寻方向。

零成本：只用已有的 CPU 参考张量。
"""
import os

import numpy as np

DIRS = [
    r"D:\ZImage_Work\p0_experiments\layer_probe2\cpu\out\Result_0",
    r"D:\ZImage_Work\p0_experiments\layer_probe\cpu\out\Result_0",
]
EXTRA = {
    "add_138": (r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1a\out\Result_0\add_138.raw", 3840),
    "unified": (r"D:\ZImage_Work\p0_experiments\testB\s0_transformer_part1b\out\Result_0\unified.raw", 3840),
}
# 张量名 -> 最后一维（通道数）
LASTDIM = {
    "mul_304": 3840, "add_153": 3840, "linear_115_fc": 3840, "linear_118_fc": 3840,
    "mul_436": 3840, "linear_150_fc": 3840, "mul_476": 3840, "mul_477": 3840,
    "add_222": 3840, "mul_479": 3840, "mul_480": 3840, "mul_483": 3840,
    "mul_484": 3840, "unified": 3840, "linear_153_fc": 3840,
    "linear_151_fc": 10240, "linear_152_fc": 10240, "val_2609": 10240,
    "silu_19": 10240, "mul_481": 10240, "node_linear_129_pre_reshape": 10240,
}


def main():
    files = {}
    for d in DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".raw") and f[:-4] in LASTDIM:
                    files.setdefault(f[:-4], (os.path.join(d, f), LASTDIM[f[:-4]]))
    files.update({k: v for k, v in EXTRA.items() if k not in files})

    print(f"{'张量':<30}{'通道数':>8}{'能量最高通道':>13}{'该通道占总能量':>15}"
          f"{'前3通道占比':>13}{'判定':>12}")
    print("=" * 96)
    localized = []
    for name, (path, C) in sorted(files.items()):
        x = np.fromfile(path, np.float32).astype(np.float64)
        if x.size % C:
            continue
        X = x.reshape(-1, C)
        e = (X ** 2).sum(axis=0)
        tot = e.sum()
        order = np.argsort(e)[::-1]
        top1 = e[order[0]] / tot
        top3 = e[order[:3]].sum() / tot
        verdict = "通道集中" if top3 > 0.5 else ("偏集中" if top3 > 0.2 else "分散")
        if top3 > 0.5:
            localized.append((name, order[:3].tolist(), top3))
        print(f"{name:<30}{C:>8}{order[0]:>13}{100*top1:>14.2f}%"
              f"{100*top3:>12.2f}%{verdict:>12}")

    print()
    print("=" * 96)
    print("判定")
    print("=" * 96)
    print(f"  能量集中在前 3 个通道（>50%）的张量：{len(localized)}/{len(files)}")
    if localized:
        from collections import Counter
        c = Counter()
        for _, ch, _ in localized:
            c.update(ch)
        print(f"  这些张量的高能通道分布：{c.most_common(8)}")
        common = [ch for ch, n in c.items() if n >= 3]
        print(f"  在 ≥3 个张量里都是高能通道的：{sorted(common)}")
    print()
    if len(localized) >= len(files) * 0.4:
        print("  => 离群在通道维上高度集中 ⇒ 属 massive activation 形态，")
        print("     '逐通道缩放迁移'这条路【成立】，值得进一步评估。")
    else:
        print("  => 离群未在通道维集中 ⇒ 该路【不成立】，须另寻方向。")


if __name__ == "__main__":
    main()
