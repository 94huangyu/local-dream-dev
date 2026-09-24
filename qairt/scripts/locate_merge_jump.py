"""
误差在合并点 add_54 出现 28 倍阶跃（两条支流各自只有 ~1.1，合并后 31.84）。
本脚本定位这个阶跃发生在 unified 序列的哪个位置：
  - 图像 token 段 [0:4096]  还是  caption token 段 [4096:4128]？
  - 是集中在少数通道，还是全面铺开？

结论用途：决定下一步该往"合并算子 select_scatter_4"还是"第一个 unified attention block"里查。
"""
import numpy as np

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
REF = np.load(EVIDENCE + r"\onnx\early_block_reference.npz")
QDIR = r"D:\ZImage_Work\p0_experiments\snpe_early_chain\Result_0"


def load_pair(name, shape):
    r = REF[name].astype(np.float32)
    q = np.fromfile(f"{QDIR}\\{name}.raw", dtype=np.float32).reshape(shape)
    return r, q


r54, q54 = load_pair("add_54", (1, 4128, 3840))
err = np.abs(q54 - r54)[0]          # (4128, 3840)
ref = r54[0]

IMG_N = 4096
img_err, cap_err = err[:IMG_N], err[IMG_N:]
img_ref, cap_ref = ref[:IMG_N], ref[IMG_N:]

print("=" * 92)
print("add_54 (unified, 4128 token) 的误差按 token 段拆分")
print("=" * 92)
print(f"{'段':<26} {'token数':>8} {'ref |max|':>12} {'误差max':>12} {'误差均值':>12} {'误差>1占比':>12}")
print("-" * 92)
for label, e, rf in [("图像 token [0:4096]", img_err, img_ref),
                     ("caption token [4096:4128]", cap_err, cap_ref)]:
    print(f"{label:<26} {e.shape[0]:>8} {np.abs(rf).max():>12.2f} {e.max():>12.4f} "
          f"{e.mean():>12.6f} {(e > 1.0).mean() * 100:>11.4f}%")

print()
print("=" * 92)
print("对照：合并前两条支流各自的误差（同一次运行实测）")
print("=" * 92)
r21, q21 = load_pair("add_21", (1, 4096, 3840))
r42, q42 = load_pair("add_42", (1, 32, 3840))
e21 = np.abs(q21 - r21)[0]
e42 = np.abs(q42 - r42)[0]
print(f"{'add_21 (图像流最后一个已测点)':<40} 误差max = {e21.max():>10.4f}  均值 = {e21.mean():.6f}")
print(f"{'add_42 (caption流最后一个已测点)':<40} 误差max = {e42.max():>10.4f}  均值 = {e42.mean():.6f}")

print()
print("=" * 92)
print("误差在 3840 个通道上的分布（add_54，找是否集中在少数通道）")
print("=" * 92)
per_ch = err.max(axis=0)                       # 每个通道在所有 token 上的最大误差
order = np.argsort(per_ch)[::-1]
print("误差最大的 15 个通道：")
print(f"{'通道':>8} {'该通道误差max':>16} {'该通道 ref |max|':>18}")
for c in order[:15]:
    print(f"{c:>8} {per_ch[c]:>16.4f} {np.abs(ref[:, c]).max():>18.4f}")
print()
print(f"误差 > 1.0 的通道数: {(per_ch > 1.0).sum()} / 3840")
print(f"误差 > 10.0 的通道数: {(per_ch > 10.0).sum()} / 3840")
print(f"全部通道误差中位数: {np.median(per_ch):.6f}")

print()
print("=" * 92)
print("判读")
print("=" * 92)
if cap_err.max() > img_err.max() * 5:
    print(">>> 阶跃集中在 caption token 段：问题出在 caption 分支并入 unified 的那一步。")
elif img_err.max() > cap_err.max() * 5:
    print(">>> 阶跃集中在图像 token 段：问题出在图像分支并入 unified 的那一步，")
    print(">>> 或出在第一个 unified attention block 对图像 token 的处理上。")
else:
    print(">>> 两段误差量级相当：阶跃不是某一条支流带进来的，")
    print(">>> 更可能来自合并之后的第一个 unified block（共用一套 per-tensor scale 覆盖 4128 个 token）。")
