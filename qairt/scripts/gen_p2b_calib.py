"""为 part2b 生成校准数据：用 FP32 跑 part2a 过全部 40 个校准样本，产出切口张量 `add_92`。

为什么必须做：part2 被切成 part2a/part2b 后，part2b 的输入 `add_92` 是**新出现的切口张量**，
项目里不存在它的校准数据。量化 part2b 必须有它，否则激活 encoding 无从校准。
这是"拆分 part2"这条路上此前没算进预算的一步（约 80 分钟）。

口径纪律：
  · 用 **FP32 ONNX** 跑 part2a，不用量化 DLC —— 与项目里其它段的校准数据同源同口径
    （part1b 的 add_138 校准数据也是上游 FP32 产物）
  · 输入直接取 part2 现成的 40 个校准样本，**逐字节复用**，不重新生成
  · 输出按 `snpe`/`qairt-quantizer` 要求写成 float32 raw

⚠️ 内存：part2a 的 fp32 权重约 5.3 GB，onnxruntime 会实际载入 ⇒ **必须独占 PC 运行**。

用法: python gen_p2b_calib.py
"""
import glob
import os

import numpy as np
import onnxruntime as ort

ONNX = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx\transformer_part2a_fixed.onnx"
SRC = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\transformer_part2\calibration_raw"
OUT = r"D:\ZImage_Work\p0_experiments\p2split\calib_part2b_fixed"
LIST = r"D:\ZImage_Work\p0_experiments\p2split\input_list_part2b_fixed.txt"

# dtype 必须与 ONNX 声明一致：unified_mask 是 bool、latents_shape 是 int32
SPEC = {
    "unified":       ((1, 4128, 3840), np.float32),
    "unified_mask":  ((1, 4128),       bool),
    "unified_freqs": ((1, 4128, 64, 2), np.float32),
    "adaln_input":   ((1, 256),        np.float32),
}
# 真实切割集（反向可达性算出，6 个张量 / 65.6 MB），不是最初以为的 1 个
CUT = ["add_92", "select", "select_1", "split_7_split_2", "split_7_split_3", "val_105"]


def load(sample, name):
    p = os.path.join(sample, name + ".raw")
    shape, dt = SPEC[name]
    if dt is bool:                       # 设备/校准侧存的是 float32 0/1
        return np.fromfile(p, np.float32).reshape(shape).astype(bool)
    if dt is np.int32:
        a = np.fromfile(p, np.int32)
        return a.reshape(shape) if a.size == 3 else np.fromfile(p, np.float32).astype(np.int32).reshape(shape)
    return np.fromfile(p, np.float32).reshape(shape)


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = sorted(glob.glob(os.path.join(SRC, "sample_*")))
    print(f"校准样本 {len(samples)} 个  <- {SRC}")
    avail = set(os.listdir(samples[0]))
    print(f"样本内含: {sorted(avail)}")

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    need = [i.name for i in sess.get_inputs()]
    print(f"part2a 需要输入: {need}")

    lines = []
    for k, s in enumerate(samples):
        feeds = {n: load(s, n) for n in need}
        outs = sess.run(CUT, feeds)
        d = os.path.join(OUT, os.path.basename(s))
        os.makedirs(d, exist_ok=True)
        parts = []
        for nm, arr in zip(CUT, outs):
            p = os.path.join(d, nm + ".raw")
            np.ascontiguousarray(arr.astype(np.float32)).tofile(p)
            parts.append(f"{nm}:={p}")
        # adaln_input 直接引用原样本（零复制）；latents_shape 被转换器折叠成常量，不是 DLC 输入
        parts.append(f"adaln_input:={os.path.join(s, 'adaln_input.raw')}")
        lines.append(" ".join(parts))
        print(f"  [{k+1:>2}/{len(samples)}] {os.path.basename(s)}  "
              f"add_92.std={outs[0].std():.4f}", flush=True)

    with open(LIST, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n写出 {LIST}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
