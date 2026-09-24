"""P0-B 第 1 步：从 per-row DLC 抽出 node_linear_99 的精确 encoding。

为什么这样抽（都是今天踩出来的）：
  · DLC 有 1400 MB，整体 qairt-dlc-to-json 再 json.load 有压机风险（本机被压死过两次）
  · snpe-dlc-info 默认只打印 channel_0；**`-d/--display_all_encodings` 才给全部 axis 量化信息**
    （约束 5：先读 --help，今天已因没读而踩过一次）
  · 用 `-s` 导出 CSV（81.74 MB），流式解析，内存安全

⚠️ 来源必须是 per-row 那份：p0_experiments/perrow/part1b_perrow_quantized.dlc
   （其 context 与设备 /data/local/tmp/htpcmp/part1b.bin 逐字节一致 = 1,472,862,256 B）
   基线 dlc_pipeline/.../transformer_part1b_quantized.dlc 里 val_1800 是 uFxp_8 per-tensor，用错即作废。
"""
import io, os, re, sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
CSV = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b",
                   "part1b_perrow_encodings.csv")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
W_NAME = "val_1800"
ACTS = ["node_linear_99_pre_reshape", "linear_99_fc"]

CH = re.compile(re.escape(W_NAME) +
                r" encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")
# 🔴 (\S+) 会把 CSV 的引号一起吞进张量名 => 永远匹配不上（2026-08-21 踩过）
ACT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                 r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def main():
    chans, acts = {}, {}
    with io.open(CSV, encoding="utf-8", errors="replace") as f:
        for line in f:
            if W_NAME in line:
                for m in CH.finditer(line):
                    i = int(m.group(1))
                    chans[i] = dict(bitwidth=int(m.group(2)), min=float(m.group(3)),
                                    max=float(m.group(4)), scale=float(m.group(5)),
                                    offset=float(m.group(6)))
            for a in ACTS:
                if a + " encoding :" in line:
                    for m in ACT.finditer(line):
                        if m.group(1) == a and a not in acts:
                            acts[a] = dict(bitwidth=int(m.group(2)), min=float(m.group(3)),
                                           max=float(m.group(4)), scale=float(m.group(5)),
                                           offset=float(m.group(6)))

    n = len(chans)
    print("权重 %s: 抓到 %d 个通道 encoding" % (W_NAME, n))
    if n == 0:
        sys.exit("❌ 一个都没抓到 —— 正则或来源不对")
    miss = [i for i in range(max(chans) + 1) if i not in chans]
    print("  通道号连续性: 0..%d，缺失 %d 个 %s" % (max(chans), len(miss), miss[:5] if miss else ""))
    if n != 3840 or miss:
        sys.exit("❌ 期望 3840 个连续通道，实得 %d ⇒ 解析不完整，不得用于重放" % n)

    scales = np.array([chans[i]["scale"] for i in range(n)], dtype=np.float64)
    offsets = np.array([chans[i]["offset"] for i in range(n)], dtype=np.float64)
    bws = set(chans[i]["bitwidth"] for i in range(n))
    print("  bitwidth 集合: %s   offset 是否全 0（对称）: %s" % (bws, bool(np.all(offsets == 0))))
    print("  scale: min %.10g  max %.10g  中位 %.10g  (max/min = %.2f 倍)"
          % (scales.min(), scales.max(), float(np.median(scales)), scales.max() / scales.min()))

    np.save(os.path.join(OUT, "val_1800_perrow_scales.npy"), scales.astype(np.float64))
    np.save(os.path.join(OUT, "val_1800_perrow_offsets.npy"), offsets.astype(np.float64))
    print("")
    print("激活 encoding：")
    for a in ACTS:
        if a not in acts:
            sys.exit("❌ 未抓到激活 encoding: %s" % a)
        e = acts[a]
        print("  %-32s bw=%d scale=%.12g offset=%g  [min %.6f, max %.6f]"
              % (a, e["bitwidth"], e["scale"], e["offset"], e["min"], e["max"]))
    json.dump({"weight": {"name": W_NAME, "axis": 0, "num_channels": n,
                          "bitwidth": sorted(bws)[0], "symmetric": bool(np.all(offsets == 0))},
               "activations": acts},
              open(os.path.join(OUT, "linear_99_encodings.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("")
    print("[落盘] p0b/val_1800_perrow_{scales,offsets}.npy + linear_99_encodings.json")


if __name__ == "__main__":
    main()
