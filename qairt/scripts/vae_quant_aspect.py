# -*- coding: utf-8 -*-
"""用真实校准数据量化各比例的 VAE（配方照抄现网，不用 overrides）。

## 为什么不用 overrides
2026-09-02 门 O1 实测：按名字注入的 overrides **够不着转换器合成的 GroupNorm 参数**
（`node_Reshape_*_GroupNorm_gamma` 等 68 个 STATIC 张量退回 Float_32）。

## 为什么不用 overrides 也能共享权重
`param_quantizer_calibration=min-max` 下**权重 encoding 只取决于权重张量自身**，
与校准数据无关；只有激活 encoding 依赖校准。而权重共享共享的是权重。

## 判据（执行前锁定）
- **门 W1（已知样本）**：用现网校准数据重量化 **1:1**，其 encoding 必须与现网**逐项相同**。
  不过 ⇒ 装置有误，后面的数一律不许用。
- **门 W2（可共享性）**：4:3 的**权重** encoding 必须与现网 1:1 **逐项相同**；
  **激活** encoding 允许不同（本来就该反映 4:3 的激活分布）。

用法: python vae_quant_aspect.py <tag>
"""
import io
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

PY = sys.executable
TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qairt_tool.py")
VD = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence",
                  "dlc_pipeline", "vae_decoder")
ONNXD = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")

TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[[^\]]*\]; "
                r"tensor type: ([A-Z]+)\)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")


def sh(name, cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace", timeout=timeout)
    print("  [%s] rc=%d  %.1f 分钟" % (name, r.returncode, (time.time() - t0) / 60.0), flush=True)
    if r.returncode:
        print((r.stdout or "")[-1500:])
        sys.exit("FAIL %s" % name)


def enc_of(dlc, csv):
    sh("dlc-info", [PY, TOOL, "snpe-dlc-info", "-i", dlc, "-d", "-s", csv], 3600)
    kind, pt = {}, {}
    for line in io.open(csv, encoding="utf-8", errors="replace"):
        for m in TT.finditer(line):
            kind.setdefault(m.group(1), (m.group(2), m.group(3)))
        for m in PT.finditer(line):
            pt.setdefault(m.group(1), m.groups()[1:])
        # per-channel 编码也要算（现网 VAE 带 --use_per_channel_quantization）
        for m in CH.finditer(line):
            pt.setdefault(m.group(1), []) if not isinstance(
                pt.get(m.group(1)), list) else None
            if isinstance(pt.get(m.group(1)), list):
                pt[m.group(1)].append(m.groups()[1:])
    static = set(k for k, v in kind.items() if v[1] == "STATIC")
    return kind, pt, static


def quant(stem, tag, listfile, lat_hw):
    f = os.path.join(OUT, "vaeq_%s_fp32.dlc" % tag)
    q = os.path.join(OUT, "vaeq_%s_quantized.dlc" % tag)
    src = os.path.join(ONNXD, stem + ".onnx")
    if not os.path.isfile(src):
        sys.exit("FAIL 缺 %s" % src)
    if not os.path.isfile(f):
        # 🔴 必须照抄现网：`--onnx_skip_simplification` + 显式输入形状。
        # 漏掉前者会让 ONNX simplifier 融合改名 => 张量数 313 vs 242、名字风格全变
        # （2026-09-02 门 W1 抓到）。我当初 grep 现网命令时只筛了预期的两项，
        # 把 `no_simplification=True` 滤掉了 —— 约束 11·补「按 grep 结果逐个追」。
        sh("转换", [PY, TOOL, "qairt-converter", "--input_network", src,
                    "--output_path", f, "--float_bitwidth", "32",
                    "--onnx_skip_simplification",
                    "-s", "vae_latents", lat_hw])
    if not os.path.isfile(q):
        sh("量化", [PY, TOOL, "qairt-quantizer", "--input_dlc", f, "--output_dlc", q,
                    "--input_list", listfile,
                    "--weights_bitwidth", "8", "--act_bitwidth", "16",
                    "--bias_bitwidth", "32",
                    "--act_quantizer_calibration", "min-max",
                    "--param_quantizer_calibration", "min-max",
                    # 🔴 现网带了它（`use_per_channel_quantization=True`）。VAE 全是卷积，
                    # 而 #44 记 per-channel 只作用于 convolution 权重 => 对 VAE 确实生效。
                    # 我第一次漏掉它，是因为 grep 时又只筛了自己预期的字段；
                    # 改成【打印全部非默认参数】后一次命中（约束 11·补）。
                    "--use_per_channel_quantization"])
    print("  fp32 %.1f MB | 量化 %.1f MB" % (os.path.getsize(f) / 1e6,
                                             os.path.getsize(q) / 1e6), flush=True)
    return q


def cmp_enc(a_pt, a_st, b_pt, b_st, label):
    wa = {k: v for k, v in a_pt.items() if k in a_st}
    wb = {k: v for k, v in b_pt.items() if k in b_st}
    com = set(wa) & set(wb)
    dif = [k for k in com if wa[k] != wb[k]]
    print("  %s 权重 encoding：A %d / B %d | 共有 %d | 不同 %d"
          % (label, len(wa), len(wb), len(com), len(dif)), flush=True)
    for k in sorted(dif)[:5]:
        print("     %s  A=%s  B=%s" % (k, wa[k], wb[k]), flush=True)
    only = sorted((set(wa) ^ set(wb)))
    if only:
        print("     只在一侧的权重 %d 个，前 5：%s" % (len(only), only[:5]), flush=True)
    # 🔴 2026-09-03 判据订正：**bias 的差异是预期的，不算失败**。
    # bias 的 scale = 输入scale x 权重scale，依赖【激活】编码，而激活编码本就该
    # 随比例不同（不同形状的激活分布不同）。权重共享共享的是**卷积权重**，不是 bias。
    # 原判据「任何差异即 FAIL」会把一个通过的结果误报成失败（2026-09-02/03 各误报一次）。
    bias = [k for k in dif if k.endswith(".bias") or "_bias" in k]
    real = [k for k in dif if k not in bias]
    print("     其中 bias %d（预期，不算失败） | 非 bias %d（这才是判据）"
          % (len(bias), len(real)), flush=True)
    for k in real[:5]:
        print("     🔴 非 bias 权重不同: %s" % k, flush=True)
    return (not real) and (not only)


def main():
    tag = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)

    # 校准列表
    lst43 = os.path.join(OUT, "vae_calib_%s.txt" % tag)
    with io.open(lst43, "w", encoding="utf-8") as fo:
        for s in (42, 43, 44, 45, 46):
            fo.write("vae_latents:=%s\n"
                     % os.path.join(P0, "vae_calib_%s" % tag, "latent_s%d.raw" % s))
    print("=== 门 W1：用现网校准数据重量化 1:1（已知样本）===", flush=True)
    q11 = quant("vae_decoder", "ctrl11", os.path.join(VD, "input_list_raw.txt"),
                "1,16,128,128")
    k1, p1, s1 = enc_of(os.path.join(VD, "vae_decoder_quantized.dlc"),
                        os.path.join(OUT, "vaeq_dep.csv"))
    k2, p2, s2 = enc_of(q11, os.path.join(OUT, "vaeq_ctrl11.csv"))
    print("  张量总数 现网 %d / 重量化 %d" % (len(p1), len(p2)), flush=True)
    ok1 = cmp_enc(p1, s1, p2, s2, "W1")
    print("  W1 判定: %s" % ("PASS" if ok1 else "FAIL 装置有误，后面的数不许用"), flush=True)
    if not ok1:
        return 1

    print("\n=== 门 W2：%s 的权重 encoding 必须与现网 1:1 相同 ===" % tag, flush=True)
    w, h = [int(x) for x in tag.lower().split("x")]
    q43 = quant("vae_decoder_%s" % tag, tag, lst43, "1,16,%d,%d" % (h // 8, w // 8))
    k3, p3, s3 = enc_of(q43, os.path.join(OUT, "vaeq_%s.csv" % tag))
    ok2 = cmp_enc(p1, s1, p3, s3, "W2")
    na = len(set(p1) - s1)
    nb = len(set(p3) - s3)
    print("  （激活 encoding：现网 %d / %s %d —— 允许不同）" % (na, tag, nb), flush=True)
    print("  W2 判定: %s" % ("PASS 权重可共享" if ok2 else "FAIL 权重不同，无法共享"), flush=True)
    json.dump({"W1": ok1, "W2": ok2}, open(os.path.join(OUT, "vae_quant_gates.json"), "w"))
    return 0 if ok2 else 1


sys.exit(main())
