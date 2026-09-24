# -*- coding: utf-8 -*-
"""从现网 VAE 量化 DLC 抽 encoding 成 overrides，再用它量化任意比例的 VAE。

## 为什么不能照抄现网 VAE 的配方
实查现网 `vae_decoder_quantized.dlc`：它用**真实校准数据**（`input_list_raw.txt`，
5 个 `[1,16,128,128]` 样本）、`act_bitwidth=16`、无 overrides、不 per-row。
⇒ 4:3 的校准样本形状不对，重新生成 5 个真实 4:3 latents 约需 3 小时。

## 改走 overrides 复用（同 transformer 的做法）
激活 encoding 是 **per-tensor 标量**、与空间形状无关；且这样量化权重逐字节相同
⇒ VAE 也能进多图共享权重。

## 🔴 门 O1（已知样本验证，约束 8）
先用抽出的 overrides **重新量化 1:1 的 VAE**，其 encoding 必须与现网**逐项相同**。
不过就说明抽取有误，**不许拿去用在 4:3 上**。

用法: python vae_ovr_build.py extract          # 抽 overrides + 跑门 O1
      python vae_ovr_build.py apply <tag>      # 用它量化 <tag> 的 VAE
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

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
PY = sys.executable
TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qairt_tool.py")
VD = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence",
                  "dlc_pipeline", "vae_decoder")
ONNXD = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
OVR = os.path.join(OUT, "vae_ovr.json")

TT = re.compile(r"([A-Za-z0-9_./-]+) \(data type: ([A-Za-z_0-9]+); tensor dimension: \[[^\]]*\]; "
                r"tensor type: ([A-Z]+)\)")
CH = re.compile(r"([A-Za-z0-9_./-]+) encoding for channel_(\d+): bitwidth (\d+), min ([-\d.]+), "
                r"max ([-\d.]+), scale ([-\d.eE+]+), offset ([-\d.]+)")
PT = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                r"scale ([-\d.eE+]+), offset ([-\d.]+)")


def sh(name, cmd, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace", timeout=timeout)
    print("  [%s] rc=%d  %.1f 分钟" % (name, r.returncode, (time.time() - t0) / 60.0), flush=True)
    for line in (r.stdout or "").splitlines():
        low = line.lower()
        if "processed" in low and "encoding" in low:
            print("    * " + line.strip(), flush=True)
    if r.returncode:
        print((r.stdout or "")[-1200:])
        sys.exit("FAIL %s" % name)
    return r.stdout or ""


def read_enc(dlc, csv):
    sh("dlc-info", [PY, TOOL, "snpe-dlc-info", "-i", dlc, "-d", "-s", csv], 3600)
    kind, ch, pt = {}, {}, {}
    with io.open(csv, encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in TT.finditer(line):
                kind.setdefault(m.group(1), (m.group(2), m.group(3)))
            for m in CH.finditer(line):
                ch.setdefault(m.group(1), {})[int(m.group(2))] = dict(
                    bitwidth=int(m.group(3)), min=float(m.group(4)), max=float(m.group(5)),
                    scale=float(m.group(6)), offset=float(m.group(7)))
            for m in PT.finditer(line):
                pt.setdefault(m.group(1), dict(bitwidth=int(m.group(2)), min=float(m.group(3)),
                                               max=float(m.group(4)), scale=float(m.group(5)),
                                               offset=float(m.group(6))))
    return kind, ch, pt


def enc1(e, sym):
    return dict(bitwidth=e["bitwidth"], min=e["min"], max=e["max"],
                scale=e["scale"], offset=e["offset"], is_symmetric=str(sym))


def build_ovr():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(VD, "vae_decoder_quantized.dlc")
    kind, ch, pt = read_enc(src, os.path.join(OUT, "vae_deployed_enc.csv"))
    static = set(k for k, v in kind.items() if v[1] == "STATIC")
    params, nch = {}, 0
    for w in sorted(static):
        if w.endswith("_bias") or kind[w][0] not in ("sFxp_8", "uFxp_8", "uFxp_16"):
            continue
        if w in ch:
            c = ch[w]
            params[w] = [enc1(c[i], all(c[j]["offset"] == 0 for j in c)) for i in sorted(c)]
            nch += len(c)
        elif w in pt:
            params[w] = [enc1(pt[w], False)]
    acts = {}
    for a, e in pt.items():
        if a in static:
            continue
        acts[a] = [enc1(e, False)]
    json.dump({"version": "0.6.1", "activation_encodings": acts, "param_encodings": params},
              io.open(OVR, "w", encoding="utf-8"))
    print("  权重 %d（per-channel 标量 %d） | 激活 %d  -> %s"
          % (len(params), nch, len(acts), os.path.basename(OVR)), flush=True)
    return params, acts


def quantize(stem, tag, ovr):
    """转换（带 overrides）+ 量化。配方照抄现网 VAE，只把 overrides 换进来。"""
    fdlc = os.path.join(OUT, "vae_%s_fp32.dlc" % tag)
    qdlc = os.path.join(OUT, "vae_%s_quantized.dlc" % tag)
    src = os.path.join(ONNXD, stem + ".onnx")
    if not os.path.isfile(src):
        sys.exit("FAIL 缺 %s" % src)
    if not os.path.isfile(fdlc):
        sh("转换", [PY, TOOL, "qairt-converter", "--input_network", src,
                    "--output_path", fdlc, "--float_bitwidth", "32",
                    "--quantization_overrides", ovr], 7200)
    if not os.path.isfile(qdlc):
        sh("量化", [PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                    "--weights_bitwidth", "8", "--act_bitwidth", "16", "--bias_bitwidth", "32",
                    "--act_quantizer_calibration", "min-max",
                    "--param_quantizer_calibration", "min-max"], 7200)
    print("  fp32 %.1f MB | 量化 %.1f MB"
          % (os.path.getsize(fdlc) / 1e6, os.path.getsize(qdlc) / 1e6), flush=True)
    return qdlc


def main():
    cmd = sys.argv[1]
    if cmd == "extract":
        print("=== 抽 overrides ===", flush=True)
        build_ovr()
        print("\n=== 门 O1：用它重量化 1:1，encoding 必须与现网逐项相同 ===", flush=True)
        q = quantize("vae_decoder", "ctrl1x1", OVR)
        _, ch2, pt2 = read_enc(q, os.path.join(OUT, "vae_ctrl_enc.csv"))
        _, ch1, pt1 = read_enc(os.path.join(VD, "vae_decoder_quantized.dlc"),
                               os.path.join(OUT, "vae_deployed_enc.csv"))
        common = set(pt1) & set(pt2)
        diff = [k for k in common if pt1[k] != pt2[k]]
        print("  张量 现网 %d / 重量化 %d | 共有 %d | 不同 %d"
              % (len(pt1), len(pt2), len(common), len(diff)), flush=True)
        for k in sorted(diff)[:6]:
            print("     %s: 现网=%s 重量化=%s" % (k, pt1[k], pt2[k]), flush=True)
        ok = (not diff) and len(common) > 50
        print("  判定: %s" % ("PASS 抽取正确，可用于其他比例" if ok
                              else "FAIL 抽取有误，不许用于 4:3"), flush=True)
        return 0 if ok else 1
    if cmd == "apply":
        tag = sys.argv[2]
        if not os.path.isfile(OVR):
            sys.exit("FAIL 先跑 extract")
        print("=== 用 overrides 量化 %s 的 VAE ===" % tag, flush=True)
        quantize("vae_decoder_%s" % tag, tag, OVR)
        return 0
    sys.exit("未知模式 %s" % cmd)


sys.exit(main())
