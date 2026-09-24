# -*- coding: utf-8 -*-
"""把 1:1 与各比例的 VAE 合成一个**多图共享权重** context。

## 为什么单独一个脚本
`aspect_multigraph_build.py` 的 `SEGS` 只有四个 transformer 段，**VAE 不在里面**。
现网那份 `vae_mg` 只含 2 个图（1:1 + 1184x896），而五比例的 mg 契约需要 5 个 ——
不补这一步，`build_aspect_contract.py --form=mg` 会在 G8（图名不在 .bin 里）判负。
🔴 这是我在编排里漏掉的一环，靠「契约生成器会硬失败」兜住了，但那已经浪费了一轮。

## 判据
V1 产物唯一
V2 图集合 == 期望的 5 个图名（不符即非零退出，不是打印一行警告）
V3 建图日志里不得出现 "available PD"（#57 红线）

用法: python aspect_vae_mg.py <tag>[,<tag>...]     # 1:1 隐式在第一位
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
UTIL = os.path.join(BIN, "qnn-context-binary-utility.exe")
AS = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect")
MG = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg")
VD = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence",
                  "dlc_pipeline", "vae_decoder")


def main():
    tags = [x for x in sys.argv[1].split(",") if x]
    d11 = os.path.join(VD, "vae_decoder_quantized.dlc")
    dlcs = [d11] + [os.path.join(AS, "vaeq_%s_quantized.dlc" % t) for t in tags]
    for f in dlcs:
        if not os.path.isfile(f):
            return print("🔴 缺 %s" % f) or 1
    gnames = ["vae_decoder_fp32"] + ["vaeq_%s_fp32" % t for t in tags]
    if len(set(gnames)) != len(gnames):
        return print("🔴 图名重复 %r" % gnames) or 1

    od = os.path.join(MG, "vae")
    if os.path.isdir(od):
        shutil.rmtree(od)
    os.makedirs(od)
    det = {"graphs": [{"graph_names": [g], "vtcm_mb": 8} for g in gnames],
           "devices": [{"soc_model": 69, "dsp_arch": "v79"}],
           "context": {"weight_sharing_enabled": True}}
    dp, ep = os.path.join(od, "d.json"), os.path.join(od, "e.json")
    json.dump(det, open(dp, "w"), indent=1)
    json.dump({"backend_extensions": {
        "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
        "config_file_path": dp}}, open(ep, "w"), indent=1)

    print("--- VAE mg : %d 图 %s ---" % (len(gnames), " + ".join(gnames)), flush=True)
    t0 = time.time()
    log = os.path.join(od, "build.log")
    with open(log, "w", encoding="utf-8", errors="replace") as fo:
        r = subprocess.run(
            [os.path.join(BIN, "qnn-context-binary-generator.exe"),
             "--backend", os.path.join(LIB, "QnnHtp.dll"),
             "--dlc_path", ",".join(dlcs), "--binary_file", "vae_mg",
             "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ep],
            stdout=fo, stderr=subprocess.STDOUT, timeout=28800)
    txt = open(log, encoding="utf-8", errors="replace").read()
    if "available PD" in txt:                                     # V3
        print("  🔴 建图期撞 PD 红线（#57）")
        return 1
    if r.returncode != 0:
        print("  FAIL rc=%d\n%s" % (r.returncode, txt[-1500:]))
        return 1
    cand = [f for f in os.listdir(od) if f.endswith(".bin")]
    if len(cand) != 1:                                            # V1
        print("  🔴 产物不唯一 %r" % cand)
        return 1
    binp = os.path.join(od, cand[0])
    info = os.path.join(od, "info.json")
    subprocess.run([UTIL, "--context_binary", binp, "--json_file", info],
                   capture_output=True, text=True)
    if not os.path.isfile(info):
        print("  🔴 元数据 dump 失败")
        return 1
    d = json.load(open(info, encoding="utf-8"))
    gn = [g["info"]["graphName"] for g in d["info"]["graphs"]]
    s1 = os.path.getsize(os.path.join(VD, "vae_decoder_ctx_sm8750.SM8750.bin"))
    sz = os.path.getsize(binp)
    print("  rc=0  %.1f 分钟  %.1f MiB（单图现网 %.1f MiB，比值 %.4f）"
          % ((time.time() - t0) / 60.0, sz / 1048576.0, s1 / 1048576.0, sz / float(s1)))
    print("  图 = %r" % gn)
    if sorted(gn) != sorted(gnames):                              # V2
        print("  🔴 图集合不符，预期 %r" % gnames)
        return 1
    print("  ✅ V1/V2/V3 全过")
    return 0


sys.exit(main())
