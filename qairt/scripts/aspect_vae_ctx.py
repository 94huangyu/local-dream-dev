# -*- coding: utf-8 -*-
"""从量化后的 VAE DLC 建**单图** context。

single 形态的回退路径需要它：mg 形态若在设备上过不了 G-MG，
就只能退回「每比例一套单图 context」，那时每个比例都得有自己的 VAE context。

🔴 必带 `--config_file`（#95）：漏了会**静默**编成 dspArch 68 / vtcm 4MB，
   产物看着正常、跑起来不对。本项目已因此作废过一整轮结论。

用法: python aspect_vae_ctx.py <tag>        # tag 如 1184x896；1x1 用 "deployed"
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


def main():
    tag = sys.argv[1]
    dlc = os.path.join(AS, "vaeq_%s_quantized.dlc" % tag)
    if not os.path.isfile(dlc):
        print("🔴 缺量化 DLC: %s（先跑 vae_quant_aspect.py %s）" % (dlc, tag))
        return 1
    gname = "vaeq_%s_fp32" % tag          # 与 aspect_multigraph_build.py 的命名一致
    od = os.path.join(AS, "ctx_vae_%s" % tag)
    if os.path.isdir(od):
        shutil.rmtree(od)
    os.makedirs(od)

    det = {"graphs": [{"graph_names": [gname], "vtcm_mb": 8}],
           "devices": [{"soc_model": 69, "dsp_arch": "v79"}]}
    dp, ep = os.path.join(od, "d.json"), os.path.join(od, "e.json")
    json.dump(det, open(dp, "w"), indent=1)
    json.dump({"backend_extensions": {
        "shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
        "config_file_path": dp}}, open(ep, "w"), indent=1)

    print("--- VAE context %s ---" % tag, flush=True)
    t0 = time.time()
    log = os.path.join(od, "build.log")
    with open(log, "w", encoding="utf-8", errors="replace") as fo:
        r = subprocess.run(
            [os.path.join(BIN, "qnn-context-binary-generator.exe"),
             "--backend", os.path.join(LIB, "QnnHtp.dll"),
             "--dlc_path", dlc, "--binary_file", "vae_%s" % tag,
             "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ep],
            stdout=fo, stderr=subprocess.STDOUT, timeout=7200)
    txt = open(log, encoding="utf-8", errors="replace").read()
    if r.returncode != 0:
        print("  FAIL rc=%d" % r.returncode)
        print(txt[-1500:])
        return 1
    cand = [f for f in os.listdir(od) if f.endswith(".bin")]
    if len(cand) != 1:
        print("  🔴 产物不唯一 %r" % cand)
        return 1
    binp = os.path.join(od, cand[0])

    # 门：产物里的图名必须就是我们要的那个（不是「建出来了」就算数）
    info = os.path.join(od, "info.json")
    subprocess.run([UTIL, "--context_binary", binp, "--json_file", info],
                   capture_output=True, text=True)
    if not os.path.isfile(info):
        print("  🔴 元数据 dump 失败，无法确认图名")
        return 1
    gs = [g["info"]["graphName"] for g in json.load(open(info, encoding="utf-8"))
          ["info"]["graphs"]]
    if gs != [gname]:
        print("  🔴 图名不符：实得 %r 期望 %r" % (gs, [gname]))
        return 1
    print("  rc=0  %.1f 分钟  %s  %.1f MiB  图=%r"
          % ((time.time() - t0) / 60.0, cand[0], os.path.getsize(binp) / 1048576.0, gs))
    return 0


sys.exit(main())
