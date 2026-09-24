# -*- coding: utf-8 -*-
"""D2-H1 · S5：把「A 变体」（只删图像流的 ScatterElements）推到交付形态 —— 五比例 mg 五图 context。

依据：`EXP_PLAN_P2_SPEED.md` §6.14 —— A 变体在两组输入下 9 个输出与部署版**逐字节相同**，加速器 0.86 倍。

🔴 三条硬性细节（错一条就白跑或装载失败）：
 1. **图名必须与部署版逐字一致**：mg 的图名取自 fp32 DLC 的文件名主干，而 app 契约按图名选比例
    （`part1a_fp16_L80_fp32` / `part1a_<比例>_fp32`）⇒ 本脚本按部署命名产出 DLC，建完读回图名逐个核对。
 2. **共享 spill-fill 组大小**：建完必须现读 `spillFillBufferSize` 并与设备 marker（331,415,552）比较，
    大于就必须更新 marker，否则 part1a 装载失败（§15.45.3 教训 1）。
 3. 建图配置照抄 mg 那次：`graphs[].vtcm_mb=8` 逐图 + `devices[soc_model 69/dsp_arch v79]` + `context.weight_sharing_enabled`。

用法: python scripts/p2_h1_deliver_build.py [--skip-mg]
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import p2_h1_build as B  # 复用 sh()/free_gb()/路径常量

ONNX_DIR = B.ONNX_DIR
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_deliver")
MG_REF = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "aspect_mg", "part1a", "part1a_mg.SM8750.bin")
MARKER_BYTES = 331415552          # 设备上 files/models/ZIMAGE/SHARE_SPILLFILL 的当前值
# (部署图名主干, 源 ONNX, 输入形状)
SPECS = [
    ("part1a_fp16_L80", "transformer_part1a_clip_L80", (128, 128, 80)),
    ("part1a_1184x896", "transformer_part1a_clip_L80_1184x896", (112, 148, 80)),
    ("part1a_896x1184", "transformer_part1a_clip_L80_896x1184", (148, 112, 80)),
    ("part1a_1280x720", "transformer_part1a_clip_L80_1280x720", (90, 160, 80)),
    ("part1a_720x1280", "transformer_part1a_clip_L80_720x1280", (160, 90, 80)),
]


def shapes(lh, lw, cap):
    return [("latents", "1,16,%d,%d" % (lh, lw)), ("timestep", "1"),
            ("caption", "1,%d,2560" % cap), ("cap_pad_mask", "1,%d" % cap)]


def main():
    os.makedirs(OUT, exist_ok=True)
    qdlcs = []
    for gname, stem, (lh, lw, cap) in SPECS:
        surg = os.path.join(ONNX_DIR, stem + "_noscatA.onnx")
        if not os.path.isfile(surg):
            B.sh("手术 " + gname, [B.PY, os.path.join(B.HERE, "p2_h1_surgery.py"),
                                   "--src", os.path.join(ONNX_DIR, stem + ".onnx"),
                                   "--only", "node_select_scatter"], 1800)
        fdlc = os.path.join(OUT, gname + "_fp32.dlc")      # 🔴 文件名决定图名
        qdlc = os.path.join(OUT, gname + "_quantized.dlc")
        if not os.path.isfile(qdlc):
            if not os.path.isfile(fdlc):
                cmd = [B.PY, B.TOOL, "qairt-converter", "--input_network", surg,
                       "--output_path", fdlc, "--float_bitwidth", "16",
                       "--quantization_overrides", B.OVR]
                for n, d in shapes(lh, lw, cap):
                    cmd += ["--source_model_input_shape", n, d]
                B.sh("转换 " + gname, cmd, 14400)
            B.sh("量化 " + gname, [B.PY, B.TOOL, "qairt-quantizer", "--input_dlc", fdlc,
                                   "--output_dlc", qdlc, "--weights_bitwidth", "8",
                                   "--bias_bitwidth", "32", "--param_quantizer_calibration", "min-max",
                                   "--use_per_row_quantization", "--keep_weights_quantized",
                                   "--enable_float_fallback"], 21600)
        if os.path.isfile(fdlc):
            os.remove(fdlc)        # fp32 DLC 是 5 分钟可再生的中间品（DISK_INVENTORY §1）
        print("  ✅ %-16s 量化 DLC %.2f GB" % (gname, os.path.getsize(qdlc) / 1e9), flush=True)
        qdlcs.append(qdlc)
    if "--skip-mg" in sys.argv:
        return 0

    od = os.path.join(OUT, "ctx_mg")
    binp = os.path.join(od, "part1a_mg_noscatA.SM8750.bin")
    if not os.path.isfile(binp):
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        gnames = [g for g, _, _ in SPECS]
        det = {"graphs": [{"graph_names": [g + "_fp32"], "vtcm_mb": 8} for g in gnames],
               "devices": [{"soc_model": 69, "dsp_arch": "v79"}],
               "context": {"weight_sharing_enabled": True}}
        dp, ep = os.path.join(od, "d.json"), os.path.join(od, "e.json")
        json.dump(det, open(dp, "w"), indent=1)
        json.dump({"backend_extensions": {
            "shared_library_path": os.path.join(B.LIB, "QnnHtpNetRunExtensions.dll"),
            "config_file_path": dp}}, open(ep, "w"), indent=1)
        o = B.sh("mg 建图（五图）", [os.path.join(B.BIN, "qnn-context-binary-generator.exe"),
                                     "--backend", os.path.join(B.LIB, "QnnHtp.dll"),
                                     "--dlc_path", ",".join(qdlcs),
                                     "--binary_file", "part1a_mg_noscatA", "--output_dir", od,
                                     "--htp_socs", "sm8750", "--config_file", ep], 43200)
        if "available PD" in o:
            raise SystemExit("🔴 撞 PD 红线（#57）")
    if not os.path.isfile(binp):
        raise SystemExit("🔴 未产出 %s" % binp)

    def dump_info(ctx, out):
        """🔴 返回码必须检查（约束 11·再补）：不查的话工具失败只会在后面变成
        一句莫名其妙的 FileNotFoundError，排查方向会被带偏。"""
        r = subprocess.run([os.path.join(B.BIN, "qnn-context-binary-utility.exe"),
                            "--context_binary", ctx, "--json_file", out],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
        if r.returncode != 0 or not os.path.isfile(out):
            sys.stderr.write((r.stdout or "")[-1200:] + (r.stderr or "")[-800:])
            raise SystemExit("🔴 读元数据失败 rc=%d：%s" % (r.returncode, ctx))
        return out

    jf = dump_info(binp, os.path.join(OUT, "info_mg_noscatA.json"))
    jr = dump_info(MG_REF, os.path.join(OUT, "info_mg_deployed.json"))

    def graphs(p):
        j = json.load(open(p, encoding="utf-8", errors="replace"))
        return {g["info"]["graphName"]: g["info"] for g in j["info"]["graphs"]}, \
               j["info"]["contextMetadata"]["info"].get("dspArch")
    gn, arch = graphs(jf)
    gr, archr = graphs(jr)
    print("\n== 交付前核对 ==")
    print("  dspArch 新 %s / 部署 %s %s" % (arch, archr, "✅" if arch == archr else "🔴"))
    ok = arch == archr
    if set(gn) != set(gr):
        print("  🔴 图名集合不一致\n     新：%s\n     部署：%s" % (sorted(gn), sorted(gr)))
        ok = False
    else:
        print("  ✅ 五个图名与部署逐个一致：%s" % sorted(gn))
    mx = 0
    for g in sorted(gn):
        a = gn[g]["graphBlobInfo"]["info"]
        b = gr[g]["graphBlobInfo"]["info"]
        mx = max(mx, a["spillFillBufferSize"])
        same = (a["vtcmSize"], a["optimizationLevel"], a["htpDlbc"]) == (b["vtcmSize"], b["optimizationLevel"], b["htpDlbc"])
        print("    %-18s vtcm %d/%d｜spillFill 新 %d / 部署 %d %s"
              % (g, a["vtcmSize"], b["vtcmSize"], a["spillFillBufferSize"], b["spillFillBufferSize"],
                 "✅" if same else "🔴 装置不一致"))
        ok &= same
    print("  新 context 的最大 spillFill = %d｜设备 marker = %d ⇒ %s"
          % (mx, MARKER_BYTES, "✅ 无需改 marker" if mx <= MARKER_BYTES else "🔴 **必须把 marker 调大到 %d**" % mx))
    print("  文件大小 新 %.3f GB / 部署 %.3f GB" % (os.path.getsize(binp) / 1e9, os.path.getsize(MG_REF) / 1e9))
    print("\n%s" % ("✅ S5 完成，可进 S6（设备：段级逐字节复核 → app 交付门）" if ok else
                    "🔴 核对未过，不得交付"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
