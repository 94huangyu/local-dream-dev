# -*- coding: utf-8 -*-
"""D2-H1 · 微探针（`EXP_PLAN_P2_SPEED.md` §6.12，判据事前锁定）。

目的：用**最小装置**回答「删掉那个数学恒等的 ScatterElements 为什么会毁数值」，并筛出可用的替代形态。
这是我该在做完整重建**之前**做的一步（见 §6.11 的 G-backend②）。

图形（与真实 part1a 同形状，无权重依赖）：
    in0[4096,3840] , in1[80,3840] --Concat(axis=0)--> [4176,3840] --Reshape--> [1,4176,3840]
      --(变体)--> --ReduceMean(axis=-1, keepdims)--> out[1,4176,1]

变体：V0 带 ScatterElements（对照）｜V1 直接删（复现 S4）｜V2 Add(0)｜V3 Mul(1)｜V4 同形 Reshape

🔴 为什么用 ReduceMean 当消费者、用「行号」当输入：
  真实图里出问题的下游是 RmsNorm（**按行归约**）；把每行填成该行的行号后，
  ReduceMean 的输出就是「行指纹」，**任何行错位一眼可见**，不需要解释数值。
  encoding 统一取 scale=1 / offset=0 ⇒ code 即数值，FP32 参考可用 numpy 直接算，不必跑 ORT。

用法:
  python scripts/p2_h1_probe.py build     # 造 5 个变体的 ONNX -> DLC -> context（纯宿主）
  python scripts/p2_h1_probe.py run       # 设备上逐个跑并比对（需插线，约 3 分钟）
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import p2_d1_profile as P  # noqa: E402

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
BIN = os.path.join(SDK, "bin", "x86_64-windows-msvc")
LIB = os.path.join(SDK, "lib", "x86_64-windows-msvc")
TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qairt_tool.py")
WORK = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "h1_probe")
OUT = os.path.join(P.REPO, "logs", "p2_20260917", "probe")
DEV = "/data/local/tmp/p2probe"
IMG, CAP, DIM = 4096, 80, 3840
UNI = IMG + CAP
VARIANTS = ["V0_scatter", "V1_none", "V2_add0", "V3_mul1", "V4_reshape"]


def build_onnx(variant, path):
    """构图。除「变体」这一段外，五份完全相同。"""
    nodes, inits = [], []
    nodes.append(helper.make_node("Concat", ["in0", "in1"], ["cat"], axis=0, name="node_cat"))
    inits.append(numpy_helper.from_array(np.array([1, UNI, DIM], dtype=np.int64), "shape3"))
    nodes.append(helper.make_node("Reshape", ["cat", "shape3"], ["resh"], name="node_reshape"))
    src = "resh"
    if variant == "V0_scatter":
        inits.append(numpy_helper.from_array(np.zeros((1, UNI, DIM), dtype=np.float32), "fill"))
        inits.append(numpy_helper.from_array(np.zeros((1, UNI, DIM), dtype=np.int32), "idx"))
        nodes.append(helper.make_node("ScatterElements", ["fill", "idx", src], ["mid"], axis=0, name="node_scatter"))
        src = "mid"
    elif variant == "V2_add0":
        inits.append(numpy_helper.from_array(np.zeros((1, 1, 1), dtype=np.float32), "zero"))
        nodes.append(helper.make_node("Add", [src, "zero"], ["mid"], name="node_add0"))
        src = "mid"
    elif variant == "V3_mul1":
        inits.append(numpy_helper.from_array(np.ones((1, 1, 1), dtype=np.float32), "one"))
        nodes.append(helper.make_node("Mul", [src, "one"], ["mid"], name="node_mul1"))
        src = "mid"
    elif variant == "V4_reshape":
        inits.append(numpy_helper.from_array(np.array([1, UNI, DIM], dtype=np.int64), "shape3b"))
        nodes.append(helper.make_node("Reshape", [src, "shape3b"], ["mid"], name="node_reshape2"))
        src = "mid"
    nodes.append(helper.make_node("ReduceMean", [src], ["out"], axes=[-1], keepdims=1, name="node_rowmean"))
    g = helper.make_graph(nodes, "probe_" + variant,
                          [helper.make_tensor_value_info("in0", TensorProto.FLOAT, [IMG, DIM]),
                           helper.make_tensor_value_info("in1", TensorProto.FLOAT, [CAP, DIM])],
                          [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, UNI, 1])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, path)
    return [n.name for n in nodes]


def overrides(path, names):
    """所有张量同一组 encoding：scale=1 / offset=0 ⇒ code 即数值，便于「行指纹」判读。
    也满足 HTP 对 ScatterElements 的『in0/in2/out0 量化必须相同』（HtpOpDefSupplement）。"""
    e = [{"bitwidth": 16, "min": 0.0, "max": 65535.0, "scale": 1.0, "offset": 0.0, "is_symmetric": "False"}]
    act = {n: e for n in names}
    json.dump({"version": "0.6.1", "activation_encodings": act, "param_encodings": {}},
              open(path, "w"), indent=1)


def calib_files(d):
    """一份校准样本（FP32）：每行填行号 ⇒ 与设备侧的原生输入语义一致。返回 input_list 路径。"""
    cd = os.path.join(d, "calib")
    os.makedirs(cd, exist_ok=True)
    a = np.tile(np.arange(IMG, dtype=np.float32).reshape(-1, 1), (1, DIM))
    b = np.tile((np.arange(CAP, dtype=np.float32) + IMG).reshape(-1, 1), (1, DIM))
    pa, pb = os.path.join(cd, "in0.raw"), os.path.join(cd, "in1.raw")
    a.tofile(pa)
    b.tofile(pb)
    lst = os.path.join(cd, "list.txt")
    open(lst, "w").write("in0:=%s in1:=%s\n" % (pa, pb))
    return lst


def sh(tag, cmd, timeout=3600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    o = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise SystemExit("🔴 %s rc=%d\n%s" % (tag, r.returncode, o[-1500:]))
    return o, time.time() - t0


def cmd_build():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for v in VARIANTS:
        d = os.path.join(WORK, v)
        os.makedirs(d, exist_ok=True)
        onnx_p = os.path.join(d, "%s.onnx" % v)
        ovr_p = os.path.join(d, "ovr.json")
        names = build_onnx(v, onnx_p)
        overrides(ovr_p, ["in0", "in1", "cat", "resh", "mid", "out"])
        fdlc, qdlc = os.path.join(d, "f.dlc"), os.path.join(d, "q.dlc")
        # 🔴 第一版用 overrides + float_fallback：未被 overrides 覆盖的张量走 FP16，
        #    建图直接失败（`resh_converted_QNN_DATATYPE_FLOAT_16 ... not sufficiently tiled to fit in TCM`）。
        #    探针只需**结构**忠实，不需要与真实 encoding 一致 ⇒ 改用一份校准样本，让量化器给所有张量定 encoding，
        #    且**不开 float fallback**（全定点，与真实图的 A16 路径同类）。
        cal = calib_files(d)
        _, t1 = sh("转换 " + v, [sys.executable, TOOL, "qairt-converter", "--input_network", onnx_p,
                                 "--output_path", fdlc,
                                 "--source_model_input_shape", "in0", "%d,%d" % (IMG, DIM),
                                 "--source_model_input_shape", "in1", "%d,%d" % (CAP, DIM)])
        _, t2 = sh("量化 " + v, [sys.executable, TOOL, "qairt-quantizer", "--input_dlc", fdlc,
                                 "--output_dlc", qdlc, "--input_list", cal,
                                 "--act_bitwidth", "16", "--weights_bitwidth", "8", "--bias_bitwidth", "32",
                                 "--act_quantizer_calibration", "min-max",
                                 "--param_quantizer_calibration", "min-max"])
        # 生效门：读回 DLC 算子表，确认「变体那个算子」真的还在（转换器可能把恒等消除）
        info, _ = sh("dlcinfo " + v, [sys.executable, TOOL, "snpe-dlc-info", "-i", qdlc])
        open(os.path.join(d, "dlcinfo.txt"), "w", encoding="utf-8").write(info)
        kinds = {"V0_scatter": "ScatterElements", "V2_add0": "Eltwise", "V3_mul1": "Eltwise", "V4_reshape": "Reshape"}
        present = kinds.get(v, "") and (kinds[v] in info)
        od = os.path.join(d, "ctx")
        if os.path.isdir(od):
            shutil.rmtree(od)
        os.makedirs(od)
        det, ext = os.path.join(od, "d.json"), os.path.join(od, "e.json")
        json.dump({"devices": [{"soc_model": 69, "dsp_arch": "v79"}]}, open(det, "w"), indent=1)
        json.dump({"backend_extensions": {"shared_library_path": os.path.join(LIB, "QnnHtpNetRunExtensions.dll"),
                                          "config_file_path": det}}, open(ext, "w"), indent=1)
        _, t3 = sh("建图 " + v, [os.path.join(BIN, "qnn-context-binary-generator.exe"), "--backend",
                                 os.path.join(LIB, "QnnHtp.dll"), "--dlc_path", qdlc, "--binary_file", v,
                                 "--output_dir", od, "--htp_socs", "sm8750", "--config_file", ext])
        b = [x for x in os.listdir(od) if x.endswith(".bin")][0]
        sz = os.path.getsize(os.path.join(od, b))
        rows.append((v, t1 + t2 + t3, sz, present))
        print("  %-12s 建完 %5.1f s｜context %5.1f MB｜变体算子在 DLC 里：%s"
              % (v, t1 + t2 + t3, sz / 1e6, "✅" if present or v == "V1_none" else "🔴 被消除"), flush=True)
    json.dump([{"variant": v, "sec": s, "bytes": b, "op_present": bool(p)} for v, s, b, p in rows],
              open(os.path.join(OUT, "build.json"), "w", encoding="utf-8"), indent=1)
    return 0


def cmd_run():
    import lab_dev
    lab_dev.require_online("probe")
    # 输入：每行填该行行号（code 即数值）⇒ 输出 = 行指纹
    os.makedirs(OUT, exist_ok=True)
    a = np.tile(np.arange(IMG, dtype=np.uint16).reshape(-1, 1), (1, DIM))
    b = np.tile((np.arange(CAP, dtype=np.uint16) + IMG).reshape(-1, 1), (1, DIM))
    pa, pb = os.path.join(OUT, "in0.raw"), os.path.join(OUT, "in1.raw")
    a.tofile(pa)
    b.tofile(pb)
    lab_dev.sh("mkdir -p %s" % DEV)
    for p_, n in ((pa, "in0"), (pb, "in1")):
        if lab_dev.sh("stat -c %%s %s/%s.raw 2>/dev/null; true" % (DEV, n)).strip() != str(os.path.getsize(p_)):
            lab_dev.push(p_, "%s/%s.raw" % (DEV, n))
    expect = np.concatenate([np.arange(IMG), np.arange(CAP) + IMG]).astype(np.float64)   # FP32 参考（numpy 直算）
    res = {}
    for v in VARIANTS:
        src = os.path.join(WORK, v, "ctx")
        b_ = [x for x in os.listdir(src) if x.endswith(".bin")][0]
        dev = "%s/%s.bin" % (DEV, v)
        if lab_dev.sh("stat -c %%s %s 2>/dev/null; true" % dev).strip() != str(os.path.getsize(os.path.join(src, b_))):
            lab_dev.push(os.path.join(src, b_), dev)
        od = "%s/o_%s" % (DEV, v)
        lab_dev.sh("rm -rf %s && mkdir -p %s && printf '%%s\\n' 'in0:=%s/in0.raw in1:=%s/in1.raw' > %s/list.txt"
                   % (od, od, DEV, DEV, od))
        o = lab_dev.sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
                       "./qnn-net-run --retrieve_context %s --backend libQnnHtp.so --input_list %s/list.txt "
                       "--output_dir %s --perf_profile burst --use_native_input_files --use_native_output_files "
                       "--profiling_level detailed --num_inferences 2 --log_level error; echo RC=$?"
                       % (P.T, P.T, P.T, dev, od, od), timeout=1800)
        if "RC=0" not in o:
            print("  🔴 %s 运行失败：%s" % (v, " ".join(o.split())[-200:]))
            res[v] = None
            continue
        f = "%s/Result_0/out_native.raw" % od
        hp = os.path.join(OUT, "out_%s.raw" % v)
        P.pull_robust(f, hp)
        got = np.fromfile(hp, dtype=np.uint16).astype(np.float64)
        logs = [x for x in lab_dev.sh("ls %s; true" % od).split() if x.startswith("qnn-profiling-data")]
        hd = os.path.join(OUT, v)
        os.makedirs(hd, exist_ok=True)
        for lg in logs:
            P.pull_robust("%s/%s" % (od, lg), os.path.join(hd, lg))
        txt = "".join(P.view(os.path.join(hd, lg)) for lg in logs)
        cyc = P.node_cycles(txt)
        res[v] = {"out": got, "cycles": cyc, "total_cycles": sum(cyc.values())}
        wrong = int((np.abs(got - expect) > 0.5).sum())
        print("  %-12s 行指纹错位 %5d / %d 行｜总 cycles %12d" % (v, wrong, UNI, sum(cyc.values())), flush=True)
    print("\n== 门 1（复现门）==")
    v0, v1 = res.get("V0_scatter"), res.get("V1_none")
    if v0 and v1:
        ok0 = np.abs(v0["out"] - expect).max() <= 0.5
        ok1 = np.abs(v1["out"] - expect).max() <= 0.5
        same01 = np.array_equal(v0["out"], v1["out"])
        print("  V0 与 FP32 参考一致：%s｜V1 与参考一致：%s｜V0==V1：%s" % (ok0, ok1, same01))
        rep = ok0 and not ok1
        print("  ⇒ %s" % ("🟢 复现了（V0 对、V1 错）⇒ 可用本探针筛变体" if rep else
                          "🔴 未复现 ⇒ 探针规模不足，不得用它筛变体（按 §6.12 扩到含第一个 unified block）"))
    else:
        print("  🔴 V0/V1 缺结果")
    print("\n== 门 2（变体筛选）==")
    for v in ("V2_add0", "V3_mul1", "V4_reshape"):
        r = res.get(v)
        if not r or not v0:
            print("  %s: 无结果" % v)
            continue
        same = np.array_equal(r["out"], v0["out"])
        ratio = r["total_cycles"] / max(1, v0["total_cycles"])
        print("  %-12s 与 V0 逐字节相同：%s｜总 cycles 比 V0 %.3f ⇒ %s"
              % (v, same, ratio, "🟢 可用" if same and ratio <= 0.8 else ("🟡 数值对但不够省" if same else "🔴 数值不同")))
    json.dump({k: ({"total_cycles": v["total_cycles"],
                    "row_mismatch": int((np.abs(v["out"] - expect) > 0.5).sum())} if v else None)
               for k, v in res.items()}, open(os.path.join(OUT, "run.json"), "w", encoding="utf-8"), indent=1)
    print("\n设备临时目录保留：%s（探针很小，约 %d MB）" % (DEV, 0))
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"build": cmd_build, "run": cmd_run}.get(c, lambda: (_ for _ in ()).throw(SystemExit(__doc__)))())
