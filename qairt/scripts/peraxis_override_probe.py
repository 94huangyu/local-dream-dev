"""P0-B 前置门：--quantization_overrides 能否注入 per-axis encoding？

方案：scripts/EXP_PLAN_PERAXIS_OVERRIDE.md（判据事前锁定）
用最小已知样本（x[1,8] @ W[8,4]），注入 4 个我自己指定的 scale，看能不能原样落到 DLC 里。
"""
import os, sys, json, glob, shutil, subprocess
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
WORK = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "peraxis_probe")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable
K, N = 8, 4
SCALES = [0.001, 0.002, 0.004, 0.008]      # 4 个互不相同、由我指定的 scale


def sh(cmd, tag):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800)
    out = (r.stdout or "") + (r.returncode and (r.stderr or "") or "")
    return r.returncode, out + (r.stderr or "")


def build_onnx(path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    rng = np.random.default_rng(7)
    W = rng.standard_normal((K, N)).astype(np.float32)
    node = helper.make_node("MatMul", ["x", "W"], ["y"], name="node_matmul")
    g = helper.make_graph([node], "peraxis_probe",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, K])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, N])],
                          [numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, path)
    return W


def make_calib(d, n=8):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(11)
    lines = []
    for i in range(n):
        p = os.path.join(d, "s%d.raw" % i)
        rng.standard_normal(K).astype(np.float32).tofile(p)
        lines.append("x:=" + p)
    lst = os.path.join(d, "input_list.txt")
    open(lst, "w").write("\n".join(lines) + "\n")
    return lst


def overrides(kind, W):
    """三条臂的 overrides 文件内容"""
    if kind == "C":                                  # V 门：0.6.1 per-tensor（已知可用）
        s = SCALES[0]
        return {"activation_encodings": {}, "version": "0.6.1",
                "param_encodings": {"W": [{"bitwidth": 8, "scale": s, "offset": -128,
                                           "min": -128 * s, "max": 127 * s,
                                           "is_symmetric": "True"}]}}
    if kind == "A":                                  # 0.6.1 AIMET per-channel（多元素列表）
        return {"activation_encodings": {}, "version": "0.6.1",
                "param_encodings": {"W": [{"bitwidth": 8, "scale": s, "offset": -128,
                                           "min": -128 * s, "max": 127 * s,
                                           "is_symmetric": "True"} for s in SCALES]}}
    if kind == "B":                                  # 2.0.0 per-channel（y_scale + axis）
        return {"version": "2.0.0",
                "encodings": [{"name": "W", "output_dtype": "int8",
                               "y_scale": list(SCALES),
                               "y_zero_point": [0] * N, "axis": 1}]}
    raise ValueError(kind)


def dlc_scales(dlc, tag):
    """用 qairt-dlc-to-json 读回 W 的 encoding，返回 (scale 列表, 原始条目)"""
    js = dlc.replace(".dlc", "_dump.json")
    rc, out = sh([PY, TOOL, "qairt-dlc-to-json", "-i", dlc, "-o", js], tag)
    if rc != 0 or not os.path.exists(js):
        return None, "dlc-to-json rc=%d\n%s" % (rc, out[-800:])
    d = json.load(open(js, encoding="utf-8"))
    # 2026-08-21：DLC dump 里张量是【按名字做键】的 /graph/tensors/<name>/quant_params，
    # 不存在 "name" 字段。第一版读取器按 name 字段找，找不到 => 误判 V2 门不过。
    qp = d.get("graph", {}).get("tensors", {}).get("W", {}).get("quant_params", {})
    found = [qp]
    scales = []
    if "scale_offset" in qp:
        scales = [float(qp["scale_offset"]["scale"])]
    for key in ("axis_scale_offset", "bw_axis_scale_offset"):
        if key in qp:
            scales = [float(e["scale"]) for e in qp[key].get("scale_offsets", [])]
    return scales, json.dumps(found, ensure_ascii=False)[:900]


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    onnx_p = os.path.join(WORK, "probe.onnx")
    W = build_onnx(onnx_p)
    lst = make_calib(os.path.join(WORK, "calib"))
    print("[建图] %s  W%s  注入 scale=%s" % (onnx_p, list(W.shape), SCALES))

    results = {}
    for kind in ("C", "A", "B"):
        print("")
        print("=" * 70)
        print("臂 %s" % kind)
        ov = os.path.join(WORK, "ovr_%s.json" % kind)
        json.dump(overrides(kind, W), open(ov, "w", encoding="utf-8"), indent=1)
        fdlc = os.path.join(WORK, "f_%s.dlc" % kind)
        qdlc = os.path.join(WORK, "q_%s.dlc" % kind)

        rc, out = sh([PY, TOOL, "qairt-converter", "--input_network", onnx_p,
                      "--output_path", fdlc, "--quantization_overrides", ov,
                      "--float_bitwidth", "32"], kind)          # #46：必须显式 32
        proc = [l.strip() for l in out.splitlines() if "quantization encoding" in l.lower()]
        print("  converter rc=%d" % rc)
        for l in proc:
            print("    %s" % l)
        if rc != 0:
            print("  ❌ converter 失败：\n%s" % out[-900:])
            results[kind] = ("CONVERTER_FAIL", out[-400:])
            continue
        # 解析必须锚到 "Processed N quantization encodings"，
        # 不能"取行内最大数字"——日志行里 278 是进程字段（2026-08-21 踩过）
        import re as _re
        n_proc = 0
        for l in proc:
            mm = _re.search(r"Processed\s+(\d+)\s+quantization encodings", l)
            if mm:
                n_proc = int(mm.group(1))
        print("  [V1] Processed N = %d  %s" % (n_proc, "OK" if n_proc >= 1 else "❌ 静默失效(#45)"))

        rc, out = sh([PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", qdlc,
                      "--input_list", lst, "--weights_bitwidth", "8", "--act_bitwidth", "16",
                      "--bias_bitwidth", "32", "--act_quantizer_calibration", "min-max",
                      "--param_quantizer_calibration", "min-max"], kind)
        print("  quantizer rc=%d" % rc)
        if rc != 0:
            print("  ❌ quantizer 失败：\n%s" % out[-900:])
            results[kind] = ("QUANTIZER_FAIL", out[-400:])
            continue

        sc, raw = dlc_scales(qdlc, kind)
        if sc is None:
            print("  ❌ 读回失败：%s" % raw)
            results[kind] = ("DUMP_FAIL", raw)
            continue
        uniq = sorted(set(round(x, 12) for x in sc))
        print("  DLC 里 W 的 scale：%s" % sc)
        print("  互不相同的个数 n_scale = %d" % len(uniq))
        print("  原始条目：%s" % raw[:400])
        results[kind] = (len(uniq), sc)

    # ---------------- 判据 ----------------
    print("")
    print("=" * 70)
    print("=== V2 门：臂 C（per-tensor，已知可用）必须注入成功 ===")
    cn, cs = results.get("C", (None, None))
    okC = isinstance(cn, int) and cn >= 1 and cs and any(
        abs(x - SCALES[0]) / SCALES[0] < 1e-6 for x in cs)
    print("  臂 C n_scale=%s  scale=%s   期望含 %.6f   %s"
          % (cn, cs, SCALES[0], "✅" if okC else "❌"))
    if not okC:
        print("  ⇒ 装置本身有问题，A/B 结果一律不可采信（方案 §五）")
        return

    print("")
    print("=== 主判据：per-axis 能否注入 ===")
    passed = None
    for kind in ("A", "B"):
        n, sc = results.get(kind, (None, None))
        if not isinstance(n, int):
            print("  臂 %s: %s ⇒ 未通过" % (kind, n))
            continue
        exact = (n == N and sc is not None and
                 all(any(abs(x - s) / s < 1e-6 for x in sc) for s in SCALES))
        print("  臂 %s: n_scale=%d  逐个吻合注入值=%s  %s"
              % (kind, n, exact, "✅ 通过" if exact else "❌"))
        if exact:
            passed = kind
    print("")
    if passed:
        print("  判定: ✅ **per-axis 可注入**（格式 = 臂 %s）⇒ P0-B 前置门通过" % passed)
    else:
        print("  判定: ❌ **per-axis 不可经 overrides 注入** ⇒ P0-B 的 exact replay 不成立，必须改设计")
        print("        （注意：'有 4 个 scale 但值是量化器自己算的' 不算通过 —— 方案 §四）")


if __name__ == "__main__":
    main()
