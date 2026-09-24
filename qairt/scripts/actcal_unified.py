"""EXP_PLAN_ACTCAL_G0 步骤 2：直接让 SDK 在 **unified** 上算各校准方法的 encoding。

比"复现算法再外推"强：不引入任何近似，SDK 自己在目标张量上算。
unified 是决策相关的那个张量（能量集中度 99.83%，percentile 灾难就发生在它身上）。
"""
import os, re, sys, json, io, shutil, subprocess
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
WORK = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b", "actcal_u")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable
U_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "vs_fp32", "unified_fp32.raw")
B, S, D, Nout = 1, 4128, 3840, 32
METHODS = ["min-max", "mse", "sqnr", "entropy"]


def sh(c, t=3600):
    r = subprocess.run(c, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=t)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((D, Nout)) * 0.02).astype(np.float32)
    p = os.path.join(WORK, "u.onnx")
    node = helper.make_node("MatMul", ["unified", "w"], ["y"], name="probe")
    g = helper.make_graph([node], "uprobe",
                          [helper.make_tensor_value_info("unified", TensorProto.FLOAT, [B, S, D])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, S, Nout])],
                          [numpy_helper.from_array(W, "w")])
    mo = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mo.ir_version = 9
    onnx.save(mo, p)
    lst = os.path.join(WORK, "list.txt")
    open(lst, "w").write("unified:=" + U_RAW + "\n")
    f = os.path.join(WORK, "u_fp32.dlc")
    rc, out = sh([PY, TOOL, "qairt-converter", "--input_network", p, "--output_path", f,
                  "--float_bitwidth", "32"])
    if rc != 0:
        sys.exit("converter rc=%d\n%s" % (rc, out[-1500:]))
    pat = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                     r"scale ([-\d.eE+]+), offset ([-\d.]+)")
    res = {}
    for m in METHODS:
        q = os.path.join(WORK, "q_%s.dlc" % m.replace("-", ""))
        rc, out = sh([PY, TOOL, "qairt-quantizer", "--input_dlc", f, "--output_dlc", q,
                      "--input_list", lst, "--weights_bitwidth", "8", "--act_bitwidth", "16",
                      "--bias_bitwidth", "32", "--act_quantizer_calibration", m,
                      "--param_quantizer_calibration", "min-max"])
        if rc != 0:
            print("  %-8s ❌ rc=%d %s" % (m, rc, [l for l in out.splitlines() if "rror" in l][:1]))
            continue
        csv = os.path.join(WORK, "e_%s.csv" % m.replace("-", ""))
        sh([PY, TOOL, "snpe-dlc-info", "-i", q, "-d", "-s", csv])
        with io.open(csv, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                for mm in pat.finditer(line):
                    if mm.group(1) == "unified" and m not in res:
                        res[m] = dict(bitwidth=int(mm.group(2)), min=float(mm.group(3)),
                                      max=float(mm.group(4)), scale=float(mm.group(5)),
                                      offset=float(mm.group(6)))
        e = res.get(m)
        if e:
            print("  %-8s unified: bw=%d min=%.6f max=%.6f scale=%.6g offset=%g"
                  % (m, e["bitwidth"], e["min"], e["max"], e["scale"], e["offset"]))
    json.dump(res, open(os.path.join(WORK, "unified_encodings.json"), "w"), indent=1)
    a = np.fromfile(U_RAW, np.float32)
    print("\n真实 unified: min=%.6f max=%.6f （min-max 应吻合，自检）" % (a.min(), a.max()))


if __name__ == "__main__":
    main()
