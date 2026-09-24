"""EXP_PLAN_ACTCAL_G0 步骤 1：让 SDK 自己在真实张量上算各校准方法的激活 encoding。

🔴 为什么不直接自己实现 mse/sqnr/entropy：与 SDK 实现不一致的话整门数字都是假的（约束 8）。
   本脚本只负责「取真值」，复现比对在 actcal_g0.py 里做（V2 门）。
"""
import os, re, sys, json, io, shutil, subprocess
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
P0B = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p0b")
WORK = os.path.join(P0B, "actcal")
TOOL = os.path.join("D:", os.sep, "LocalDreamZImage", "scripts", "qairt_tool.py")
PY = sys.executable
X_RAW = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "fc_inputs",
                     "cpu", "out", "Result_0", "node_linear_99_pre_reshape.raw")
M, K, N = 4128, 3840, 3840
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
    W = np.load(os.path.join(P0B, "val_1800_fp32.npy")).astype(np.float32)
    onnx_p = os.path.join(WORK, "fc99.onnx")
    node = helper.make_node("MatMul", ["x", "val_1800"], ["linear_99_fc"], name="node_linear_99")
    g = helper.make_graph([node], "fc99",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
                          [helper.make_tensor_value_info("linear_99_fc", TensorProto.FLOAT, [M, N])],
                          [numpy_helper.from_array(W, "val_1800")])
    mo = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mo.ir_version = 9
    onnx.save(mo, onnx_p)
    lst = os.path.join(WORK, "list.txt")
    open(lst, "w").write("x:=" + X_RAW + "\n")
    fdlc = os.path.join(WORK, "fc99_fp32.dlc")
    rc, out = sh([PY, TOOL, "qairt-converter", "--input_network", onnx_p,
                  "--output_path", fdlc, "--float_bitwidth", "32"])
    if rc != 0:
        sys.exit("converter rc=%d\n%s" % (rc, out[-1200:]))

    pat = re.compile(r"([A-Za-z0-9_./-]+) encoding : bitwidth (\d+), min ([-\d.]+), max ([-\d.]+), "
                     r"scale ([-\d.eE+]+), offset ([-\d.]+)")
    res = {}
    for m in METHODS:
        q = os.path.join(WORK, "q_%s.dlc" % m.replace("-", ""))
        rc, out = sh([PY, TOOL, "qairt-quantizer", "--input_dlc", fdlc, "--output_dlc", q,
                      "--input_list", lst, "--weights_bitwidth", "8", "--act_bitwidth", "16",
                      "--bias_bitwidth", "32", "--act_quantizer_calibration", m,
                      "--param_quantizer_calibration", "min-max"])
        if rc != 0:
            print("  %-8s ❌ rc=%d %s" % (m, rc, [l for l in out.splitlines() if "rror" in l][:1]))
            continue
        csv = os.path.join(WORK, "e_%s.csv" % m.replace("-", ""))
        sh([PY, TOOL, "snpe-dlc-info", "-i", q, "-d", "-s", csv])
        got = {}
        with io.open(csv, encoding="utf-8", errors="replace") as f:
            for line in f:
                for mm in pat.finditer(line):
                    if mm.group(1) in ("x", "linear_99_fc") and mm.group(1) not in got:
                        got[mm.group(1)] = dict(bitwidth=int(mm.group(2)), min=float(mm.group(3)),
                                                max=float(mm.group(4)), scale=float(mm.group(5)),
                                                offset=float(mm.group(6)))
        res[m] = got
        e = got.get("x", {})
        print("  %-8s x: bw=%s min=%.6f max=%.6f scale=%.6g offset=%g"
              % (m, e.get("bitwidth"), e.get("min", 0), e.get("max", 0),
                 e.get("scale", 0), e.get("offset", 0)))
    json.dump(res, open(os.path.join(WORK, "sdk_encodings.json"), "w"), indent=1)
    x = np.fromfile(X_RAW, np.float32)
    print("\n真实张量 x: min=%.6f max=%.6f  ⇒ min-max 应给出这两个值（自检）" % (x.min(), x.max()))


if __name__ == "__main__":
    main()
