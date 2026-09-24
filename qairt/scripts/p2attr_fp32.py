"""EXP_PLAN_P2_ATTR 宿主半边（改道版）：用 FP32 ONNX 算 part2a 的 12 个探针**真值**。

为什么不用 SNPE CPU 参考（原方案）：
  `snpe-net-run` 对 per-row DLC 直接报
  `error_code=202 ... Dequantization of axis-quant tensor is not supported for FullyConnected`
  ⇒ CPU 运行时不支持 per-axis 权重反量化，跑不了部署用的那份。见 MAINLINE #105。

改道后的好处：真值不含任何量化误差，也绕开 #35 的「CPU 是浮点执行」前提问题。
⚠️ 但参照系变了：`cos(X)` 是「HTP vs FP32 真值」，**包含量化误差本身**，不只是 HTP 特有误差。

做法：把探针张量加进 graph.output（**只改 .onnx，外部权重文件不动**，
新文件写进同一目录 ⇒ 相对 location 仍有效，不重写 GB 级 .data）。
"""
import os, sys, hashlib
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
SRC = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "testB", "s0_transformer_part2")
OUT = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr", "fp32")
BASE = "transformer_part2a_fixed"
IN_SHAPES = {"unified": (1, 4128, 3840), "unified_mask": (1, 4128),
             "unified_freqs": (1, 4128, 64, 2), "adaln_input": (1, 256)}
# ONNX 侧的名字（设备侧 DLC 名带 _fc 后缀，配对时换算）
PROBES = ["mul_1", "linear_1", "mul_4", "mul_6", "scaled_dot_product_attention", "linear_4",
          "mul_16", "add_8", "mul_19", "linear_5", "mul_23", "linear_9",
          # 第二轮：mul_23 是放大点，其 RmsNorm 输入是 linear_7（massive activation, |a|max=76493）
          "linear_7", "mul_21", "linear_6"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(D, BASE + ".onnx")
    dst = os.path.join(D, BASE + "_taps.onnx")
    m = onnx.load(src, load_external_data=False)
    have = {o.name for o in m.graph.output}
    produced = {o for n in m.graph.node for o in n.output}
    missing = [t for t in PROBES if t not in produced]
    if missing:
        sys.exit("❌ ONNX 里没有这些张量：%s" % missing)
    for t in PROBES:
        if t not in have:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    onnx.save(m, dst)
    print("[tap] 写出 %s（新增 %d 个输出，外部权重未动）"
          % (os.path.basename(dst), len([t for t in PROBES if t not in have])), flush=True)

    feeds = {}
    for n, s in IN_SHAPES.items():
        p = os.path.join(SRC, n + ".raw")
        a = np.fromfile(p, np.float32).reshape(s)
        feeds[n] = a
        print("  输入 %-14s %-20s md5=%s" % (n, s, md5(p)[:16]), flush=True)
    # unified_mask 在 ONNX 侧的 dtype 需按图声明取
    for i in m.graph.input:
        want = i.type.tensor_type.elem_type
        if want == TensorProto.BOOL:
            feeds[i.name] = feeds[i.name].astype(bool)
            print("    (%s 按图声明转 bool)" % i.name)

    print("[run] onnxruntime CPU ...", flush=True)
    s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    outs = [o.name for o in s.get_outputs() if o.name in PROBES]
    r = s.run(outs, feeds)
    print("\n真值导出成功（**FP32 ONNX**，不含任何量化）：")
    for n, a in zip(outs, r):
        a = np.ascontiguousarray(a.astype(np.float32))
        a.tofile(os.path.join(OUT, n + ".raw"))
        print("  %-30s %-22s std=%.6f  |a|max=%.1f" % (n, str(a.shape), a.std(), np.abs(a).max()))


if __name__ == "__main__":
    main()
