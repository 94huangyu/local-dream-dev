"""验证切分的数值等价性：完整 part2 == part2a → part2b（FP32）

这是切分操作**最基本的正确性检查**，而我在切完之后直接去做转换/量化，
把这一步整个跳过了（2026-08-16，被用户质问后补做）。
若不等价，则后续所有基于切分的结论（包括 OOM 排查）全部无效。

输入取 part2 现成的校准样本 sample_0000，逐字节复用，不新造数据。
"""
import gc, os
import numpy as np, onnxruntime as ort

E = r"D:\ZImage_Work\ZImage_QNN_Evidence\onnx"
S = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline\transformer_part2\calibration_raw\sample_0000"
OUT = r"D:\ZImage_Work\p0_experiments\p2split"
SPEC = {"unified": ((1,4128,3840), np.float32), "unified_mask": ((1,4128), bool),
        "unified_freqs": ((1,4128,64,2), np.float32), "adaln_input": ((1,256), np.float32)}
CUT = ["add_92","select","select_1","split_7_split_2","split_7_split_3","val_105"]

def load(n):
    sh, dt = SPEC[n]
    a = np.fromfile(os.path.join(S, n+".raw"), np.float32).reshape(sh)
    return a.astype(bool) if dt is bool else a

def run(model, feeds, outs):
    s = ort.InferenceSession(os.path.join(E, model), providers=["CPUExecutionProvider"])
    r = s.run(outs, feeds); del s; gc.collect(); return r

feeds = {n: load(n) for n in SPEC}
print("跑完整 part2 (FP32) ...", flush=True)
full = run("transformer_part2_fixed_dce.onnx", feeds, ["latents"])[0]
print(f"  latents std={full.std():.6f}", flush=True)

print("跑 part2a (FP32) ...", flush=True)
mid = run("transformer_part2a_fixed.onnx", feeds, CUT)
print("跑 part2b (FP32) ...", flush=True)
f2 = dict(zip(CUT, mid)); f2["adaln_input"] = feeds["adaln_input"]
split = run("transformer_part2b_fixed.onnx", f2, ["latents"])[0]
print(f"  latents std={split.std():.6f}", flush=True)

a = full.astype(np.float64).ravel(); b = split.astype(np.float64).ravel()
rel = np.linalg.norm(b-a)/np.linalg.norm(a)
mx = np.abs(b-a).max()
cos = float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
print("\n" + "="*60)
print(f"元素数 {a.size}")
print(f"相对 L2  = {rel*100:.8f}%")
print(f"最大绝对差 = {mx:.3e}")
print(f"余弦     = {cos:.10f}")
print(f"逐位相同 = {np.array_equal(full, split)}")
print("判定：", "✅ 等价（差异在 float 复算噪声量级 <1e-5）" if rel < 1e-5
      else "❌ 不等价——切分有问题，后续所有结论作废")
np.ascontiguousarray(split.astype(np.float32)).tofile(os.path.join(OUT,"latents_split_s0.raw"))
np.ascontiguousarray(full.astype(np.float32)).tofile(os.path.join(OUT,"latents_full_s0.raw"))
