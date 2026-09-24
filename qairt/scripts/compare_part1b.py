"""
Verify transformer_part1b's split correctness the same way part1a was verified:
feed it the (already bit-exact-verified) split-boundary reference tensors,
compare its "unified" output against the unsplit reference model's own
native "unified" output for the same calibration sample.
"""
import time
import numpy as np
import onnxruntime as ort

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
CALIB = EVIDENCE + r"\calibration\transformer_part1\sample_0000"

def load_calib_inputs():
    return {
        "latents": np.load(CALIB + r"\latents.npy").astype(np.float32),
        "timestep": np.load(CALIB + r"\timestep.npy").astype(np.float32),
        "caption": np.load(CALIB + r"\caption.npy").astype(np.float32),
        "cap_pad_mask": np.load(CALIB + r"\cap_pad_mask.npy").astype(bool),
    }

def main():
    print("=== Loading transformer_part1_fixed.onnx (unsplit reference) for its native 'unified' output ===")
    t0 = time.time()
    sess_ref = ort.InferenceSession(EVIDENCE + r"\onnx\transformer_part1_fixed.onnx",
                                     providers=["CPUExecutionProvider"])
    print(f"Loaded in {time.time()-t0:.1f}s")
    inputs = load_calib_inputs()
    t1 = time.time()
    unified_ref = sess_ref.run(["unified"], inputs)[0]
    print(f"Inference done in {time.time()-t1:.1f}s")
    print(f"Reference 'unified': shape={unified_ref.shape} min={unified_ref.min():.4f} "
          f"max={unified_ref.max():.4f} mean={unified_ref.mean():.4f} std={unified_ref.std():.4f}")

    print("\n=== Loading transformer_part1b.onnx (split) ===")
    t2 = time.time()
    sess_b = ort.InferenceSession(EVIDENCE + r"\onnx\transformer_part1a.onnx",
                                   providers=["CPUExecutionProvider"])
    part1a_out = dict(zip([o.name for o in sess_b.get_outputs()],
                          sess_b.run([o.name for o in sess_b.get_outputs()], inputs)))
    print(f"part1a re-run in {time.time()-t2:.1f}s (reusing to get exact split-boundary tensors)")

    sess_1b = ort.InferenceSession(EVIDENCE + r"\onnx\transformer_part1b.onnx",
                                    providers=["CPUExecutionProvider"])
    part1b_inputs = {name: part1a_out[name] for name in
                      ["adaln_input", "add_131", "add_138", "select_45", "select_46", "tanh_19"]}
    t3 = time.time()
    unified_1b = sess_1b.run(["unified"], part1b_inputs)[0]
    print(f"part1b inference done in {time.time()-t3:.1f}s")
    print(f"part1b 'unified':     shape={unified_1b.shape} min={unified_1b.min():.4f} "
          f"max={unified_1b.max():.4f} mean={unified_1b.mean():.4f} std={unified_1b.std():.4f}")

    diff = np.abs(unified_ref.astype(np.float64) - unified_1b.astype(np.float64))
    max_abs = diff.max()
    rel = max_abs / (np.abs(unified_ref.astype(np.float64)).max() + 1e-8)
    print(f"\nmax_abs_diff={max_abs:.6f} rel={rel:.6f} -> {'OK (split correct)' if max_abs < 1e-2 else '*** MISMATCH -- part1b split is WRONG ***'}")

if __name__ == "__main__":
    main()
