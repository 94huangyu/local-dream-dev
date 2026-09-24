"""
Verify the transformer_part1 -> part1a/part1b split preserved semantics.

Loads the ORIGINAL unsplit transformer_part1_fixed.onnx, feeds it a real
calibration sample, and requests the exact intermediate tensor names that
became the split-boundary outputs (add_138, add_131, tanh_19, select_45,
select_46, adaln_input, unified_freqs, unified_mask, latents_shape).

Then loads transformer_part1a.onnx alone with the SAME inputs and compares
its actual outputs against those reference intermediate values.

If they match: the split operation is correct, and the on-device bug is
elsewhere (part1b/part2 internal logic, quantization, or the App side).
If they diverge: the split introduced the bug, and that's exactly which
tensor(s) to fix in the export pipeline.
"""
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
CALIB = EVIDENCE + r"\calibration\transformer_part1\sample_0000"
SPLIT_BOUNDARY_TENSORS = [
    "add_138", "add_131", "tanh_19", "adaln_input",
    "select_45", "select_46", "unified_freqs", "unified_mask", "latents_shape",
]

def load_inputs():
    latents = np.load(CALIB + r"\latents.npy").astype(np.float32)
    timestep = np.load(CALIB + r"\timestep.npy").astype(np.float32)
    caption = np.load(CALIB + r"\caption.npy").astype(np.float32)
    cap_pad_mask = np.load(CALIB + r"\cap_pad_mask.npy").astype(bool)
    print("Input shapes:", latents.shape, timestep.shape, caption.shape, cap_pad_mask.shape)
    return {
        "latents": latents,
        "timestep": timestep,
        "caption": caption,
        "cap_pad_mask": cap_pad_mask,
    }

def main():
    inputs = load_inputs()

    print("\n=== Loading transformer_part1_fixed.onnx (unsplit reference, ~13.7GB weights) ===")
    print("This may take a while and use significant RAM...")
    t0 = time.time()
    unsplit_path = EVIDENCE + r"\onnx\transformer_part1_fixed.onnx"

    # Add the split-boundary tensors as extra graph outputs so we can fetch
    # their reference values directly from the ORIGINAL unsplit graph.
    model = onnx.load(unsplit_path, load_external_data=False)
    existing_outputs = {o.name for o in model.graph.output}
    value_info_by_name = {vi.name: vi for vi in list(model.graph.value_info) + list(model.graph.output)}
    added = []
    for name in SPLIT_BOUNDARY_TENSORS:
        if name in existing_outputs:
            continue
        vi = value_info_by_name.get(name)
        if vi is not None:
            model.graph.output.append(vi)
        else:
            # No shape/type info available; add a bare ValueInfoProto by name.
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        added.append(name)
    print("Added as extra outputs:", added)

    patched_path = EVIDENCE + r"\onnx\transformer_part1_fixed_with_taps.onnx"
    onnx.save(model, patched_path, save_as_external_data=False)
    print(f"Patched graph saved (weights still external, referenced from original .data file): {patched_path}")

    sess = ort.InferenceSession(patched_path, providers=["CPUExecutionProvider"])
    print(f"Loaded in {time.time()-t0:.1f}s")

    output_names = [o.name for o in sess.get_outputs()]
    print("Session outputs:", output_names)

    t1 = time.time()
    results = sess.run(output_names, inputs)
    print(f"Inference done in {time.time()-t1:.1f}s")
    ref = dict(zip(output_names, results))

    print("\n=== Reference (unsplit) split-boundary tensor stats ===")
    for name in SPLIT_BOUNDARY_TENSORS:
        if name in ref:
            arr = ref[name]
            if arr.dtype == bool:
                print(f"  {name}: shape={arr.shape} dtype=bool true_frac={arr.mean():.4f}")
            else:
                a = arr.astype(np.float64) if arr.dtype != np.int64 else arr.astype(np.float64)
                print(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
                      f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} std={a.std():.4f}")
        else:
            print(f"  {name}: MISSING from reference run output")

    np.savez(EVIDENCE + r"\onnx\split_boundary_reference.npz",
              **{k: v for k, v in ref.items() if k in SPLIT_BOUNDARY_TENSORS})
    print("\nSaved reference tensors to split_boundary_reference.npz")

    print("\n=== Loading transformer_part1a.onnx (split, actual on-device graph) ===")
    t2 = time.time()
    sess_a = ort.InferenceSession(EVIDENCE + r"\onnx\transformer_part1a.onnx",
                                   providers=["CPUExecutionProvider"])
    print(f"Loaded in {time.time()-t2:.1f}s")
    a_output_names = [o.name for o in sess_a.get_outputs()]
    a_results = sess_a.run(a_output_names, inputs)
    part1a_out = dict(zip(a_output_names, a_results))

    print("\n=== Comparison: part1a actual output vs. unsplit reference ===")
    all_match = True
    for name in SPLIT_BOUNDARY_TENSORS:
        if name not in part1a_out or name not in ref:
            print(f"  {name}: SKIP (missing on one side)")
            continue
        a = part1a_out[name]
        b = ref[name]
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            all_match = False
            continue
        if a.dtype == bool:
            same = np.array_equal(a, b)
            print(f"  {name}: bool exact_match={same}")
            all_match = all_match and same
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = diff.max()
        rel = max_abs / (np.abs(b.astype(np.float64)).max() + 1e-8)
        ok = max_abs < 1e-2  # allow small float non-associativity noise
        print(f"  {name}: max_abs_diff={max_abs:.6f} rel={rel:.6f} {'OK' if ok else '*** MISMATCH ***'}")
        all_match = all_match and ok

    print("\n" + ("=== SPLIT IS CORRECT: part1a reproduces the unsplit model's intermediate tensors ==="
                   if all_match else
                   "=== SPLIT INTRODUCED A DIVERGENCE: see *** MISMATCH *** lines above ==="))

if __name__ == "__main__":
    main()
