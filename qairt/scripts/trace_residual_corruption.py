"""
Binary-search the transformer_part1a residual stream to find which layer
first introduces the severe "outlier activation" quantization corruption
seen at add_138 (add_138 = 14th residual add; chain: add_54 (1st) -> add_66
-> add_78 -> add_90 -> add_102 -> add_114 -> add_126 -> add_138 (14th)).

For each checkpoint tensor, computes the same "bulk vs outlier" error split
used for add_138: elements with |ref| < 5 are the "normal" bulk; elements
with |ref| >= 5 are the "outlier activations" (the ones that broke).
"""
import time
import numpy as np
import onnx
import onnxruntime as ort

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
CALIB = EVIDENCE + r"\calibration\transformer_part1\sample_0000"
CALIB_RAW = EVIDENCE + r"\dlc_pipeline\transformer_part1a\calibration_raw\sample_0000"

# Every 2nd residual add along the traced chain (earliest -> latest), plus
# the already-known-broken add_138 endpoint for reference.
CHECKPOINTS = ["add_54", "add_66", "add_78", "add_90", "add_102", "add_114", "add_126", "add_138"]

def load_fp32_inputs():
    return {
        "latents": np.load(CALIB + r"\latents.npy").astype(np.float32),
        "timestep": np.load(CALIB + r"\timestep.npy").astype(np.float32),
        "caption": np.load(CALIB + r"\caption.npy").astype(np.float32),
        "cap_pad_mask": np.load(CALIB + r"\cap_pad_mask.npy").astype(bool),
    }

def get_reference(checkpoints):
    print("=== Loading transformer_part1_fixed.onnx (unsplit reference) ===")
    t0 = time.time()
    unsplit_path = EVIDENCE + r"\onnx\transformer_part1_fixed.onnx"
    model = onnx.load(unsplit_path, load_external_data=False)
    existing = {o.name for o in model.graph.output}
    vi_by_name = {vi.name: vi for vi in list(model.graph.value_info) + list(model.graph.output)}
    for name in checkpoints:
        if name in existing:
            continue
        vi = vi_by_name.get(name)
        model.graph.output.append(vi if vi is not None else onnx.helper.make_empty_tensor_value_info(name))
    patched = EVIDENCE + r"\onnx\transformer_part1_fixed_with_chain_taps.onnx"
    onnx.save(model, patched, save_as_external_data=False)

    sess = ort.InferenceSession(patched, providers=["CPUExecutionProvider"])
    print(f"Loaded in {time.time()-t0:.1f}s")
    output_names = [o.name for o in sess.get_outputs()]
    t1 = time.time()
    results = dict(zip(output_names, sess.run(output_names, load_fp32_inputs())))
    print(f"Inference done in {time.time()-t1:.1f}s")
    return {k: results[k] for k in checkpoints}

def main():
    ref = get_reference(CHECKPOINTS)
    np.savez(EVIDENCE + r"\onnx\residual_chain_reference.npz", **ref)
    print("Saved reference checkpoints.\n")

    print(f"{'tensor':<10}{'shape':<20}{'ref_max':>10}{'bulk_mean_err':>15}{'bulk_max_err':>14}"
          f"{'outlier_n':>10}{'outlier_mean_err':>18}{'outlier_max_err':>16}")
    for name in CHECKPOINTS:
        r = ref[name]
        print(f"{name:<10}{str(r.shape):<20}{np.abs(r).max():>10.2f}"
              f"{'--- (need quantized run, see part 2) ---':>60}")

if __name__ == "__main__":
    main()
