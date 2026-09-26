#!/usr/bin/env python3
"""Convert a BF16/FP16 DiT checkpoint to FP8 E4M3 "scaled" safetensors.

This produces the same kind of file as the built-in Z-Image Turbo package
(Kijai's z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors), which is what the
Hexagon DiT engine runs fastest: every transformer-block linear weight is
stored as FP8 E4M3 with one FP32 scale per tensor, and everything else keeps
its original dtype.

    pip install numpy ml_dtypes
    python tools/convert_dit_fp8.py input.safetensors output_fp8.safetensors

    # Only report what a file is (architecture, dtypes, required text
    # encoder, problems the app would reject it for); writes nothing:
    python tools/convert_dit_fp8.py --inspect some.safetensors

    # Recommended: copy the per-layer precision of the file the app ships and
    # the engine was validated with (only its header is downloaded):
    python tools/convert_dit_fp8.py in.safetensors out.safetensors --like zimage
    python tools/convert_dit_fp8.py in.safetensors out.safetensors --like klein4b

    # Diagnose a converted file: layer-precision diff against the reference,
    # and a numeric check of every tensor against the original:
    python tools/convert_dit_fp8.py --inspect out.safetensors --like zimage
    python tools/convert_dit_fp8.py out.safetensors --verify in.safetensors

No PyTorch or GPU is needed. The input is memory-mapped and the output is
streamed to disk one tensor at a time, so peak RAM stays around a few hundred
MB even for a 12 GB checkpoint.

Supported inputs
  * Z-Image Turbo fine-tunes/merges in the ComfyUI single-file layout
    ("layers.N.attention.qkv.weight") or the diffusers layout
    ("layers.N.attention.to_q.weight"), which is fused and renamed on the way.
  * FLUX.2 Klein fine-tunes in the original BFL/ComfyUI layout
    ("double_blocks.N...", "single_blocks.N...").
  * All-in-one checkpoints: only the "model.diffusion_model." part is kept.
  * Inputs that are already FP8 (E4M3 or E5M2, with or without scales) are
    dequantized and requantized, which also makes E5M2 files loadable.
  * ComfyUI int8_tensorwise layers are dequantized and requantized.
    ConvRot int8 and packed 4-bit formats (NVFP4/MXFP4) are rejected: use the
    BF16 or FP8 release of the model instead.
"""

import argparse
import json
import re
import os
import struct
import sys
import urllib.request

try:
    import ml_dtypes
    import numpy as np
except ImportError:
    sys.exit("This script needs numpy and ml_dtypes: pip install numpy ml_dtypes")

FP8_MAX = 448.0  # largest finite float8_e4m3fn value
DIFFUSION_PREFIX = "model.diffusion_model."

DTYPES = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "F8_E5M2": np.dtype(ml_dtypes.float8_e5m2),
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}
FLOAT_TYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
FP8_TYPES = {"F8_E4M3", "F8_E5M2"}
# Weight dtypes that carry a separate dequantization scale.
SCALED_TYPES = FP8_TYPES | {"I8"}

# Linear weights inside the transformer blocks: the bulk of the parameters and
# what the engine's FP8 matmul accelerates. Norms, modulation, embedders and
# the final layer are small and precision-sensitive, so they keep their dtype.
QUANTIZE_PATTERNS = [
    # Z-Image (Lumina2-style) main layers and refiners.
    re.compile(r"(^|\.)(layers|noise_refiner|context_refiner)\.\d+\."
               r"(attention\.(qkv|out)|feed_forward\.w[123])\.weight$"),
    # FLUX.2 double-stream blocks.
    re.compile(r"(^|\.)double_blocks\.\d+\.(img|txt)_(attn\.(qkv|proj)|mlp\.\d+)\.weight$"),
    # FLUX.2 single-stream blocks.
    re.compile(r"(^|\.)single_blocks\.\d+\.linear[12]\.weight$"),
]

# Scale tensors and markers of an FP8 input; they are rebuilt, not copied.
SCALE_SUFFIXES = (".scale_weight", ".weight_scale", ".input_scale", ".scale_input", ".comfy_quant")
MARKER_KEYS = {"scaled_fp8"}

# Diffusers -> original names for Z-Image, mirroring what stable-diffusion.cpp
# does on load (convert_diffusers_dit_to_original_lumina2). q/k/v are fused
# here because separately scaled pieces cannot be merged after quantization.
ZIMAGE_DIFFUSERS_RENAMES = [
    ("all_x_embedder.2-1.", "x_embedder."),
    ("all_final_layer.2-1.", "final_layer."),
    ("attention.norm_q.", "attention.q_norm."),
    ("attention.norm_k.", "attention.k_norm."),
    ("attention.to_out.0.", "attention.out."),
]


class SafetensorsFile:
    """Read-only, memory-mapped view of a .safetensors file."""

    def __init__(self, path):
        with open(path, "rb") as f:
            (header_size,) = struct.unpack("<Q", f.read(8))
            if header_size > 256 << 20:
                sys.exit(f"{path}: not a safetensors file")
            header = json.loads(f.read(header_size))
        self.metadata = header.pop("__metadata__", None) or {}
        self.entries = header
        self.data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)

    def dtype(self, name):
        return self.entries[name]["dtype"]

    def shape(self, name):
        return tuple(self.entries[name]["shape"])

    def raw(self, name):
        begin, end = self.entries[name]["data_offsets"]
        return self.data[begin:end]

    def array(self, name):
        dtype = self.dtype(name)
        if dtype not in DTYPES:
            sys.exit(f"unsupported dtype {dtype} for tensor {name}")
        return self.raw(name).view(DTYPES[dtype]).reshape(self.shape(name))

    def comfy_quant(self, module):
        """The ComfyUI quantization config of a module, or None."""
        name = module + ".comfy_quant"
        if name not in self.entries:
            return None
        try:
            return json.loads(bytes(self.raw(name)).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            sys.exit(f"unreadable ComfyUI quantization metadata: {name}")

    def as_float32(self, name):
        """The tensor in float32, with any stored dequantization scale applied."""
        dtype = self.dtype(name)
        values = self.array(name).astype(np.float32)
        if dtype not in SCALED_TYPES or not name.endswith(".weight"):
            return values
        module = name[: -len(".weight")]
        if dtype == "I8":
            config = self.comfy_quant(module) or {}
            if config.get("convrot"):
                sys.exit(f"{name}: ConvRot int8 weights cannot be converted; "
                         "use the BF16 or FP8 release of this model")
        for suffix in (".scale_weight", ".weight_scale"):
            if module + suffix not in self.entries:
                continue
            scale = self.array(module + suffix).astype(np.float32)
            if scale.size == 1:
                values *= float(scale.reshape(-1)[0])
            elif values.ndim == 2 and scale.size == values.shape[0]:
                # One scale per output row: broadcast along the input axis.
                values *= scale.reshape(-1, 1)
            else:
                sys.exit(f"{name}: unsupported scale shape {list(scale.shape)} "
                         f"for weight shape {list(values.shape)}")
            return values
        if dtype == "I8":
            sys.exit(f"{name}: int8 weight without a scale cannot be converted")
        return values


# The FP8 files the app's built-in packages download, i.e. the layer layouts the
# Hexagon engine has actually been validated with.
REFERENCES = {
    "zimage": "https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/"
              "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors",
    "klein4b": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/"
               "flux-2-klein-4b-fp8.safetensors",
}
# Largest magnitude F16 can hold. The Hexagon backend stores BF16 weights as
# F16, so any kept BF16 weight above this becomes inf on the NPU.
F16_MAX = 65504.0


def normalize_name(name):
    """Name used to match tensors across files: no prefix, one scale spelling."""
    if name.startswith(DIFFUSION_PREFIX):
        name = name[len(DIFFUSION_PREFIX):]
    if name.endswith(".scale_weight"):
        name = name[: -len(".scale_weight")] + ".weight_scale"
    return name


def read_header(source):
    """Tensor entries of a local .safetensors file or of a URL (header bytes only)."""
    source = REFERENCES.get(source, source)
    if source.startswith(("http://", "https://")):
        headers = {"User-Agent": "convert_dit_fp8"}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]

        def fetch(start, length):
            request = urllib.request.Request(
                source, headers=dict(headers, Range=f"bytes={start}-{start + length - 1}"))
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status == 200 and start > 0:
                        # Range ignored: skip ahead without keeping the bytes.
                        response.read(start)
                    return response.read(length)
            except OSError as e:
                sys.exit(f"cannot download the header of {source}: {e}\n"
                         "(gated repositories need HF_TOKEN set to a Hugging Face token)")

        (size,) = struct.unpack("<Q", fetch(0, 8))
        raw = fetch(8, size)
    else:
        with open(source, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            if size > 256 << 20:
                sys.exit(f"{source}: not a safetensors file")
            raw = f.read(size)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def reference_layout(source):
    """normalized name -> (dtype, shape) for the weights of a reference file."""
    layout = {}
    for name, info in read_header(source).items():
        norm = normalize_name(name)
        if norm.endswith(SCALE_SUFFIXES) or norm in MARKER_KEYS:
            continue
        layout[norm] = (info["dtype"], tuple(info["shape"]))
    return layout


class Output:
    """One output tensor: its dtype, shape and how to produce its bytes."""

    def __init__(self, dtype, shape, produce):
        self.dtype = dtype
        self.shape = shape
        self.produce = produce

    @property
    def nbytes(self):
        return int(np.prod(self.shape, dtype=np.int64)) * DTYPES[self.dtype].itemsize


def should_quantize(name, shape):
    return len(shape) == 2 and any(p.search(name) for p in QUANTIZE_PATTERNS)


def fp8_scale(values):
    """Per-tensor scale mapping the largest magnitude onto the FP8 E4M3 range."""
    amax = float(np.max(np.abs(values))) if values.size else 0.0
    return amax / FP8_MAX if amax > 0 else 1.0


def quantize(values, scale):
    return np.clip(values / scale, -FP8_MAX, FP8_MAX).astype(ml_dtypes.float8_e4m3fn)


def plan_sources(src):
    """Maps each output tensor name to the input tensor(s) it is built from."""
    names = list(src.entries)
    if any(n.startswith(DIFFUSION_PREFIX) for n in names):
        dropped = [n for n in names if not n.startswith(DIFFUSION_PREFIX)]
        if dropped:
            print(f"dropping {len(dropped)} non-diffusion tensors (text encoder / VAE)")
        names = [n for n in names if n.startswith(DIFFUSION_PREFIX)]

    diffusers_qkv = [n for n in names if n.endswith("attention.to_q.weight")]
    if diffusers_qkv and any("transformer_blocks." in n for n in names):
        sys.exit("diffusers-layout FLUX.2 checkpoints are not supported; "
                 "use the original BFL/ComfyUI single file instead")
    if diffusers_qkv:
        print(f"diffusers layout detected: fusing q/k/v in {len(diffusers_qkv)} attention blocks")

    sources = {}
    for name in names:
        bare = name[len(DIFFUSION_PREFIX):] if name.startswith(DIFFUSION_PREFIX) else name
        if name.endswith(SCALE_SUFFIXES) or bare in MARKER_KEYS:
            continue
        if ".attention.to_k." in name or ".attention.to_v." in name:
            continue
        parts = [name]
        out_name = name
        if ".attention.to_q." in name:
            template = name.replace(".attention.to_q.", ".attention.{}.")
            parts = [template.format(p) for p in ("to_q", "to_k", "to_v")]
            out_name = template.format("qkv")
        if diffusers_qkv:
            for old, new in ZIMAGE_DIFFUSERS_RENAMES:
                out_name = out_name.replace(old, new)
        sources[out_name] = parts
    return sources


def build_outputs(src, sources, check, reference=None):
    outputs = {}
    stats = {"converted": 0, "kept": 0, "worst": 0.0, "not_in_reference": [], "shape_differs": 0}

    for out_name, parts in sources.items():
        dtype = src.dtype(parts[0])
        shape = list(src.shape(parts[0]))
        if len(parts) > 1:
            shape[0] = sum(src.shape(p)[0] for p in parts)

        # With a reference, every tensor takes the reference's dtype: FP8 where
        # it is FP8 and the same BF16/F16/F32 elsewhere, so the engine sees the
        # exact op/dtype mix it was validated on.
        target = None
        if reference is not None:
            ref = reference.get(normalize_name(out_name))
            if ref is None:
                stats["not_in_reference"].append(out_name)
            else:
                target = ref[0]
                if tuple(ref[1]) != tuple(shape):
                    stats["shape_differs"] += 1

        def load(parts=parts):
            if len(parts) == 1:
                return src.as_float32(parts[0])
            return np.concatenate([src.as_float32(p) for p in parts], axis=0)

        if dtype not in DTYPES:
            sys.exit(f"{out_name}: unsupported dtype {dtype} (packed 4-bit formats such as "
                     "NVFP4/MXFP4 are not supported); use the BF16 or FP8 release")
        if dtype == "U8" and out_name.endswith(".weight"):
            sys.exit(f"{out_name}: packed U8 weights (NVFP4/MXFP4 style) are not supported; "
                     "use the BF16 or FP8 release of this model")
        wants_fp8 = (target == "F8_E4M3") if target is not None else (
            reference is None and should_quantize(out_name, shape))
        if (dtype in FLOAT_TYPES or dtype == "I8") and wants_fp8:
            # The header needs every scale before any data is written, so the
            # scales are computed in this first pass and each weight is read
            # (and quantized) again while streaming, instead of holding all
            # quantized weights in memory.
            values = load()
            scale = fp8_scale(values)
            if check:
                q = quantize(values, scale)
                err = float(np.linalg.norm(q.astype(np.float32) * scale - values) /
                            max(float(np.linalg.norm(values)), 1e-12))
                stats["worst"] = max(stats["worst"], err)
                print(f"  {out_name}: rel. error {err:.4%}")
                del q
            del values
            outputs[out_name] = Output(
                "F8_E4M3", shape,
                lambda load=load, scale=scale: quantize(load(), scale).tobytes())
            scale_bytes = np.array([scale], dtype=np.float32).tobytes()
            outputs[out_name[: -len(".weight")] + ".weight_scale"] = Output(
                "F32", [1], lambda b=scale_bytes: b)
            stats["converted"] += 1
        elif (target in ("BF16", "F16", "F32") and target != dtype
              and (dtype in FLOAT_TYPES or dtype == "I8")):
            outputs[out_name] = Output(
                target, shape,
                lambda load=load, t=DTYPES[target]: load().astype(t).tobytes())
            stats["kept"] += 1
        elif dtype in SCALED_TYPES:
            # A scaled tensor outside the blocks: restore it to BF16.
            outputs[out_name] = Output(
                "BF16", shape, lambda load=load: load().astype(ml_dtypes.bfloat16).tobytes())
            stats["kept"] += 1
        elif len(parts) > 1:
            outputs[out_name] = Output(
                dtype, shape,
                lambda parts=parts: b"".join(bytes(src.raw(p)) for p in parts))
            stats["kept"] += 1
        else:
            outputs[out_name] = Output(dtype, shape, lambda p=parts[0]: bytes(src.raw(p)))
            stats["kept"] += 1
    return outputs, stats


# Hidden size of the text encoders the built-in packages ship (Qwen3-4B).
BUILTIN_TE_HIDDEN = 2560
TE_BY_HIDDEN = {2560: "Qwen3-4B", 4096: "Qwen3-8B"}
BUNDLED_PREFIXES = ("first_stage_model.", "cond_stage_model.", "conditioner.",
                    "text_encoders.", "text_encoder.", "vae.")


def describe(src):
    """Architecture, text-encoder requirement and app-import problems of a file."""
    names = [n for n in src.entries]

    def find(*suffixes):
        return next((n for n in names if n.endswith(suffixes)), None)

    report = {"arch": "unknown", "te_hidden": None, "problems": [], "notes": []}
    if any("cap_embedder." in n for n in names):
        report["arch"] = "Z-Image"
        key = find("cap_embedder.1.weight")
        if key:
            report["te_hidden"] = src.shape(key)[-1]
    elif any("double_stream_modulation_img" in n for n in names):
        big = any("single_blocks.47." in n for n in names)
        report["arch"] = "FLUX.2 dev" if big else "FLUX.2 Klein"
        key = find("txt_in.weight", "context_embedder.weight")
        if key:
            report["te_hidden"] = src.shape(key)[-1] // 3
        if big:
            report["problems"].append("FLUX.2 dev is not one of the app's DiT kinds")
    elif any("txt_in.text_norm" in n for n in names):
        report["arch"] = "Qwen Image 2.1"
    else:
        report["problems"].append("no Z-Image / FLUX.2 Klein / Qwen Image 2.1 tensor names found")

    te = report["te_hidden"]
    if te is not None and te != BUILTIN_TE_HIDDEN:
        report["problems"].append(
            f"needs a {TE_BY_HIDDEN.get(te, f'{te}-dim')} text encoder; the built-in "
            f"{TE_BY_HIDDEN[BUILTIN_TE_HIDDEN]} llm.gguf will not work, so choose a matching "
            "llm.gguf for the text encoder when importing")

    dtypes = {}
    for n in names:
        dtypes[src.dtype(n)] = dtypes.get(src.dtype(n), 0) + 1
    report["dtypes"] = dtypes
    if "F8_E5M2" in dtypes:
        report["problems"].append("contains FP8 E5M2 weights: convert this file")
    if "I8" in dtypes:
        report["problems"].append("contains INT8 weights: convert this file")
    if any(d not in DTYPES for d in dtypes):
        report["problems"].append(f"unsupported dtypes {sorted(d for d in dtypes if d not in DTYPES)}")
    if any(n.startswith(BUNDLED_PREFIXES) for n in names):
        report["problems"].append("bundles a text encoder or VAE: convert this file to extract the DiT")

    fp8 = dtypes.get("F8_E4M3", 0)
    if fp8 and any(n.endswith((".weight_scale", ".scale_weight")) for n in names):
        report["notes"].append("already FP8 E4M3 with scales (the fast format)")
    elif fp8:
        report["notes"].append("FP8 E4M3 without scales (loads, but converting adds scales)")
    elif dtypes.get("BF16", 0) or dtypes.get("F16", 0):
        report["notes"].append("BF16/F16: convert it for speed and to halve the size")
    return report


def print_report(path, report):
    print(f"{path}")
    print(f"  architecture : {report['arch']}")
    if report["te_hidden"] is not None:
        te = report["te_hidden"]
        print(f"  text encoder : {TE_BY_HIDDEN.get(te, 'unknown')} (hidden size {te})")
    print(f"  tensor dtypes: {report['dtypes']}")
    for note in report["notes"]:
        print(f"  note         : {note}")
    for problem in report["problems"]:
        print(f"  PROBLEM      : {problem}")
    if not report["problems"]:
        print("  import       : OK for the app's DiT import")


def write_safetensors(path, outputs, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    for name, out in outputs.items():
        header[name] = {"dtype": out.dtype, "shape": list(out.shape),
                        "data_offsets": [offset, offset + out.nbytes]}
        offset += out.nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)  # 8-byte aligned data start

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name, out in outputs.items():
            data = out.produce()
            if len(data) != out.nbytes:
                sys.exit(f"internal error: {name} produced {len(data)} bytes, expected {out.nbytes}")
            f.write(data)
    return offset


def compare_layout(src, reference):
    """Prints where a file's per-tensor dtypes differ from a reference layout."""
    mine = {}
    for name in src.entries:
        norm = normalize_name(name)
        if norm.endswith(SCALE_SUFFIXES) or norm in MARKER_KEYS:
            continue
        mine[norm] = (src.dtype(name), tuple(src.shape(name)))
    differ = [(n, mine[n][0], reference[n][0]) for n in mine
              if n in reference and mine[n][0] != reference[n][0]]
    only_mine = [n for n in mine if n not in reference]
    only_ref = [n for n in reference if n not in mine]
    shapes = sum(1 for n in mine if n in reference and mine[n][1] != reference[n][1])
    print("  layout vs reference:")
    print(f"    same dtype      : {len(mine) - len(only_mine) - len(differ)} tensors")
    print(f"    different dtype : {len(differ)} tensors")
    for n, d_mine, d_ref in differ[:15]:
        print(f"      {n}: {d_mine} here, {d_ref} in reference")
    if len(differ) > 15:
        print(f"      ... and {len(differ) - 15} more")
    if only_mine:
        print(f"    only here       : {len(only_mine)} tensors, e.g. {only_mine[:3]}")
    if only_ref:
        print(f"    only in reference: {len(only_ref)} tensors, e.g. {only_ref[:3]}")
    if shapes:
        print(f"    different shape : {shapes} tensors (a different model size than the reference)")


def verify(converted, original):
    """Numeric check of a converted file against the file it was made from."""
    sources = plan_sources(original)
    bad = []
    worst = (0.0, None)
    overflow = []
    checked = 0
    for name in converted.entries:
        norm = normalize_name(name)
        if norm.endswith(SCALE_SUFFIXES) or norm in MARKER_KEYS:
            continue
        dtype = converted.dtype(name)
        if dtype not in FLOAT_TYPES:
            continue
        mine = converted.as_float32(name)
        if not np.all(np.isfinite(mine)):
            bad.append(f"{name}: contains NaN/Inf")
            continue
        if dtype in ("BF16", "F32") and mine.size and float(np.max(np.abs(mine))) > F16_MAX:
            overflow.append(name)
        parts = sources.get(name)
        if parts is None:
            continue
        ref = (original.as_float32(parts[0]) if len(parts) == 1 else
               np.concatenate([original.as_float32(p) for p in parts], axis=0))
        if ref.shape != mine.shape:
            bad.append(f"{name}: shape {list(mine.shape)} vs original {list(ref.shape)}")
            continue
        err = float(np.linalg.norm(mine - ref) / max(float(np.linalg.norm(ref)), 1e-12))
        checked += 1
        if err > worst[0]:
            worst = (err, name)
        if err > 0.10:
            bad.append(f"{name}: relative error {err:.1%} against the original")
    print(f"verified {checked} tensors against the original")
    if worst[1]:
        print(f"  worst relative error: {worst[0]:.2%} ({worst[1]})")
    for line in bad[:20]:
        print(f"  PROBLEM: {line}")
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20} more problems")
    for name in overflow[:10]:
        print(f"  PROBLEM: {name} exceeds the F16 range; the NPU stores it as F16, so it becomes inf")
    if not bad and not overflow:
        print("  OK: the converted weights match the original (FP8 rounding only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="source .safetensors (BF16/FP16/FP32 or FP8)")
    parser.add_argument("output", nargs="?", help="destination .safetensors")
    parser.add_argument("--check", action="store_true",
                        help="print the relative quantization error of every converted tensor")
    parser.add_argument("--inspect", action="store_true",
                        help="only describe the input file; write nothing")
    parser.add_argument("--like", metavar="REF",
                        help="copy the per-tensor precision of a reference file: 'zimage', "
                             "'klein4b', a local .safetensors or a URL (header only)")
    parser.add_argument("--verify", metavar="ORIGINAL",
                        help="check the input (a converted file) against the file it was made from")
    args = parser.parse_args()

    src = SafetensorsFile(args.input)
    reference = reference_layout(args.like) if args.like else None
    if args.inspect or args.verify:
        print_report(args.input, describe(src))
        if reference is not None:
            compare_layout(src, reference)
        if args.verify:
            print()
            verify(src, SafetensorsFile(args.verify))
        return
    if not args.output:
        parser.error("an output path is required unless --inspect or --verify is given")
    sources = plan_sources(src)
    outputs, stats = build_outputs(src, sources, args.check, reference)
    if reference is not None:
        if stats["not_in_reference"]:
            print(f"warning: {len(stats['not_in_reference'])} tensors are not in the reference "
                  f"and keep their dtype, e.g. {stats['not_in_reference'][:3]}")
        if stats["shape_differs"]:
            print(f"note: {stats['shape_differs']} tensors differ in shape from the reference "
                  "(a different model size); only the per-layer precision is copied")
    if stats["converted"] == 0:
        sys.exit("no transformer-block linear weights matched; "
                 "is this a Z-Image or FLUX.2 Klein DiT?")

    metadata = {k: str(v) for k, v in src.metadata.items()}
    metadata["localdream.fp8"] = "e4m3fn per-tensor scale"
    total = write_safetensors(args.output, outputs, metadata)
    print(f"converted {stats['converted']} linear weights to FP8 E4M3, "
          f"kept {stats['kept']} tensors as is")
    print(f"{src.data.size / 2**30:.2f} GiB -> {total / 2**30:.2f} GiB, written to {args.output}")
    if args.check:
        print(f"worst relative error: {stats['worst']:.4%}")
    print()
    out = SafetensorsFile(args.output)
    print_report(args.output, describe(out))
    if reference is not None:
        compare_layout(out, reference)


if __name__ == "__main__":
    main()
