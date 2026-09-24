#!/usr/bin/env python3
"""Convert a BF16/FP16 DiT checkpoint to FP8 E4M3 "scaled" safetensors.

This produces the same kind of file as the built-in Z-Image Turbo package
(Kijai's z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors), which is what the
Hexagon DiT engine runs fastest: every transformer-block linear weight is
stored as FP8 E4M3 with one FP32 scale per tensor, and everything else keeps
its original dtype.

    pip install numpy ml_dtypes
    python tools/convert_dit_fp8.py input.safetensors output_fp8.safetensors

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
"""

import argparse
import json
import re
import struct
import sys

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
SCALE_SUFFIXES = (".scale_weight", ".weight_scale", ".input_scale", ".scale_input")
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

    def as_float32(self, name):
        """The tensor in float32, with any stored FP8 dequantization scale applied."""
        values = self.array(name).astype(np.float32)
        if self.dtype(name) in FP8_TYPES and name.endswith(".weight"):
            module = name[: -len(".weight")]
            for suffix in (".scale_weight", ".weight_scale"):
                if module + suffix in self.entries:
                    values *= self.array(module + suffix).astype(np.float32)
                    break
        return values


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


def build_outputs(src, sources, check):
    outputs = {}
    stats = {"converted": 0, "kept": 0, "worst": 0.0}

    for out_name, parts in sources.items():
        dtype = src.dtype(parts[0])
        shape = list(src.shape(parts[0]))
        if len(parts) > 1:
            shape[0] = sum(src.shape(p)[0] for p in parts)

        def load(parts=parts):
            if len(parts) == 1:
                return src.as_float32(parts[0])
            return np.concatenate([src.as_float32(p) for p in parts], axis=0)

        if dtype in FLOAT_TYPES and should_quantize(out_name, shape):
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
        elif dtype in FP8_TYPES:
            # An FP8 tensor outside the blocks: restore it to BF16.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="source .safetensors (BF16/FP16/FP32 or FP8)")
    parser.add_argument("output", help="destination .safetensors")
    parser.add_argument("--check", action="store_true",
                        help="print the relative quantization error of every converted tensor")
    args = parser.parse_args()

    src = SafetensorsFile(args.input)
    sources = plan_sources(src)
    outputs, stats = build_outputs(src, sources, args.check)
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


if __name__ == "__main__":
    main()
