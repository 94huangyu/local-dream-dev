"""
通用 tap 工具：给 transformer_part1a.onnx 加任意 tap 输出，用 onnxruntime 跑 FP32 参考值。

用法:
    python trace_taps.py <输出npz名> <张量1> <张量2> ...

配套的量化侧（同一份输入字节）：
    snpe-net-run --container transformer_part1a_quantized.dlc --input_list <带%前缀的列表>

注意事项（踩过的坑）：
  1. tap 后的 onnx 必须存回 ZImage_QNN_Evidence\onnx\ 同一目录，否则外部权重相对路径解析不到。
  2. 请求的张量必须在已部署 DLC 里真实存在，否则 snpe-net-run 整个运行失败（error_code=204）。
     权威名单 = qairt-quantizer --dump_encoding_json 导出的那份 JSON。
  3. 千万不要 tap attention 的 score 张量（如 val_953/val_954，shape 含 4128x4128），
     单个张量就是 TB 级别。
"""
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
ONNX_DIR = EVIDENCE + r"\onnx"
RAW = EVIDENCE + r"\dlc_pipeline\transformer_part1a\calibration_raw\sample_0000"


def load_inputs():
    return {
        "latents": np.fromfile(RAW + r"\latents.raw", dtype=np.float32).reshape(1, 16, 128, 128),
        "timestep": np.fromfile(RAW + r"\timestep.raw", dtype=np.float32).reshape(1),
        "caption": np.fromfile(RAW + r"\caption.raw", dtype=np.float32).reshape(1, 32, 2560),
        "cap_pad_mask": np.fromfile(RAW + r"\cap_pad_mask.raw", dtype=np.int32).reshape(1, 32).astype(bool),
    }


def main():
    out_name, checkpoints = sys.argv[1], sys.argv[2:]
    src = ONNX_DIR + r"\transformer_part1a.onnx"
    patched = ONNX_DIR + f"\\transformer_part1a_taps_{out_name}.onnx"

    model = onnx.load(src, load_external_data=False)
    existing = {o.name for o in model.graph.output}
    for name in checkpoints:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    onnx.save(model, patched, save_as_external_data=False)

    t0 = time.time()
    sess = ort.InferenceSession(patched, providers=["CPUExecutionProvider"])
    print(f"session 加载 {time.time()-t0:.1f}s")

    t1 = time.time()
    res = sess.run(checkpoints, load_inputs())
    print(f"推理 {time.time()-t1:.1f}s")

    ref = dict(zip(checkpoints, res))
    dst = ONNX_DIR + f"\\{out_name}.npz"
    np.savez(dst, **ref)
    print(f"已保存 {dst}\n")
    for k in checkpoints:
        a = ref[k]
        print(f"{k:<34} {str(a.shape):<24} min={a.min():>12.4f} max={a.max():>12.4f} std={a.std():>10.4f}")


if __name__ == "__main__":
    main()
