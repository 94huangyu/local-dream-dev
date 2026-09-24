"""
定位量化误差的真实起点：测量 part1a 前 4 个 attention block 的残差张量。

背景（HANDOVER 13.8 / 13.9）：
  - 8.4 节只测了 add_54~add_138（attention #5~#12），并错误地把 add_54 标注成"第1层"。
  - 实际 part1a 共 23 个残差加法 / 12 个 attention，add_54 前面还有 4 个完整 block 从未测量。
  - 13.8 已证明误差不来自残差张量自身的编码（差 5,600~24,700 倍），所以必须往前找真正的起点。

本脚本负责 FP32 参考侧；量化侧用：
  snpe-net-run --container transformer_part1a_quantized.dlc --input_list <带 % 前缀的列表>
  （已验证 % 可以导出非图输出的中间张量）
"""
import time
import numpy as np
import onnx
import onnxruntime as ort

EVIDENCE = r"D:\ZImage_Work\ZImage_QNN_Evidence"
ONNX_DIR = EVIDENCE + r"\onnx"
RAW = EVIDENCE + r"\dlc_pipeline\transformer_part1a\calibration_raw\sample_0000"

# add_24 / add_45 在量化 DLC 里被融合掉了（不存在同名张量），所以不测
CHECKPOINTS = ["add_9", "add_12", "add_21", "add_32", "add_35", "add_42", "add_54"]

OUT_NPZ = ONNX_DIR + r"\early_block_reference.npz"


def load_inputs():
    """直接读 .raw，保证和 snpe-net-run 吃的是同一份字节"""
    return {
        "latents": np.fromfile(RAW + r"\latents.raw", dtype=np.float32).reshape(1, 16, 128, 128),
        "timestep": np.fromfile(RAW + r"\timestep.raw", dtype=np.float32).reshape(1),
        "caption": np.fromfile(RAW + r"\caption.raw", dtype=np.float32).reshape(1, 32, 2560),
        # cap_pad_mask.raw = 128 bytes / 32 元素 = int32，模型输入声明是 bool
        "cap_pad_mask": np.fromfile(RAW + r"\cap_pad_mask.raw", dtype=np.int32).reshape(1, 32).astype(bool),
    }


def main():
    src = ONNX_DIR + r"\transformer_part1a.onnx"
    # 必须存回同一目录，否则外部权重数据的相对路径解析不到
    patched = ONNX_DIR + r"\transformer_part1a_with_early_taps.onnx"

    print(f"=== 给 {src} 加 tap 输出 ===")
    model = onnx.load(src, load_external_data=False)
    existing = {o.name for o in model.graph.output}
    for name in CHECKPOINTS:
        if name not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    onnx.save(model, patched, save_as_external_data=False)
    print(f"已写出 {patched}")

    t0 = time.time()
    sess = ort.InferenceSession(patched, providers=["CPUExecutionProvider"])
    print(f"session 加载耗时 {time.time()-t0:.1f}s")

    inputs = load_inputs()
    for k, v in inputs.items():
        print(f"  input {k:<14} {str(v.shape):<20} {v.dtype}")

    t1 = time.time()
    res = sess.run(CHECKPOINTS, inputs)
    print(f"推理耗时 {time.time()-t1:.1f}s")

    ref = dict(zip(CHECKPOINTS, res))
    np.savez(OUT_NPZ, **ref)
    print(f"\n参考值已保存到 {OUT_NPZ}")
    print()
    print(f"{'张量':<10} {'shape':<22} {'min':>12} {'max':>12} {'std':>12}")
    print("-" * 72)
    for k in CHECKPOINTS:
        a = ref[k]
        print(f"{k:<10} {str(a.shape):<22} {a.min():>12.4f} {a.max():>12.4f} {a.std():>12.4f}")


if __name__ == "__main__":
    main()
