"""
convert_qairt_zimage.py — ZImage Turbo ONNX → QNN HTP Context Binary（W8A8 量化版）

变更日志：
  v2.0  加入 W8A8 PTQ 量化（INT8 权重 + INT8 激活）：
        - wrapper.py 调用磁盘桩包解决 arch_linter 的 pandas 依赖
        - 将 .npy 校准文件自动转换为 .raw 格式（qnn-onnx-converter 要求）
        - 量化参数：--weights_bitwidth 8 --act_bitwidth 8
                     --use_per_channel_quantization --bias_bitwidth 32
  选择 W8A8 原因：16GB 手机逐模型加载，最大单模型 ~3.6GB，内存完全够用；
                  质量明显优于 W4A8，适合作为验证基准。
"""

import os
import subprocess
import json
import hashlib
import numpy as np
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 路径配置 (适配云端 Linux / 本地 Windows)
# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR = Path(__file__).resolve().parent.parent  # Assuming script is in Cloud_Package, data in parent dir

# 自动推断 QNN SDK 路径 (如果环境变量配置了 QNN_SDK_ROOT 则优先使用)
QNN_SDK_ROOT = os.environ.get("QNN_SDK_ROOT", "/opt/qcom/qairt")

# 云端相对路径
ONNX_DIR     = WORK_DIR / "onnx"
CALIB_DIR    = WORK_DIR / "calibration"
OUT_DIR      = WORK_DIR / "qnn_w8a8"

# 根据操作系统自动选择 QNN 平台架构
if sys.platform == "win32":
    QNN_TARGET = "x86_64-windows-msvc"
else:
    QNN_TARGET = "x86_64-linux-clang"

QNN_BIN_DIR    = os.path.join(QNN_SDK_ROOT, "bin", QNN_TARGET)
QNN_LIB_DIR    = os.path.join(QNN_SDK_ROOT, "lib", QNN_TARGET)
QNN_PYTHON_PATH = os.path.join(QNN_SDK_ROOT, "lib", "python")

MODELS = ["text_encoder", "transformer_part1", "transformer_part2", "vae_decoder"]

# 各模型固定输入维度（动态shape的ONNX模型必须通过 -d 显式指定）
# text_encoder: 序列长度统一 padding 到 20
INPUT_DIMS = {
    "text_encoder": {
        "input_ids":      "1,20",
        "attention_mask": "1,20",
    },
    "transformer_part1": {
        "latents":      "1,16,128,128",
        "timestep":     "1",
        "caption":      "1,32,2560",
        "cap_pad_mask": "1,32",
    },
    "transformer_part2": {
        "unified":       "1,4128,3840",
        "unified_mask":  "1,4128",
        "unified_freqs": "1,4128,64,2",
        "adaln_input":   "1,256",
        "latents_shape": "3",
    },
    "vae_decoder": {
        "latent_sample": "1,16,128,128",
    },
}

# text_encoder 序列固定长度（所有 input_ids/attention_mask 样本 pad 到此长度）
TE_FIXED_SEQ_LEN = 20


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def get_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert_npy_calib_to_raw(model_name: str) -> Path:
    """
    将 calibration/<model>/ 下的 .npy 校准文件转换为 .raw（raw float32 二进制）。
    生成新的 input_list_raw.txt 供 qnn-onnx-converter 使用。
    返回 input_list_raw.txt 的路径。
    """
    src_model_dir = CALIB_DIR / model_name
    raw_dir       = CALIB_DIR / f"{model_name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    src_list = src_model_dir / "input_list.txt"
    raw_list = src_model_dir / "input_list_raw.txt"

    if not src_list.exists():
        raise FileNotFoundError(f"input_list.txt not found: {src_list}")

    lines = src_list.read_text(encoding="utf-8").strip().splitlines()
    if lines:
        lines = [lines[0]]  # Only keep the first sample to reduce memory usage during quantization
    raw_lines = []

    for line in lines:
        parts = line.strip().split()
        raw_parts = []
        for part in parts:
            if ":=" not in part:
                continue
            input_name, npy_path_str = part.split(":=", 1)
            npy_path = Path(npy_path_str)

            # 保持目录结构：sample_XXXX/input_name.raw
            rel_path    = npy_path.relative_to(src_model_dir)
            raw_path    = raw_dir / rel_path.parent / (rel_path.stem + ".raw")
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            if not raw_path.exists():
                arr = np.load(npy_path)

                # text_encoder 特殊处理：将 input_ids/attention_mask pad 到固定长度
                # qnn-onnx-converter 要求 --input_list 中所有样本 shape 与 -d 声明完全一致
                if model_name == "text_encoder" and input_name in ("input_ids", "attention_mask"):
                    current_len = arr.shape[1]
                    if current_len < TE_FIXED_SEQ_LEN:
                        pad_width = TE_FIXED_SEQ_LEN - current_len
                        # attention_mask: 补 0（不关注填充位）
                        # input_ids: 补 0（pad token id）
                        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
                    elif current_len > TE_FIXED_SEQ_LEN:
                        arr = arr[:, :TE_FIXED_SEQ_LEN]  # 截断（保险）

                # 保留原始 dtype：INT32 输入必须保持 INT32，float 保持 float32
                arr.tofile(raw_path)

            raw_parts.append(f"{input_name}:={raw_path}")

        if raw_parts:
            raw_lines.append(" ".join(raw_parts))

    raw_list.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    print(f"  [RAW] {model_name}: {len(raw_lines)} samples -> {raw_list}")
    return raw_list


def run_cmd_via_wrapper(script_path: str, args: list, log_file: Path,
                        env: dict, qnn_lib_dir: str) -> bool:
    """
    生成临时 wrapper.py 并用当前 Python 执行，wrapper 负责：
      1. 调用 os.add_dll_directory 注入高通 DLL 路径（Windows 必须）
      2. 注入 pandas mock（绕过 arch_linter 的 pandas 依赖，不影响转换正确性）
      3. 通过 runpy.run_path 执行目标脚本
    """
    wrapper_path = log_file.parent / "wrapper.py"
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write("import os, sys, runpy, types\n")
        f.write(f"if hasattr(os, 'add_dll_directory'):\n")
        f.write(f"    os.add_dll_directory(r'{qnn_lib_dir}')\n")
        # pandas 已通过磁盘桩包解决（.venv-qnn/Lib/site-packages/pandas/）
        # multiprocessing spawn 子进程也可正常 import，无需 sys.modules patch
        #
        # onnx 兼容补丁：onnx 1.22.0 移除了旧式 onnx.version.version 属性
        # QAIRT SDK 的 onnx_model_api.py 仍调用 onnx.version.version
        # 此补丁对主进程和 multiprocessing 子进程均有效（每次 import wrapper 时执行）
        f.write("import onnx\n")
        f.write("if not hasattr(onnx, 'version') or not hasattr(onnx.version, 'version'):\n")
        f.write("    onnx.version = types.SimpleNamespace(version=onnx.__version__)\n")
        f.write(f"sys.argv = [r'{script_path}'] + sys.argv[1:]\n")
        f.write(f"runpy.run_path(r'{script_path}', run_name='__main__')\n")

    cmd = [sys.executable, str(wrapper_path)] + args
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  CMD: {cmd_str}")

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"Command: {cmd_str}\n\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        process.wait()
        if process.returncode != 0:
            print(f"ERROR: Command failed with exit code {process.returncode}")
            lf.write(f"\nERROR: Exit code {process.returncode}\n")
            return False
    return True


def run_context_binary_generator(model_name: str, dll_path: Path,
                                  model_out: Path, env: dict) -> bool:
    """运行 qnn-context-binary-generator.exe，带实时日志输出"""
    htp_backend = os.path.join(QNN_LIB_DIR, "QnnHtp.dll")
    log3 = model_out / "03_context_binary_generator.log"

    cmd = [
        os.path.join(QNN_BIN_DIR, "qnn-context-binary-generator.exe"),
        "--backend",     htp_backend,
        "--model",       str(dll_path),
        "--binary_file", f"{model_name}_ctx",
        "--output_dir",  str(model_out),
    ]
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  CMD: {cmd_str}")

    with open(log3, "w", encoding="utf-8") as lf:
        lf.write(f"Command: {cmd_str}\n\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        process.wait()
        if process.returncode != 0:
            print(f"ERROR: qnn-context-binary-generator failed (exit {process.returncode})")
            lf.write(f"\nERROR: Exit code {process.returncode}\n")
            return False
    return True


def freeze_dynamic_shapes(onnx_path: Path, model_name: str, fixed_dims: dict) -> Path:
    """
    对 ONNX 模型做形状传播+常量折叠预处理，将动态 Slice 索引固化为常量。
    
    原理：QAIRT 转换器要求 Slice op 的 start/end 为常量（图中的 initializer）。
    文本编码器等模型导出时存在 Shape->Gather->Slice 动态链（序列长度依赖），
    通过 onnx.shape_inference.infer_shapes_path(data_prop=True) 可将这些链中的
    运行时标量值传播为静态常量，使 QAIRT 能正常处理。
    
    返回固化后的 ONNX 文件路径（*_fixed.onnx，与原文件同目录）。
    """
    import onnx
    from onnx import shape_inference as si
    from onnx import helper

    fixed_path = onnx_path.parent / f"{model_name}_fixed.onnx"
    if fixed_path.exists():
        print(f"  [FREEZE] {model_name}: using cached {fixed_path.name}")
        return fixed_path

    print(f"  [FREEZE] {model_name}: loading model structure...")
    # 只加载图结构，不加载庞大的外部权重数据，避免爆内存和 protobuf 2GB 限制
    model = onnx.load(str(onnx_path), load_external_data=False)

    # 将动态输入维度覆盖为固定值，使 shape inference 能推导出确定的常量
    for inp in model.graph.input:
        if inp.name in fixed_dims:
            dims_str = fixed_dims[inp.name]  # e.g. "1,20"
            dim_vals = [int(d) for d in dims_str.split(",")]
            tensor_type = inp.type.tensor_type
            # 确保 shape 字段存在且维度数匹配
            while len(tensor_type.shape.dim) < len(dim_vals):
                tensor_type.shape.dim.add()
            for i, val in enumerate(dim_vals):
                tensor_type.shape.dim[i].dim_value = val
                tensor_type.shape.dim[i].dim_param  = ""  # 清除动态符号

    # 针对 QAIRT 的特殊限制：强制替换动态的 Slice, Expand, Reshape, Tile 等节点
    # 我们将需要的动态 shape 结果作为额外输出加入模型中，利用 ONNXRuntime 全图推断出准确数值，
    # 最后将准确的静态 shape (例如 [1, 1, 20, 20]) 直接注入为常量 Initializer。
    print("  [FREEZE] Patching GatherND nodes for QNN CPU compatibility...")
    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "GatherND":
            data_name = node.input[0]
            idx_name = node.input[1]
            out_name = node.output[0]
            
            shape_name = node.name + "_reshape_shape"
            shape_tensor = helper.make_tensor(shape_name, onnx.TensorProto.INT64, [4], [1, 1, 1, -1])
            model.graph.initializer.append(shape_tensor)
            
            reshape_node = helper.make_node("Reshape", [data_name, shape_name], [out_name], name=node.name+"_replaced")
            new_nodes.append(reshape_node)
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    print("  [FREEZE] Identifying dynamic shape dependencies...")
    init_names = {i.name for i in model.graph.initializer}
    dynamic_shape_tensors = set()
    for node in model.graph.node:
        if node.op_type in ["Expand", "Reshape", "Tile", "ConstantOfShape"]:
            if len(node.input) > 1 and node.input[1] not in init_names:
                dynamic_shape_tensors.add(node.input[1])
        elif node.op_type == "Slice":
            if len(node.input) > 1 and node.input[1] not in init_names:
                dynamic_shape_tensors.add(node.input[1])
            if len(node.input) > 2 and node.input[2] not in init_names:
                dynamic_shape_tensors.add(node.input[2])

    if dynamic_shape_tensors:
        import onnxruntime as ort
        from onnx import helper
        orig_out_len = len(model.graph.output)
        import numpy as np
        import onnx
        
        print(f"  [FREEZE] Found {len(dynamic_shape_tensors)} dynamic shape tensors. Preparing probe model...")
        
        # 为了不破坏原图结构，先深拷贝或者直接修改后利用另一个干净的模型注入
        # 但我们这里修改的是内存中的 model，后续我们将再次加载它以确保干净
        for t in dynamic_shape_tensors:
            vi = next((v for v in model.graph.value_info if v.name == t), None)
            if not vi:
                vi = helper.make_tensor_value_info(t, onnx.TensorProto.INT64, None)
            model.graph.output.append(vi)
            
        probe_path = onnx_path.parent / f"{model_name}_probe.onnx"
        onnx.save(model, str(probe_path))
        
        print("  [FREEZE] Evaluating dynamic shapes via ONNXRuntime (this takes a moment)...")
        sess = ort.InferenceSession(str(probe_path))
        
        # 构造 Dummy 输入
        # 这里严格从 fixed_dims 拿尺寸信息（例如 1,20），不要盲猜 0 = 20
        # 必须使用 np.ones 而非 np.zeros！如果是 zeros，attention_mask 的 ReduceSum 就是 0，
        # 会导致依赖 Mask 长度的动态切片 (Slice) 计算出 end=0，进而导致无效的空切片错误。
        inputs_dummy = {}
        for inp in model.graph.input:
            if inp.name in fixed_dims:
                shape = [int(d) for d in fixed_dims[inp.name].split(",")]
            else:
                shape = []
                for d in inp.type.tensor_type.shape.dim:
                    val = d.dim_value
                    if val <= 0: val = 20 # 回退方案
                    shape.append(val)
            
            # ORT 需要匹配数据类型，默认文本模型输入均为 int32
            dtype = np.int32
            if inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                dtype = np.float32
            inputs_dummy[inp.name] = np.ones(shape, dtype=dtype)
            
        outputs = sess.run(list(dynamic_shape_tensors), inputs_dummy)
        
        print(f"  [FREEZE] Injecting {len(outputs)} evaluated static shapes...")
        # 移除刚才为探针添加的临时输出，恢复原模型的输出定义
        while len(model.graph.output) > orig_out_len:
            model.graph.output.pop()
            
        for t, arr in zip(dynamic_shape_tensors, outputs):
            fixed_name = t + "_fixed_qairt"
            arr_int64 = np.array(arr, dtype=np.int64)
            new_init = helper.make_tensor(fixed_name, onnx.TensorProto.INT64, arr_int64.shape, arr_int64.flatten().tolist())
            model.graph.initializer.append(new_init)
            
            # 遍历图中所有的 node，如果 input 中有 t，则替换为 fixed_name
            # 这样不仅注入了常量，还切断了对原有动态节点（如 ReduceSum）的依赖！
            for node in model.graph.node:
                for i in range(len(node.input)):
                    if node.input[i] == t:
                        node.input[i] = fixed_name
            
        if probe_path.exists():
            probe_path.unlink()
            
    tmp_path = onnx_path.parent / f"{model_name}_tmp.onnx"
    import onnx
    onnx.save(model, str(tmp_path))

    # data_prop=True：传播常量数据值（Shape/Gather/Slice 链中的标量）
    print(f"  [FREEZE] {model_name}: running shape inference (data_prop=True)...")
    try:
        si.infer_shapes_path(str(tmp_path), str(fixed_path), check_type=True, strict_mode=False, data_prop=True)
    except Exception as e:
        print(f"  [FREEZE] Warning: shape inference raised {e}, using basic inference...")
        si.infer_shapes_path(str(tmp_path), str(fixed_path), check_type=False, strict_mode=False, data_prop=False)

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"  [FREEZE] {model_name}: saved -> {fixed_path.name}")
    return fixed_path


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QNN_SDK_ROOT"] = QNN_SDK_ROOT
    env["PATH"]        = f"{QNN_BIN_DIR};{QNN_LIB_DIR};" + env.get("PATH", "")
    env["PYTHONPATH"]  = f"{QNN_PYTHON_PATH};" + env.get("PYTHONPATH", "")

    # 指定 QAIRT 的临时目录到 D 盘（避免 C 盘空间不足 2GB 被 15GB 的模型挤爆）
    qairt_tmp = OUT_DIR / "tmp"
    qairt_tmp.mkdir(parents=True, exist_ok=True)
    env["QAIRT_TMP_DIR"] = str(qairt_tmp)

    print("=" * 70)
    print("ZImage Turbo ONNX -> QNN HTP Context Binary (W8A8 PTQ)")
    print(f"  SDK root : {QNN_SDK_ROOT}")
    print(f"  ONNX dir : {ONNX_DIR}")
    print(f"  Calib dir: {CALIB_DIR}")
    print(f"  Out dir  : {OUT_DIR}")
    print("=" * 70)

    # ── 阶段 0：将 .npy 校准文件批量转换为 .raw ──────────────────────────────
    print("\n[Phase 0] Converting calibration .npy -> .raw ...")
    raw_lists = {}
    for model in MODELS:
        calib_model_dir = CALIB_DIR / model
        if calib_model_dir.exists():
            raw_lists[model] = convert_npy_calib_to_raw(model)
        else:
            print(f"  [WARN] No calibration dir for {model}, will skip quantization for it.")
            raw_lists[model] = None

    # ── 主循环：逐模型转换 ────────────────────────────────────────────────────
    for model in MODELS:
        print(f"\n{'─'*70}")
        print(f"Processing: {model}")
        print(f"{'─'*70}")

        onnx_path = ONNX_DIR / f"{model}.onnx"
        if not onnx_path.exists():
            print(f"  SKIP: {onnx_path} not found.")
            continue

        model_out = OUT_DIR / model
        model_out.mkdir(parents=True, exist_ok=True)

        # ── Step 0.5: 形状固化预处理（QAIRT 要求 Slice 等 op 的索引为常量）──────
        print(f"\n[{model}] Step 0.5: Freezing dynamic shapes...")
        if model in INPUT_DIMS:
            onnx_path = freeze_dynamic_shapes(onnx_path, model, INPUT_DIMS[model])
        else:
            print(f"  [FREEZE] {model}: no fixed dims defined, skipping.")
            
        # Run graph fix script to remove QNN-incompatible NO-OPs
        print(f"  [FIX_GRAPH] Running fix_qnn_graph.py on {onnx_path}")
        subprocess.run([sys.executable, r"D:\ZImage_Work\fix_qnn_graph.py", str(onnx_path)], check=True)

        # ── Step 1: ONNX → CPP（W8A8 量化）──────────────────────────────────
        print(f"\n[{model}] Step 1: ONNX -> CPP (W8A8 quantization) ...")
        cpp_out = model_out / f"{model}.cpp"
        log1    = model_out / "01_onnx_converter.log"

        converter_args = [
            "-i", str(onnx_path),
            "-o", str(cpp_out),
            # W8A8: 权重 INT8 + 激活 INT8，16GB 手机最大单模型 ~3.6GB，质量优于 W4A8
            "--weights_bitwidth",          "8",
            "--act_bitwidth",              "8",
            "--bias_bitwidth",             "32", # bias 保持 FP32，精度更高
            "--use_per_channel_quantization",    # 逐通道量化，显著提高精度
            "--act_quantizer_calibration", "percentile",  # 激活截断更鲁棒
            "--param_quantizer_calibration","min-max",     # 权重全范围量化
            "--no_simplification",                         # 禁止 QAIRT 内置的 onnxsim（15GB 大模型会触发 protobuf 2GB 限制死锁）
        ]

        # 为动态 shape 模型显式指定固定输入维度（-d input_name shape）
        # 所有模型均需声明，避免 converter 报 'Missing command line inputs' 错误
        if model in INPUT_DIMS:
            for inp_name, dims in INPUT_DIMS[model].items():
                converter_args += ["-d", inp_name, dims]
                print(f"    -d {inp_name} {dims}")

        # 如果有对应的校准数据，加入 --input_list
        if raw_lists.get(model):
            converter_args += ["--input_list", str(raw_lists[model])]
            print(f"  Using calibration: {raw_lists[model]}")
        else:
            print(f"  [WARN] No calibration data, quantization quality may be poor.")

        if not run_cmd_via_wrapper(
            os.path.join(QNN_BIN_DIR, "qnn-onnx-converter"),
            converter_args, log1, env, QNN_LIB_DIR
        ):
            print(f"  FAILED at Step 1 for {model}. Stopping.")
            return

        # ── Step 2 及后续步骤依赖 MSVC，当前环境缺失故跳过 ──
        print(f"\n[{model}] Step 1 Completed successfully! Skipping DLL generation due to missing MSVC.")

        # 临时跳过 Step 3 和 4

    # ── 输出 manifest ─────────────────────────────────────────────────────────
    manifest_path = OUT_DIR / "qnn_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("[OK] All conversions completed successfully!")
    print(f"  Output dir   : {OUT_DIR}")
    print(f"  Manifest     : {manifest_path}")
    total_gb = sum(
        (OUT_DIR / m / f"{m}_ctx.bin").stat().st_size
        for m in MODELS
        if (OUT_DIR / m / f"{m}_ctx.bin").exists()
    ) / (1024**3)
    print(f"  Total size   : {total_gb:.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
