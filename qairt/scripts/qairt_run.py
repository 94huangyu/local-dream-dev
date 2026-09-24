"""qairt-converter / qairt-quantizer 的调用垫片。

存在的理由（①实测）：本机 onnx 1.22 没有 `onnx.version.version` 属性，
SDK 2.48 的入口脚本直接访问它 ⇒ `Encountered Error: module 'onnx' has no attribute 'version'`。
原 `qairt_wrapper.py` 只能跑 quantizer（路径写死），这里改成第一个参数选工具。

用法:
    python qairt_run.py converter  <args...>
    python qairt_run.py quantizer  <args...>
    python qairt_run.py dlc-info   <dlc>
需要 PYTHONPATH=<SDK>/lib/python。
"""
import runpy
import sys
import types

import onnx

if not hasattr(onnx, "version") or not hasattr(onnx.version, "version"):
    onnx.version = types.SimpleNamespace(version=onnx.__version__)

BIN = r"D:\qairt\2.48.0.260626\bin\x86_64-windows-msvc"
TOOLS = {
    "converter": BIN + r"\qairt-converter",
    "quantizer": BIN + r"\qairt-quantizer",
    "dlc-info": BIN + r"\snpe-dlc-info",
}

tool = TOOLS[sys.argv[1]]
rest = sys.argv[2:]
if sys.argv[1] == "dlc-info":
    rest = ["-i"] + rest
sys.argv = [tool] + rest
runpy.run_path(tool, run_name="__main__")
