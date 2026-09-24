"""通用 QAIRT 工具启动器：`python qairt_tool.py <工具名> <参数...>`

为什么需要它（①实测，不是预防性代码）：
SDK 2.48 的 `qairt-converter` 里有
  `if parse_version(onnx.version.version) >= parse_version("1.19.0")`
而本机 onnx 1.22.0 **不再提供 `onnx.version` 子模块** ⇒ 直接跑报
`AttributeError: module 'onnx' has no attribute 'version'`。

项目里早有 `dlc_pipeline/*/qairt_wrapper.py` 在做同一件事，但它把工具名
**硬编码成 qairt-quantizer**，没法用于 converter。这里做成通用版。

用法:
  python scripts/qairt_tool.py qairt-converter --input_network a.onnx --output_path a.dlc
  python scripts/qairt_tool.py qairt-quantizer --input_dlc a.dlc ...
"""
import os
import runpy
import sys
import types

import onnx

SDK = r"D:\qairt\2.48.0.260626"
BIN = SDK + r"\bin\x86_64-windows-msvc"
LIB = SDK + r"\lib\x86_64-windows-msvc"
PYLIB = SDK + r"\lib\python"

# 2026-08-21：原版没设 PYTHONPATH => ModuleNotFoundError: No module named "qti"。
# qairt_run.py 的文档串里写着"需要 PYTHONPATH=<SDK>/lib/python"，但要人记得去设。
# 按 #73 的做法根治：让遗漏不可能——这里自己装好，调用方什么都不用管。
# 同上：SDK 工具会往 stdout 打非 GBK 字符，在中文 Windows 控制台直接 UnicodeEncodeError。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if PYLIB not in sys.path:
    sys.path.insert(0, PYLIB)
os.environ["PYTHONPATH"] = PYLIB + os.pathsep + os.environ.get("PYTHONPATH", "")
# 臂 B 那条路径需要它，否则 getPathForTargetConfig 抛 RuntimeError（2026-08-21 实测）
os.environ.setdefault("QNN_SDK_ROOT", SDK)
os.environ.setdefault("SNPE_ROOT", SDK)
os.environ["PATH"] = LIB + os.pathsep + BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory") and os.path.isdir(LIB):
    os.add_dll_directory(LIB)

if not hasattr(onnx, "version") or not hasattr(onnx.version, "version"):
    onnx.version = types.SimpleNamespace(version=onnx.__version__)

tool = sys.argv[1]
path = f"{BIN}\\{tool}"
sys.argv = [path] + sys.argv[2:]
runpy.run_path(path, run_name="__main__")
