"""
L1：生成机器可读的 DLC 契约（唯一事实来源）。

见 MASTERY_PLAN.md。契约**只能**由 snpe-dlc-info 转储解析得到，
禁止从 ONNX 推导、禁止手工维护。

用法：
    # 1) 为某个 DLC 生成转储（重活，需要内存，别和大任务并行）
    python dlc_contracts.py dump <part_name>
    # 2) 把已有转储解析成 dlc_contracts.json
    python dlc_contracts.py build
    # 3) 查看
    python dlc_contracts.py show [part_name]
"""
import json
import os
import re
import subprocess
import sys

SDK = r"D:\qairt\2.48.0.260626"
DLC_DIR = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline"
DUMP_DIR = r"D:\ZImage_Work\p0_experiments"
OUT_JSON = r"D:\LocalDreamZImage\scripts\dlc_contracts.json"

PARTS = ["text_encoder_part1", "text_encoder_part2", "text_encoder_part3", "text_encoder_part4",
         "transformer_part1a", "transformer_part1b", "transformer_part2", "vae_decoder",
         # 2026-08-22：part2 已按 #57 拆分部署，归因实验需要这两段的契约
         "transformer_part2a", "transformer_part2b"]

# 不在 DLC_DIR 约定路径下的 DLC，在这里登记真实位置（仍由 snpe-dlc-info 自动转储，
# 不违反"禁止手工维护契约"——手工登记的只是**文件在哪**，不是契约内容）
DLC_OVERRIDE = {
    "transformer_part2a": r"D:\ZImage_Work\p0_experiments\p2split\part2a_fixed_perrow.dlc",
    "transformer_part2b": r"D:\ZImage_Work\p0_experiments\p2split\part2b_fixed_perrow.dlc",
}

PAT = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*) \(data type: ([A-Za-z_0-9]+); "
    r"tensor dimension: \[([0-9,]*)\]; tensor type: (APP_WRITE|APP_READ)\)")

# ../docs/EXECUTION_MODEL.md 规则 1/2：宿主侧读写一律 float32，与 DLC 内部声明的类型无关。
HOST_ITEMSIZE = 4


def dump(part):
    """跑 snpe-dlc-info 生成转储。占内存，别和大任务并行。"""
    dlc = DLC_OVERRIDE.get(part) or f"{DLC_DIR}\\{part}\\{part}_quantized.dlc"
    if not os.path.isfile(dlc):
        raise FileNotFoundError(dlc)
    helper = f"{DUMP_DIR}\\_dlcinfo_helper.py"
    with open(helper, "w") as f:
        f.write(
            "import sys, runpy, types, onnx\n"
            "if not hasattr(onnx,'version') or not hasattr(onnx.version,'version'):\n"
            "    onnx.version = types.SimpleNamespace(version=onnx.__version__)\n"
            f"sys.argv=[r'{SDK}\\bin\\x86_64-windows-msvc\\snpe-dlc-info','-i',sys.argv[1]]\n"
            f"runpy.run_path(r'{SDK}\\bin\\x86_64-windows-msvc\\snpe-dlc-info', run_name='__main__')\n")
    env = dict(os.environ)
    env["QNN_SDK_ROOT"] = SDK
    env["PYTHONPATH"] = f"{SDK}\\lib\\python;" + env.get("PYTHONPATH", "")
    env["PATH"] = f"{SDK}\\bin\\x86_64-windows-msvc;{SDK}\\lib\\x86_64-windows-msvc;" + env["PATH"]
    env["PYTHONIOENCODING"] = "utf-8"
    out = f"{DUMP_DIR}\\{part}_dlcinfo.txt"
    with open(out, "w", encoding="utf-8") as fo:
        subprocess.run([sys.executable, helper, dlc], env=env, stdout=fo,
                       stderr=subprocess.STDOUT, check=False)
    print(f"dumped -> {out} ({os.path.getsize(out)} bytes)")


def parse_dump(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    ins, outs = {}, {}
    for name, dt, dims, kind in PAT.findall(txt):
        dims = [int(x) for x in dims.split(",") if x != ""]
        cnt = 1
        for d in dims:
            cnt *= d
        rec = {"dlc_dtype": dt, "shape": dims, "elems": cnt,
               "host_bytes": cnt * HOST_ITEMSIZE}
        (ins if kind == "APP_WRITE" else outs)[name] = rec
    return {"inputs": ins, "outputs": outs}


def build():
    contracts = {}
    for p in PARTS:
        path = f"{DUMP_DIR}\\{p}_dlcinfo.txt"
        if not os.path.isfile(path):
            print(f"  [跳过] {p}: 尚无转储（先跑 dump）")
            continue
        c = parse_dump(path)
        if not c["inputs"]:
            print(f"  [警告] {p}: 转储里没解析到输入，可能 dump 失败")
            continue
        contracts[p] = c
        print(f"  [OK] {p}: {len(c['inputs'])} 输入 / {len(c['outputs'])} 输出")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2, ensure_ascii=False)
    print(f"\n已写出 {OUT_JSON}（{len(contracts)}/{len(PARTS)} 个 DLC）")
    missing = [p for p in PARTS if p not in contracts]
    if missing:
        print(f"仍缺：{missing}")


def show(part=None):
    c = json.load(open(OUT_JSON, encoding="utf-8"))
    for p, v in c.items():
        if part and p != part:
            continue
        print("=" * 96)
        print(p)
        for kind in ("inputs", "outputs"):
            print(f"  [{kind}]")
            for n, r in sorted(v[kind].items()):
                print(f"    {n:<18} {r['dlc_dtype']:<10} {str(r['shape']):<22} "
                      f"host_bytes={r['host_bytes']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "dump":
        for p in (sys.argv[2:] or PARTS):
            dump(p)
    elif cmd == "build":
        build()
    elif cmd == "show":
        show(sys.argv[2] if len(sys.argv) > 2 else None)
