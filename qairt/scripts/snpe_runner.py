"""
L2：跑 DLC 的【唯一入口】。见 MASTERY_PLAN.md。

本项目禁止各脚本自己拼 input_list / 自己写 .raw —— 那样做已经导致过多次静默错误。

本模块把四类错误从"撞了才知道"变成"跑不起来"：

  1. feed 的键集合 != 契约输入集合        -> 抛错（曾因多喂 latents_shape 失败）
  2. 输入 .raw 字节数 != 契约要求          -> 抛错（曾把 Bool_8 写成 uint8）
  3. 请求的输出名不在契约/encoding 名单     -> 抛错（曾请求被融合掉的 add_24）
  4. 输出 .raw 字节数 != 元素数 x 4        -> 抛错【最关键】
     snpe 在输入尺寸不对时【不报错】，只把所有输出按比例缩小；
     这是唯一能抓住它的检查。2026-08-14 实测：32/128 的输入导致全部输出缩到 1/4。
"""
import json
import os
import subprocess
import sys

import numpy as np

SDK = r"D:\qairt\2.48.0.260626"
SNPE = f"{SDK}\\bin\\x86_64-windows-msvc\\snpe-net-run.exe"
QNN_LIB = f"{SDK}\\lib\\x86_64-windows-msvc"
QNN_BIN = f"{SDK}\\bin\\x86_64-windows-msvc"
DLC_DIR = r"D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline"
CONTRACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlc_contracts.json")

HOST_ITEMSIZE = 4   # ../docs/EXECUTION_MODEL.md 规则 1/2：宿主侧一律 float32


class ContractError(RuntimeError):
    pass


def load_contract(part):
    if not os.path.isfile(CONTRACTS):
        raise ContractError(f"契约文件不存在：{CONTRACTS}，先跑 dlc_contracts.py build")
    c = json.load(open(CONTRACTS, encoding="utf-8"))
    if part not in c:
        raise ContractError(
            f"契约里没有 {part}。已有：{sorted(c)}。\n"
            f"请先跑：python dlc_contracts.py dump {part} && python dlc_contracts.py build")
    return c[part]


def run(part, feeds, want, workdir, extra_shapes=None, dlc_path=None):
    """
    part        : 如 "transformer_part1a"
    feeds       : {输入名: ndarray}，键集合必须恰好等于契约的输入集合
    want        : 要导出的张量名列表（可含非图输出的中间张量，见规则 5）
    workdir     : 本次运行的工作目录
    extra_shapes: {中间张量名: shape}，用于契约里没有的中间张量
    返回        : {张量名: ndarray}
    """
    ct = load_contract(part)
    extra_shapes = extra_shapes or {}

    # ---- 检查 1：输入键集合必须完全一致 ----
    need, got = set(ct["inputs"]), set(feeds)
    if need != got:
        raise ContractError(
            f"[{part}] 输入集合不匹配\n"
            f"  契约要求 : {sorted(need)}\n"
            f"  实际提供 : {sorted(got)}\n"
            f"  缺少     : {sorted(need - got)}\n"
            f"  多余     : {sorted(got - need)}   <- 多喂会报 invalid input size")

    os.makedirs(workdir, exist_ok=True)
    raws = {}
    for name, arr in feeds.items():
        spec = ct["inputs"][name]
        a = np.ascontiguousarray(np.asarray(arr).astype(np.float32))
        # ---- 检查 2：输入字节数 ----
        if a.size != spec["elems"]:
            raise ContractError(
                f"[{part}] 输入 {name} 元素数 {a.size} != 契约 {spec['elems']} "
                f"(shape 应为 {spec['shape']})")
        p = os.path.join(workdir, f"{name}.raw")
        a.tofile(p)
        n = os.path.getsize(p)
        if n != spec["host_bytes"]:
            raise ContractError(f"[{part}] 输入 {name} 写出 {n} 字节 != 契约 {spec['host_bytes']}")
        raws[name] = p

    # ---- 检查 3：输出名 ----
    unknown = [w for w in want if w not in ct["outputs"] and w not in extra_shapes]
    if unknown:
        raise ContractError(
            f"[{part}] 请求了契约里没有的输出：{unknown}\n"
            f"  契约输出：{sorted(ct['outputs'])}\n"
            f"  若是中间张量，请用 extra_shapes 显式给出 shape，并确认它在\n"
            f"  qairt-quantizer --dump_encoding_json 的名单里（被融合的张量不存在）")

    lst = os.path.join(workdir, "input_list.txt")
    with open(lst, "w") as f:
        f.write("%" + " ".join(want) + "\n")
        f.write(" ".join(f"{k}:={v}" for k, v in raws.items()) + "\n")

    outdir = os.path.join(workdir, "out")
    dlc = dlc_path or f"{DLC_DIR}\\{part}\\{part}_quantized.dlc"
    env = dict(os.environ, PATH=f"{QNN_LIB};{QNN_BIN};" + os.environ["PATH"])
    r = subprocess.run([SNPE, "--container", dlc, "--input_list", lst, "--output_dir", outdir],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise ContractError(f"[{part}] snpe-net-run 失败 (rc={r.returncode})\n"
                            f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    res = {}
    for w in want:
        shape = ct["outputs"][w]["shape"] if w in ct["outputs"] else list(extra_shapes[w])
        elems = int(np.prod(shape))
        path = os.path.join(outdir, "Result_0", f"{w}.raw")
        nbytes = os.path.getsize(path)
        # ---- 检查 4：输出字节数（抓静默缩批的唯一防线）----
        if nbytes != elems * HOST_ITEMSIZE:
            raise ContractError(
                f"[{part}] 输出 {w} 为 {nbytes} 字节，期望 {elems*HOST_ITEMSIZE}"
                f"（比值 {nbytes/(elems*HOST_ITEMSIZE):.4f}）。\n"
                f"  这通常【不是】输出的问题，而是某个输入的 .raw 尺寸不对，\n"
                f"  snpe 用「输入字节数 / 期望字节数」推断批次并静默缩小了整张图。\n"
                f"  见 ../docs/EXECUTION_MODEL.md 规则 1。")
        res[w] = np.fromfile(path, dtype=np.float32).reshape(shape)
    return res


if __name__ == "__main__":
    c = json.load(open(CONTRACTS, encoding="utf-8"))
    print(f"已载入契约：{sorted(c)}")
    for p, v in c.items():
        print(f"  {p}: 输入 {sorted(v['inputs'])}")
