"""
从 snpe-dlc-info 的转储里提取每个 DLC 的【完整输入/输出契约】。

动机：本项目多次因为"用 ONNX 的知识推断 DLC 行为"而喂错数据（Bool_8 被当 float32、
latents_shape 在 DLC 里不存在、输出一律 float32 等）。这些都是撞了才查。
本脚本把契约变成可查的事实表，而不是每次现推。

用法：
    python extract_dlc_contract.py <dlcinfo转储文件> [更多...]
"""
import re
import sys

# APP_WRITE = 图的输入（宿主写入）；APP_READ = 图的输出（宿主读取）
PAT = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*) \(data type: ([A-Za-z_0-9]+); "
    r"tensor dimension: \[([0-9,]*)\]; tensor type: (APP_WRITE|APP_READ)\)")

# snpe 数据类型 -> 每元素字节数（这是喂 .raw 时唯一要紧的东西）
WIDTH = {"Bool_8": 1, "uFxp_8": 1, "sFxp_8": 1,
         "uFxp_16": 2, "sFxp_16": 2, "Float_16": 2,
         "uFxp_32": 4, "sFxp_32": 4, "Float_32": 4, "Int_32": 4, "Int_64": 8}


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    seen = {}
    for name, dt, dims, kind in PAT.findall(txt):
        dims = tuple(int(x) for x in dims.split(",") if x != "")
        seen.setdefault((kind, name), (dt, dims))
    return seen


for path in sys.argv[1:]:
    print("=" * 100)
    print(path.split("\\")[-1].replace("_dlcinfo.txt", ""))
    print("=" * 100)
    seen = parse(path)
    for kind, label in [("APP_WRITE", "输入（宿主写 .raw）"), ("APP_READ", "输出（宿主读 .raw）")]:
        rows = [(n, dt, dims) for (k, n), (dt, dims) in sorted(seen.items()) if k == kind]
        if not rows:
            print(f"\n  【{label}】(未在转储中找到)")
            continue
        print(f"\n  【{label}】")
        print(f"  {'张量名':<18} {'DLC声明类型':<12} {'shape':<22} {'元素数':>10} "
              f"{'声明字节宽':>10} {'.raw应为':>12}")
        print("  " + "-" * 92)
        for n, dt, dims in rows:
            cnt = 1
            for d in dims:
                cnt *= d
            w = WIDTH.get(dt)
            if kind == "APP_WRITE":
                note = f"{cnt*w} 字节" if w else "未知宽度"
            else:
                # 实测：snpe-net-run 的输出一律写 float32，即使张量声明是 Bool_8
                note = f"{cnt*4} 字节(f32)"
            print(f"  {n:<18} {dt:<12} {str(dims):<22} {cnt:>10} "
                  f"{(str(w)+' B') if w else '?':>10} {note:>12}")
    print()
