"""在改任何量化配置【之前】，先查官方 OpDef 支持表：本模型用到的算子，
在目标数据类型组合下到底支不支持。

依据：D:\\qairt\\<ver>\\docs\\QAIRT-Docs\\QNN\\OpDef\\HtpOpDefSupplement.html

教训（2026-08-15，真实）：没查这张表就用 `--param_quantizer_schema symmetric`
重量化 part1b，花了 1 小时量化 + 10 分钟建图，最后撞上
"RmsNorm ... None of the combinations match" —— 而表里早就写明
UFxp16 激活下权重只允许 UFxp16 / SFxp16 / UFxp8，没有 SFxp8。

用法: python check_opdef_support.py [算子名 ...]
"""
import html
import io
import re
import sys

DOC = r"D:\qairt\2.48.0.260626\docs\QAIRT-Docs\QNN\OpDef\HtpOpDefSupplement.html"

# 本项目 transformer 三段实际用到的算子（由 map_opid 统计得出）
OUR_OPS = ["RmsNorm", "MatMul", "FullyConnected", "Softmax", "ElementWiseNeuron",
           "LayerNorm", "Gather", "Concat", "Reshape", "Transpose", "Split",
           "Convert", "ElementWiseAdd", "ElementWiseMultiply"]

SHORT = [("QNN_DATATYPE_", ""), ("FIXED_POINT_", "Fxp"),
         ("FLOAT_", "FP"), ("BFLOAT_", "BF")]


def short(t):
    for a, b in SHORT:
        t = t.replace(a, b)
    return t


def load_lines():
    s = io.open(DOC, encoding="utf-8", errors="replace").read()
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", "\n", s)
    s = html.unescape(s)
    return [l.strip() for l in s.splitlines() if l.strip()]


def table_for(L, op):
    """返回 [(cfg, in0, in1, in2, out0), ...]；找不到返回 None"""
    idxs = [i for i, l in enumerate(L) if l == op]
    for st in reversed(idxs):
        # 该段后面应有 Datatypes / Configuration / in[0] ... out[0]
        win = L[st:st + 25]
        if "Datatypes" not in win:
            continue
        try:
            j = st + win.index("out[0]") + 1
        except ValueError:
            continue
        # in[0]/in[1]/... 的个数：数以 "in[" 开头的整行
        ncol = sum(1 for x in win if x.startswith("in["))
        if ncol == 0:
            continue
        rows = []
        while j + ncol + 1 <= len(L):
            cfg = L[j]
            if cfg not in ("INT16", "INT8", "INT4", "FP16", "BF16"):
                break
            vals = [short(L[j + k]) for k in range(1, ncol + 2)]
            rows.append((cfg, *vals))
            j += ncol + 2
        if rows:
            return rows
    return None


def main():
    L = load_lines()
    ops = sys.argv[1:] or OUR_OPS
    print(f"文档: {DOC}\n")
    for op in ops:
        rows = table_for(L, op)
        print("=" * 78)
        if not rows:
            print(f"{op}: 文档中未找到数据类型表（可能是数据搬运类算子，不受量化类型约束）")
            continue
        print(f"{op}: {len(rows)} 种支持组合")
        # 只关心 16-bit 激活的情形
        a16 = [r for r in rows if r[1] in ("UFxp16", "SFxp16")]
        if a16:
            print(f"  16-bit 激活下的组合：")
            for r in a16:
                print(f"    {r[0]:<6} " + "  ".join(f"{v:<10}" for v in r[1:]))
            w = sorted({r[2] for r in a16})
            print(f"  ⇒ 16-bit 激活时，第二个输入（权重）允许: {w}")
            print(f"     A16W16(SFxp16 权重) 支持? {'是' if 'SFxp16' in w else '否'}")
            print(f"     A16W8 (UFxp8  权重) 支持? {'是' if 'UFxp8' in w else '否'}")
            print(f"     SFxp8 权重        支持? {'是' if 'SFxp8' in w else '否'}")
        else:
            print("  （无 16-bit 激活组合）")


if __name__ == "__main__":
    main()
