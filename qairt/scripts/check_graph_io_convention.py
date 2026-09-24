"""换新模型必跑：检查 ONNX 图的头尾有没有【烘焙进去的前/后处理】。

## 为什么存在这个脚本（2026-08-21，代价：误判 2 天 + 产品一直出劣化图）
`vae_decoder.onnx` 的第一个节点是 `Div(vae_latents, 0.3611)` —— 反缩放图内已做。
而全部 6 个宿主脚本 + app 的 C++ 流水线按框架惯例又做了一遍 `lat/s + shift`。
**除了两遍。** 后果：
  · 解码本身错了（对官方 PyTorch 23.51 dB，修正后 79.92 dB）
  · 输入落到量化校准量程外 2.77 倍，被量化器硬钳（这部分损害一直被记在"量化"账上）
它极难被发现，因为：图结构与权重的静态检查**全部通过**（导出确实是对的），
成图只是"差不多但质感不对"，而所有"我们 A vs 我们 B"的对照都**共模抵消**。

详见 HANDOVER §15.22 与 QNN_CONVERSION_GUIDE §二十。

## 用法
    python check_graph_io_convention.py <dir_or_onnx> [...]
不带参数则扫描本项目的 evidence/onnx 目录。
"""
import os, sys, glob

# 带常量的逐元素算子出现在图的头/尾 => 极可能是被烘焙进去的前/后处理
SCALE_OPS = {"Div", "Mul", "Add", "Sub", "Pow", "Clip", "Cast"}
HEAD_DEPTH = 6
TAIL_DEPTH = 6


def const_operands(node, init, const):
    """返回该节点里取自常量的输入（名字, 元素数, 首值）"""
    from onnx import numpy_helper, TensorProto
    out = []
    for name in node.input:
        arr = None
        try:
            t = None
            if name in init:
                t = init[name]
            elif name in const:
                t = const[name].attribute[0].t
            if t is None:
                continue
            # 外部权重一律跳过：它们是大权重，不可能是我们要找的标量常量
            if t.data_location == TensorProto.EXTERNAL:
                continue
            arr = numpy_helper.to_array(t)
        except Exception:
            continue
        if arr is not None:
            v = float(arr.ravel()[0]) if arr.size else float("nan")
            out.append((name, int(arr.size), v))
    return out


def check(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    init = {t.name: t for t in g.initializer}
    const = {n.output[0]: n for n in g.node if n.op_type == "Constant"}
    inputs = {i.name for i in g.input} - set(init)
    outputs = {o.name for o in g.output}

    print("=" * 78)
    print("%s" % os.path.basename(path))
    print("  inputs : %s" % sorted(inputs))
    print("  outputs: %s" % sorted(outputs))

    findings = []

    def scan(nodes, where):
        for n in nodes:
            if n.op_type not in SCALE_OPS:
                continue
            cs = const_operands(n, init, const)
            if not cs:
                continue
            touches = (any(i in inputs for i in n.input) if where == "头"
                       else any(o in outputs for o in n.output))
            small = [c for c in cs if c[1] <= 4]
            if small:
                findings.append((where, n.op_type, n.name, small, touches))

    scan(g.node[:HEAD_DEPTH], "头")
    scan(g.node[-TAIL_DEPTH:], "尾")

    print("  --- 头 %d 个节点 ---" % HEAD_DEPTH)
    for n in g.node[:HEAD_DEPTH]:
        print("    %-10s %-28s %s -> %s" % (n.op_type, n.name[:27], list(n.input)[:3], list(n.output)))
    print("  --- 尾 %d 个节点 ---" % TAIL_DEPTH)
    for n in g.node[-TAIL_DEPTH:]:
        print("    %-10s %-28s %s -> %s" % (n.op_type, n.name[:27], list(n.input)[:3], list(n.output)))

    if findings:
        print("  🔴 发现疑似【图内已做的前/后处理】——宿主侧【不要】再做一遍：")
        for where, op, name, cs, touches in findings:
            for cname, size, v in cs:
                print("     [%s] %-6s %-24s 常量 %s = %.6f%s"
                      % (where, op, name[:23], cname[:20], v,
                         "   ← 直接接图的输入/输出" if touches else ""))
        print("  ⇒ 判读（2026-08-21 实测校准，避免假阳性）：")
        print("     · 真嫌疑 = 常量作用在【宿主真正喂的输入】上，且值是模型特有的尺度")
        print("       例：vae_latents 的 Div 0.3611（真 bug）、timestep 的 Mul 1000（查过，对的）")
        print("     · 常见假阳性 = 从网络中间切开的分段，头部会露出 RmsNorm 内部：")
        print("       Pow 2.0 / Add 1e-5(eps) / Add 1.0(1+scale)。这些不是宿主接口，忽略。")
        print("  ⇒ 必做的已知答案验证：**第一个算子的输出**。两侧权重与输入逐位相同 ⇒")
        print("     输出必须逐位相同。本项目正是靠这条抓到 1/0.3611 = 2.7693 这个比值。")
    else:
        print("  ✅ 头尾未发现带常量的逐元素算子（不代表没有别的约定差异，仍须做已知答案验证）")
    return findings


def main():
    args = sys.argv[1:]
    if not args:
        args = [os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")]
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(os.path.join(a, "*.onnx"))) if os.path.isdir(a) else [a])
    if not paths:
        sys.exit("没有找到 .onnx")
    total = 0
    for p in paths:
        total += len(check(p))
    print("=" * 78)
    print("扫描 %d 个图，疑似烘焙前后处理 %d 处。" % (len(paths), total))
    if total:
        print("每一处都必须去调用方核对：宿主/app 是不是又做了一遍。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
