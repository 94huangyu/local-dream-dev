"""EXP_PLAN_GAMMA_BW 第 1 步：造 B 臂 ONNX —— 只把 205 个 RmsNorm gamma 做 8-bit Q/DQ。

判据见 scripts/EXP_PLAN_GAMMA_BW.md（已定稿，不得事后修改）。

关键实现决定（都有理由，不是随手选的）：
  · gamma 用**结构检测**定位，不靠张量名（约束 8）。已知样本：part1b 必须检出 44 个（#13）
  · 只把这 205 个小张量**内联**，其余初始化器保持 EXTERNAL 引用不变
    ⇒ 新 .onnx 写进**同一目录**，相对 location 仍有效，**不重写 13.7 GB**
  · scale 必须 **float32**（float64 版在已知样本门上 FAIL 4/4，见方案 §三）
  · V1 单变量门内建：除那 205 个外，其余初始化器必须逐字段相同
"""
import os, sys, hashlib
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

sys.stdout.reconfigure(encoding="utf-8")
D = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
PARTS = ["transformer_part1a", "transformer_part1b", "transformer_part2"]
KNOWN_COUNT = {"transformer_part1b": 44}          # #13 实测：part1b 有 44 个 norm


def find_gammas(g):
    """结构检测 RmsNorm 的 gamma：Mul(A, init) 且 A = Mul(x, Reciprocal(...))"""
    prod = {o: n for n in g.node for o in n.output}
    inits = {t.name: t for t in g.initializer}
    out = []
    for n in g.node:
        if n.op_type != "Mul":
            continue
        gi = [i for i in n.input if i in inits]
        ot = [i for i in n.input if i not in inits]
        if len(gi) != 1 or len(ot) != 1:
            continue
        a = prod.get(ot[0])
        if a is None or a.op_type != "Mul":
            continue
        if not any((prod.get(i) is not None and prod[i].op_type == "Reciprocal") for i in a.input):
            continue
        out.append(gi[0])
    return sorted(set(out))


def read_external(t, base):
    d = {e.key: e.value for e in t.external_data}
    with open(os.path.join(base, d["location"]), "rb") as f:
        f.seek(int(d["offset"]))
        buf = f.read(int(d["length"]))
    assert t.data_type == TensorProto.FLOAT, t.data_type
    return np.frombuffer(buf, "<f4").reshape([d for d in t.dims])


def qdq8(x):
    """量化器规则，已通过已知样本门 4/4（方案 §三）"""
    x = x.astype(np.float64)
    lo = min(float(x.min()), 0.0)
    hi = max(float(x.max()), 0.0)
    s = float(np.float32((hi - lo) / 255.0))
    o = float(np.round(lo / s))
    q = np.clip(np.round(x / s) - o, 0, 255)
    return ((q + o) * s).astype(np.float32), s, o


def sig(t):
    """初始化器的可比指纹：外部引用比字段，内联比字节"""
    if t.data_location == TensorProto.EXTERNAL:
        return ("EXT", tuple(sorted((e.key, e.value) for e in t.external_data)),
                tuple(t.dims), t.data_type)
    return ("INL", hashlib.sha256(t.raw_data).hexdigest(), tuple(t.dims), t.data_type)


def main():
    total = 0
    for part in PARTS:
        src = os.path.join(D, part + ".onnx")
        dst = os.path.join(D, part + "_gq8.onnx")
        m = onnx.load(src, load_external_data=False)
        g = m.graph
        names = find_gammas(g)
        if part in KNOWN_COUNT and len(names) != KNOWN_COUNT[part]:
            raise SystemExit("已知样本门 FAIL: %s 检出 %d 个 gamma，应为 %d"
                             % (part, len(names), KNOWN_COUNT[part]))
        before = {t.name: sig(t) for t in g.initializer}

        inits = {t.name: t for t in g.initializer}
        stats = []
        for nm in names:
            t = inits[nm]
            assert t.data_location == TensorProto.EXTERNAL, nm
            x = read_external(t, D)
            y, s, o = qdq8(x)
            rel = 100.0 * np.linalg.norm(y.astype(np.float64) - x.astype(np.float64)) \
                  / np.linalg.norm(x.astype(np.float64))
            stats.append((nm, x.size, s, o, rel))
            t.ClearField("external_data")
            t.data_location = TensorProto.DEFAULT
            t.raw_data = y.tobytes()
        onnx.save(m, dst)

        # ---- V1 单变量门 ----
        m2 = onnx.load(dst, load_external_data=False)
        after = {t.name: sig(t) for t in m2.graph.initializer}
        changed = {k for k in before if before[k] != after[k]}
        assert set(before) == set(after), "初始化器集合变了"
        ok = (changed == set(names))
        print("%-22s gamma=%-4d 改动初始化器=%-4d  V1单变量门=%s"
              % (part, len(names), len(changed), "PASS" if ok else "FAIL"))
        if not ok:
            print("   意外改动:", sorted(changed - set(names))[:10])
            raise SystemExit("V1 FAIL")
        r = np.array([s[4] for s in stats])
        print("   gamma 自身 Q/DQ 相对误差: 中位 %.4f%%  最大 %.4f%%  最小 %.4f%%"
              % (np.median(r), r.max(), r.min()))
        print("   写出 %s (%.1f MB)" % (os.path.basename(dst), os.path.getsize(dst) / 1e6))
        total += len(names)
    print("\n合计 gamma:", total)


if __name__ == "__main__":
    main()
