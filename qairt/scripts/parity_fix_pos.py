# -*- coding: utf-8 -*-
"""主线 A 根因验证 · 把 L=80 图的图像块位置常数 33 -> 81（单变量）。

## 已实测的事实（每条都是直接读出来/跑出来的）
| | caption 槽位置 | 图像块位置 | 结果 |
|---|---|---|---|
| 基座 L=32 | `val_520` = **1..32** | `stack_1` 轴0 = **33** | 紧接其后，不重叠；**对官方 1.79%** |
| 现网 L=80 | `val_520` = **1..80** | `stack_1` 轴0 = **33**（手术没改）| **与 caption 槽 33..80 位置重叠**；**对官方 15.52%** |

⇒ Tier 2 的 32→80 手术把 caption 序列拉长，**位置常数没跟着往后挪**。
   在 RoPE 下，图像 token 与 caption 的第 33~80 槽拿到同一批位置编码。

## 本实验
只把 `transformer_part1a_L80.onnx` 里 `stack_1` 的轴0 从 33 改成 **81**（= 80 槽之后），
其余一切不动，重跑 step 0，与官方臂比。

## 判据（事前锁定）
基线：L=80 现状对官方 **15.5172%**；基座 L=32 对官方 **1.7901%**（可达下界）。

| 结果 | 判读 |
|---|---|
| 降到 **<= 3%** | 🟢🟢 **根因确认且已定位到这一个常数**，修法明确 |
| 3% ~ 8% | 🟡 是主因，但手术还有别的伤 |
| **> 8%** | 🔴 不是主因，位置重叠只是并存现象 |

🔴 无论结果如何都不改判据。

用法:
  python scripts/parity_fix_pos.py make     # 生成 _pos81 变体（纯改一个 initializer）
  python scripts/parity_fix_pos.py run      # 跑 step 0 并比对
"""
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx
from onnx import numpy_helper

P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
W = os.path.join(P0, "parity")
E = os.path.join("D:", os.sep, "ZImage_Work", "ZImage_QNN_Evidence", "onnx")
SEGS = ["part1a", "part1b", "part2a", "part2b"]
TAG = "pos81"
NEW_POS = 81
BASE_L80 = 0.155172      # 现状对官方
FLOOR_L32 = 0.017901     # 基座 L=32 对官方（可达下界）


def src_name(seg):
    import canonical_sources
    p = canonical_sources.onnx_for(seg, "base")
    stem = os.path.splitext(os.path.basename(p))[0]
    return os.path.join(os.path.dirname(p), "%s_L80.onnx" % stem)


def make():
    for seg in SEGS:
        s = src_name(seg)
        d = s.replace("_L80.onnx", "_L80_%s.onnx" % TAG)
        if seg != "part1a":
            # 其余三段原样复制（external data 引用是相对同目录，仍然有效）
            shutil.copyfile(s, d)
            print("复制 %-8s -> %s" % (seg, os.path.basename(d)))
            continue
        # 🔴 必须 load_external_data=False：否则 onnx.save 会把 13.76 GB 权重全写进内部
        m = onnx.load(s, load_external_data=False)
        hit = [i for i in m.graph.initializer if i.name == "stack_1"]
        if len(hit) != 1:
            raise SystemExit("🔴 stack_1 不唯一（%d 个）" % len(hit))
        t = hit[0]
        # 读原值需要外部数据；单独把这一个张量读出来
        m_full = onnx.load(s, load_external_data=True)
        old = numpy_helper.to_array([i for i in m_full.graph.initializer
                                     if i.name == "stack_1"][0]).copy()
        del m_full
        print("原 stack_1: shape=%s 轴0 取值=%s" % (old.shape, np.unique(old[..., 0])))
        if not np.all(old[..., 0] == 33):
            raise SystemExit("🔴 轴0 不是恒 33，先查清楚再改")
        new = old.copy()
        new[..., 0] = NEW_POS
        nt = numpy_helper.from_array(new.astype(old.dtype), name="stack_1")
        # 就地替换：清掉外部引用，改成内部 raw_data（仅 49 KB）
        t.ClearField("external_data")
        t.data_location = onnx.TensorProto.DEFAULT
        t.CopyFrom(nt)
        onnx.save(m, d)   # 其余 initializer 仍指向同目录的 .data
        print("写出 %-8s -> %s（stack_1 轴0 = %d，其余不动）"
              % (seg, os.path.basename(d), NEW_POS))
    # 生效门：读回来确认
    chk = onnx.load(src_name("part1a").replace("_L80.onnx", "_L80_%s.onnx" % TAG),
                    load_external_data=True)
    a = numpy_helper.to_array([i for i in chk.graph.initializer if i.name == "stack_1"][0])
    print("\n生效门：读回 stack_1 轴0 = %s（轴1 %d~%d 轴2 %d~%d，应与原图相同）"
          % (np.unique(a[..., 0]), a[..., 1].min(), a[..., 1].max(),
             a[..., 2].min(), a[..., 2].max()))
    if not np.all(a[..., 0] == NEW_POS):
        raise SystemExit("🔴 改动没落盘")
    return 0


def run():
    out = os.path.join(W, "s0_L80_%s.raw" % TAG)
    env = dict(os.environ, ZI_TAG=TAG)
    cmd = [sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "t2_fp32_ref.py"), "step", "80", "0",
           os.path.join(P0, "latents_cxx_seed42.raw"), out,
           os.path.join(P0, "cap_r4x3.raw"), os.path.join(P0, "mask_r4x3.raw")]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=7200)
    for ln in (r.stdout or "").strip().splitlines()[-3:]:
        print("  " + ln)
    if r.returncode != 0 or not os.path.isfile(out):
        print((r.stdout or "")[-1200:]); print((r.stderr or "")[-1200:])
        raise SystemExit("🔴 跑失败 rc=%d" % r.returncode)

    ts = [1000.0, 954.5455]
    dt = (ts[1] - ts[0]) / 1000.0
    lat = np.fromfile(os.path.join(P0, "latents_cxx_seed42.raw"), np.float32).astype(np.float64)
    nxt = np.fromfile(out, np.float32).astype(np.float64)
    ours = (lat - nxt) / dt
    off = -np.fromfile(os.path.join(W, "noise_s0_official_ourcap.raw"),
                       np.float32).astype(np.float64)   # 符号对齐
    rel = np.linalg.norm(ours - off) / np.linalg.norm(off)
    cos = float(off @ ours / (np.linalg.norm(off) * np.linalg.norm(ours)))
    mk = np.abs(off) <= np.quantile(np.abs(off), 0.99)
    relb = np.linalg.norm(ours[mk] - off[mk]) / np.linalg.norm(off[mk])
    print("\n=== 判据（事前锁定）===")
    print("  现状 L=80 对官方      : %.4f%%" % (BASE_L80 * 100))
    print("  基座 L=32 对官方(下界): %.4f%%" % (FLOOR_L32 * 100))
    print("  **本次 pos=81 对官方  : %.4f%%（主体 %.4f%%，余弦 %.6f）**"
          % (rel * 100, relb * 100, cos))
    v = ("🟢🟢 根因确认，修法明确" if rel <= 0.03 else
         ("🟡 是主因但还有别的伤" if rel <= 0.08 else "🔴 不是主因"))
    print("  ⇒ %s（%.2f%% -> %.2f%%，消掉了原差异的 %.1f%%）"
          % (v, BASE_L80 * 100, rel * 100, 100 * (BASE_L80 - rel) / (BASE_L80 - FLOOR_L32)))
    return 0


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "make"
    sys.exit(make() if c == "make" else run())
