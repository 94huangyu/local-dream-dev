"""无人值守夜间链：dtype 机制单段验证 -> 四段重建 -> 端到端 -> 三层指标。

设计原则（都是今天踩出来的）：
  · 任何设备判断走 lab_dev（设备不可达 != 任务结束）
  · 单段验证不通过就**停止**，不浪费 4 小时建另外三段
  · 每一步都落盘 overnight_state.json，醒来后一眼看到走到哪、为什么停
  · 设备操作一律 --retrieve_context（不做在线建图 => 不压手机内存）
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import lab_dev as L
from wait_for import wait_build

import numpy as np

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
W = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "p2attr")
STATE = os.path.join(W, "overnight_state.json")
PY = sys.executable
T = "/data/local/tmp/htpcmp"
BASE_COS = 0.945511          # part2a 单段基线（旧机制 26 张量版实测）
BASE_PSNR = 15.53            # 端到端基线 L3


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def save(**kw):
    st = {}
    if os.path.exists(STATE):
        st = json.load(open(STATE, encoding="utf-8"))
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def run(cmd, timeout, tag):
    log("RUN", tag)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(W, "overnight_%s.log" % tag), "w", encoding="utf-8").write(out)
    for line in out.splitlines()[-25:]:
        if line.strip():
            print("    " + line.strip()[:160], flush=True)
    log(tag, "rc=%d" % r.returncode)
    return r.returncode, out


def bulk_cos(a, b):
    m = np.abs(a) <= np.percentile(np.abs(a), 99)
    x, y = a[m], b[m]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def device_seg(ctx, tag, outname):
    """单段上机：推 context -> 跑 -> 拉。全部 --retrieve_context，不做在线建图。"""
    L.require_online("device_seg " + tag)
    log("push", os.path.basename(ctx), "%.0f MB" % (os.path.getsize(ctx) / 1e6))
    L.push(ctx, "%s/ov_%s.bin" % (T, tag))
    od = "%s/ov_%s_out" % (T, tag)
    L.sh("rm -rf %s && mkdir -p %s" % (od, od))
    o = L.sh("cd %s && export LD_LIBRARY_PATH=%s && export ADSP_LIBRARY_PATH=%s && "
             "./qnn-net-run --retrieve_context %s/ov_%s.bin --backend libQnnHtp.so "
             "--input_list %s/p2attr/in/list.txt --output_dir %s --log_level error 2>&1 | tail -3"
             % (T, T, T, T, tag, T, od))
    if "Finished Executing Graphs" not in o:
        raise RuntimeError("qnn-net-run 失败 [%s]:\n%s" % (tag, o))
    rp = "%s/Result_0/%s.raw" % (od, outname)
    nb = int(L.sh("stat -c %%s %s" % rp).strip())
    exp = 4128 * 3840 * 4
    if nb != exp:
        raise RuntimeError("输出字节 %d != %d（静默缩批）" % (nb, exp))
    loc = os.path.join(W, "htp", "%s_%s.raw" % (outname, tag))
    L.pull(rp, loc)
    return loc


def main():
    save(phase="start", note="dtype 机制单段验证 -> 四段 -> 端到端")

    # ---------- 1. 等 part2a 量化 + 建 context ----------
    ctxdir = os.path.join(W, "ctx_part2a_dt")
    # 🔴 必须同时盯进程和产物（wait_for）：只盯产物会在构建失败后空等数小时（2026-08-24 教训）
    st = wait_build(ctxdir, marker="seg_dtype_build",
                    logfile=os.path.join(W, "p2a_dt3.log"), timeout=5 * 3600)
    if st != "done":
        save(phase="stop", reason="part2a context 未产出（wait_for=%s）" % st)
        log("STOP part2a:", st)
        return
    bins = [x for x in os.listdir(ctxdir) if x.endswith(".bin")]
    ctx = os.path.join(ctxdir, bins[0])
    save(phase="p2a_built", ctx=ctx)

    # ---------- 2. 单段上机验证 ----------
    try:
        loc = device_seg(ctx, "p2adt", "add_92")
    except Exception as e:
        save(phase="stop", reason="设备单段验证失败: %s" % str(e)[:300])
        log("STOP", e)
        return
    ref = np.fromfile(os.path.join(W, "fp32", "add_92.raw"), np.float32).astype(np.float64)
    got = np.fromfile(loc, np.float32).astype(np.float64)
    cos = bulk_cos(ref, got)
    nbad = int((~np.isfinite(got)).sum())
    log("单段 part2a(dtype) 主体余弦 = %.6f （旧机制基线 %.6f）  inf/nan=%d"
        % (cos, BASE_COS, nbad))
    save(phase="p2a_verified", cos=cos, baseline=BASE_COS, inf=nbad)
    if nbad or cos < BASE_COS - 0.010:
        save(phase="stop", reason="dtype 机制单段未优于旧机制（cos %.6f < %.6f）⇒ 不推四段"
             % (cos, BASE_COS))
        log("STOP: dtype 机制没有改善，按设计不浪费 4 小时推四段")
        return

    # ---------- 3. 另外三段 ----------
    for seg in ("part2b", "part1b", "part1a"):
        d = os.path.join(W, "ctx_%s_dt" % seg)
        if os.path.isdir(d) and [x for x in os.listdir(d) if x.endswith(".bin")]:
            log("跳过已建好的", seg)
            continue
        rc, _ = run([PY, "scripts/seg_dtype_build.py", seg], 28800, seg + "_dt")
        if rc:
            save(phase="stop", reason="%s dtype 构建失败" % seg)
            return
        save(phase="built_" + seg)

    # ---------- 4. 端到端 ----------
    names = {"part1a": "part1a", "part1b": "part1b",
             "part2a": "part2a_fixed", "part2b": "part2b_fixed"}
    try:
        L.require_online("push 4 seg")
        for seg, dev in names.items():
            d = os.path.join(W, "ctx_%s_dt" % seg)
            b = [x for x in os.listdir(d) if x.endswith(".bin")][0]
            p = os.path.join(d, b)
            log("push", seg, "%.0f MB" % (os.path.getsize(p) / 1e6))
            L.push(p, "%s/%s_dt.bin" % (T, dev))
    except Exception as e:
        save(phase="stop", reason="推送四段失败: %s" % str(e)[:300])
        return
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MSYS_NO_PATHCONV="1",
               INLOOP_TAG="dtseg", INLOOP_P2="split", INLOOP_BIN_SUFFIX="_dt")
    log("RUN 端到端在环（dtype 四段）")
    r = subprocess.run([PY, "scripts/htp_inloop_pipeline.py"], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=7200)
    open(os.path.join(W, "overnight_inloop.log"), "w", encoding="utf-8").write(
        (r.stdout or "") + (r.stderr or ""))
    if r.returncode:
        save(phase="stop", reason="端到端失败，见 overnight_inloop.log")
        return
    save(phase="inloop_done")

    # ---------- 5. 三层指标 ----------
    rc, out = run([PY, "scripts/panel3.py"], 3600, "panel")
    save(phase="done", panel=out[-1500:])
    log("全部完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        save(phase="crash", reason=str(e)[:500])
        log("CRASH", e)
        raise
