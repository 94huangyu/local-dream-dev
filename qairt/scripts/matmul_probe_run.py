"""#30 探针的设备侧执行：把 3 个极小 context binary 推上去各跑 32 个测试样本。

方案 `scripts/EXP_PLAN_MATMUL_PROBE.md`。设备占用：三个 `.bin` 合计 < 0.5 MB，
**内存占用可忽略**，不是把手机压死的那类任务。

纪律：
  · 每段执行前确认设备在线（`sh` 在 adb 返回码非零时直接抛，不返回空串
    —— 2026-08-17 实测：静默返回空串会把一次 USB 掉线伪装成模型执行失败）
  · 输出字节数强制校验（约束 3：这是抓静默缩批的唯一防线）

用法: python matmul_probe_run.py
"""
import os
import subprocess
import sys

import numpy as np

ROOT = r"D:\ZImage_Work\p0_experiments\matmul_probe"
ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEV = "3B1F65EA9BBUMSHZ"
T = "/data/local/tmp/htpcmp"
D = f"{T}/mmprobe"
KS = [64, 1024, 3840]
N_OUT = 64
N_TEST = 32


def sh(cmd, timeout=900):
    r = subprocess.run([ADB, "-s", DEV, "shell", cmd], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"adb shell 失败 (rc={r.returncode}): {cmd[:120]}\n"
                           f"stderr: {r.stderr.strip()}")
    return r.stdout


def push(local, remote):
    r = subprocess.run([ADB, "-s", DEV, "push", local, remote],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"adb push 失败: {local}\n{r.stderr}")


def require_device():
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    if not any(DEV in l and "device" in l.split() for l in out.splitlines()[1:]):
        raise SystemExit(f"设备 {DEV} 不在线。adb devices:\n{out}")


def main():
    require_device()
    sh(f"mkdir -p {D}")
    for k in KS:
        d = os.path.join(ROOT, f"k{k}")
        print(f"\n=== K={k} ===", flush=True)
        push(os.path.join(d, f"probe_k{k}_ctx.SM8750.bin"), f"{D}/k{k}.bin")

        # 测试输入逐个推上去，设备侧现拼 input_list（宿主路径不能直接用）
        sh(f"rm -rf {D}/in{k} {D}/out{k} && mkdir -p {D}/in{k}")
        push(os.path.join(d, "test"), f"{D}/in{k}")
        lines = "\\n".join(f"x:={D}/in{k}/test/x_{i:04d}.raw" for i in range(N_TEST))
        sh(f"printf '{lines}\\n' > {D}/list{k}.txt")
        n = sh(f"wc -l < {D}/list{k}.txt").strip()
        assert int(n) == N_TEST, f"input_list 行数 {n} != {N_TEST}"

        out = sh(f"cd {T} && export LD_LIBRARY_PATH={T} && export ADSP_LIBRARY_PATH={T} && "
                 f"./qnn-net-run --retrieve_context {D}/k{k}.bin --backend libQnnHtp.so "
                 f"--input_list {D}/list{k}.txt --output_dir {D}/out{k} --log_level error "
                 f"2>&1 | tail -5")
        if "Finished Executing Graphs" not in out:
            raise RuntimeError(f"qnn-net-run 失败 [K={k}]:\n{out}")

        # 约束 3：输出字节数校验
        sizes = sh(f"stat -c '%s' {D}/out{k}/Result_*/y.raw").split()
        exp = N_OUT * 4
        assert len(sizes) == N_TEST, f"输出结果数 {len(sizes)} != {N_TEST}"
        bad = [s for s in sizes if int(s) != exp]
        assert not bad, f"K={k} 输出字节数异常 {bad[:3]} != {exp}，疑似静默缩批"
        print(f"  {N_TEST} 个结果，每个 {exp} 字节 —— 校验通过")

        os.makedirs(os.path.join(d, "htp"), exist_ok=True)
        for i in range(N_TEST):
            r = subprocess.run([ADB, "-s", DEV, "pull", f"{D}/out{k}/Result_{i}/y.raw",
                                os.path.join(d, "htp", f"y_{i:04d}.raw")],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"adb pull 失败 Result_{i}: {r.stderr}")
        a = np.stack([np.fromfile(os.path.join(d, "htp", f"y_{i:04d}.raw"), np.float32)
                      for i in range(N_TEST)])
        assert a.shape == (N_TEST, N_OUT), f"拉回形状 {a.shape}"
        print(f"  y_htp std={a.std():.4f}")

    print("\n设备侧完成。下一步: python scripts/matmul_probe_analyze.py")


if __name__ == "__main__":
    sys.exit(main())
