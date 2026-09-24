"""等待一个后台构建完成 —— **同时盯进程和产物**，缺一不可。

存在的唯一理由（2026-08-24 代价：一整夜）：
  夜间链只盯产物文件，构建进程 02:45 就已失败退出，链子却空等到 06:37。
  CLAUDE.md 约束 9·补 早已写明「判断任务死活只看两样：进程表 + 产物/日志时间戳」，
  当时只实现了一样。

用法（作为库）：
    from wait_for import wait_build
    st = wait_build(product=..., logfile=..., procname="python.exe", marker="seg_dtype_build")
    # st in {"done", "proc_gone", "timeout"}
  · done       产物出现（可信）
  · proc_gone  **进程没了但产物没出现 => 构建失败，立刻返回，不要再等**
  · timeout    超时
"""
import os
import subprocess
import time


def _proc_alive(marker):
    """marker 出现在某个 python 进程的命令行里 => 认为构建仍在跑。"""
    try:
        r = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60)
    except Exception:
        return True          # 查不到就别误判成死了
    return marker in (r.stdout or "")


def wait_build(product, marker, poll=60, timeout=6 * 3600, logfile=None, grace=180):
    """product: 产物路径（文件或目录）。marker: 构建进程命令行里的特征串。"""
    t0 = time.time()
    started = False
    while True:
        if os.path.isdir(product):
            ok = any(x.endswith(".bin") for x in os.listdir(product))
        else:
            ok = os.path.exists(product)
        if ok:
            return "done"
        alive = _proc_alive(marker)
        if alive:
            started = True
        elif started or time.time() - t0 > grace:
            # 🔴 进程不在、产物也没有 => 失败，立刻返回（不要空等）
            tail = ""
            if logfile and os.path.exists(logfile):
                try:
                    tail = "".join(open(logfile, encoding="utf-8", errors="replace")
                                   .readlines()[-12:])
                except Exception:
                    pass
            print("[wait_for] 🔴 进程已不在且产物未生成 => 构建失败\n%s" % tail, flush=True)
            return "proc_gone"
        if time.time() - t0 > timeout:
            return "timeout"
        print("[wait_for] %s 仍在跑（%.0f 分钟）" % (time.strftime("%H:%M:%S"), (time.time() - t0) / 60),
              flush=True)
        time.sleep(poll)
