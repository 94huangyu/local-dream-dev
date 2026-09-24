"""设备操作的统一入口：**让「报错说谎」在结构上不可能发生**。

为什么存在（不重复叙述历史，只说规则）：
  设备类判断有三种截然不同的状态，混淆任何两个都会得出错误结论：
    A. 设备不可达（拔线 / 休眠 / adb server 挂了）
    B. 设备可达，进程不在
    C. 设备可达，进程在
  裸用 `adb shell "ps | grep X"` 时，A 和 B 的返回值**完全一样**。

规则（本模块强制）：
  · 任何"进程是否还在"的判断，必须先确认设备可达；不可达时抛 DeviceOffline，
    **不得退化成"进程没了"**。
  · 任何 adb 调用失败必须抛异常并透传 stderr，不得返回空串。
  · 宿主侧路径一律用 Windows 形式（D:/...），设备侧用 Unix 形式；
    本模块内部处理，调用方不必再关心 MSYS_NO_PATHCONV。
"""
import os
import subprocess

ADB = r"C:\Users\sinai\AppData\Local\Android\Sdk\platform-tools\adb.exe"
SERIAL = "3B1F65EA9BBUMSHZ"


class DeviceOffline(Exception):
    """设备不可达。**这不等于任务结束。**"""


class AdbFailed(Exception):
    """adb 自身失败（非模型失败）。"""


def _run(args, timeout=1800):
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    return subprocess.run([ADB, "-s", SERIAL] + args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=env)


def online():
    """设备是否可达。唯一允许用来判断 A 状态的函数。"""
    try:
        r = subprocess.run([ADB, "-s", SERIAL, "get-state"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30)
    except Exception:
        return False
    return r.returncode == 0 and "device" in (r.stdout or "")


def require_online(where=""):
    if not online():
        raise DeviceOffline("设备不可达 @ %s —— 这不等于任务结束；"
                            "设备回来后请核实产物再下结论" % (where or "?"))


def sh(cmd, timeout=1800):
    """设备 shell。失败必抛，绝不返回空串冒充成功。"""
    require_online("sh: %s" % cmd[:60])
    r = _run(["shell", cmd], timeout)
    if r.returncode != 0:
        raise AdbFailed("adb shell rc=%d\nCMD: %s\nSTDOUT:%s\nSTDERR:%s"
                        % (r.returncode, cmd[:200], (r.stdout or "")[-600:], (r.stderr or "")[-600:]))
    return r.stdout


def proc_alive(name):
    """进程是否在。**设备不可达时抛异常，不返回 False。**"""
    require_online("proc_alive(%s)" % name)
    out = sh("ps -A -o NAME 2>/dev/null | grep -ci %s" % name)
    return int(out.strip() or "0") > 0


def mem_available_mb():
    require_online("mem_available")
    out = sh("grep MemAvailable /proc/meminfo")
    return int(out.split()[1]) // 1024


def push(host_path, dev_path):
    require_online("push")
    r = _run(["push", str(host_path).replace(os.sep, "/"), dev_path])
    if r.returncode != 0:
        raise AdbFailed("adb push 失败\n%s\n%s" % ((r.stdout or "")[-400:], (r.stderr or "")[-400:]))
    return r.stdout


def pull(dev_path, host_path):
    require_online("pull")
    r = _run(["pull", dev_path, str(host_path).replace(os.sep, "/")])
    if r.returncode != 0:
        raise AdbFailed("adb pull 失败\n%s\n%s" % ((r.stdout or "")[-400:], (r.stderr or "")[-400:]))
    return r.stdout


def wait_done(proc="qnn-net-run", product_check=None, poll=60, mem_floor_mb=800):
    """阻塞等待设备任务真结束。

    返回值三选一，**调用方必须分别处理**：
      "done"     进程已退出（设备在线时确认，可信）
      "offline"  设备中途不可达 —— **任务可能仍在跑**，需设备回来后核实产物
      "lowmem"   设备可用内存跌破下限 —— 告警，**不自动杀进程**（该由人决定）
    """
    import time
    while True:
        if not online():
            return "offline"
        try:
            if not proc_alive(proc):
                return "done"
            m = mem_available_mb()
            if m < mem_floor_mb:
                return "lowmem"
            n = ""
            if product_check:
                n = sh("ls %s 2>/dev/null | wc -l" % product_check).strip()
            print("[wait] %s 仍在跑  MemAvailable=%d MB  产物=%s"
                  % (time.strftime("%H:%M:%S"), m, n or "-"), flush=True)
        except DeviceOffline:
            return "offline"
        time.sleep(poll)
