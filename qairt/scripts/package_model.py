# -*- coding: utf-8 -*-
"""把设备上正在跑的那份 Z-Image 模型打包成 app 可导入的 zip。

## 三步，各自独立，可单独重跑
    --copy   设备 → 本地文件夹（逐文件，带 sha256 核对）   约 9 分钟
    --zip    本地文件夹 → zip（用 zipfile.write，不手搓流）约 5 分钟
    --push   zip → 设备 /sdcard/Download                  约 5 分钟
    --all    三步连做

🔴 **为什么是三步而不是一条流水线**（2026-09-20 用户指出，我原先做错了）：
本质就是「拷贝 → 压缩 → 推送」三件成熟的事。我原先为了省一次磁盘写，把拷贝和压缩
耦合成一个流式管道（边从设备读边往 zip 里写），代价是：
  · 写 zip 的参数错一个（`force_zip64`），**已经传好的 4.34 GB 一起废掉**；
  · 没有中间产物可检查；
  · 下次再打包要把 13 GB **重新传一遍**。
分成三步后，本地文件夹 `STAGE` 留着，再出包只是本地压缩几分钟的事。

## 本轮在这件小事上翻车四次，根因都一样：没弄清就动手
| # | 做法 | 结果 | 错在哪 |
|---|---|---|---|
| 1 | app `cp` 到 `/data/local/tmp` → `adb pull` | Permission denied | 没查根因就换路 |
| 2 | `exec-out tar` 一次流 13 GB | 两次卡死在同一字节数 | 手搓方案替代成熟工具，且未先验证 |
| 3 | 回到 #1 加 `chmod 777` | 仍然 denied | **根因判断错**：真因是 SELinux（`untrusted_app` 不得写 `shell_data_file`），权限位改不了 |
| 4 | 流式写 zip | `File size unexpectedly exceeded ZIP64 limit` | 没读 `zipfile` 文档：流式写单个 >2 GiB 成员必须 `force_zip64=True`（`allowZip64` 只管整包） |

⊕ 顺带修掉一个真 bug：`subprocess.PIPE` 接了 stderr 却只读 stdout ⇒ 大数据量下必然死锁。
   接了哪个流就必须有人读，否则重定向到文件。

## zip 结构要求（读 `ModelListScreen.kt::extractNpuModel` 得出，① 代码事实）
  · **不能带顶层目录**：解压是 `File(modelDir, entry.name)`
  · 以 `.` 开头的路径段会被跳过 ⇒ `.verified_cache` 自动不进包（正合适）
  · `npucustom` 由 app 解压后自建
  · 🔴 `LOAD_MMAP` / `SHARE_SPILLFILL` **必须打包**：少了前者每次生图慢 17.5 s，
    少了后者不共享 spill-fill、内存可能不够
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dprime_measure as D

SRC = "files/models/ZIMAGE"
PKG_DIR = os.path.join("D:", os.sep, "ZImage_Work", "package")
STAGE = os.path.join(PKG_DIR, "ZIMAGE")          # 本地副本，**打完包保留**
ZIP = os.path.join(PKG_DIR, "ZIMAGE_sm8750_v2.zip")
DEV_ZIP = "/sdcard/Download/ZIMAGE_sm8750_v2.zip"
FROZEN = os.path.join(D.REPO, "docs", "DELIVERY_FROZEN.md")
DROP = re.compile(r"backup|\.4seg\.json$|\.3seg|^\.")


def frozen_hashes():
    """设备侧 sha256 对照表，直接读冻结快照。"""
    out = {}
    if not os.path.isfile(FROZEN):
        return out
    for line in open(FROZEN, encoding="utf-8"):
        m = re.match(r"\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`", line.strip())
        if m:
            out[m.group(1)] = (int(m.group(2)), m.group(3))
    return out


def do_copy(dev):
    """设备 → 本地文件夹。逐个 `adb exec-out run-as cat`。

    ✅ 上机前已用已知样本验证过（约束 8）：小文件 154,981 B / 0.1 s、
       大文件 443,772,928 B / 14.5 s，sha256 都与冻结快照一致 ⇒ 约 26~30 MB/s。
    🔴 不能走 `adb pull`：app 私有目录 shell 读不到，中转 `/data/local/tmp` 被 SELinux 挡死。
    ⊕ 已存在且 sha256 对得上的文件**跳过**，所以中断后重跑只补缺的那些。
    """
    print("== 拷贝：设备 → %s ==" % STAGE, flush=True)
    out = dev.sh("run-as %s find %s -type f" % (D.PKG, SRC), timeout=900)
    rels = sorted(line.strip()[len(SRC) + 1:] for line in out.splitlines()
                  if line.strip().startswith(SRC + "/"))
    rels = [r for r in rels if not any(DROP.search(p) for p in r.split("/"))]
    want = frozen_hashes()
    print("  %d 个文件（对照表 %d 条 sha256）" % (len(rels), len(want)), flush=True)
    os.makedirs(STAGE, exist_ok=True)
    errlog = os.path.join(PKG_DIR, "cat_stderr.log")
    total = skipped = 0
    t0 = time.time()
    with open(errlog, "wb") as ef:
        for i, rel in enumerate(rels, 1):
            dst = os.path.join(STAGE, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if rel in want and os.path.isfile(dst) and os.path.getsize(dst) == want[rel][0] \
                    and D.sha256(dst) == want[rel][1]:
                skipped += 1
                continue
            h = hashlib.sha256()
            n = 0
            p = subprocess.Popen(
                [dev.ADB, "exec-out", "run-as", D.PKG, "cat", "%s/%s" % (SRC, rel)],
                stdout=subprocess.PIPE, stderr=ef)   # stderr 走文件，PIPE 会死锁
            with open(dst, "wb") as f:
                while True:
                    chunk = p.stdout.read(4 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    f.write(chunk)
                    n += len(chunk)
            p.wait()
            if p.returncode != 0:
                raise SystemExit("🔴 拉 %s 失败 rc=%d（stderr 见 %s）" % (rel, p.returncode, errlog))
            if rel in want and (n, h.hexdigest()) != want[rel]:
                raise SystemExit("🔴 %s sha256 与设备不符" % rel)
            total += n
            if n > (100 << 20) or i % 30 == 0 or i == len(rels):
                print("    [%3d/%d] %-42s %7.1f MB  累计 %5.2f GB  %.1f 分钟"
                      % (i, len(rels), rel[-42:], n / 1e6, total / 1e9,
                         (time.time() - t0) / 60), flush=True)
    print("  ✅ 新拉 %.2f GB，跳过已有 %d 个，%.1f 分钟"
          % (total / 1e9, skipped, (time.time() - t0) / 60), flush=True)

    miss = [r for r in want if not os.path.isfile(os.path.join(STAGE, r.replace("/", os.sep)))]
    print("  %s 对照表 %d 项%s" % ("✅" if not miss else "🔴", len(want),
                                   "全部就位" if not miss else "缺 %s" % miss))
    return 0 if not miss else 1


def do_zip():
    """本地文件夹 → zip。用 `ZipFile.write`，它知道文件大小、**自动处理 ZIP64**。"""
    print("\n== 压缩：%s → zip ==" % os.path.basename(STAGE), flush=True)
    if not os.path.isdir(STAGE):
        raise SystemExit("🔴 还没拷贝，先跑 --copy")
    files = []
    for root, _, names in os.walk(STAGE):
        for nm in names:
            full = os.path.join(root, nm)
            rel = os.path.relpath(full, STAGE).replace(os.sep, "/")
            if not any(DROP.search(part) for part in rel.split("/")):
                files.append((full, rel))
    files.sort(key=lambda x: x[1])
    total = sum(os.path.getsize(f) for f, _ in files)
    print("  %d 个文件 / %.2f GB（ZIP_STORED）" % (len(files), total / 1e9), flush=True)
    t0 = time.time()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        done = 0
        for i, (full, rel) in enumerate(files, 1):
            z.write(full, rel)
            done += os.path.getsize(full)
            if i % 25 == 0 or i == len(files):
                print("    %d/%d  %.2f GB  %.1f 分钟"
                      % (i, len(files), done / 1e9, (time.time() - t0) / 60), flush=True)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    assert "ZIMAGE" in names, "🔴 缺 ZIMAGE 标记"
    assert "final_qnn_contract.json" in names, "🔴 缺契约"
    assert "LOAD_MMAP" in names and "SHARE_SPILLFILL" in names, "🔴 缺 marker"
    assert not any(p.startswith(".") for n in names for p in n.split("/")), "🔴 含 . 开头的段"
    assert sum(1 for n in names if n.startswith("models/")) == 9, "🔴 models/ 不是 9 个"
    print("  ✅ %.2f GB，结构自证通过（无顶层目录、marker 齐全、models/ 9 个）"
          % (os.path.getsize(ZIP) / 1e9), flush=True)
    return 0


def do_push(dev):
    print("\n== 推送：zip → 设备 ==", flush=True)
    if not os.path.isfile(ZIP):
        raise SystemExit("🔴 还没打包，先跑 --zip")
    t0 = time.time()
    r = subprocess.run([dev.ADB, "push", ZIP, DEV_ZIP], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=7200)
    if r.returncode != 0:
        sys.stderr.write((r.stdout or "")[-2000:] + (r.stderr or "")[-2000:])
        raise SystemExit("🔴 push 失败 rc=%d" % r.returncode)
    back = int(dev.sh("stat -c %%s %s" % DEV_ZIP).strip())
    ok = back == os.path.getsize(ZIP)
    print("  %s 设备 %d / 宿主 %d 字节，%.1f 分钟"
          % ("✅" if ok else "🔴", back, os.path.getsize(ZIP), (time.time() - t0) / 60))
    if ok:
        print("\n  📱 导入：模型列表 →「添加自定义 NPU 模型」→ 选 下载/%s"
              % os.path.basename(DEV_ZIP))
        print("     🔴 模型名用**新名字**：同名会先 deleteRecursively 再解压，失败就没了。")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 0
    import lab_dev as dev
    rc = 0
    if "--copy" in args or "--all" in args:
        dev.require_online("package")
        rc |= do_copy(dev)
    if rc == 0 and ("--zip" in args or "--all" in args):
        rc |= do_zip()
    if rc == 0 and ("--push" in args or "--all" in args):
        dev.require_online("package")
        rc |= do_push(dev)
    return rc


if __name__ == "__main__":
    sys.exit(main())
