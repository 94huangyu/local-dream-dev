"""对照实验的【装置等同性】检查器：读回 context binary 的构建画像并 diff。

## 为什么存在（2026-08-22，代价：一整轮结论作废并翻转）
建 standalone context 时漏了 `--config_file`，产物被静默编译成 **dspArch 68 / vtcm 4MB**，
而部署的是 **79 / 8MB** —— 尽管两者都传了 `--htp_socs sm8750`，
**尽管两者文件名都是 `.SM8750.bin`**（项目原来的判别方法就是看这个后缀，完全失效）。
同一份 DLC、同一份输入、同一套 encoding，只差这个画像，单算子误差差 **15 倍**（1.5544% vs 0.1034%）。

## 用法
    python check_ctx_identity.py <ctx.bin> [<ctx.bin> ...]      # 打印画像
    python check_ctx_identity.py --expect-arch 79 --expect-vtcm 8 <ctx.bin> ...   # 断言
    python check_ctx_identity.py --diff <参照.bin> <待检.bin>     # 两两 diff（对照实验必做）
"""
import os, sys, json, argparse, subprocess, tempfile

SDK = os.environ.get("QAIRT_SDK", os.path.join("D:", os.sep, "qairt", "2.48.0.260626"))
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")

# 🔴 装置字段：对照实验两臂**必须相同**，不同即实验不成立
DEVICE_FIELDS = ["dspArch", "vtcmSize", "optimizationLevel", "htpDlbc", "numHvxThreads"]
# 模型字段：不同模型之间本来就会不同，仅供参考（同一模型的两臂则应相同）
MODEL_FIELDS = ["spillFillBufferSize", "graphNames"]
FIELDS = DEVICE_FIELDS + MODEL_FIELDS


def profile(path):
    """读回一个 context binary 的构建画像"""
    if not os.path.exists(UTIL):
        sys.exit("找不到 qnn-context-binary-utility：%s（可用 QAIRT_SDK 环境变量指定 SDK）" % UTIL)
    with tempfile.TemporaryDirectory() as td:
        js = os.path.join(td, "info.json")
        r = subprocess.run([UTIL, "--context_binary", path, "--json_file", js],
                           capture_output=True, text=True, timeout=1200)
        # 🔴 返回码必须查：工具非零退出却留下一份旧/半截 json 时，
        #    只看 os.path.exists 会让 G3 装置门**空过**（#95/#96：装置画像错误
        #    曾让一整轮结论作废）。code_lint C5 报的就是这里。
        if r.returncode != 0:
            return {"error": "rc=%d %s" % (r.returncode, (r.stdout + r.stderr)[-260:])}
        if not os.path.exists(js):
            return {"error": (r.stdout + r.stderr)[-300:]}
        d = json.load(open(js, encoding="utf-8"))
    i = d.get("info", {})
    p = {"file": os.path.basename(path), "bytes": os.path.getsize(path)}
    p["dspArch"] = i.get("contextMetadata", {}).get("info", {}).get("dspArch")
    names, blob = [], {}
    for g in i.get("graphs", []):
        gi = g.get("info", {})
        names.append(gi.get("graphName"))
        b = gi.get("graphBlobInfo", {}).get("info", {})
        if b and not blob:
            blob = b
    p["graphNames"] = names
    for k in ("vtcmSize", "optimizationLevel", "htpDlbc", "numHvxThreads", "spillFillBufferSize"):
        p[k] = blob.get(k)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bins", nargs="+")
    ap.add_argument("--expect-arch", type=int, default=None)
    ap.add_argument("--expect-vtcm", type=int, default=None)
    ap.add_argument("--diff", action="store_true",
                    help="以第一个为参照，逐项 diff 其余（对照实验必做）")
    a = ap.parse_args()

    profs = [profile(b) for b in a.bins]
    for p in profs:
        if "error" in p:
            print("❌ %s 读取失败: %s" % (p.get("file"), p["error"]))
            continue
        print("=== %s  (%.2f MB) ===" % (p["file"], p["bytes"] / 1e6))
        for k in FIELDS:
            print("    %-20s %s" % (k, p.get(k)))

    bad = 0
    if a.expect_arch is not None or a.expect_vtcm is not None:
        print("")
        print("=== 断言 ===")
        for p in profs:
            if "error" in p:
                continue
            ok = True
            if a.expect_arch is not None and p.get("dspArch") != a.expect_arch:
                print("  ❌ %s dspArch=%s，期望 %d" % (p["file"], p.get("dspArch"), a.expect_arch))
                ok = False
            if a.expect_vtcm is not None and p.get("vtcmSize") != a.expect_vtcm:
                print("  ❌ %s vtcmSize=%s，期望 %d" % (p["file"], p.get("vtcmSize"), a.expect_vtcm))
                ok = False
            if ok:
                print("  ✅ %s" % p["file"])
            else:
                bad += 1
        if bad:
            print("")
            print("  ⇒ 🔴 **画像不符。产物文件名带 .SM8750 后缀【不能】证明画像正确**")
            print("     漏 --config_file（detail 里要有 devices）就会静默编译成 v68/4MB。")

    if a.diff and len(profs) >= 2:
        ref = profs[0]
        print("")
        print("=== 装置等同性 diff（参照：%s）===" % ref["file"])
        for p in profs[1:]:
            if "error" in p:
                continue
            dev = [(k, ref.get(k), p.get(k)) for k in DEVICE_FIELDS if ref.get(k) != p.get(k)]
            mod = [(k, ref.get(k), p.get(k)) for k in MODEL_FIELDS if ref.get(k) != p.get(k)]
            if not dev:
                print("  ✅ %s 装置画像与参照一致" % p["file"])
            else:
                bad += 1
                print("  🔴 %s **装置画像不同 ⇒ 对照实验不成立**：" % p["file"])
                for k, x, y in dev:
                    print("       %-20s 参照 %-14s 待检 %s" % (k, x, y))
            if mod:
                print("     （模型字段差异，不同模型间属正常；同一模型的两臂则应相同）")
                for k, x, y in mod:
                    print("       %-20s 参照 %-14s 待检 %s" % (k, str(x)[:14], str(y)[:40]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
