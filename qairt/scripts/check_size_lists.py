# -*- coding: utf-8 -*-
"""三处「已交付尺寸」清单必须一致：C++ / Kotlin / 交付契约。

## 为什么需要它
台账 #73 的教训是「同一份清单出现在多处 ⇒ 一律改成从单一数据源派生」，
但 UI 在后端启动前读不到契约，所以 Kotlin 侧不得不有一份。
折中：**保留三份，但用这个检查把「漂移会响」从运行时提前到构建期。**
运行时的两道硬失败仍然保留（RequestParser 打回 1024 / PipelineZImage 抛
"delivery has no graphs for size"），本检查只是让漂移**更早**暴露。

🔴 别把「运行时会硬失败」当成不做这个检查的理由：
   2026-09-04 那次交付就是在真机上才发现 UI 门没打开（`isSdxl` 参数名误导），
   而那本可以在宿主上查出来。**能在宿主发现的，就不要留到插线时发现。**

用法: python check_size_lists.py [契约.json]     非零退出 = 不一致
"""
import io
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP = os.path.join(ROOT, "local-dream", "app", "src", "main", "cpp", "src", "Config.hpp")
KT = os.path.join(ROOT, "local-dream", "app", "src", "main", "java", "io", "github",
                  "xororz", "localdream", "data", "ZImageSizes.kt")


def cpp_sizes():
    t = io.open(CPP, encoding="utf-8").read()
    m = re.search(r"zimage_sizes\[\]\s*=\s*\{(.*?)\};", t, re.S)
    if not m:
        raise SystemExit("🔴 Config.hpp 里找不到 zimage_sizes")
    body = re.sub(r"//[^\n]*", "", m.group(1))          # 去掉行注释里的数字
    return [(int(a), int(b)) for a, b in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", body)]


def kt_sizes():
    t = io.open(KT, encoding="utf-8").read()
    m = re.search(r"zimageDeliveredSizes[^=]*=\s*listOf\((.*?)\n\)", t, re.S)
    if not m:
        raise SystemExit("🔴 ZImageSizes.kt 里找不到 zimageDeliveredSizes")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return [(int(a), int(b)) for a, b in re.findall(r"Resolution\(\s*(\d+)\s*,\s*(\d+)\s*\)", body)]


def contract_sizes(p):
    c = json.load(io.open(p, encoding="utf-8"))
    out = [(1024, 1024)]                                 # 基准始终在 models 里
    for k in (c.get("size_variants") or {}):
        w, h = k.lower().split("x")
        out.append((int(w), int(h)))
    return out


def main():
    a, b = cpp_sizes(), kt_sizes()
    print("C++    Config.hpp        : %s" % (a,))
    print("Kotlin ZImageSizes.kt    : %s" % (b,))
    bad = 0
    if a != b:
        print("🔴 C++ 与 Kotlin 不一致：仅 C++ 有 %s ；仅 Kotlin 有 %s"
              % (sorted(set(a) - set(b)), sorted(set(b) - set(a))))
        bad = 1
    else:
        print("✅ C++ 与 Kotlin 逐项一致（含顺序）")

    if len(sys.argv) > 1:
        c = contract_sizes(sys.argv[1])
        print("契约 %-18s : %s" % (os.path.basename(sys.argv[1]), c))
        # 契约可以少于代码（尚未交付的比例），但**不得多于**：代码没列的尺寸
        # 契约里有，说明代码会把它挡在 RequestParser 外，那份图就是死重量。
        extra = sorted(set(c) - set(a))
        miss = sorted(set(a) - set(c))
        if extra:
            print("🔴 契约有而代码没有：%s —— 这些图永远用不到（死重量）" % (extra,))
            bad = 1
        if miss:
            print("⚠️ 代码有而契约没有：%s —— 用户选中会抛 "
                  "\"delivery has no graphs for size\"（硬失败，非静默）" % (miss,))
    print("\n%s" % ("🔴 不一致" if bad else "✅ 一致"))
    return bad


sys.exit(main())
