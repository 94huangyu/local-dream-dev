# -*- coding: utf-8 -*-
"""订正 #129 的「步数已到下限」——那是误读，不是实测。"""
import io
import os
import shutil

ROOT = os.path.join("D:", os.sep, "LocalDreamZImage")
MAIN = os.path.join(ROOT, "MAINLINE.md")

OLD = (u"\u2461 **\u6b65\u6570\u5df2\u5230\u4e0b\u9650**"
       u"\uff08`zimage_turbo_steps=8`\uff0c\u539f\u751f\u5f3a\u5236\uff09\u3002")
NEW = (u"\u2461 ~~**\u6b65\u6570\u5df2\u5230\u4e0b\u9650**"
       u"\uff08`zimage_turbo_steps=8`\uff0c\u539f\u751f\u5f3a\u5236\uff09~~ "
       u"\U0001F534 **\u8be5\u53e5\u5df2\u8ba2\u6b63\uff082026-08-31\uff09\uff1a\u8bef\u8bfb\uff0c\u975e\u5b9e\u6d4b**\u3002"
       u"\u5b9e\u67e5\u6e90\u7801\uff1a`Config.hpp:75` \u662f\u4e00\u4e2a**\u786c\u7f16\u7801\u5e38\u91cf** "
       u"`inline constexpr int zimage_turbo_steps = 8;`\uff0c"
       u"`RequestParser.hpp:90` \u7528\u5b83**\u8986\u76d6\u7528\u6237\u8bf7\u6c42**\u7684 steps\u3002"
       u"\u21d2 \u300c\u539f\u751f\u5f3a\u5236\u300d\u7684\u771f\u5b9e\u542b\u4e49\u662f"
       u"\u300c**\u6211\u4eec\u7684 app \u5199\u6b7b\u4e86 8**\u300d\uff0c"
       u"**\u4e0d\u662f\u300c\u6a21\u578b\u8981\u6c42 8 \u6b65\u300d**\u3002"
       u"\u5b98\u65b9\u6a21\u578b\u76ee\u5f55\uff08`D:\\Z-Image-Turbo`\uff09\u91cc"
       u"**\u6ca1\u6709\u4efb\u4f55\u5730\u65b9\u89c4\u5b9a\u6b65\u6570**"
       u"\uff08`num_inference_steps` \u662f\u8fd0\u884c\u65f6\u53c2\u6570\uff09\u3002<br>"
       u"\u21d2 \U0001F534 **\u300c\u6b65\u6570\u80fd\u4e0d\u80fd\u51cf\u300d\u672c\u9879\u76ee\u4ece\u672a\u6d4b\u8fc7**\uff0c"
       u"\u5b83\u662f\u76ee\u524d**\u5e45\u5ea6\u6700\u5927\u7684\u672a\u6d4b\u6760\u6746**"
       u"\uff088\u21926 \u7ea6 \u221225%\uff0c8\u21924 \u7ea6 \u221250%\uff09\u3002"
       u"\u26a0\ufe0f \u4f46\u5b83\u662f**\u62ff\u753b\u8d28\u6362\u901f\u5ea6**\uff0c"
       u"\u4e0e\u7528\u6237 2026-08-25 \u5b9a\u7684\u300c\u4e0d\u5f97\u62ff\u901f\u5ea6\u6362\u7cbe\u5ea6\u300d\u662f**\u53cd\u65b9\u5411**\uff0c"
       u"\u7528\u6237\u672a\u8868\u6001 \u21d2 **\u5c5e\u4e8e\u9700\u8981\u7528\u6237\u5b9a\u7684\u53d6\u820d\uff08\u7ea6\u675f 10\u00b7\u8865\uff09**\u3002<br>"
       u"\u2295 **\u6559\u8bad**\uff1a\u672c\u6761\u628a\u300c\u4ee3\u7801\u5199\u6b7b\u300d\u5199\u6210\u4e86\u300c\u5df2\u5230\u4e0b\u9650\u300d\uff0c"
       u"\u4e4b\u540e\u591a\u8f6e\u88ab\u5f53\u6210\u5b9e\u6d4b\u4e8b\u5b9e\u5f15\u7528\uff08\u542b\u6211\u5199\u7684\u300c\u901f\u5ea6\u7ebf\u89c1\u5e95\u300d\uff09\u3002"
       u"\u540c\u7c7b\uff1a\u7ea6\u675f 1\u3002")


def main():
    s = io.open(MAIN, encoding="utf-8").read()
    assert s.count(OLD) == 1, "锚点未唯一命中"
    assert u"该句已订正" not in s
    shutil.copyfile(MAIN, os.path.join(ROOT, "logs", "MAINLINE.bak_steps.md"))
    io.open(MAIN, "w", encoding="utf-8", newline="").write(s.replace(OLD, NEW, 1))
    print("OK #129 的「步数已到下限」已订正")


if __name__ == "__main__":
    main()
