"""交接文档体检：把「留给下一个线程的坑」变成可机检的东西。

存在理由：MAINLINE §6 的过期执行表**连续两次**（08-22、08-24）被留在文档里，
下一个线程照它行动就会重跑已关闭的路。眼睛看不可靠。

用法: python doc_audit.py [--strict]

检查项（每条都对应真实发生过的事故）：
  D1 重复/缺失的小节编号，以及**编号顺序颠倒**（§2.5 曾排在 §2.4 之前）
  D2 §6 里出现多于一张执行表（两次事故的直接形态）
  D3 引用了不存在的文件/脚本路径
  D4 引用了不存在的 §小节号（HANDOVER 目录曾把未写的 §15.30 标为"最新"）
  D5 关键数字在全文中不一致（当前最优 L3 / L1 / 参照上限）
  D6 markdown 表格被切断或行被合并（2026-08-24：按偏移插入把 3 行并成 1 行、丢了 2 条台账）
"""
import io
import os
import re
import sys

# 🔴 本工具是强制门（MAINLINE §0.4）。它在 GBK 控制台上打印 ⚠️ 会抛
#    UnicodeEncodeError 而**整个崩掉**——门崩掉和门放行一样危险，都不报问题。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MAIN = os.path.join(ROOT, "MAINLINE.md")
HAND = os.path.join(ROOT, "docs", "HANDOVER_2026-08-13_ZIMAGE_MVP.md")
GUIDE = os.path.join(ROOT, "docs", "QNN_CONVERSION_GUIDE.md")

def _load_baseline():
    """已知历史缺陷基线（scripts/known_d6.json）。

    只登记"已确认存在、且决定**不自动修**"的旧缺陷 —— 见台账 #118。
    新出现的表格损坏不在基线里，仍会照报。要清理基线里的条目必须**人工逐处确认**。
    """
    import json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_d6.json")
    if not os.path.exists(p):
        return set()
    return {(a, b) for a, b in json.load(io.open(p, encoding="utf-8"))}


KNOWN_D6 = _load_baseline()

KEY_NUMBERS = {
    # 2026-08-25：#111 Clip+FP16 把最优从 15.53 推到 18.67 dB。
    # 🔴 守卫自身过期也是事故（本行原先仍盯着 15.53）=> 换最优时必须同步改这里。
    "当前最优 L3 PSNR": ("18.67 dB", r"18\.67\s*dB"),
    "当前最优 L1 E_all": ("24.37%", r"24\.37\s*%"),
    "参照上限": ("41.61 dB", r"41\.61\s*dB"),
}


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def check_sections(txt, name, hits):
    """D1: 小节编号重复 / 顺序颠倒"""
    secs = re.findall(r"^#{2,3}\s+(\d+(?:\.\d+)*)", txt, re.M)
    seen, dup = set(), []
    for s in secs:
        if s in seen:
            dup.append(s)
        seen.add(s)
    if dup:
        hits.append(("D1", name, "小节编号重复: %s" % sorted(set(dup))))
    # 顺序：同一父章节内的子节应递增
    nums = [tuple(int(x) for x in s.split(".")) for s in secs]
    for i in range(1, len(nums)):
        a, b = nums[i - 1], nums[i]
        if len(a) == len(b) == 2 and a[0] == b[0] and b[1] < a[1]:
            hits.append(("D1", name, "小节顺序颠倒: §%d.%d 排在 §%d.%d 之后"
                         % (b[0], b[1], a[0], a[1])))


def check_exec_tables(txt, hits):
    """D2: §6 里只允许一张执行表"""
    m = re.search(r"^## 6\..*?(?=^## 7\.)", txt, re.M | re.S)
    if not m:
        hits.append(("D2", "MAINLINE", "找不到 §6 执行顺序"))
        return
    seg = m.group(0)
    # 执行表的特征：表头含「设备」或「成本」，且行首是 P0/P1/P2/P3 或 序号
    n = len(re.findall(r"^\|\s*(?:\*\*)?P[0-3]", seg, re.M))
    tables = len(re.findall(r"^\|\s*序\s*\|", seg, re.M))
    if tables > 1:
        hits.append(("D2", "MAINLINE", "§6 里有 %d 张「序|条目|设备」表 —— 只允许一张" % tables))
    if n and tables:
        hits.append(("D2", "MAINLINE", "§6 同时存在 P 级列表与「序」表 —— 疑似新旧两张并存"))


def check_paths(txt, name, hits):
    """D3: 引用的脚本/文档是否存在"""
    for m in re.finditer(r"`((?:scripts|docs|archive)/[A-Za-z0-9_./-]+\.(?:py|md|sh|json))`", txt):
        rel = m.group(1)
        if not os.path.exists(os.path.join(ROOT, rel)):
            hits.append(("D3", name, "引用了不存在的文件: %s" % rel))


def check_section_refs(txt, name, target_txt, target_name, hits):
    """D4: 引用的 §15.x 小节是否真的存在正文"""
    have = set(re.findall(r"^##\s+(15\.\d+)", target_txt, re.M))
    for m in re.finditer(r"§\s*(15\.\d+)", txt):
        if m.group(1) not in have:
            hits.append(("D4", name, "引用了 %s 的 §%s，但该小节无正文"
                         % (target_name, m.group(1))))


def check_numbers(txts, hits):
    """D5: 关键数字是否出现相互矛盾的旧值"""
    joined = "\n".join(txts)
    # 旧的「当前最优」值若仍以「当前/最优/现在」修饰出现，即为冲突
    for bad, why in [(r"最优[^\n]{0,30}14\.56\s*dB", "14.56 dB 曾被误当作最优"),
                     (r"当前最优[^\n]{0,30}15\.53\s*dB", "15.53 dB 已被 #111 的 18.67 dB 取代"),
                     (r"当前最优[^\n]{0,30}11\.93\s*dB", "11.93 dB 是起点不是最优")]:
        if re.search(bad, joined):
            hits.append(("D5", "全文", why))
    for label, (val, pat) in KEY_NUMBERS.items():
        if not re.search(pat, joined):
            hits.append(("D5", "全文", "关键数字缺失: %s 应为 %s" % (label, val)))



def check_tables(txt, name, hits):
    """D6: markdown 表格结构完整性（2026-08-24 事故：按偏移插入把行切断/合并）

    两类致命形态，眼睛都看不出来：
      · 行被**合并** —— 两条台账挤进一行，渲染时后一条整个消失
      · 表格中间出现**不以 `|` 开头的孤儿行** —— 表格在此被截断，后面的行不再是表格

    竖线计数必须扣掉转义的 `\|`（正文里 `|a|max` 这类写法本来就带竖线）。
    """
    lines = txt.split("\n")
    # 🔴 必须跳过 ``` 代码围栏：代码块里也会有以 | 开头的行（HANDOVER §7 就有）。
    #    2026-08-24 我写的"自动修表格"脚本正是漏了这一条，把代码块的结束围栏
    #    并进了正文、还给代码里的 | 加了转义 —— 事后整份回滚才消除。
    fence = [False] * len(lines)
    inb = False
    for k, x in enumerate(lines):
        if x.lstrip().startswith("```"):
            inb = not inb
            fence[k] = True
            continue
        fence[k] = inb
    i = 0
    while i < len(lines):
        if fence[i] or not lines[i].startswith("|"):
            i += 1
            continue
        start = i
        # 表头 + 分隔行
        if i + 1 >= len(lines) or not re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            i += 1
            continue
        ncol = lines[i].replace("\\|", "").count("|")
        i += 2
        while i < len(lines) and lines[i].strip():
            ln = lines[i]
            if not ln.startswith("|"):
                hits.append(("D6", name, "第 %d 行：表格（起于第 %d 行）中间出现非表格行 "
                             "=> 表格被截断。开头: %s" % (i + 1, start + 1, ln[:40])))
                break
            n = ln.replace("\\|", "").count("|")
            if n != ncol:
                hits.append(("D6", name, "第 %d 行：竖线数 %d ≠ 表头 %d "
                             "=> 行被合并或列数不符。开头: %s"
                             % (i + 1, n, ncol, ln[:40])))
            i += 1
        i += 1

def check_dup_blocks(txt, name, hits, win=200):
    """D7: 同一表格行里出现**重复的长子串** —— `str.replace` 忘了 count 的签名。

    2026-08-29 事故：给台账行追加内容时写了
        row.replace(" |", EXTRA + " |")          # 没有 count=1
    `" |"` 在一行里出现 4 次（每个列边界一次）=> EXTRA 被插了 4 遍，行结构被撑烂。
    🔴 **D6 完全没拦住**：每个单元格都被插入了同样的东西，**列数因此仍然正确**
    => 需要一条正交的检查。

    🔴 本规则的**第一版是错的**（约束 8 的又一个例子）：第一版按 `<br>` 切段比对，
    在已知坏样本上**一条都没命中** —— 因为重复块后面跟着不同的尾巴，切不出相同的段。
    现改为定长滑窗查重复子串，并已用 bak4（已知坏）与当前 MAINLINE（已知好）双向验证。
    🔴 **第二版也错了**：滑窗按 step=20 采样，两次出现的**偏移差不是 20 的倍数**
    就永远对不上（实测：坏样本里锚点出现 4 次，采样版命中 0 条）。必须逐位置扫。
    """
    lines = txt.split(chr(10))
    inb = False
    for k, ln in enumerate(lines):
        if ln.lstrip().startswith('```'):
            inb = not inb
            continue
        if inb or not ln.startswith('|') or len(ln) < 2 * win:
            continue
        seen = {}
        for p in range(0, len(ln) - win + 1):
            w = ln[p:p + win]
            q = seen.get(w)
            if q is not None and p - q >= win:
                hits.append(('D7', name,
                             '第 %d 行：长度 %d 的子串在同一行内重复出现（位置 %d 与 %d）'
                             ' => 极可能是 str.replace 漏了 count。片段开头: %s'
                             % (k + 1, win, q, p, w[:40])))
                break
            if q is None:
                seen[w] = p


# ── D7：硬性结论必须能回答「怎么知道的」与「什么条件下成立」 ────────────────
#
# 存在理由（2026-09-21）：本轮花了大量精力**重新验证之前线程已经做过的结论**。
# 逐条归类后，其中可避免的那部分只有两种成因：
#   A 证据/推理没写 —— 例：§2.4「转换器只支持到 opset 17」只有断言没有依据，
#     实测发现 text_encoder/vae 全是 opset 18 且已交付跑通；
#   B 适用范围没写 —— 例：`--use_native_*`，§五 与 §45.8 **两条都有扎实实测**，
#     但都写成了无条件祈使句，于是互相矛盾，照任一条做都可能踩坑。
# （另有两种**写清也免不掉**：结论有时效性（L=32 时代的校准结论，L80 后必须重测）、
#   以及验证者自己选错样本 —— 那两类不是文档能解决的。）
#
# ⇒ 本检查只管 A/B：**硬性断言所在的段落里，必须出现证据标记或适用条件。**
#   不是要求每句话都带出处，而是不允许"光下命令、不说来源"。
IMPERATIVE = ("必须", "一律", "不得", "禁止", "只能", "绝不", "永远不")
EVIDENCE_MARK = ("①", "②", "③", "实测", "官方文档", "文档实查", "原文", "见 §", "见§",
                 "§", "台账", "HANDOVER", "EXP_PLAN", "未验证", "尚未验证", "本项目")


def check_claims(txt, name, hits):
    """D7（⚠️ 实验性，`--claims` 才启用，**不在强制门里**）。

    🔴 **当前状态：操作化未经已知样本验证，不可依赖。**
    2026-09-21 实测：GUIDE 上命中 44 处，其中大量假阳性（段落被空行切开，
    证据在相邻段落；"三个必须遵守的点："这类引导句后面才是证据）。
    更关键的是，关键词表**恰好漏掉了它本该抓的两个真实案例**：
      · §2.4 写的是「只支持到 opset 16/17」——"只支持到"不在 IMPERATIVE 里
      · §五 2.5a 写的是「**不要**加 `--use_native_*`」——"不要"也不在
    ⇒ 要让它可用，必须先拿这两处的**修改前原文**当已知样本调参（约束 8）。
    在那之前它只是个提示器，不许当判据。
    """
    for blk in re.split(r"\n\s*\n", txt):
        if not any(k in blk for k in IMPERATIVE):
            continue
        if any(k in blk for k in EVIDENCE_MARK):
            continue
        if blk.lstrip().startswith(("```", "#", "|")):   # 代码块/标题/表格行另算
            continue
        head = re.sub(r"\s+", " ", blk.strip())[:56]
        hits.append(("D7", name, "硬性断言但段落内无证据标记/适用范围 —— 开头: " + head))


def main():
    hits = []
    mt, ht, gt = read(MAIN), read(HAND), read(GUIDE)
    # 🔴 D7 **默认不跑**（`--claims` 才启用）。理由见 check_claims 的文档串：
    #    它的操作化**尚未用已知样本验证**，实测 44 处命中里大量是假阳性，
    #    而且关键词表**恰好漏掉了它本该抓的两个真实案例**（§2.4 用「只支持到」、
    #    §五 2.5a 用「不要加」，都不在 IMPERATIVE 里）。
    #    doc_audit 是强制门（MAINLINE §0.4）——**门常红等于门失效**，
    #    没验证过的判据不许进门。这本身就是 §47.10 模式 4。
    if "--claims" in sys.argv:
        check_claims(gt, "GUIDE", hits)
    check_sections(mt, "MAINLINE", hits)
    check_exec_tables(mt, hits)
    for t, n in ((mt, "MAINLINE"), (ht, "HANDOVER"), (gt, "GUIDE")):
        check_paths(t, n, hits)
    check_section_refs(mt, "MAINLINE", ht, "HANDOVER", hits)
    check_numbers([mt], hits)
    for t, n in ((mt, "MAINLINE"), (ht, "HANDOVER"), (gt, "GUIDE")):
        check_dup_blocks(t, n, hits)
    d6 = []
    for t, n in ((mt, "MAINLINE"), (ht, "HANDOVER"), (gt, "GUIDE")):
        check_tables(t, n, d6)
    # 已知历史缺陷（2026-08-24 基线）：不让门常红，但**新出现的仍会报**。
    # 🔴 基线只登记"我确认过、且决定不动"的旧缺陷 —— 见台账 #118：
    #    我曾试图自动修 HANDOVER，结果又弄坏一次（代码围栏被并行），已整份回滚。
    #    => 结论：HANDOVER 的表格缺陷**只登记不自动修**，要修必须人工逐处确认。
    known = 0
    for rid, where, msg in d6:
        key = re.search(r"开头: (.{0,36})", msg)
        k = (where, key.group(1) if key else msg)
        if k in KNOWN_D6:
            known += 1
            continue
        hits.append((rid, where, msg))
    if known:
        print("（D6 跳过 %d 处已知历史缺陷，见台账 #118；新出现的仍会报）" % known)

    print("文档体检：MAINLINE / HANDOVER / GUIDE")
    if not hits:
        print("✅ 无问题")
        return 0
    print("\n发现 %d 处：\n" % len(hits))
    for rid, where, msg in hits:
        print("  [%s] %-9s %s" % (rid, where, msg))
    return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc if "--strict" in sys.argv else 0)
