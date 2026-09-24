# -*- coding: utf-8 -*-
"""往 QNN_CONVERSION_GUIDE.md 追加 §38 / §39（主线 B：可复用结论入库）。

按 MAINLINE §0.4 的纪律：内容锚定、先备份、改完跑 doc_audit。
"""
import io
import os
import shutil

P = os.path.join("D:", os.sep, "LocalDreamZImage", "docs", "QNN_CONVERSION_GUIDE.md")
BAK = os.path.join("D:", os.sep, "LocalDreamZImage", "logs", "GUIDE.bak_before3839.md")

ADD = u"""

---

## 三十八、🔴 **context 内存里有一块叫 spill-fill 的开销，它可以跨 context 共享**（2026-08-30 实测 + 官方文档）

转大模型（本项目下一个是 Flux.2 Klein）撞内存时，这是一条**现成的、官方支持的**省量手段。

### 38.1 先把开销拆开看（宿主，零设备，几分钟）

```bash
qnn-context-binary-utility.exe --context_binary <x.bin> --json_file out.json
```

里面 `graphs[0].info.graphBlobInfo.info.spillFillBufferSize` 就是那块开销；
`graphBlobInfoV2` 还会给 `constSize / opDataSize / ioTensorSize / ddrTensorSize`。

①**本项目四段实测**（L=80，A16W8 + 部分 FP16）：

| 段 | context 文件 | **spillFill** | const | 实测 ION | ION/文件 |
|---|---|---|---|---|---|
| part1a | 2263 MiB | **289** | 2159 | 2928 | 1.29× |
| part1b | 1398 | **266** | 1325 | 1989 | 1.42× |
| part2a | 1354 | **273** | 1273 | 1954 | 1.44× |
| part2b | 1398 | **258** | 1327 | 1812 | 1.30× |
| 合计 | 6413 | **1085** | | 8683 | |

⇒ ①**spillFill 占四段总 ION 的 12.5%**，是「ION 比文件大 29~44%」里最大的单项。

### 38.2 官方机制：一组 context 可以共用一份

`docs/QAIRT-Docs/QNN/general/htp/htp_shared_buffer_tutorial.html` 原文：

> **76.** External spill-fill buffers can also be **shared between graphs of multiple contexts**
> by registering the same external spill-fill buffer with multiple contexts.
> **74-75.** the required size for this buffer is the **largest** out of all the spill-fill buffers.
> **81.** Graphs sharing the same external spill-fill buffer **cannot be executed in parallel**.

⇒ N 个 context 从「各自一份」变成「共用最大的那一份」。
①**本项目算下来可省 1085 − 289 = 796 MiB**。

🔴 **两个必须先确认的前提**：

1. **你的段必须是串行执行的**（第 81 行）。扩散模型的步循环天然串行，满足；
   若你想并行跑两个图，这条路直接封死。
2. **它与 `REGISTER_MULTI_CONTEXTS` 互斥**（第 50/95 行明写 *is not supported together with*）。
   官方其实有**两条**共享路径：①外置 buffer（客户端 `rpcmem_alloc` + 把**同一个 fd**
   `QnnMem_register` 给多个 context）；②`QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS`
   + `QnnHtpContext_GroupRegistration_t{firstGroupHandle, maxSpillFillBuffer}`（QNN 内部分配）。
   **选一条，不能叠用。**

⚠️ ③**「共享后系统 ION 总量真的下降」本项目尚未实测**（四段本就装得下，没用上）。
已知的只有：机制在本机**注册成功**（台账 #60 实测，260 MB 外置 spill-fill 注册 OK）。
`scripts/quadctx_probe/` 已内置 `share` 模式，要用时直接验。

⚠️ **别把它与 #60 的结论搞混**：#60 测的是「外置 spill-fill 能否降低 **PD 容量估算**」
（单个 context 能不能装进 DSP 保护域）—— 结论是 **PD 估算一字节未降**。
本节问的是另一个量：**多个 context 同时驻留时的系统 ION 总量**。两者不相干。

---

## 三十九、⚡ **把所有 context 提出步循环：本项目最大的单笔提速**（2026-08-30 交付）

### 39.1 结果

多段扩散模型的典型写法是在步循环里 `loadGraph → run → 释放`，
因为「全部常驻会爆内存」。本项目分两步把它提了出去：

| 阶段 | 做法 | 耗时 | 精度 |
|---|---|---|---|
| 原始 | 四段都在循环内装卸 | 271 s | — |
| 方案 A | part1a+1b 常驻 | 198～204 s | sha256 逐字节相同 |
| **方案 B** | **四段全常驻** | **160.9 s** | **sha256 逐字节相同** |

①**累计 −40.6%，零精度代价**。代码改动就是把 `loadGraph` 移出 `for`，变量改成循环外的 `unique_ptr`。

### 39.2 为什么安全（这一条要自己核）

重复执行同一个已建图不带一次性状态：I/O 张量只建一次后复用，
执行时只往持久 client buffer 里 `memcpy`。**判据就是出图 sha256 逐字节相同**——
同一份 context、同一份输入，常驻与否在数值上就应当一模一样。
🔴 **sha256 不同 ⇒ 那是 bug（状态污染），不是「收益/代价权衡」。不许退而用「指标在噪声内」放行。**

### 39.3 预估模型不准，判据里别写预期值

用「每段加载耗时 × 步数」推收益，本项目**两次都没推准**：
方案 A 预估 47~54 s、实得 **73 s**（偏大）；方案 B 预估 52 s、实得 **43.4 s**（偏小）。
⇒ **判据只写方向 + 下限**（如「快 ≥15% 算有效」），不写预期数字。

### 39.4 验收必须包含「连续多张」

单张成功不够。常驻把内存压力从「来回起落」变成「**全程维持高位**」，风险形态不同。
①本项目验收：连续 3 张均 HTTP 200、**后端进程 PID 全程未变**（没被杀）、三张 sha256 相同。

⚠️ 读 logcat 时注意假阳性：`am_kill ... due to installPackageLI` 是你自己装 APK 触发的；
`fastrpc ... domain_deinit (kill time 155 us)` 是通道拆除的措辞，**都不是进程被杀**。

### 39.5 `O` / `dlbc` 图优化级别：`O:3` 约 −9% 计算，`dlbc` 无效

①**实测**（单段 part1b、`qnn-net-run` 离线、N=1 vs N=9 斜率、正反两趟）：
CTRL 15.903 s/次｜**O3 14.472 = 0.910×**｜DLBC 16.034 = 1.008×。
三次独立测量的 O3/CTRL = 0.939 / 0.890 / 0.930。

三个配套事实：

- ①**默认就是 `O=0`**。只传 `devices` 段、不传 `graphs` 段时，`O`/`dlbc` **根本不生效**（同 §九十三）。
- ①**加 `graphs` 段但只填无效果的 `vtcm_mb:8`，产物与不传时 md5 逐字节相同** ⇒ 可以直接拿部署版当 CTRL。
- ①**`O:3` 的代价是建图时间与体积**：四段建图 17.9~32.4 分钟/段（CTRL 约 10），体积 **+2.34%**
  ⇒ ION 同比上升，**会吃掉常驻方案的余量**，两个优化叠加时必须重测内存。

⚠️ 措辞：以上是 `qnn-net-run` **离线单段**计时，**不等于** app 内每步耗时（同 §三十二的分母陷阱）；
③端到端收益**尚未实测**。
"""


def main():
    s = io.open(P, encoding="utf-8").read()
    assert u"## 三十八、" not in s, "§38 已存在，不重复追加"
    assert u"### 37.6" in s, "§37.6 应当已在"
    shutil.copyfile(P, BAK)
    io.open(P, "w", encoding="utf-8", newline="").write(s.rstrip() + ADD)
    t = io.open(P, encoding="utf-8").read()
    for k in (u"## 三十八、", u"## 三十九、", u"spillFillBufferSize", u"160.9 s"):
        assert k in t, k
    print("OK 已追加 §38 / §39；备份 %s" % BAK)


if __name__ == "__main__":
    main()
