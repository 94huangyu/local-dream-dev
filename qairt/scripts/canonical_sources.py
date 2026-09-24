# -*- coding: utf-8 -*-
"""🔴 权威来源解析器 —— **任何要读「某段的源 ONNX / DLC / context」的脚本都必须走这里**。

## 为什么存在（同一个错误犯了两次）

| 次数 | 事故 |
|---|---|
| #61 | 用错源 ONNX（`base` 而非 `_fixed`），结论作废 |
| 2026-08-27 | Tier 2 手术做在 `transformer_part1a.onnx` 上，而部署用的是 `transformer_part1a_clip.onnx`（带显式 Clip + FP16 白名单，值 **+6.74 dB**）。八段转换白做，删掉 23 GB 产物 |

#61 当时的结论是「**不得由文件名推断**」——但只写成了提醒，**没有做成机制**，于是又犯。
本文件把它变成可执行的解析：**名字由这里给出，脚本不许自己拼。**

## 现状有多容易取错（实测计数）

`onnx/` 里共 **72 个 .onnx**，其中 `transformer_part1a*` 就有 **14 个**变体
（`_clip / _exc / _gq8 / _simA..M / _taps* / _with_early_taps / _L80 ...`）。
且命名**不一致**：part1a/1b 是 `<base>_clip`，part2a/2b 是 `<base>_fixed_clip`。
量化 DLC 同样有 `part1a_fp16 / part1a_perrow / part1a_dt` 三套长得一样的名字。

## 事实来源（每条都标出处，不是我编的）

- ONNX base 映射：`scripts/insert_clip.py:29-30` 的 `ONNX = {...}` —— **就是造出部署产物的那个脚本**
- Clip 变体命名：`insert_clip.py` 的产物规则 `<base>_clip.onnx`
- 部署产物文件名与 `qnn_graph_name`：设备上的 `final_qnn_contract.json`
- 白名单构成：契约的 `quantization_granularity` 字段
  = 「old-26 的 26 个 + 18 个被钳的 massive activation」
  ⚠️ **指南 §29.8 只记到 old-26（15.53 dB），比部署配置旧一代**，照抄会漏掉 Clip 那一层

## 🔴 序列长度：`onnx_for(seg)` **不是**当前部署的那张图（2026-08-30 修）

Tier 2（2026-08-29）交付后，**设备上跑的是 L=80**（`*_clip_L80.onnx` 产出的 context），
而本模块的表只记到 L=32。谁把 `onnx_for('part1b')` 读成「部署用的图」谁就错了 ——
这正是 #61/#136/#138 那三次取错源的同一个形态：**交付变了、权威表没跟上。**

规则：

- `onnx_for(seg, variant)` —— 返回 **L=32 的原始表**（`_L80` 手术的**输入**）。
  `dit_seq_surgery.py` 依赖这个语义（它有 stem 断言），**不要改**。
- `onnx_for(seg, variant, L=80)` —— 返回**当前部署长度**的那张图（`*_L80.onnx`）。
- `DEPLOYED_L` —— 当前部署的序列长度。**换代时改这一个常量。**

用法：
    from canonical_sources import onnx_for, describe, DEPLOYED_L
    p = onnx_for('part1a')                      # L=32 原始表（手术输入）
    p = onnx_for('part1a', L=DEPLOYED_L)        # 当前真正部署的那张图
    print(describe('part1a'))
"""
import os

ONNX_DIR = os.path.join('D:', os.sep, 'ZImage_Work', 'ZImage_QNN_Evidence', 'onnx')

# 🔴 当前部署的序列长度（caption 槽位）。Tier 2 于 2026-08-29 把它从 32 改到 80。
# 依据：台账 #134/#139 + `scripts/t2_deliver.sh` 下发的 `*_L80_ctx_sm8750.SM8750.bin`。
DEPLOYED_L = 80

# 段 -> (Clip 注入前的 base, 部署实际使用的 ONNX 主干)
# 🔴 base 一列抄自 insert_clip.py 的 ONNX 字典；deployed 一列是它加 Clip 后的产物名。
TRANSFORMER = {
    'part1a': ('transformer_part1a',       'transformer_part1a_clip'),
    'part1b': ('transformer_part1b',       'transformer_part1b_clip'),
    'part2a': ('transformer_part2a_fixed', 'transformer_part2a_fixed_clip'),
    'part2b': ('transformer_part2b_fixed', 'transformer_part2b_fixed_clip'),
}
# TE 没有 clip 变体，源就是 base
TEXT_ENCODER = {'te%d' % i: ('text_encoder_part%d' % i, 'text_encoder_part%d' % i)
                for i in (1, 2, 3, 4)}
ALL = dict(TRANSFORMER, **TEXT_ENCODER)

# 🚫 这些**不是**部署源，出现在参数里一律拒绝（防止手滑）
TRAP_SUFFIXES = ('_exc', '_gq8', '_simA', '_simC', '_simF', '_simG', '_simH', '_simM',
                 '_simZ', '_taps', '_with_early_taps', '_stats', '_maskfix', '_dce',
                 '_taps_caption_ffn_reference', '_taps_merge_window_reference')


def _check(stem):
    for s in TRAP_SUFFIXES:
        if stem.endswith(s):
            raise ValueError('%s 是实验变体，不是部署源。用 canonical_sources.onnx_for()' % stem)


def onnx_for(seg, variant='deployed', L=None):
    """返回该段的源 ONNX 绝对路径。

    variant: deployed（带 Clip）| base（Clip 注入前）
    L:       None = L=32 的原始表（`_L80` 手术的**输入**，dit_seq_surgery 依赖此语义）
             80   = **当前真正部署**的那张图（`*_L80.onnx`）
    """
    if seg not in ALL:
        raise KeyError('未知段 %r，可选 %s' % (seg, sorted(ALL)))
    stem = ALL[seg][0 if variant == 'base' else 1]
    _check(stem)
    if L is not None and L != 32:
        stem = '%s_L%d' % (stem, L)
    p = os.path.join(ONNX_DIR, stem + '.onnx')
    if not os.path.isfile(p):
        raise FileNotFoundError('%s 不存在：%s' % (seg, p))
    return p


def describe(seg):
    base, dep = ALL[seg]
    return ('%s: L=32 原始表 = %s.onnx（base = %s.onnx）；'
            '当前部署 L=%d ⇒ %s_L%d.onnx'
            % (seg, dep, base, DEPLOYED_L, dep, DEPLOYED_L))


def audit():
    """列出所有段的解析结果，并报告 onnx 目录里的变体数量（易取错程度）。"""
    import glob
    print('%-8s %-34s %-30s %s' % ('段', '部署源 ONNX', 'base', '存在'))
    for seg in ('part1a', 'part1b', 'part2a', 'part2b', 'te1', 'te2', 'te3', 'te4'):
        base, dep = ALL[seg]
        ok = os.path.isfile(os.path.join(ONNX_DIR, dep + '.onnx'))
        print('%-8s %-34s %-30s %s' % (seg, dep + '.onnx', base + '.onnx', '✅' if ok else '🔴'))
    print()
    tot = len(glob.glob(os.path.join(ONNX_DIR, '*.onnx')))
    for pre in ('transformer_part1a', 'transformer_part1b',
                'transformer_part2a', 'transformer_part2b', 'text_encoder_part1'):
        n = len(glob.glob(os.path.join(ONNX_DIR, pre + '*.onnx')))
        print('   %-24s 有 %2d 个同前缀变体' % (pre, n))
    print('   onnx 目录共 %d 个 .onnx ⇒ **靠名字挑必错，必须走本模块**' % tot)


if __name__ == '__main__':
    audit()
