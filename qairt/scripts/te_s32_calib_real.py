# -*- coding: utf-8 -*-
"""Tier 1 修正版校准：用**真实运行时分布**生成 seq-32 校准数据。

## 为什么重做（2026-08-27 实测，EXP_PLAN_S32_VS_FP32 §六）

第一版 `te_s32_calib.py` 把中间层 hidden 的尾部槽位用「**复制最后一行**」补齐。
理由是「旧图自己就这么干（`Expand(hidden[19:20],[1,12,1])`）、且不引入新极值」。
**实测结果是倒退 1.276 dB**（32 槽 17.486 vs 20 槽 18.763，同 FP32 参考）。

**根因假设（③未验证，本脚本就是它的检验）**：校准分布与运行时不符——
运行时那些槽装的是**真实 pad token（`kQwenPadId=151643`）的 hidden state**，
不是最后一行的复制品。量化器据错误分布定 encoding ⇒ 真实输入落在次优量程上。

## 本脚本的做法

对 5 个原校准样本，取出其真实 token（原 20 槽数据里 `attention_mask==1` 的部分），
按**运行时的方式**重建 seq-32 输入（尾部填 `151643`、mask 填 0），
然后用 **FP32 ONNX（onnxruntime）跑整条 TE 链**，逐段导出真实中间激活：

    input_ids/attention_mask -> part1 -> add_2452 -> part2 -> add_4828 -> part3 -> add_7204

⚠️ 约束 3：宿主侧 `.raw` 一律 **float32**（原校准文件实测就是 float32，
按 int32 读会得到 1065353216 这类乱码）。

⚠️ 口径差异必须记住：**旧的部署版校准用 pad=0**（实测 sample_0002/0003 尾部是 0），
而运行时是 151643。本脚本改成与运行时一致 —— 这是**一处刻意的变更**，
不是「恢复原状」。它是否真的更好，**由设备实测的 L3 PSNR 判定**。
"""
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort

ONNX = r'D:\ZImage_Work\ZImage_QNN_Evidence\onnx'
SRC_ROOT = r'D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline'
DST_ROOT = r'D:\ZImage_Work\TIER1_S32'
OLD_LEN, NEW_LEN, HIDDEN = 20, 32, 2560
PAD = 151643            # kQwenPadId，与 TextEncoder.hpp:613 一致
CHAIN = [('text_encoder_part1_s32.onnx', ['input_ids', 'attention_mask'], 'add_2452'),
         ('text_encoder_part2_s32.onnx', ['add_2452', 'attention_mask'], 'add_4828'),
         ('text_encoder_part3_s32.onnx', ['add_4828', 'attention_mask'], 'add_7204'),
         ('text_encoder_part4_s32.onnx', ['add_7204', 'attention_mask'], 'caption')]
PART_INPUTS = {
    'text_encoder_part1': ['input_ids', 'attention_mask'],
    'text_encoder_part2': ['add_2452', 'attention_mask'],
    'text_encoder_part3': ['add_4828', 'attention_mask'],
    'text_encoder_part4': ['add_7204', 'attention_mask'],
}


def real_tokens(sample_dir):
    """从原 20 槽校准样本取出真实 token 序列。"""
    ids = np.fromfile(os.path.join(sample_dir, 'input_ids.raw'), np.float32)
    msk = np.fromfile(os.path.join(sample_dir, 'attention_mask.raw'), np.float32)
    if ids.size != OLD_LEN or msk.size != OLD_LEN:
        raise ValueError('%s 元素数不是 %d' % (sample_dir, OLD_LEN))
    n = int((msk > 0.5).sum())
    return ids[:n].astype(np.int64), n


def main():
    sess = {}
    cwd = os.getcwd()
    os.chdir(ONNX)                      # external data 按当前工作目录解析
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        for mdl, _, _ in CHAIN:
            sess[mdl] = ort.InferenceSession(mdl, so, providers=['CPUExecutionProvider'])
            print('  已载入 %s' % mdl, flush=True)
    finally:
        os.chdir(cwd)

    src1 = os.path.join(SRC_ROOT, 'text_encoder_part1', 'calibration_raw')
    samples = sorted(os.listdir(src1))
    # 先清空四段的目标目录
    for part in PART_INPUTS:
        d = os.path.join(DST_ROOT, part, 'calibration_raw')
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    per_part_lines = {p: [] for p in PART_INPUTS}
    for s in samples:
        ids, n = real_tokens(os.path.join(src1, s))
        a = np.full((1, NEW_LEN), PAD, dtype=np.int64)
        a[0, :n] = ids
        m = np.zeros((1, NEW_LEN), dtype=np.int64)
        m[0, :n] = 1

        t = {'input_ids': a, 'attention_mask': m}
        for mdl, ins, out in CHAIN:
            se = sess[mdl]
            feeds = {}
            for i in se.get_inputs():
                arr = t[i.name]
                if 'int32' in i.type:
                    arr = arr.astype(np.int32)
                elif 'float' in i.type:
                    arr = arr.astype(np.float32)
                feeds[i.name] = arr
            for o, v in zip(se.get_outputs(), se.run(None, feeds)):
                t[o.name] = v
        print('  %s 真实 token=%2d  caption %s' % (s, n, t['caption'].shape), flush=True)

        # 逐段写出该段自己的输入
        for part, ins in PART_INPUTS.items():
            d = os.path.join(DST_ROOT, part, 'calibration_raw', s)
            os.makedirs(d, exist_ok=True)
            entry = []
            for name in ins:
                v = t[name]
                arr = np.ascontiguousarray(v.astype(np.float32)).reshape(-1)
                want = NEW_LEN * 4 if name in ('input_ids', 'attention_mask') \
                    else NEW_LEN * HIDDEN * 4
                p = os.path.join(d, name + '.raw')
                arr.tofile(p)
                got = os.path.getsize(p)
                if got != want:
                    print('FAIL %s/%s/%s: %d B != %d B' % (part, s, name, got, want))
                    return 1
                entry.append('%s:=%s' % (name, p))
            per_part_lines[part].append(' '.join(entry))

    for part, lines in per_part_lines.items():
        il = os.path.join(DST_ROOT, part, 'input_list_raw.txt')
        with open(il, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('%-20s %d 个样本 -> %s' % (part, len(lines), il))
    return 0


if __name__ == '__main__':
    sys.exit(main())
