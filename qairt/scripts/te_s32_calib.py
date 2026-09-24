# -*- coding: utf-8 -*-
"""Tier 1 步2：把 text_encoder 四段的校准数据从 20 槽补到 32 槽。

约束 3：宿主侧一律 float32（实测确认——现有 `input_ids.raw` 就是 float32，
按 int32 读会得到 1065353216 这类乱码）。

补齐规则（每段两类输入）：
  · `input_ids` / `attention_mask`  ——  补 **0**，与现有样本的 padding 写法一致
    （实测 sample_0002/0003 的尾部就是 0，mask 也是 0）
  · 中间层 hidden（`add_2452` / `add_4828` / `add_7204`，[1,20,2560]）
    ——  **重复最后一行 12 次**

🔴 为什么用「重复最后一行」而不是真跑一遍 part1→2→3：
  ① 这正是**旧图自己的做法**：原 part4 图尾用 `Expand(hidden[:,19:20,:],[1,12,1])`
     把第 20 行复制 12 份填满 32 槽 ⇒ 这个分布是部署模型本来就见过的；
  ② 重复行**不引入新的极值** ⇒ min-max 量程与现有部署版逐位一致
     ⇒ encoding 基本不变 ⇒ 对 T3（短 prompt 回归应尽量一致）最有利；
  ③ 真跑链式推理需要改 `snpe_runner.py` 的契约与硬编码路径（共享基础设施）。
⚠️ ③**未验证假设**：运行时那 12 个槽装的是真实 pad token 的 hidden state，
   与复制行不同。**这条由设备上的 T3/T4 检验**，不得当成已验证。
"""
import os
import shutil
import sys

import numpy as np

SRC_ROOT = r'D:\ZImage_Work\ZImage_QNN_Evidence\dlc_pipeline'
DST_ROOT = r'D:\ZImage_Work\TIER1_S32'
OLD_LEN, NEW_LEN = 20, 32
HIDDEN = 2560
PARTS = {
    'text_encoder_part1': ['input_ids', 'attention_mask'],
    'text_encoder_part2': ['add_2452', 'attention_mask'],
    'text_encoder_part3': ['add_4828', 'attention_mask'],
    'text_encoder_part4': ['add_7204', 'attention_mask'],
}


def pad_tensor(a, name):
    """a 是按 float32 读出的一维数组。返回补到 NEW_LEN 的一维数组。"""
    if name in ('input_ids', 'attention_mask'):
        if a.size != OLD_LEN:
            raise ValueError('%s 元素数 %d != %d' % (name, a.size, OLD_LEN))
        return np.concatenate([a, np.zeros(NEW_LEN - OLD_LEN, dtype=np.float32)])
    # 中间层 hidden: [1, OLD_LEN, HIDDEN]
    if a.size != OLD_LEN * HIDDEN:
        raise ValueError('%s 元素数 %d != %d' % (name, a.size, OLD_LEN * HIDDEN))
    m = a.reshape(OLD_LEN, HIDDEN)
    tail = np.repeat(m[-1:], NEW_LEN - OLD_LEN, axis=0)
    return np.concatenate([m, tail], axis=0).reshape(-1)


def main():
    for part, inputs in PARTS.items():
        src = os.path.join(SRC_ROOT, part, 'calibration_raw')
        dst = os.path.join(DST_ROOT, part, 'calibration_raw')
        if not os.path.isdir(src):
            print('FAIL: 缺少 %s' % src)
            return 1
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)

        samples = sorted(os.listdir(src))
        lines = []
        for s in samples:
            os.makedirs(os.path.join(dst, s))
            entry = []
            for name in inputs:
                f = os.path.join(src, s, name + '.raw')
                a = np.fromfile(f, dtype=np.float32)
                out = pad_tensor(a, name).astype(np.float32)
                of = os.path.join(dst, s, name + '.raw')
                out.tofile(of)
                # 字节数硬校验（约束 3：尺寸不对 snpe 不报错，只静默缩批）
                want = NEW_LEN * 4 if name in ('input_ids', 'attention_mask') \
                    else NEW_LEN * HIDDEN * 4
                got = os.path.getsize(of)
                if got != want:
                    print('FAIL %s/%s/%s: %d B != %d B' % (part, s, name, got, want))
                    return 1
                entry.append('%s:=%s' % (name, of))
            lines.append(' '.join(entry))

        il = os.path.join(DST_ROOT, part, 'input_list_raw.txt')
        with open(il, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('%-20s %d 个样本, 每样本 %s, input_list -> %s'
              % (part, len(samples), '+'.join(inputs), os.path.basename(il)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
