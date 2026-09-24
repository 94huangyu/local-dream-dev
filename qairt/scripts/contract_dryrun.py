# -*- coding: utf-8 -*-
"""宿主侧「干跑」：照 app 的校验逻辑逐图走一遍契约，不需要设备。

## 为什么有这个脚本（2026-08-27 事故）

seq-32 交付后真机报 `Z-Image manifest input byte mismatch: input_ids`：
我更新契约时只改了 sha256/size_bytes，漏了 `inputs/outputs` 的 `exact_bytes`。
**这个错误完全可以在宿主上、零设备成本地提前发现** —— 它只是字节数对不对得上。

本脚本复刻 `PipelineZImage.hpp` 的两处校验：

  · `valueFor()`  (行 ~402)：每个图的每个声明输入，必须能在 values 里找到生产者，
                              且 `bytes.size() == spec.bytes`，否则抛
                              "Z-Image manifest input byte mismatch: <name>"
  · `runGraph()`  (行 ~442)：宿主直接喂的（无 spec 的）值，字节数必须等于 input.bytes

以及宿主喂入的字节数由 app 源码决定：

  · input_ids / attention_mask : zimage_text_max_length * 4   (int32)
  · cap_pad_mask               : 32 * 1                        (uint8 数组，固定 32)
  · latents                    : 16*128*128 * 4                (float32)
  · timestep                   : 1 * 4

用法：
    python scripts/contract_dryrun.py <contract.json> [--text-max-length 32]
"""
import argparse
import io
import json
import os
import re
import sys

# 执行顺序照 PipelineZImage::generate()
TE = ["text_encoder_part1", "text_encoder_part2", "text_encoder_part3", "text_encoder_part4"]
DIT = ["transformer_part1a", "transformer_part1b", "transformer_part2a", "transformer_part2b"]
VAE = ["vae_decoder"]


def parse_config_len():
    """从 app 的 Config.hpp 解析 zimage_text_max_length —— **唯一真相来源**。

    不允许在本文件里再写一份副本（#73：根治 = 让清单不存在）。
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local-dream",
                     "app", "src", "main", "cpp", "src", "Config.hpp")
    p = os.path.normpath(p)
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*inline\s+constexpr\s+int\s+zimage_text_max_length\s*=\s*(\d+)",
                         line)
            if m:
                return int(m.group(1))
    raise RuntimeError("在 %s 里找不到 zimage_text_max_length" % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contract")
    ap.add_argument("--text-max-length", type=int, default=None,
                    help="覆盖 zimage_text_max_length；默认**从 Config.hpp 解析**")
    a = ap.parse_args()

    d = json.load(open(a.contract, encoding="utf-8"))
    g = {m["internal_graph_name"]: m for m in d["models"]}

    # 🔴 2026-08-29：原来这里是 `default=32` 的**写死副本**。
    #     Tier 2 把 Config.hpp 改成 80 后，本脚本仍照 32 建模，
    #     于是给出 **6 条假 FAIL** —— 与 #73/#138 同一个模式：
    #     **同一个常量在两处各写一遍，追不上就会说谎**。
    #     → 改成从 Config.hpp 解析，命令行只作覆盖。
    L = a.text_max_length if a.text_max_length else parse_config_len()
    # 🔴 只有 putBytes 喂的张量才可能与契约错位（字节数由 app 源码独立决定）。
    #    latents / timestep / vae_latents 走 putQuantized，其字节数**由契约自身推导**
    #    （`count * sizeof(uint16_t) != spec.bytes` 当场断言），因此不可能不一致 ——
    #    把它们建模成「等于契约声明」，否则会产生假警（2026-08-27 第一版就报了 3 条假警）。
    LAT = 16 * 128 * 128
    host = {                       # putBytes：app 侧独立决定，**真正的风险点**
        "input_ids": L * 4,        # sizeof(int32_t)
        "attention_mask": L * 4,
        # cap_pad_mask 也从 L 派生：PipelineZImage.hpp 已于 2026-08-29 改为
        # `std::array<uint8_t, zimage_text_max_length>`（原先写死 32）。
        "cap_pad_mask": L * 1,
    }
    derived = {                    # putQuantized：uint16 × 元素数，随契约自适应
        "latents": LAT * 2,
        "timestep": 1 * 2,
        "vae_latents": LAT * 2,
    }
    values = dict(host); values.update(derived)
    print("宿主 putBytes 喂入（**解析自 Config.hpp** zimage_text_max_length=%d）：" % L)
    for k, v in host.items():
        print("   %-16s %d B" % (k, v))
    print("宿主 putQuantized 喂入（随契约自适应）：")
    for k, v in derived.items():
        print("   %-16s %d B" % (k, v))

    order = TE + DIT + VAE
    fails = []
    print("\n逐图校验：")
    for name in order:
        if name not in g:
            fails.append("契约里没有图 %s" % name)
            continue
        m = g[name]
        for spec in m["inputs"]:
            src = spec.get("source") or spec["name"]
            if src not in values:
                fails.append('%s: 输入 "%s" 无生产者（app 会抛 '
                             '"Z-Image manifest input has no producer"）' % (name, src))
                continue
            got, want = values[src], spec["exact_bytes"]
            flag = "ok " if got == want else "🔴 "
            if got != want:
                fails.append('%s: 输入 "%s" 字节数 %d != 契约 %d ⇒ app 会抛 '
                             '"Z-Image manifest input byte mismatch: %s"'
                             % (name, spec["name"], got, want, spec["name"]))
            print("   %s%-20s %-16s 送 %-8d 契约 %-8d" % (flag, name, spec["name"], got, want))
        for out in m["outputs"]:
            values[out["name"]] = out["exact_bytes"]

    print()
    if fails:
        print("🔴 发现 %d 个问题：" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 1
    print("✅ 全链路字节数自洽：%d 个图、%d 个张量衔接全部对得上"
          % (len(order), len(values)))
    print("   （这不保证数值正确，只保证 app 的契约校验不会拦下来）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
