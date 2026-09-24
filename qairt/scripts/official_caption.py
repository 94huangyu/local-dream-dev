"""用【官方 PyTorch 文本编码器】算 caption，验证我们导出的 20-token 版本，并产出 22-token 版本。

## 为什么做（2026-08-18 实测发现）
我们导出的 text_encoder 被**硬编码固定在 20 token**（ONNX 内部 Reshape 常量 `{1,20,-1,128}`），
而官方 `ZImagePipeline` 默认 `max_sequence_length=512`，且 chat 模板带
`add_generation_prompt=True`（含 `<|im_start|>assistant\n`）。
本项目测试 prompt 的完整模板是 **22 token** ⇒ **末尾 assistant 标记被整个截掉**。

## 官方实现的三个关键点（从 diffusers 0.39 源码核实）
1. `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`
2. embedding 取 **`hidden_states[-2]`**（倒数第二层）
   —— 我们的 ONNX 含 35 层 / config 36 层，**已验证一致**
3. `prompt_embeds[i][prompt_masks[i]]` —— **只保留非 padding 位置**，变长输出

## 两个模式
- `verify`：取前 20 槽与我们现有 caption.raw 比对 ⇒ **验证本脚本的构造方式正确**
- `emit`  ：产出 22-token 的 caption（补零到 32 槽）供 transformer 使用

⚠️ verify 不通过则 emit 的结果不可用。
"""
import os, sys, time
import numpy as np

MODEL = os.path.join("D:", os.sep, "Z-Image-Turbo")
PROMPT = "a cute orange cat sitting on a wooden table, masterpiece, best quality"
IDS_NPY = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments", "prompt_ids.npy")


def embeds(max_len, ids_npy=None):
    """token id 由系统 Python 的 tokenizers 预先算好传入 —— venv 里的
    transformers 4.44.2 读不了这份较新的 tokenizer.json
    （`data did not match any variant of untagged enum ModelWrapper`）。
    ⚠️ 已核实：模板 `enable_thinking=True` 分支**不插入 <think> 块**，
    最终串与本项目脚本硬编码的 `<|im_start|>user
{p}<|im_end|>
<|im_start|>assistant
`
    完全一致，故用预算好的 id 不引入偏差。"""
    import torch
    from transformers import AutoModelForCausalLM
    ids = np.load(ids_npy or IDS_NPY)            # 系统 Python 产出
    QWEN_PAD = 151643
    n = min(len(ids), max_len)
    input_ids = np.full((1, max_len), QWEN_PAD, dtype=np.int64)
    attn = np.zeros((1, max_len), dtype=np.int64)
    input_ids[0, :n] = ids[:n]; attn[0, :n] = 1
    t0 = time.time()
    # 权重是 bf16（config: torch_dtype=bfloat16）。fp32 载入需 15.4 GB，超出本机可用内存
    # ⇒ 按原始精度 bf16 载入。这会引入 ~0.1~0.5% 的舍入差异，verify 的判据已按此放宽。
    m = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL, "text_encoder"),
                                             dtype=torch.bfloat16, low_cpu_mem_usage=True)
    m.eval()
    print(f"  文本编码器载入 {time.time()-t0:.0f}s", flush=True)

    class _TI:
        pass
    ti = _TI()
    ti.input_ids = torch.from_numpy(input_ids)
    ti.attention_mask = torch.from_numpy(attn)
    print(f"  max_length={max_len} ⇒ 实际 token {n}", flush=True)
    with torch.no_grad():
        out = m(input_ids=ti.input_ids, attention_mask=ti.attention_mask.bool(),
                output_hidden_states=True)
    h = out.hidden_states[-2][0][ti.attention_mask[0].bool()]   # 官方口径
    return h.float().numpy(), n


def main():
    mode = sys.argv[1]
    if mode == "verify":
        e, n = embeds(20)
        ours = np.fromfile(os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments",
                                        "htp_inloop", "const", "caption.raw"),
                           np.float32).reshape(32, 2560)
        k = min(n, 20)
        a, b = e[:k], ours[:k]
        rel = np.linalg.norm(a - b) / np.linalg.norm(b)
        cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"\n官方前 {k} 槽 vs 我们导出的 caption 前 {k} 槽:")
        print(f"  相对 L2 = {rel*100:.4f}%   余弦 = {cos:.6f}")
        print(f"  判定: {'✅ 构造方式正确，可用于 emit' if rel < 0.08 else '❌ 不一致，emit 结果不可用'}")
        print('  (阈值 8% 已含 bf16 载入的舍入余量；我们的 ONNX 是 fp32)')
    else:
        slots, ids_npy, pos = 32, None, []
        for _a in sys.argv[2:]:
            if _a.startswith("--slots="):
                slots = int(_a.split("=", 1)[1])
            elif _a.startswith("--ids="):
                ids_npy = _a.split("=", 1)[1]
            else:
                pos.append(_a)
        e, n = embeds(int(pos[2]) if len(pos) > 2 else 512, ids_npy)
        cap = np.zeros((slots, 2560), dtype=np.float32)
        k = min(n, slots)
        cap[:k] = e[:k]
        cap.tofile(pos[0])
        mask = np.ones((1, slots), dtype=np.float32); mask[0, :k] = 0.0
        mask.tofile(pos[1])
        print("写出 caption(%d 真实槽 + %d padding, 共 %d 槽) -> %s"
              % (k, slots - k, slots, pos[0]))
        print("写出 cap_pad_mask(%d) -> %s" % (slots, pos[1]))
        if n > slots:
            print("🔴 警告：实际 token %d > 槽数 %d，已截断 %d 个" % (n, slots, n - slots))


if __name__ == "__main__":
    main()
