#ifndef PIPELINEZIMAGE_HPP
#define PIPELINEZIMAGE_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Pipeline.hpp"
#include "QnnRuntime.hpp"
#include "ZImageFlowMatchScheduler.hpp"
#include "ZImageQnnContract.hpp"

// The Z-Image Turbo graphs do not share Anima's IO contract.  Every tensor is
// bound by its name and source in qnn_manifest.json, produced alongside the
// QNN binaries.  A mismatched conversion therefore fails before it can copy
// bytes into a wrong QNN tensor.
class PipelineZImage final : public Pipeline {
 public:
  PipelineZImage(TextEncoder &text_encoder, const std::string &model_dir)
      : Pipeline(text_encoder, model_dir, /*sdxl=*/false,
                 /*use_v_pred=*/false) {}

  // Deliberately does not load any of the 8 QNN contexts. Their combined file
  // size is ~11GB and QNN's resident runtime footprint per context runs well
  // above that on-disk size (graph structures, spill-fill scratch, etc.);
  // holding all 8 at once is enough to exhaust RAM on a phone and get the
  // whole app killed by the system's low-memory killer (confirmed on-device:
  // free memory fell from ~9GB to <100MB within seconds of loading them all).
  // generate() below loads only the stage it is currently running and
  // releases it before moving to the next, exactly like PipelineAnima's
  // lowram mode -- except Z-Image has no non-lowram mode, since three ~1-3GB
  // context groups can never coexist in memory on this class of device.
  bool initialize() override {
    const auto model_dir = std::filesystem::path(model_dir_);
    const auto final_contract = model_dir / "final_qnn_contract.json";
    contract_ = ZImageQnnContract::load(
        std::filesystem::is_regular_file(final_contract) ? final_contract
                                                          : model_dir / "qnn_manifest.json");
    if (!contract_.isFinalDelivery())
      throw std::runtime_error("Z-Image requires the final eight-graph QNN contract");
    // 🔴 #178：这一步是**冷启动的大头**，而此前从未计时 —— 我只能②按「12.04 GiB ÷ 实测
    //    吞吐」推断它占 148~210 s 启动开销的大部分。加上这行以后就是①实测。
    const auto t_v0 = std::chrono::steady_clock::now();
    contract_.validateGraphFiles(model_dir_);
    QNN_INFO("[startup] 契约基准校验 %d ms（sha256: %s）", elapsedMs(t_v0),
             sha256hw::enabled() ? "ARMv8 硬件加速" : "软件回落");
    QNN_INFO("Z-Image Turbo QNN contract validated; graphs load per-stage in generate().");
    return true;
  }

  bool supportsImg2Img() const override { return false; }
  bool isZImage() const override { return true; }

  GenerationResult generate(GenerationRequest &req,
                            const ProgressCallback &progress_callback) override {
    if (req.img2img || req.has_mask)
      throw std::invalid_argument("Z-Image Turbo supports text-to-image only");
    // 🔴 2026-09-03（#86）：尺寸不再写死 1024，但仍**只放行已交付的比例**
    // （zimage_sizes）。步数与 CFG 仍然锁死 —— 那两项是蒸馏模型的固有 profile。
    if (!zimageSizeSupported(req.width, req.height) ||
        req.steps != zimage_turbo_steps || req.cfg != 0.0f)
      throw std::invalid_argument("Z-Image request is outside the delivered size/8/CFG0 profile");
    // 本次生成使用的尺寸键，用来把契约切到该比例的那套图规格（图名 + 形状 + 字节数）。
    // 🔴 zimage_sizes 只是「代码允许的尺寸」，**交付里有没有这套图由契约说了算**。
    //    二者不一致时必须硬失败：否则会静默按 1:1 的形状去喂 4:3 的图（#73 类失效）。
    const std::string size_key = zimageSizeKey(req.width, req.height);
    const bool is_base = req.width == zimage_canvas_size && req.height == zimage_canvas_size;
    if (!is_base && !contract_.hasSize(size_key))
      throw std::invalid_argument("Z-Image delivery has no graphs for size " + size_key);
    contract_.setActiveSize(contract_.hasSize(size_key) ? size_key : std::string());
    // 🔴 惰性校验：启动时只算了基准的 sha256（约 10.4 GiB / 170 秒）。首次用到某个
    //    新尺寸时，在**装载它的任何图之前**把它引用的文件校验掉。已算过的不重复算。
    //    这样交付规模再大，后端启动时间也不随之增长（多比例总量可达 35.6 GiB，
    //    一次性全算要 9.8 分钟）。安全性不变：仍是「装载前必已校验」。
    if (contract_.hasSize(size_key)) {
      const auto t_v = std::chrono::steady_clock::now();
      contract_.validateSize(model_dir_, size_key);
      QNN_INFO("[zimage] size %s files validated in %d ms", size_key.c_str(),
               elapsedMs(t_v));
    }

    const auto started = std::chrono::steady_clock::now();
    // H5：每次生成从零累计。同一进程里连出两张时若不清零，第二张的 [segsplit]
    // 会把第一张的量加进来 —— 那是"看起来正常但没意义"的数字，最难发现。
    seg_timing_.clear();
    const auto text = text_encoder_.processZImagePrompt(req.prompt);
    ValueMap values;
    putBytes(values, "input_ids", text.input_ids.data(),
             text.input_ids.size() * sizeof(int32_t));
    putBytes(values, "attention_mask", text.attention_mask.data(),
             text.attention_mask.size() * sizeof(int32_t));
    {
      // Stage 1/3: the four text-encoder contexts (~4.2GB on disk) are only
      // needed to produce "caption"; load them, run once, then let them fall
      // out of scope so their HTP memory is freed before the transformer
      // stage loads.
      const auto text_encoder_part1 = loadGraph("text_encoder_part1");
      const auto text_encoder_part2 = loadGraph("text_encoder_part2");
      const auto text_encoder_part3 = loadGraph("text_encoder_part3");
      const auto text_encoder_part4 = loadGraph("text_encoder_part4");
      runGraph("text_encoder_part1", *text_encoder_part1, values);
      runGraph("text_encoder_part2", *text_encoder_part2, values);
      runGraph("text_encoder_part3", *text_encoder_part3, values);
      runGraph("text_encoder_part4", *text_encoder_part4, values);
      QNN_INFO("[lowram] Z-Image text encoder released");
    }

    // The Transformer mask is zimage_text_max_length wide and the text encoder
    // feeds all of those slots.  Tier 1 (2026-08-27) took it 20 -> 32; Tier 2
    // (2026-08-29, ledger #134/#139) re-cut all eight graphs to 80, so a prompt can
    // now occupy up to 72 content tokens (80 minus the 8-token chat template).
    // NOTE: the width is DERIVED from zimage_text_max_length on purpose -- this file
    // used to hardcode 32 in two places, exactly the duplicated-constant pattern that
    // cost this project an "Unknown Z-Image graph" incident (ledger #73).
    // Slots past token_count stay marked as padding here exactly as before.
    // Polarity confirmed 2026-08-13 by tracing transformer_part1a.onnx:
    // the "cap_pad_mask" input (torch.onnx.original_node_name-tagged, i.e. the
    // exact tensor from diffusers' _pad_with_ids/_prepare_sequence) feeds
    // Gather->Unsqueeze->Where(mask, cap_pad_token, real_caption_feats) --
    // Where(condition, X, Y) picks X when condition is true, so mask=1 means
    // "replace this slot with the padding embedding". An earlier version of
    // this code had the polarity inverted (1 = real token), which discarded
    // essentially all real prompt conditioning and kept only padding slots.
    std::array<uint8_t, zimage_text_max_length> cap_pad_mask{};
    std::fill(cap_pad_mask.begin(), cap_pad_mask.end(), 1);
    std::fill_n(cap_pad_mask.begin(),
                std::min<int>(text.token_count, zimage_text_max_length), 0);
    putBytes(values, "cap_pad_mask", cap_pad_mask.data(), cap_pad_mask.size());

    // 🔴 曾写死 128*128（=1024/8）。多比例下必须由请求尺寸派生，否则 4:3 会拿
    //    1:1 的元素数去初始化 latents，然后在契约的字节数检查处才炸（#86）。
    const size_t latent_count = static_cast<size_t>(zimage_latent_channels) *
                                (req.height / 8) * (req.width / 8);
    std::vector<float> latents(latent_count);
    std::mt19937 generator(req.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float &value : latents) value = normal(generator);
    logStats("initial latents (N(0,1))", latents);

    ZImageFlowMatchScheduler scheduler;
    scheduler.set_timesteps(zimage_turbo_steps);
    const auto &timesteps = scheduler.get_timesteps();
    int first_step_ms = 0;
    // DIAGNOSTIC (temporary): isolate whether the striped-artifact output is
    // coming from the VAE decode/pixel-unpack path or from the transformer
    // denoising loop. A marker file (not an env var -- simpler to drop into
    // the model dir from adb without touching how BackendService launches
    // the process) skips denoising entirely and VAE-decodes the raw N(0,1)
    // latents as-is. Striping surviving this => bug is in VAE input/output
    // tensor layout, independent of the transformer. Clean "TV static" noise
    // instead => VAE path is fine, bug is upstream in the transformer stage.
    if (!std::filesystem::is_regular_file(
            std::filesystem::path(model_dir_) / "DEBUG_SKIP_TRANSFORMER")) {
    // Stage 2/3: the three transformer contexts together are ~6.8GB on disk.
    // Loading all three at once (like the 4.2GB text-encoder stage) measured
    // ~7.3GB resident and got the process killed by the device's low-memory
    // manager. part1a+part1b together (~3.84GB) fit safely under that same
    // demonstrated-safe envelope, so they load/run/release as one group each
    // step; part2 (~2.93GB) never co-resides with them and loads separately.
    // Peak resident stays under ~3.84GB (vs. ~6.8GB for all three at once)
    // while halving the per-step context-load count in the previous
    // fully-separated ("sequential DiT" style, matching PipelineAnima's
    // seq_dit) version from 3 down to 2.
    // 🔴 2026-08-26 (#132 "方案 A") + 2026-08-30 (#145 "方案 B"): ALL FOUR
    // transformer contexts are hoisted OUT of the step loop and stay resident
    // for all 8 steps. Measured in the real app on 2026-08-26 (logcat, SM8750): every
    // step re-loaded all four transformer contexts -- "Initializing QNN App
    // from Buffer" appeared 8x for each of part1a/1b/2a/2b in a single
    // generation -- and context loading cost 13.2s of the 32.4s step.
    //
    // Safe because repeated execution of one QnnModel carries no one-shot
    // state: QnnModel::ensureIoTensors() returns early once inputs/outputs
    // exist, and executeNamedGraph() only memcpys into those persistent client
    // buffers, runs, and memcpys the outputs back out.
    //
    // Memory: 🔴 the paragraph that used to sit here estimated the peak at
    // ~5.77GB from context FILE sizes and warned "do not extend this to
    // keeping part2 resident (~6.78GB) without that measurement". Both the
    // basis and the warning are now superseded -- the measurement was done on
    // 2026-08-30 (ledger #145) and it reversed the conclusion. See the block
    // right below the part1a/1b loads for the measured numbers.
    //
    // 🔴 Do NOT measure this with process RSS (ps -o RSS / dumpsys meminfo):
    // the backend's VmRSS peaks at only ~2.3GB, and that peak is just the
    // transient CPU-side buffer holding one .bin during createFromBinary. The
    // contexts live in DSP/ION memory, which process RSS cannot see. Use
    // IonTotalUsed from /proc/meminfo (ledger #144).
    {
    // 低内存模式（台账 #162 的 D'，2026-09-06）：marker 文件在则**不常驻 part1a**，
    // 改为每步装卸。沿用本文件既有的 marker 开关做法（见上面的
    // DEBUG_SKIP_TRANSFORMER），这样同一个 APK 就能跑两条臂 ⇒ 对照是严格单变量，
    // 不需要为基线单独编一版（约束 3.6 / 约束 11 铁律 1）。
    //
    // 🔴🔴 2026-09-06 订正：**只装卸一段省不到任何峰值内存。**
    //   峰值 = Σ(常驻) + max(每步装卸的段) —— 装卸段被装上的那一刻，常驻段全都还在。
    //   ⇒ 省量 = Σ(装卸) − max(装卸)，只装卸一段时恒为 0。
    //   模型已用两个实测点回归验证：1a/1b 常驻+part2 装卸 算 6943（#145 实测 7148~7202）；
    //   四段全常驻 算 8879（今天实测 9116，含约 200 后端本底）。
    //   ⇒ 要省内存必须**成对以上**地装卸。最划算的一对是 part1a+part1b：
    //     省 1986 MiB，而它俩恰是装载最快的（1.88+1.37 s vs part2 的 2.16+2.60）。
    //   marker 文件内容 = 逗号分隔的段名，例：`part1a,part1b`；文件不在 = 现网行为。
    const auto evict = evictSet();
    // 跨 context 共享 spill-fill（台账 #146）：四段各带一份 258~289 MiB 的暂存区，
    // 合计 1085 MiB；共享后只留组内最大的一份 ⇒ ③预测省约 797 MiB（公式口径），
    // **且时间代价为零**。机制在 PipelineAnima/PipelineSdxl 上已在用，Z-Image 一直没接。
    const uint64_t sf_bytes = spillFillGroupBytes();
    Qnn_ContextHandle_t sf_head = nullptr;
    auto hoist = [&](const char *name) -> std::unique_ptr<QnnModel> {
      auto m = loadGraph(name, sf_bytes, sf_head);
      // 组长 = 第一个常驻下来的段；它必须在其余成员存活期间一直活着。
      if (sf_bytes && !sf_head && m) sf_head = m->getContextHandle();
      return m;
    };
    std::unique_ptr<QnnModel> transformer_part1a;
    if (!evict.count("part1a")) transformer_part1a = hoist("transformer_part1a");
    std::unique_ptr<QnnModel> transformer_part1b;
    if (!evict.count("part1b")) transformer_part1b = hoist("transformer_part1b");
    // #132 方案 B（台账 F/#145）2026-08-30: part2 也提出步循环。
    // 原先每步把两个半段各装卸一次 = 6.5 s/步 x 8 步 = 52 s 纯加载开销。
    //
    // 内存前提**已实测**（scripts/quadctx_probe/，独立 ARM64 探针，四段依次
    // createFromBinary 并全部保持存活）：ION 357 -> 9040 MiB，逐段 +2928 /
    // +1989 / +1954 / +1812 MiB，四段全部加载成功，到顶时仍余 MemAvailable
    // 2010 MiB。此前那句「估计 6.78 GB、必须先测量」用的是 context **文件
    // 大小**，而真实占用是 ION，比文件大 31~47%；同时我把「峰值 + 峰值时
    // MemAvailable = 8057 MiB」当成天花板，那不是天花板——内核还有
    // KReclaimable 与 zram 可回收。三个量必须分清：①文件字节 ②PD 分配估算
    // ③ION 实际占用，只有 ③ 能和内存压力比较。
    const bool split_part2 = contract_.hasSplitPart2();
    std::unique_ptr<QnnModel> transformer_part2a;
    std::unique_ptr<QnnModel> transformer_part2b;
    std::unique_ptr<QnnModel> transformer_part2;
    if (split_part2) {
      if (!evict.count("part2a")) transformer_part2a = hoist("transformer_part2a");
      if (!evict.count("part2b")) transformer_part2b = hoist("transformer_part2b");
    } else {
      transformer_part2 = hoist("transformer_part2");
    }
    for (int step = 0; step < zimage_turbo_steps; ++step) {
      const auto step_started = std::chrono::steady_clock::now();
      putQuantized(values, "latents", latents,
                   inputSpec("transformer_part1a", "latents"));
      // Diffusers sends (1000 - scheduler_timestep) / 1000 to Z-Image's
      // Transformer; the scheduler itself still advances in sigma space.
      const float timestep = 1.0f - timesteps(step) / 1000.0f;
      putQuantized(values, "timestep", &timestep, 1,
                   inputSpec("transformer_part1a", "timestep"));
      // 每步现装现卸的段：跑完立刻出作用域 ⇒ 下一段执行前它已经释放。
      // 🔴 装卸段**不加入 spill-fill 组**：组长必须在成员存活期间一直活着，
      //    而装卸段会反复析构 ⇒ 让它自带一份暂存区（多占 258~289 MiB，
      //    但省量本来就按 max 算，这一份就是那个 max，不影响结论）。
      runSeg("transformer_part1a", transformer_part1a, values);
      runSeg("transformer_part1b", transformer_part1b, values);
      // part2 ships either as one context or as two halves. The split exists
      // because the per-row quantized single part2 context (3535 MB) exceeds
      // the unsigned PD ceiling (measured between 3506 and 3535 MB) and fails
      // to load, while the halves are ~1.9 GB each. runGraph resolves inputs
      // by tensor name from `values`, so the 6-tensor / 65.6 MB cut between
      // part2a and part2b needs no explicit plumbing here; the contract's
      // validatePart2Split() is what guarantees none of it is dropped.
      if (split_part2) {
        runSeg("transformer_part2a", transformer_part2a, values);
        runSeg("transformer_part2b", transformer_part2b, values);
      } else {
        runSeg("transformer_part2", transformer_part2, values);
      }

      const auto noise = requireFloat(values, "latents", latent_count);
      if (step == 0 || step == zimage_turbo_steps - 1)
        logStats(("transformer predicted velocity, step " + std::to_string(step)).c_str(), noise);
      const float sigma = timesteps(step) / 1000.0f;
      const float next_sigma = step + 1 < zimage_turbo_steps
                                   ? timesteps(step + 1) / 1000.0f
                                   : 0.0f;
      const float dt = next_sigma - sigma;
      // The official pipeline_z_image.py negates the transformer's raw
      // output (`noise_pred = -noise_pred`) before handing it to
      // FlowMatchEulerDiscreteScheduler.step(), which itself computes
      // `sample + dt * model_output`. Combined: sample - dt * v_raw. This
      // "-=" is that combined formula applied directly to our un-negated
      // "noise" (v_raw), confirmed 2026-08-13 against diffusers' real
      // pipeline_z_image.py:564 + scheduling_flow_match_euler_discrete.py:514
      // (an earlier "+=" here was a regression that briefly reverted this).
      for (size_t i = 0; i < latent_count; ++i) latents[i] -= dt * noise[i];
      if (step == 0 || step == zimage_turbo_steps - 1)
        logStats(("latents after step " + std::to_string(step)).c_str(), latents);
      if (step == 0) first_step_ms = elapsedMs(step_started);
      progress_callback(step + 1, zimage_turbo_steps, "");
    }
    }  // all four transformer contexts released here, after the last step
       // (#132 方案 A hoisted part1a/1b; #145 方案 B added part2a/2b)
    QNN_INFO("[lowram] Z-Image transformer stage complete");
    logSegTiming();   // H5：逐段拆时（prep/in/exec/out/post）
    } else {
      QNN_INFO("[diagnostic] LOCALDREAM_ZIMAGE_SKIP_TRANSFORMER set: VAE-decoding raw N(0,1) latents");
    }

    // Official Z-Image VAE decode input: latents / scaling_factor + shift.
    //
    // 🔴 2026-08-21 FIX (HANDOVER 15.22, ledger #80/#81): our exported
    // vae_decoder graph ALREADY starts with Div(vae_latents, 0.3611) -- the
    // de-scaling is baked into the graph. Doing it again here divided by the
    // scaling factor TWICE. Two separate damages followed:
    //   1. the decode itself was wrong (vs official PyTorch VAE: 23.68 dB);
    //   2. the input landed 2.77x outside the range the VAE was quantized
    //      against (calibration latents were raw: std ~1.0, +-4.5; we fed
    //      std 3.045, +-12.2) => hard-clipped by the quantizer.
    // Solving graph(x/s) == official(lat/s + shift) gives x = lat + shift*s.
    // Measured on device (SM8750, same vae.bin, only this line changed):
    //   before 13.284 mean|px| / 23.68 dB / HF 1.3904x
    //   after   0.200 mean|px| / 54.33 dB / HF 0.9999x   (improvement 98.5%)
    for (float &value : latents)
      value = value + zimage_vae_shift_factor * zimage_vae_scaling_factor;
    putQuantized(values, "vae_latents", latents,
                 inputSpec("vae_decoder", "vae_latents"));
    std::vector<float> pixels_storage;
    {
      // Stage 3/3: VAE decoder (~103MB), loaded last and released immediately
      // after use.
      const auto vae_decoder = loadGraph("vae_decoder");
      runGraph("vae_decoder", *vae_decoder, values);
      pixels_storage = requireFloat(values, "pixels",
                                    static_cast<size_t>(3) * req.width * req.height);
    }
    const auto &pixels = pixels_storage;

    GenerationResult result;
    result.width = req.width;
    result.height = req.height;
    result.channels = 3;
    result.first_step_time_ms = first_step_ms;
    result.generation_time_ms = elapsedMs(started);
    result.image_data.resize(static_cast<size_t>(result.width) * result.height * 3);
    for (int y = 0; y < result.height; ++y) {
      for (int x = 0; x < result.width; ++x) {
        const size_t out = (static_cast<size_t>(y) * result.width + x) * 3;
        for (int c = 0; c < 3; ++c) {
          const float value = pixels[(static_cast<size_t>(c) * result.height + y) *
                                     result.width + x];
          result.image_data[out + c] = static_cast<uint8_t>(std::lround(
              std::clamp((value + 1.0f) * 127.5f, 0.0f, 255.0f)));
        }
      }
    }
    return result;
  }

 protected:
  void encodeText(const ProcessedPromptPair &, bool, bool,
                  Conditioning &) override {
    throw std::logic_error("Z-Image uses its dedicated Qwen text graph");
  }
  void vaeEncode(const GenerationRequest &, const float *, float *,
                 float *) override {
    throw std::logic_error("Z-Image does not support image encoding");
  }
  void vaeDecode(const GenerationRequest &, const float *, float *) override {
    throw std::logic_error("Z-Image uses its dedicated VAE decode path");
  }
  void runUnetStep(const GenerationRequest &, const float *, float, bool,
                   Conditioning &, float *) override {
    throw std::logic_error("Z-Image uses its dedicated Transformer path");
  }
  bool canSkipUncond() const override { return true; }
  bool previewSupported() const override { return false; }

 private:
  struct TensorValue {
    std::vector<uint8_t> bytes;
    const ZImageTensorSpec *spec = nullptr;  // null means a host-produced value.
  };
  using ValueMap = std::unordered_map<std::string, TensorValue>;

  // DIAGNOSTIC (temporary): coarse sanity stats to spot a saturated/degenerate
  // quantization scale (e.g. every value pinned to the same clipped extreme)
  // without needing a host-side reference to compare against.
  static void logStats(const char *label, const std::vector<float> &v) {
    float lo = v[0], hi = v[0], sum = 0.0, sumsq = 0.0;
    for (float x : v) {
      lo = std::min(lo, x);
      hi = std::max(hi, x);
      sum += x;
      sumsq += static_cast<double>(x) * x;
    }
    const double mean = sum / v.size();
    const double var = sumsq / v.size() - mean * mean;
    QNN_INFO("[diagnostic] %s: min=%.4f max=%.4f mean=%.4f std=%.4f n=%zu",
             label, lo, hi, mean, var > 0 ? std::sqrt(var) : 0.0, v.size());
  }

  static int elapsedMs(const std::chrono::steady_clock::time_point &from) {
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - from).count());
  }

  // ---- 低内存实验开关（台账 #162）。都用 marker 文件，理由见 §D' 那段注释：
  //      同一个 APK 能跑多条臂 ⇒ 对照严格单变量，不必为基线单独编一版。----

  // `LOWMEM_EVICT` 的内容 = 逗号分隔的段名（part1a/part1b/part2a/part2b）。
  // 文件不存在或为空 ⇒ 全部常驻 = 现网行为。
  std::set<std::string> evictSet() const {
    std::set<std::string> out;
    const auto p = std::filesystem::path(model_dir_) / "LOWMEM_EVICT";
    if (!std::filesystem::is_regular_file(p)) return out;
    std::ifstream f(p);
    std::string all((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    std::string cur;
    for (char c : all + ",") {
      if (c == ',' || c == '\n' || c == '\r' || c == ' ') {
        if (!cur.empty()) out.insert(cur);
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
    if (!out.empty())
      QNN_INFO("[lowmem] evict set has %zu segment(s)", out.size());
    return out;
  }

  // `SHARE_SPILLFILL` 存在 ⇒ 四段共用一份 spill-fill 暂存区（台账 #146）。
  //
  // 🔴 2026-09-17 设备实测翻车：这里原来**写死** 302,645,248 B（single 形态 1:1 四段的最大者），
  //   上面还写着「换比例后必须重新读，别把这个常数带走」——交付 mg 时还是带走了。
  //   mg 的 part1a 各图需要 311,492,608 B（1:1）~ 331,415,552 B（1184x896）⇒ 超出组上限
  //   ⇒ 交付后第一张图 `Failed init QNN context: transformer_part1a`，自动回滚。
  //   官方要求（htp_backend.html「shared spill-fill」）：组大小须为**所有成员 context 的最大者**，
  //   且只取**第一个注册者**的值 ⇒ 装第一段之前就必须知道，不能边装边求。
  // ⇒ 数值改由宿主从**实际交付的 .bin 元数据**算出（`oneshot_mg_session.py::sf_group_bytes`，
  //   取契约引用的全部 transformer 图、全部比例的最大者），写进 marker 文件内容。
  //   marker 在但读不出正整数 ⇒ **不共享**并报错（宁可 H1/H2 显式不过，也不拿错的值去注册）。
  uint64_t spillFillGroupBytes() const {
    const auto p = std::filesystem::path(model_dir_) / "SHARE_SPILLFILL";
    if (!std::filesystem::is_regular_file(p)) return 0;
    std::ifstream f(p);
    long long v = 0;
    if (!(f >> v) || v <= 0) {
      QNN_ERROR("[spill-fill] SHARE_SPILLFILL present but holds no positive byte count "
                "-> group sharing DISABLED");
      return 0;
    }
    QNN_INFO("[spill-fill] group sharing enabled: %llu bytes (from marker)",
             (unsigned long long)v);
    return static_cast<uint64_t>(v);
  }

  // 常驻的就直接跑；被列入 evict 的现装现卸（跑完出作用域即释放）。
  //
  // 🔴 逐段计时（2026-09-08 加，为主线 P2「生图速度」）：
  //   在此之前**没有任何 app 内的逐段耗时数据** —— #147 那些是 `qnn-net-run`
  //   离线单段的数，它自己标了「不等于 app 内每步耗时」。表一只能给到
  //   「步循环占 82.9%、每步 16.22 s」，**再往下就没有分辨力了**。
  //   这里只加两行日志、不碰任何数值；惰性由交付门自证：
  //   G2/G3 要求出图 sha256 与现网**逐字节相同**，改动若影响数值必然被抓。
  //   ⊕ 一次生图 32 行日志，开销可忽略；`scripts/timing_breakdown.py --seg`
  //     直接从同一份 logcat 解析，不需要额外的设备时间。
  void runSeg(const std::string &name, const std::unique_ptr<QnnModel> &hoisted,
              ValueMap &values) {
    const auto t0 = std::chrono::steady_clock::now();
    bool transient_load = false;
    if (hoisted) {
      runGraph(name, *hoisted, values);
    } else {
      transient_load = true;
      const auto transient = loadGraph(name);
      runGraph(name, *transient, values);
    }
    QNN_INFO("[segtime] %s %d ms%s", name.c_str(), elapsedMs(t0),
             transient_load ? " (含本步装载)" : "");
  }

  std::unique_ptr<QnnModel> loadGraph(const std::string &name, uint64_t sf_bytes = 0,
                                      Qnn_ContextHandle_t sf_head = nullptr) {
    const auto path = std::filesystem::path(model_dir_) / contract_.graph(name).file;
    if (!std::filesystem::is_regular_file(path))
      throw std::runtime_error("Z-Image graph file missing: " + path.string());
    // Every graph in the final/models contract schema is sourced from its
    // "context_binary" field, i.e. it is always a precompiled QNN context
    // binary, never a model library .so. Route there unconditionally instead
    // of guessing from the filename (a former ".ctx." substring check silently
    // never matched real delivery names like "..._ctx.SM8550.bin" and sent
    // every graph through the model-library loader instead).
    // 多比例：contract_ 已按本次请求的尺寸切到对应的一套图规格，这里取到的
    // spec.qnn_graph 就是该比例要 enable 的 QNN 图名（旧契约该字段为空 = 单图行为）。
    const auto &spec = contract_.graph(name);
    return qnn_runtime::createAndInitContext(path.string(), name, spec.qnn_graph,
                                             sf_bytes, sf_head);
  }

  static void putBytes(ValueMap &values, const std::string &source,
                       const void *data, size_t bytes) {
    if (!data || bytes == 0) throw std::invalid_argument("Empty Z-Image tensor: " + source);
    auto &target = values[source];
    target.bytes.resize(bytes);
    target.spec = nullptr;
    std::memcpy(target.bytes.data(), data, bytes);
  }

  const ZImageTensorSpec &inputSpec(const std::string &graph,
                                    const std::string &name) const {
    for (const auto &input : contract_.graph(graph).inputs) {
      if (input.name == name) return input;
    }
    throw std::runtime_error("Z-Image graph missing required input: " + graph + "/" + name);
  }

  static void putQuantized(ValueMap &values, const std::string &source,
                           const float *data, size_t count,
                           const ZImageTensorSpec &spec) {
    if (!spec.quantized || spec.dtype != "uint16" || count * sizeof(uint16_t) != spec.bytes)
      throw std::runtime_error("Z-Image host tensor has unsupported quantized ABI: " + source);
    std::vector<uint16_t> encoded(count);
    for (size_t i = 0; i < count; ++i) encoded[i] = quantize(data[i], spec);
    putBytes(values, source, encoded.data(), encoded.size() * sizeof(uint16_t));
  }

  static void putQuantized(ValueMap &values, const std::string &source,
                           const std::vector<float> &data,
                           const ZImageTensorSpec &spec) {
    putQuantized(values, source, data.data(), data.size(), spec);
  }

  static uint16_t quantize(float value, const ZImageTensorSpec &spec) {
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite Z-Image host value");
    // The manifest supplies scale/offset but no converter rounding rule.
    // Nearest-and-clamp is the standard affine-quantized fallback, pending
    // a device trace that can prove the converter's exact implementation.
    const long raw = std::lround(value / spec.scale - static_cast<float>(spec.offset));
    return static_cast<uint16_t>(std::clamp(raw, 0L, 65535L));
  }

  static float dequantize(uint16_t value, const ZImageTensorSpec &spec) {
    return (static_cast<float>(value) + static_cast<float>(spec.offset)) * spec.scale;
  }

  static bool sameQuantization(const ZImageTensorSpec &from,
                               const ZImageTensorSpec &to) {
    return from.dtype == to.dtype && from.bytes == to.bytes &&
           from.quantized == to.quantized && (!from.quantized ||
             (from.scale == to.scale && from.offset == to.offset));
  }

  static std::vector<uint8_t> requantize(const TensorValue &value,
                                         const ZImageTensorSpec &target) {
    const auto &source = *value.spec;
    if (!source.quantized || !target.quantized || source.dtype != "uint16" ||
        target.dtype != "uint16" || source.bytes != target.bytes)
      throw std::runtime_error("Z-Image contract requires an unsupported tensor conversion");
    std::vector<uint8_t> converted(target.bytes);
    const auto *input = reinterpret_cast<const uint16_t *>(value.bytes.data());
    auto *output = reinterpret_cast<uint16_t *>(converted.data());
    for (size_t i = 0; i < target.bytes / sizeof(uint16_t); ++i)
      output[i] = quantize(dequantize(input[i], source), target);
    return converted;
  }

  static const TensorValue &valueFor(const ValueMap &values,
                                     const ZImageTensorSpec &spec) {
    const std::string &source = spec.source.empty() ? spec.name : spec.source;
    const auto it = values.find(source);
    if (it == values.end())
      throw std::runtime_error("Z-Image manifest input has no producer: " + source);
    if (it->second.bytes.size() != spec.bytes)
      throw std::runtime_error("Z-Image manifest input byte mismatch: " + spec.name);
    return it->second;
  }

  static const std::vector<float> requireFloat(const ValueMap &values,
                                                const std::string &source,
                                                size_t elements) {
    const auto it = values.find(source);
    if (it == values.end())
      throw std::runtime_error("Z-Image manifest did not produce: " + source);
    if (!it->second.spec || !it->second.spec->quantized ||
        it->second.spec->dtype != "uint16" ||
        it->second.bytes.size() != elements * sizeof(uint16_t))
      throw std::runtime_error("Z-Image output byte mismatch: " + source);
    std::vector<float> output(elements);
    const auto *input = reinterpret_cast<const uint16_t *>(it->second.bytes.data());
    for (size_t i = 0; i < elements; ++i) output[i] = dequantize(input[i], *it->second.spec);
    return output;
  }

  // H5（P2 生图速度）：按段累计「宿主准备 / 拷入 / graphExecute / 拷出 / 落库」。
  //   动机：D1 只能由「加速器周期 vs 墙钟」②推出 part1a 有 41.2% 非算力开销，
  //   但推不出那 41.2% 落在哪一段拷贝上。这里是①实测，且是**在真实 app 环境里**
  //   （约束 11 第 4 条：run-as / qnn-net-run 换了执行环境就不是对照）。
  //   开销：每段 5 次 steady_clock ≈ 亚微秒，相对 16 s/步可忽略；不改任何数值。
  struct SegTiming {
    double prep_ms = 0, in_ms = 0, exec_ms = 0, out_ms = 0, post_ms = 0;
    int calls = 0;
  };
  std::map<std::string, SegTiming> seg_timing_;

  void logSegTiming() const {
    for (const auto &entry : seg_timing_) {
      const auto &t = entry.second;
      QNN_INFO("[segsplit] %s n=%d prep %.0f | in %.0f | exec %.0f | out %.0f | post %.0f ms"
               "（合计 %.0f，非算力占比 %.1f%%）",
               entry.first.c_str(), t.calls, t.prep_ms, t.in_ms, t.exec_ms, t.out_ms, t.post_ms,
               t.prep_ms + t.in_ms + t.exec_ms + t.out_ms + t.post_ms,
               100.0 * (t.prep_ms + t.in_ms + t.out_ms + t.post_ms) /
                   std::max(1e-9, t.prep_ms + t.in_ms + t.exec_ms + t.out_ms + t.post_ms));
    }
  }

  void runGraph(const std::string &name, QnnModel &model, ValueMap &values) {
    const auto t_prep0 = std::chrono::steady_clock::now();
    const auto &spec = contract_.graph(name);
    std::vector<QnnModel::NamedTensorView> inputs;
    std::vector<std::vector<uint8_t>> converted_inputs;
    inputs.reserve(spec.inputs.size());
    converted_inputs.reserve(spec.inputs.size());
    for (const auto &input : spec.inputs) {
      const auto &value = valueFor(values, input);
      const std::vector<uint8_t> *data = &value.bytes;
      if (value.spec && !sameQuantization(*value.spec, input)) {
        converted_inputs.push_back(requantize(value, input));
        data = &converted_inputs.back();
      }
      if (!value.spec && data->size() != input.bytes)
        throw std::runtime_error("Z-Image host input byte mismatch: " + input.name);
      inputs.push_back({input.name, data->data(), data->size()});
    }
    std::vector<QnnModel::NamedTensorValue> outputs;
    QnnModel::ExecSplit split;
    const auto t_exec0 = std::chrono::steady_clock::now();
    if (model.executeNamedGraph(name.c_str(), inputs, outputs, &split) !=
        qnn::tools::sample_app::StatusCode::SUCCESS)
      throw std::runtime_error("Z-Image QNN graph failed: " + name);
    const auto t_post0 = std::chrono::steady_clock::now();
    if (outputs.size() != spec.outputs.size())
      throw std::runtime_error("Z-Image graph output count mismatch: " + name);
    for (const auto &expected : spec.outputs) {
      auto actual = std::find_if(outputs.begin(), outputs.end(), [&](const auto &value) {
        return value.name == expected.name;
      });
      if (actual == outputs.end() || actual->bytes.size() != expected.bytes)
        throw std::runtime_error("Z-Image graph output mismatch: " + expected.name);
      values[expected.name] = {actual->bytes, &expected};
      if (!expected.source.empty() && expected.source != expected.name)
        values[expected.source] = {actual->bytes, &expected};
    }
    const auto ms = [](const std::chrono::steady_clock::time_point &a,
                       const std::chrono::steady_clock::time_point &b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto &t = seg_timing_[name];
    t.calls += 1;
    t.prep_ms += ms(t_prep0, t_exec0);
    t.in_ms += split.in_us / 1000.0;
    t.exec_ms += split.exec_us / 1000.0;
    t.out_ms += split.out_us / 1000.0;
    t.post_ms += ms(t_post0, std::chrono::steady_clock::now());
  }

  ZImageQnnContract contract_;
};

#endif  // PIPELINEZIMAGE_HPP
