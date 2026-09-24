// Strict runtime contract for the QNN binaries produced for Z-Image Turbo.
// A model directory is invalid unless it carries this manifest; filenames on
// their own are not proof that the graphs form one executable pipeline.
#ifndef ZIMAGEQNNCONTRACT_HPP
#define ZIMAGEQNNCONTRACT_HPP

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json.hpp"
#include "Sha256.hpp"
#include "Sha256Hw.hpp"   // ARMv8 sha2 加速 + 启动自检 + 自动回落（#178）

struct ZImageTensorSpec {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  size_t bytes = 0;
  std::string source;
  bool quantized = false;
  float scale = 0.0f;
  int32_t offset = 0;
};

struct ZImageGraphSpec {
  std::string file;
  std::string sha256;
  // Name of the graph to enable inside a multi-graph context binary (ledger #86).
  // Empty = legacy single-graph binary; behaviour then is exactly as before.
  // 🔴 This is NOT `internal_graph_name` (that one is the pipeline-side logical
  // name used to find the entry); this is the QNN graph name inside the .bin.
  std::string qnn_graph;
  std::vector<ZImageTensorSpec> inputs;
  std::vector<ZImageTensorSpec> outputs;
};

class ZImageQnnContract final {
 public:
  static constexpr int kVersion = 1;

  static ZImageQnnContract load(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Z-Image QNN contract not found: " + path.string());
    nlohmann::json root;
    stream >> root;

    if (root.contains("models")) return loadFinalDelivery(root);
    if (root.value("contract_version", 0) != kVersion)
      throw std::runtime_error("Unsupported Z-Image QNN contract version");
    const auto &profile = root.at("profile");
    if (profile.value("width", 0) != 1024 || profile.value("height", 0) != 1024 ||
        profile.value("steps", 0) != 8 || profile.value("cfg", -1.0) != 0.0)
      throw std::runtime_error("Z-Image QNN contract is not the required 1024/8/CFG0 profile");

    ZImageQnnContract contract;
    const auto &graphs = root.at("graphs");
    for (const char *name : kRequiredGraphs) {
      if (!graphs.contains(name))
        throw std::runtime_error(std::string("Z-Image QNN contract missing graph: ") + name);
      contract.graphs_.emplace(name, parseGraph(name, graphs.at(name)));
    }
    contract.validatePartSplit();
    contract.validateRuntimeSources();
    return contract;
  }

  // ---- 多比例（台账 #86）--------------------------------------------------
  // 1:1 仍然留在 `models` 原位不动（老交付/老 app 原样可读）；其他比例整份放进
  // `size_variants: {"1184x896": {"models":[...]}}`，**用同一段解析代码**读。
  // 🔴 为什么整份而不是「只打补丁改几个形状」：形状、exact_bytes、量化 scale/offset、
  //    QNN 图名都随比例变，逐字段打补丁等于在两处维护同一份清单（#73 的教训）。
  // 尺寸未登记时静默回落到 1:1 —— 由 RequestParser/zimageSizeSupported 挡非法尺寸，
  // 这里不做第二次校验，避免两套判定不一致。
  void setActiveSize(const std::string &key) { active_size_ = key; }

  const ZImageGraphSpec &graph(const std::string &name) const {
    if (!active_size_.empty()) {
      const auto v = variants_.find(active_size_);
      if (v != variants_.end()) {
        const auto g = v->second.find(name);
        if (g != v->second.end()) return g->second;
      }
    }
    const auto it = graphs_.find(name);
    if (it == graphs_.end()) throw std::out_of_range("Unknown Z-Image graph: " + name);
    return it->second;
  }

  bool hasSize(const std::string &key) const { return variants_.count(key) != 0; }

  bool isFinalDelivery() const { return final_delivery_; }

  // The manifest's SHA-256 values are an executable integrity contract, not
  // merely conversion provenance. Verify before QNN loads an untrusted file.
  // 🔴🔴 2026-09-06：改为**只校验当前尺寸用到的文件**，且每个文件一进程只算一次。
  //   起因：多比例交付把引用文件总量推到 12.3 GiB（多图）乃至 35.6 GiB（单图），
  //   而设备现算 sha256 约 62 MiB/s ⇒ 后端每次启动要 3.4~9.8 分钟，不可接受。
  //   现在：启动只校验基准（约 10.4 GiB / 170 秒，与交付前持平），
  //   首次切到某个新尺寸时再校验那个尺寸多出来的文件。
  //   安全性不变：**任何文件在被 QNN 装载之前都已校验过**——loadGraph 之前
  //   PipelineZImage 会先调 validateActiveSize()。
  void validateGraphFiles(const std::filesystem::path &directory) const {
    validateSize(directory, "");
  }

  // 校验某个尺寸（""=基准）引用到的文件。已校验过的文件不重复算。
  void validateSize(const std::filesystem::path &directory,
                    const std::string &sizeKey) const {
    const std::unordered_map<std::string, ZImageGraphSpec> *table = &graphs_;
    if (!sizeKey.empty()) {
      const auto v = variants_.find(sizeKey);
      if (v == variants_.end())
        throw std::runtime_error("Z-Image delivery has no graphs for size " + sizeKey);
      table = &v->second;
    }
    // 按 (相对路径 -> sha256) 去重：同一比例内多段可能共用一个多图 context 文件，
    // 不去重会把同一个几 GB 的文件哈希多遍。去重仍抓得住「同一文件两处声明了
    // 不同 sha256」——那会在这里判冲突。
    std::map<std::string, std::string> want;
    for (const auto &entry : *table) {
      const auto ins = want.emplace(entry.second.file, entry.second.sha256);
      if (!ins.second && !equalsIgnoreCase(ins.first->second, entry.second.sha256))
        throw std::runtime_error("Z-Image contract declares two SHA-256 for: " +
                                 entry.second.file);
    }
    // 🔴 台账 #182：校验结果缓存。
    //   动机：这 12.04 GiB / 19.4 s 的启动校验是**本项目自己加的**——原始 local-dream
    //   根本不校验模型文件（查 git HEAD 确认）。它真正的价值在开发期（推错文件/推一半
    //   当场抓住）；对用户而言模型在 app 私有目录、非 root 无写入途径，每次开机全量清点
    //   是在防一件不会发生的事。用户 2026-09-20 决策：加缓存，理由是**模型可重新导入，
    //   损坏的代价低**。
    //   失效机制：字节数或 mtime 变、契约期望值变、缓存缺失或损坏 —— 任一情况都回落全量校验。
    auto cache = loadVerifiedCache(directory);
    bool cache_dirty = false;
    for (const auto &entry : want) {
      const std::string &file = entry.first;
      const std::string &sha = entry.second;
      if (validated_.count(file)) continue;         // 本进程已校验过，不重复算
      const std::filesystem::path relative(file);
      if (relative.empty() || relative.is_absolute() ||
          std::find(relative.begin(), relative.end(), std::filesystem::path("..")) !=
              relative.end())
        throw std::runtime_error("Z-Image graph has unsafe relative path: " + file);
      const auto path = directory / file;
      if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("Z-Image graph file missing: " + path.string());
      const auto stamp = fileStamp(path);
      const auto hit = cache.find(file);
      // 三个条件同时成立才敢跳过：字节数一致、mtime 一致、**缓存里记的哈希正是契约要的那个**。
      // 第三条让「契约改了」也能自动失效 —— 不需要额外记录契约版本。
      if (hit != cache.end() && hit->second.size == stamp.size &&
          hit->second.mtime == stamp.mtime && equalsIgnoreCase(hit->second.sha, sha)) {
        validated_.insert(file);
        continue;
      }
      if (!equalsIgnoreCase(hashFile(path), sha))
        throw std::runtime_error("Z-Image graph SHA-256 mismatch: " + path.string());
      cache[file] = CacheEntry{stamp.size, stamp.mtime, sha};
      cache_dirty = true;
      validated_.insert(file);
    }
    if (cache_dirty) saveVerifiedCache(directory, cache);
  }

 private:
  // graphs_ 与所有 variants_ 的统一遍历入口，避免每处校验各写一遍循环而漏掉 variant
  // ——「同一份清单出现在多处」正是 #73 的失效模式。
  std::vector<const std::unordered_map<std::string, ZImageGraphSpec> *> allGraphTables() const {
    std::vector<const std::unordered_map<std::string, ZImageGraphSpec> *> out{&graphs_};
    for (const auto &v : variants_) out.push_back(&v.second);
    return out;
  }

  static constexpr const char *kRequiredGraphs[4] = {
      "text_encoder", "transformer_part1", "transformer_part2", "vae_decoder"};

  static ZImageGraphSpec parseGraph(const char *graph_name,
                                    const nlohmann::json &json) {
    ZImageGraphSpec graph;
    graph.file = json.at("file").get<std::string>();
    graph.sha256 = json.at("sha256").get<std::string>();
    if (graph.file.empty() || graph.file.find('/') != std::string::npos ||
        graph.file.find('\\') != std::string::npos)
      throw std::runtime_error(std::string(graph_name) + ": invalid binary filename");
    if (!isSha256(graph.sha256))
      throw std::runtime_error(std::string(graph_name) + ": invalid SHA-256");
    graph.inputs = parseTensors(graph_name, "inputs", json.at("inputs"));
    graph.outputs = parseTensors(graph_name, "outputs", json.at("outputs"));
    return graph;
  }

  // Two delivery layouts are accepted. The 4-segment one exists because the
  // per-row quantized transformer_part2 context (3535 MB) does not fit the
  // unsigned PD ceiling measured at 3506-3535 MB, so part2 was split at a
  // 6-tensor / 65.6 MB boundary into two ~1.9 GB halves. Layout is detected
  // from the manifest rather than configured, so a bundle cannot claim one
  // shape and ship the other.
  static ZImageQnnContract loadFinalDelivery(const nlohmann::json &root) {
    const auto &models = root.at("models");
    const bool split_part2 =
        std::any_of(models.begin(), models.end(), [](const auto &entry) {
          return entry.value("internal_graph_name", "") == "transformer_part2a";
        });

    std::vector<std::string> final_graphs = {
        "text_encoder_part1", "text_encoder_part2", "text_encoder_part3",
        "text_encoder_part4", "transformer_part1a", "transformer_part1b"};
    if (split_part2) {
      final_graphs.emplace_back("transformer_part2a");
      final_graphs.emplace_back("transformer_part2b");
    } else {
      final_graphs.emplace_back("transformer_part2");
    }
    final_graphs.emplace_back("vae_decoder");

    ZImageQnnContract contract;
    contract.final_delivery_ = true;
    contract.split_part2_ = split_part2;
    contract.graphs_ = parseFinalModels(models, final_graphs);
    // 其他比例：同一个解析器、同一份必需图清单，只是换一个 models 数组。
    if (root.contains("size_variants")) {
      for (const auto &item : root.at("size_variants").items())
        contract.variants_.emplace(
            item.key(), parseFinalModels(item.value().at("models"), final_graphs));
    }
    // 结构校验（切口闭合、runtime source 对得上）**逐比例各跑一遍**：这两个校验
    // 都经由 graph()，而 graph() 是按 active_size_ 取的，只跑一次等于只验了 1:1。
    for (const std::string &key : contract.sizeKeys()) {
      contract.setActiveSize(key);
      contract.validateFinalRuntimeSources();
      if (split_part2) contract.validatePart2Split();
    }
    contract.setActiveSize("");
    return contract;
  }

  // "" = 基准（1:1），其余为 size_variants 的键
  std::vector<std::string> sizeKeys() const {
    std::vector<std::string> out{""};
    for (const auto &v : variants_) out.push_back(v.first);
    return out;
  }

  static std::unordered_map<std::string, ZImageGraphSpec> parseFinalModels(
      const nlohmann::json &models, const std::vector<std::string> &final_graphs) {
    std::unordered_map<std::string, ZImageGraphSpec> out;
    for (const std::string &name_str : final_graphs) {
      const char *name = name_str.c_str();
      const auto model = std::find_if(models.begin(), models.end(), [name](const auto &entry) {
        return entry.value("internal_graph_name", "") == name;
      });
      if (model == models.end())
        throw std::runtime_error(std::string("Final Z-Image contract missing graph: ") + name);
      ZImageGraphSpec graph;
      graph.file = model->at("context_binary").get<std::string>();
      graph.sha256 = model->at("sha256").get<std::string>();
      graph.qnn_graph = model->value("qnn_graph_name", "");
      graph.inputs = parseFinalTensors(name, "inputs", model->at("inputs"));
      graph.outputs = parseFinalTensors(name, "outputs", model->at("outputs"));
      out.emplace(name, std::move(graph));
    }
    return out;
  }

  // Every part2a output must be consumed by part2b. A dropped cut tensor would
  // otherwise surface as a silently wrong image rather than a load error.
  void validatePart2Split() const {
    const auto &a = graph("transformer_part2a");
    const auto &b = graph("transformer_part2b");
    for (const auto &out : a.outputs) {
      const bool consumed =
          std::any_of(b.inputs.begin(), b.inputs.end(), [&out](const auto &in) {
            return in.name == out.name || in.source == out.name;
          });
      if (!consumed)
        throw std::runtime_error("Z-Image part2 split drops part2a output: " + out.name);
    }
  }

  static std::vector<ZImageTensorSpec> parseFinalTensors(
      const char *graph_name, const char *kind, const nlohmann::json &json) {
    if (!json.is_array() || json.empty())
      throw std::runtime_error(std::string(graph_name) + ": empty " + kind);
    std::vector<ZImageTensorSpec> tensors;
    for (const auto &item : json) {
      ZImageTensorSpec tensor;
      tensor.name = item.at("name").get<std::string>();
      tensor.dtype = normalizeFinalDtype(item.at("physical_dtype").get<std::string>());
      tensor.shape = item.at("fixed_shape").get<std::vector<size_t>>();
      const auto &bytes = item.at("exact_bytes");
      tensor.bytes = bytes.is_number_integer() ? bytes.get<size_t>() : tensorBytes(tensor);
      tensor.source = tensor.name;
      const auto &quantization = item.at("quantization");
      if (!quantization.at("scale").is_null()) {
        tensor.quantized = true;
        tensor.scale = quantization.at("scale").get<float>();
        tensor.offset = quantization.at("offset").get<int32_t>();
        if (tensor.scale <= 0.0f)
          throw std::runtime_error(std::string(graph_name) + ": invalid quantization scale");
      }
      if (tensor.name.empty() || tensor.shape.empty() || tensor.bytes == 0)
        throw std::runtime_error(std::string(graph_name) + ": invalid final " + kind + " tensor");
      if (tensorBytes(tensor) != tensor.bytes)
        throw std::runtime_error(std::string(graph_name) + ": final tensor byte mismatch for " + tensor.name);
      tensors.push_back(std::move(tensor));
    }
    return tensors;
  }

  static std::string normalizeFinalDtype(const std::string &dtype) {
    if (dtype == "QNN_DATATYPE_INT_32") return "int32";
    if (dtype == "QNN_DATATYPE_UFIXED_POINT_16") return "uint16";
    if (dtype == "QNN_DATATYPE_BOOL_8") return "bool";
    throw std::runtime_error("Unsupported final Z-Image QNN dtype: " + dtype);
  }

  static size_t tensorBytes(const ZImageTensorSpec &tensor) {
    size_t elements = 1;
    for (size_t dim : tensor.shape) {
      if (dim == 0 || elements > SIZE_MAX / dim)
        throw std::runtime_error("Invalid final Z-Image tensor shape");
      elements *= dim;
    }
    const size_t item_size = dtypeBytes(tensor.dtype);
    if (elements > SIZE_MAX / item_size)
      throw std::runtime_error("Final Z-Image tensor size overflow");
    return elements * item_size;
  }

  static std::vector<ZImageTensorSpec> parseTensors(
      const char *graph_name, const char *kind, const nlohmann::json &json) {
    if (!json.is_array() || json.empty())
      throw std::runtime_error(std::string(graph_name) + ": empty " + kind);
    std::vector<ZImageTensorSpec> tensors;
    for (const auto &item : json) {
      ZImageTensorSpec tensor;
      tensor.name = item.at("name").get<std::string>();
      tensor.dtype = item.at("dtype").get<std::string>();
      tensor.shape = item.at("shape").get<std::vector<size_t>>();
      tensor.bytes = item.at("bytes").get<size_t>();
      tensor.source = item.value("source", "");
      if (tensor.name.empty() || tensor.shape.empty() || tensor.bytes == 0)
        throw std::runtime_error(std::string(graph_name) + ": invalid " + kind + " tensor");
      size_t elements = 1;
      for (size_t dim : tensor.shape) {
        if (dim == 0 || elements > SIZE_MAX / dim)
          throw std::runtime_error(std::string(graph_name) + ": invalid tensor shape");
        elements *= dim;
      }
      const size_t item_size = dtypeBytes(tensor.dtype);
      if (elements > SIZE_MAX / item_size || elements * item_size != tensor.bytes)
        throw std::runtime_error(std::string(graph_name) + ": tensor byte count mismatch for " + tensor.name);
      tensors.push_back(std::move(tensor));
    }
    return tensors;
  }

  void validatePartSplit() const {
    const auto &part1 = graph("transformer_part1");
    const auto &part2 = graph("transformer_part2");
    for (const auto &out : part1.outputs) {
      bool consumed = false;
      for (const auto &in : part2.inputs) {
        if (in.name == out.name || in.source == out.name) {
          consumed = true;
          break;
        }
      }
      if (!consumed)
        throw std::runtime_error("Z-Image transformer split drops part1 output: " + out.name);
    }
  }

  void validateRuntimeSources() const {
    // These sources are the complete host-side ABI implemented by
    // PipelineZImage. Export wrappers must internalize any RoPE/grid helper
    // tensors rather than leaving Android to infer them from graph names.
    requireInputSource("text_encoder", "input_ids");
    requireInputSource("text_encoder", "attention_mask");
    requireOutputSource("text_encoder", "caption");
    requireInputSource("transformer_part1", "latents");
    requireInputSource("transformer_part1", "timestep");
    requireInputSource("transformer_part1", "caption");
    requireInputSource("transformer_part1", "caption_length");
    requireOutputSource("transformer_part2", "noise");
    requireInputSource("vae_decoder", "vae_latents");
    requireOutputSource("vae_decoder", "pixels");
  }

  void validateFinalRuntimeSources() const {
    requireInputSource("text_encoder_part1", "input_ids");
    requireInputSource("text_encoder_part1", "attention_mask");
    requireOutputSource("text_encoder_part4", "caption");
    requireInputSource("transformer_part1a", "latents");
    requireInputSource("transformer_part1a", "timestep");
    requireInputSource("transformer_part1a", "caption");
    requireInputSource("transformer_part1a", "cap_pad_mask");
    requireOutputSource(finalTransformerGraph(), "latents");
    requireInputSource("vae_decoder", "vae_latents");
    requireOutputSource("vae_decoder", "pixels");
  }

 public:
  bool hasSplitPart2() const { return split_part2_; }

  // Callers must never keep their own copy of the graph list. main.cpp used to
  // re-list the eight final graph names to build its file-existence check, and
  // that copy silently went stale the moment part2 was split in two: the
  // contract loaded fine, then the hardcoded list asked for "transformer_part2"
  // and aborted with "Unknown Z-Image graph". Iterate this instead.
  std::vector<std::string> graphNames() const {
    std::vector<std::string> names;
    names.reserve(graphs_.size());
    for (const auto &entry : graphs_) names.push_back(entry.first);
    std::sort(names.begin(), names.end());
    return names;
  }

  // The graph that produces the noise prediction, whichever layout shipped.
  const char *finalTransformerGraph() const {
    return split_part2_ ? "transformer_part2b" : "transformer_part2";
  }

 private:

  void requireInputSource(const char *graph_name, const char *source) const {
    for (const auto &input : graph(graph_name).inputs) {
      if (input.source == source) return;
    }
    throw std::runtime_error(std::string("Z-Image contract missing ") +
                             graph_name + " input source: " + source);
  }

  void requireOutputSource(const char *graph_name, const char *source) const {
    for (const auto &output : graph(graph_name).outputs) {
      if (output.source == source) return;
    }
    throw std::runtime_error(std::string("Z-Image contract missing ") +
                             graph_name + " output source: " + source);
  }

  static size_t dtypeBytes(const std::string &dtype) {
    if (dtype == "float32" || dtype == "int32") return 4;
    if (dtype == "float16" || dtype == "int16" || dtype == "uint16") return 2;
    if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
    throw std::runtime_error("Unsupported Z-Image QNN tensor dtype: " + dtype);
  }

  static bool isSha256(const std::string &value) {
    if (value.size() != 64) return false;
    for (unsigned char c : value) {
      if (!std::isxdigit(c)) return false;
    }
    return true;
  }

  // ---- #182 校验结果缓存 ----
  struct CacheEntry {
    uintmax_t size = 0;
    long long mtime = 0;
    std::string sha;
  };
  struct Stamp {
    uintmax_t size = 0;
    long long mtime = 0;
  };

  static Stamp fileStamp(const std::filesystem::path &p) {
    Stamp s;
    std::error_code ec;
    s.size = std::filesystem::file_size(p, ec);
    if (ec) s.size = 0;
    const auto t = std::filesystem::last_write_time(p, ec);
    s.mtime = ec ? 0 : static_cast<long long>(t.time_since_epoch().count());
    return s;
  }

  static std::filesystem::path cachePath(const std::filesystem::path &dir) {
    return dir / ".verified_cache";
  }

  // 读缓存。**任何解析异常都当作"没有缓存"**（回落全量校验），绝不因缓存坏了而放行。
  static std::map<std::string, CacheEntry> loadVerifiedCache(
      const std::filesystem::path &dir) {
    std::map<std::string, CacheEntry> out;
    std::ifstream in(cachePath(dir));
    if (!in) return out;
    std::string line;
    while (std::getline(in, line)) {
      const auto a = line.find('\t');
      if (a == std::string::npos) continue;
      const auto b = line.find('\t', a + 1);
      if (b == std::string::npos) continue;
      const auto c = line.find('\t', b + 1);
      if (c == std::string::npos) continue;
      CacheEntry e;
      try {
        e.size = static_cast<uintmax_t>(std::stoull(line.substr(a + 1, b - a - 1)));
        e.mtime = std::stoll(line.substr(b + 1, c - b - 1));
      } catch (...) {
        continue;                                   // 这一行坏了就丢掉它，不影响其余
      }
      e.sha = line.substr(c + 1);
      if (!e.sha.empty()) out[line.substr(0, a)] = e;
    }
    return out;
  }

  static void saveVerifiedCache(const std::filesystem::path &dir,
                                const std::map<std::string, CacheEntry> &cache) {
    // 写失败不算错误：下次全量校验即可（缓存只是加速，不是正确性的一部分）。
    std::ofstream out(cachePath(dir), std::ios::trunc);
    if (!out) return;
    for (const auto &kv : cache)
      out << kv.first << '\t' << kv.second.size << '\t' << kv.second.mtime << '\t'
          << kv.second.sha << '\n';
  }

  // 🔴 台账 #178：冷启动要校验 12.04 GiB，旧实现（纯软件 + 逐字节 `update`）实测
  //    **95.7 MiB/s ⇒ 约 129 秒**，比生图本身还长。改用 ARMv8 `sha2` 指令后
  //    实测 **1486 MiB/s（15.5 倍）**，端到端读文件 1022 MiB/s ⇒ 约 12 秒。
  //    ⊕ 读缓冲 64 KiB → 4 MiB：`sha_bench` 实测这个块大小才吃得满 I/O。
  //    ⊕ 正确性三重闭合（2026-09-19 设备实测）：硬件版 == 本仓库软件版（5 组向量含分片边界）
  //      == 系统 `sha256sum`（64 MiB 随机数据逐位一致）；`sha256hw::enabled()` 内含启动自检，
  //      任何不一致都**自动回落软件版**（哈希算错会让契约校验全失败，不许赌）。
  static std::string hashFile(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read Z-Image graph: " + path.string());
    const bool hw = sha256hw::enabled();
    Sha256 sha;
    sha256hw::Hasher shahw;
    std::vector<uint8_t> buffer(4u << 20);
    while (file.read(reinterpret_cast<char *>(buffer.data()),
                     static_cast<std::streamsize>(buffer.size())) || file.gcount()) {
      const size_t n = static_cast<size_t>(file.gcount());
      if (hw) shahw.update(buffer.data(), n);
      else sha.update(buffer.data(), n);
    }
    if (!file.eof()) throw std::runtime_error("Cannot hash Z-Image graph: " + path.string());
    const auto digest = hw ? shahw.finalize() : sha.finalize();
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (uint8_t byte : digest) {
      result.push_back(hex[byte >> 4]);
      result.push_back(hex[byte & 0x0f]);
    }
    return result;
  }

  static bool equalsIgnoreCase(const std::string &left, const std::string &right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(left[i])) !=
          std::tolower(static_cast<unsigned char>(right[i]))) return false;
    }
    return true;
  }

  std::unordered_map<std::string, ZImageGraphSpec> graphs_;
  // 尺寸键 -> 该比例下的整套图规格（台账 #86）。见 setActiveSize 处的说明。
  std::map<std::string, std::unordered_map<std::string, ZImageGraphSpec>> variants_;
  std::string active_size_;
  // 本进程已校验过 sha256 的文件（相对路径）。mutable：校验是 const 语义上的
  // 「确认」，不改变契约内容，但需要记住做过了以免重复哈希几十 GiB。
  mutable std::set<std::string> validated_;
  bool final_delivery_ = false;
  bool split_part2_ = false;
};

#endif  // ZIMAGEQNNCONTRACT_HPP
