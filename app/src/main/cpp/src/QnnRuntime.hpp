#ifndef QNNRUNTIME_HPP
#define QNNRUNTIME_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "MemUtils.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnModel.hpp"
#include "QnnSampleAppUtils.hpp"
#include "zstd.h"

// Owns the process-wide QNN backend state (system function pointers and the
// HTP backend stub path) and provides creation/initialization of QnnModel
// instances on top of it. init() must succeed before any model is created.
namespace qnn_runtime {

inline QnnFunctionPointers g_systemFuncs;
inline std::string g_backendPath;
inline bool g_initialized = false;

// Resolves libQnnHtp.so / libQnnSystem.so inside `lib_dir`.
inline bool init(const std::string &lib_dir) {
  using namespace qnn::tools;
  std::filesystem::path lib(lib_dir);
  g_backendPath = (lib / "libQnnHtp.so").string();
  dynamicloadutil::StatusCode status =
      dynamicloadutil::getQnnSystemFunctionPointers(
          (lib / "libQnnSystem.so").string(), &g_systemFuncs);
  g_initialized = (status == dynamicloadutil::StatusCode::SUCCESS);
  return g_initialized;
}

inline std::unique_ptr<QnnModel> createModel(const std::string &modelPath,
                                             const std::string &modelName) {
  using namespace qnn::tools;
  ::QnnFunctionPointers funcs = g_systemFuncs;
  void *backendHandle = nullptr;
  void *modelHandle = nullptr;
  dynamicloadutil::StatusCode drvStatus =
      dynamicloadutil::getQnnFunctionPointers(g_backendPath, modelPath, &funcs,
                                              &backendHandle, false,
                                              &modelHandle);
  if (drvStatus != dynamicloadutil::StatusCode::SUCCESS) {
    QNN_ERROR("Failed get QNN func ptrs for %s.", modelName.c_str());
    if (modelHandle) dlclose(modelHandle);
    return nullptr;
  }
  std::string inputListPaths, opPackagePaths, outputPath, saveBinaryName;
  bool debug = false;
  bool dumpOutputs = false;
  iotensor::OutputDataType outputDataType =
      iotensor::OutputDataType::FLOAT_ONLY;
  iotensor::InputDataType inputDataType = iotensor::InputDataType::FLOAT;
  sample_app::ProfilingLevel profilingLevel = sample_app::ProfilingLevel::OFF;
  auto app = std::make_unique<QnnModel>(
      funcs, inputListPaths, opPackagePaths, backendHandle, outputPath, debug,
      outputDataType, inputDataType, profilingLevel, dumpOutputs, modelPath,
      saveBinaryName);
  // Hand off the model library handle so the QnnModel destructor can dlclose
  // it. Otherwise lowram mode leaks one .so handle per load cycle.
  if (app) app->m_modelHandle = modelHandle;
  return app;
}

// Runs the full QnnModel bring-up sequence. When `buffer` is non-null the
// context is created from that in-memory binary instead of the model file.
inline int initializeApp(const std::string &modelName,
                         std::unique_ptr<QnnModel> &app,
                         const uint8_t *buffer = nullptr,
                         uint64_t bufferSize = 0) {
  using qnn::tools::sample_app::StatusCode;
  if (!app) return EXIT_FAILURE;

  if (buffer && bufferSize > 0) {
    QNN_INFO("Initializing QNN App from Buffer: %s (size: %llu bytes)",
             modelName.c_str(), bufferSize);
  } else {
    QNN_INFO("Initializing QNN App from Cache: %s", modelName.c_str());
  }

  if (StatusCode::SUCCESS != app->initialize())
    return app->reportError(modelName + " Init failure");
  if (StatusCode::SUCCESS != app->initializeBackend())
    return app->reportError(modelName + " Backend Init failure");
  auto devPropStat = app->isDevicePropertySupported();
  if (StatusCode::FAILURE != devPropStat) {
    if (StatusCode::SUCCESS != app->createDevice())
      return app->reportError(modelName + " Device Creation failure");
  }
  if (StatusCode::SUCCESS != app->initializeProfiling())
    return app->reportError(modelName + " Profiling Init failure");
  if (StatusCode::SUCCESS != app->registerOpPackages())
    return app->reportError(modelName + " Register Op Packages failure");

  if (buffer && bufferSize > 0) {
    if (StatusCode::SUCCESS != app->createFromBuffer(buffer, bufferSize))
      return app->reportError(modelName + " Create From Buffer failure");
  } else {
    if (StatusCode::SUCCESS != app->createFromBinary())
      return app->reportError(modelName + " Create From Binary failure");
  }

  if (StatusCode::SUCCESS != app->enablePerformaceMode())
    return app->reportError(modelName + " Enable Performance Mode failure");

  if (buffer && bufferSize > 0) {
    QNN_INFO("QNN App Initialized from Buffer: %s", modelName.c_str());
  } else {
    QNN_INFO("QNN App Initialized from Cache: %s", modelName.c_str());
  }
  return EXIT_SUCCESS;
}

// Convenience wrapper used by lazy (lowram) loading: create + initialize in
// one call, throwing on failure.
inline std::unique_ptr<QnnModel> createAndInitModel(
    const std::string &modelPath, const std::string &modelName) {
  auto app = createModel(modelPath, modelName);
  if (!app) throw std::runtime_error("Failed create QNN model: " + modelName);
  if (initializeApp(modelName, app) != EXIT_SUCCESS)
    throw std::runtime_error("Failed init QNN model: " + modelName);
  return app;
}

// Context binaries already contain their compiled graph; unlike model-library
// loading, they must be supplied to QNN as an in-memory binary.
inline std::unique_ptr<QnnModel> createAndInitContext(
    const std::string &contextPath, const std::string &modelName,
    const std::string &enabledGraph = std::string(),
    // 跨 context 共享 HTP spill-fill 暂存区（台账 #146）。
    // sfBytes = 组内最大的 spillFillBufferSize；groupHead = 组长的 context handle
    // （组长自己传 nullptr）。两者都必须在 createFromBinary **之前**设好。
    // 机制在本仓库的 PipelineAnima / PipelineSdxl 上已在用，Z-Image 一直没接。
    uint64_t sfBytes = 0, Qnn_ContextHandle_t groupHead = nullptr) {
  std::ifstream file(contextPath, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Failed open QNN context: " + contextPath);
  const auto size = static_cast<uint64_t>(file.tellg());
  if (size == 0) throw std::runtime_error("Empty QNN context: " + contextPath);

  // 🔴 P2/H2（台账 #174）：默认路径是「整文件读进 vector」——先把 2~3 GB 匿名内存清零，
  //    再整份拷贝一遍，然后才交给 QNN。实测四段在标记前的空白合计 12.19 s（`Initializing`
  //    标记之前那段），远大于 createFromBuffer 自己的 7.97 s。
  //    mmap 路径省掉「清零 + 一次拷贝」，且不占匿名内存（#144：装载期 VmRSS 峰值就是这份缓冲）。
  //    ⚠️ 用 marker 文件做开关，**同一个 APK 跑两臂**，保证 A/B 是严格单变量；
  //       QNN 在 createFromBinary 返回后不再引用该缓冲（现有实现本来就在函数返回时释放 vector），
  //       所以映射只需活到 initializeApp 结束。
  // marker 两处都认：`files/models/ZIMAGE/LOAD_MMAP`（与 SHARE_SPILLFILL 同级，推荐）
  // 或 `files/models/ZIMAGE/models/LOAD_MMAP`（与 .bin 同级）。
  const auto ctx_dir = std::filesystem::path(contextPath).parent_path();
  const bool use_mmap = std::filesystem::is_regular_file(ctx_dir / "LOAD_MMAP") ||
                        std::filesystem::is_regular_file(ctx_dir.parent_path() / "LOAD_MMAP");
  std::vector<uint8_t> bytes;
  const uint8_t *data = nullptr;
  struct MapGuard {
    void *addr = nullptr;
    size_t len = 0;
    int fd = -1;
    ~MapGuard() {
      if (addr && addr != MAP_FAILED) munmap(addr, len);
      if (fd >= 0) close(fd);
    }
  } guard;

  // 🔴 这一行必须打在「读入 / 映射」**之前**。第一版打在之后，于是
  //    `[loadpath] → QNN App Initialized` 这个区间**恰好把要省的那段排除在外**，
  //    两臂会量出几乎一样的数 ⇒ H2 无论真假都测不出来（约束 9 自审第 1 条抓出）。
  QNN_INFO("[loadpath] %s %s (%llu bytes)", use_mmap ? "mmap" : "read-into-vector",
           modelName.c_str(), (unsigned long long)size);
  if (use_mmap) {
    guard.fd = open(contextPath.c_str(), O_RDONLY);
    if (guard.fd < 0) throw std::runtime_error("Failed open(2) QNN context: " + contextPath);
    guard.len = static_cast<size_t>(size);
    guard.addr = mmap(nullptr, guard.len, PROT_READ, MAP_PRIVATE, guard.fd, 0);
    if (guard.addr == MAP_FAILED)
      throw std::runtime_error("Failed mmap QNN context: " + contextPath);
    madvise(guard.addr, guard.len, MADV_SEQUENTIAL);
    madvise(guard.addr, guard.len, MADV_WILLNEED);
    data = static_cast<const uint8_t *>(guard.addr);
  } else {
    bytes.resize(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size)))
      throw std::runtime_error("Failed read QNN context: " + contextPath);
    data = bytes.data();
  }

  // Context creation needs the backend and system libraries, not a per-model
  // shared library. DynamicLoadUtil accepts an empty optional model path.
  auto app = createModel("", modelName);
  if (!app) throw std::runtime_error("Failed create QNN context model: " + modelName);
  // Multi-graph context (ledger #86): must be set BEFORE createFromBinary.
  // Two device-measured facts drive this:
  //   1. Without ENABLE_GRAPHS the whole binary is deserialized and
  //      createFromBinary fails with 0x3ea (PD capacity) -- required, not optional.
  //   2. graphRetrieve() succeeds even for graphs that were not enabled, so the
  //      graph must be picked BY NAME; QnnModel does that internally.
  if (!enabledGraph.empty()) app->setEnabledGraph(enabledGraph.c_str());
  if (sfBytes) app->setSpillFillGroup(sfBytes, groupHead);
  if (initializeApp(modelName, app, data, size) != EXIT_SUCCESS)
    throw std::runtime_error("Failed init QNN context: " + modelName);
  return app;
}

struct PatchedModelBuffer {
  std::shared_ptr<uint8_t> buffer;
  uint64_t size;

  PatchedModelBuffer() : buffer(nullptr), size(0) {}

  PatchedModelBuffer(uint8_t *buf, uint64_t sz)
      : buffer(buf, std::default_delete<uint8_t[]>()), size(sz) {}

  void reset() {
    buffer.reset();
    size = 0;
  }
};

inline std::vector<char> readFileForPatch(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filePath);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (size > 0) {
    if (!file.read(buffer.data(), size)) {
      throw std::runtime_error("Failed to read file: " + filePath);
    }
  }
  return buffer;
}

inline std::unique_ptr<PatchedModelBuffer> applyZstdPatchToBuffer(
    const std::string &oldFilePath, const std::string &patchFilePath) {
  try {
    // The old model is only read (as the zstd dictionary), so map it read-only
    // instead of pulling the whole multi-GB file into an anonymous buffer.
    MmapFile oldFile(oldFilePath);
    if (!oldFile.valid()) {
      throw std::runtime_error("Failed to map old file: " + oldFilePath);
    }
    QNN_INFO("Mapped old file (%s): %zu bytes.", oldFilePath.c_str(),
             oldFile.size);

    std::vector<char> patchFileBuffer = readFileForPatch(patchFilePath);
    QNN_INFO("Read patch file (%s): %zu bytes.", patchFilePath.c_str(),
             patchFileBuffer.size());

    if (patchFileBuffer.empty()) {
      throw std::runtime_error("Patch file (" + patchFilePath +
                               ") is empty or could not be read.");
    }

    unsigned long long const decompressedSize = ZSTD_getFrameContentSize(
        patchFileBuffer.data(), patchFileBuffer.size());

    if (decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
      throw std::runtime_error("Patch file (" + patchFilePath +
                               ") is not a valid zstd frame.");
    }
    if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
      throw std::runtime_error(
          "Decompressed size is unknown. Cannot proceed with this simple "
          "implementation.");
    }

    if (decompressedSize == 0) {
      QNN_ERROR("Patch resulted in empty buffer.");
      return nullptr;
    }

    uint8_t *newBuffer = new uint8_t[decompressedSize];

    ZSTD_DCtx *const dctx = ZSTD_createDCtx();
    if (dctx == nullptr) {
      delete[] newBuffer;
      throw std::runtime_error("ZSTD_createDCtx() failed!");
    }

    size_t const actualDecompressedSize = ZSTD_decompress_usingDict(
        dctx, newBuffer, decompressedSize, patchFileBuffer.data(),
        patchFileBuffer.size(), oldFile.data, oldFile.size);

    ZSTD_freeDCtx(dctx);

    if (ZSTD_isError(actualDecompressedSize)) {
      delete[] newBuffer;
      throw std::runtime_error(
          "ZSTD_decompress_usingDict() failed: " +
          std::string(ZSTD_getErrorName(actualDecompressedSize)));
    }

    QNN_INFO("Successfully applied patch to buffer. Decompressed %zu bytes.",
             actualDecompressedSize);

    return std::make_unique<PatchedModelBuffer>(newBuffer,
                                                actualDecompressedSize);

  } catch (const std::exception &e) {
    QNN_ERROR("Error applying patch to buffer: %s", e.what());
    return nullptr;
  }
}

// Legacy on-disk patched models ("unet.bin.<res>") are superseded by the
// in-memory patch flow; remove any leftovers next to the patch file.
inline void cleanupOldPatchedFiles(const std::string &patchPath) {
  try {
    std::filesystem::path patchFile(patchPath);
    std::filesystem::path patchDir = patchFile.parent_path();

    size_t totalFreed = 0;
    int filesRemoved = 0;

    for (const auto &entry : std::filesystem::directory_iterator(patchDir)) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();

        if (filename.rfind("unet.bin.", 0) == 0 && filename.length() > 9) {
          try {
            auto fileSize = entry.file_size();
            std::filesystem::remove(entry.path());
            totalFreed += fileSize;
            filesRemoved++;
            QNN_INFO("Cleaned up old patched file: %s (%.2f MB)",
                     entry.path().string().c_str(),
                     fileSize / (1024.0 * 1024.0));
          } catch (const std::exception &e) {
            QNN_WARN("Failed to remove file %s: %s",
                     entry.path().string().c_str(), e.what());
          }
        }
      }
    }

    if (filesRemoved > 0) {
      QNN_INFO("Total: cleaned up %d old patched file(s), freed %.2f MB",
               filesRemoved, totalFreed / (1024.0 * 1024.0));
    } else {
      QNN_DEBUG("No old patched files to clean up");
    }
  } catch (const std::exception &e) {
    QNN_WARN("Failed to clean up old patched files: %s", e.what());
  }
}

}  // namespace qnn_runtime

#endif  // QNNRUNTIME_HPP
