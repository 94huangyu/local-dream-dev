// 最小 external buffer probe —— 只回答一个问题：
//   完整的 part2 per-row context（2934 MB，普通 createFromBinary 报 err=1002
//   "Failed to find available PD ... context size estimate 3652345600"）
//   在 external spill-fill / weights buffer 下能否 finalize？
//
// 不跑推理、不接 I/O、不接调度器。四档单变量：
//   T0 普通 createFromBinary                 （预期复现 err 1002）
//   T1 DEFER + external spill-fill only
//   T2 DEFER + external weights only
//   T3 DEFER + external spill-fill + weights
//
// 依据 docs/QAIRT-Docs/QNN/general/htp/htp_shared_buffer_tutorial.html：
//   Context 必须用 QnnContext_createFromBinary + DEFER_GRAPH_INIT 创建，
//   此时不 deserialize，只拿 handle；然后查 buffer 尺寸、分配 DMA buffer、
//   QnnMem_register，最后 QnnContext_finalize 才真正 deserialize。
//   限制：仅 Android FastRPC；不可与 graph switching / udma64 /
//   securePD / REGISTER_MULTI_CONTEXTS 同用（本项目均未使用）。
//
// 顺带产出两个本项目从未见过的数：
//   QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE
//   QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnMem.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpMem.h"
#include "HTP/QnnHtpDevice.h"

// rpcmem（DMA buffer）——从 libcdsprpc.so 动态取，避免链接期依赖
// ⚠️ rpcmem_alloc 的 size 是 **int**，2730 MB 的权重 buffer 会 32 位溢出（实测分配失败）。
//    设备上 libcdsprpc.so 导出了 64 位的 rpcmem_alloc2，必须用它。
typedef void* (*rpcmem_alloc_t)(int heapid, uint32_t flags, size_t size);
typedef void  (*rpcmem_free_t)(void* po);
typedef int   (*rpcmem_to_fd_t)(void* po);
static rpcmem_alloc_t rpcmem_alloc_f = nullptr;
static rpcmem_free_t  rpcmem_free_f  = nullptr;
static rpcmem_to_fd_t rpcmem_to_fd_f = nullptr;
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

#define CK(msg, e)                                                        \
  do {                                                                    \
    if ((e) != QNN_SUCCESS) {                                             \
      printf("  [FAIL] %s -> 0x%x\n", msg, (unsigned)(e));                \
      return (int)(e);                                                    \
    }                                                                     \
  } while (0)

static QNN_INTERFACE_VER_TYPE I;

static int loadBackend(const char* lib) {
  void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
  if (!h) { printf("dlopen %s: %s\n", lib, dlerror()); return -1; }
  auto getProviders = (Qnn_ErrorHandle_t(*)(const QnnInterface_t***, uint32_t*))
      dlsym(h, "QnnInterface_getProviders");
  if (!getProviders) { printf("dlsym QnnInterface_getProviders 失败\n"); return -1; }
  const QnnInterface_t** p = nullptr; uint32_t n = 0;
  if (getProviders(&p, &n) != QNN_SUCCESS || n == 0) { printf("getProviders 失败\n"); return -1; }
  I = p[0]->QNN_INTERFACE_VER_NAME;
  printf("backend 已加载: %s (providers=%u)\n", lib, n);

  void* r = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (r) {
    rpcmem_alloc_f = (rpcmem_alloc_t)dlsym(r, "rpcmem_alloc2");
    if (!rpcmem_alloc_f) rpcmem_alloc_f = (rpcmem_alloc_t)dlsym(r, "rpcmem_alloc");
    rpcmem_free_f  = (rpcmem_free_t) dlsym(r, "rpcmem_free");
    rpcmem_to_fd_f = (rpcmem_to_fd_t)dlsym(r, "rpcmem_to_fd");
  }
  printf("rpcmem: alloc=%p to_fd=%p\n", (void*)rpcmem_alloc_f, (void*)rpcmem_to_fd_f);
  return 0;
}

static std::vector<char> readFile(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { printf("打不开 %s\n", p); return {}; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> b((size_t)n);
  size_t rd = fread(b.data(), 1, (size_t)n, f);
  fclose(f);
  printf("读取 %s: %.1f MB\n", p, rd / 1e6);
  return b;
}

// 用 external buffer 建 context。sf/wt: <0 不启用；否则作为该 buffer 的字节数
static int probe(const std::vector<char>& bin, bool defer, long sfSize, long wtSize,
                 const char* tag) {
  printf("\n=== %s (defer=%d spillfill=%ld weights=%ld) ===\n",
         tag, (int)defer, sfSize, wtSize);
  Qnn_BackendHandle_t be = nullptr;
  Qnn_DeviceHandle_t dev = nullptr;
  CK("backendCreate", I.backendCreate(nullptr, nullptr, &be));
  CK("deviceCreate",  I.deviceCreate(nullptr, nullptr, &dev));

  std::vector<const QnnContext_Config_t*> cfgs;
  QnnContext_Config_t deferCfg = QNN_CONTEXT_CONFIG_INIT;
  if (defer) {
    deferCfg.option = QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT;
    deferCfg.isGraphInitDeferred = 1;
    cfgs.push_back(&deferCfg);
  }
  cfgs.push_back(nullptr);

  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_ErrorHandle_t e = I.contextCreateFromBinary(
      be, dev, cfgs.data(), (void*)bin.data(), bin.size(), &ctx, nullptr);
  if (e != QNN_SUCCESS) { printf("  [FAIL] createFromBinary -> 0x%x\n", (unsigned)e); return (int)e; }
  printf("  createFromBinary OK (defer=%d)\n", (int)defer);

  if (!defer) { printf("  [PASS] T0 直接成功（说明 PD 容量本来就够）\n"); return 0; }

  // 查两个关键尺寸（本项目从未见过的数）
  auto getProp = [&](QnnHtpContext_GetPropertyOption_t opt, const char* name) -> uint64_t {
    QnnHtpContext_CustomProperty_t cp = QNN_HTP_CONTEXT_CUSTOM_PROPERTY_INIT;
    cp.option = opt;
    QnnContext_Property_t qp = QNN_CONTEXT_PROPERTY_INIT;
    qp.option = QNN_CONTEXT_PROPERTY_OPTION_CUSTOM;
    qp.customProperty = &cp;
    QnnContext_Property_t* arr[] = {&qp, nullptr};
    Qnn_ErrorHandle_t r = I.contextGetProperty(ctx, arr);
    // 直接读联合体的具名成员；此前用 memcpy 按偏移猜位置，读出天文数字（结构体对齐）
    uint64_t v = (opt == QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE)
                     ? cp.weightsBufferSize
                     : cp.spillfillBufferSize;
    printf("  %-34s rc=0x%-6x value=%llu (%.1f MB)\n", name, (unsigned)r,
           (unsigned long long)v, v / 1e6);
    return v;
  };
  uint64_t sfNeed = getProp(QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE,
                            "MAX_SPILLFILL_BUFFER_SIZE");
  uint64_t wtNeed = getProp(QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE,
                            "WEIGHTS_BUFFER_SIZE");

  // 分配并注册 external buffer
  auto reg = [&](uint64_t bytes, QnnHtpMem_Type_t type, const char* name) -> bool {
    if (!bytes || !rpcmem_alloc_f) return false;
    void* p = rpcmem_alloc_f(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)bytes);
    if (!p) { printf("  rpcmem_alloc(%s, %.1f MB) 失败\n", name, bytes / 1e6); return false; }
    int fd = rpcmem_to_fd_f(p);
    QnnMemHtp_Descriptor_t hd{};
    hd.type = type;
    hd.size = bytes;
    // 联合体成员按 type 选择；两者都是 QnnHtpMem_SharedBufferConfig_t{fd, offset}
    if (type == QNN_HTP_MEM_WEIGHTS_BUFFER) {
      hd.weightsBufferConfig.fd = fd;
      hd.weightsBufferConfig.offset = 0;
    } else {
      hd.sharedSpillfillBufferConfig.fd = fd;
      hd.sharedSpillfillBufferConfig.offset = 0;
    }
    Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT;
    md.memShape = {0, nullptr, nullptr};
    md.dataType = QNN_DATATYPE_UNDEFINED;
    md.memType = QNN_MEM_TYPE_CUSTOM;
    md.customInfo = &hd;
    Qnn_MemHandle_t mh = nullptr;
    Qnn_ErrorHandle_t r = I.memRegister(ctx, &md, 1, &mh);
    printf("  memRegister(%s, %.1f MB) rc=0x%x %s\n", name, bytes / 1e6,
           (unsigned)r, r == QNN_SUCCESS ? "[OK]" : "[FAIL]");
    return r == QNN_SUCCESS;
  };
  if (sfSize == 0) reg(sfNeed, QNN_HTP_MEM_SHARED_SPILLFILL_BUFFER, "spill-fill");
  if (wtSize == 0) reg(wtNeed, QNN_HTP_MEM_WEIGHTS_BUFFER, "weights");

  printf("  finalize ...\n");
  e = I.contextFinalize(ctx, nullptr);
  printf("  contextFinalize -> 0x%x %s\n", (unsigned)e,
         e == QNN_SUCCESS ? "[PASS] 装进 PD 了" : "[FAIL]");
  return (int)e;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("用法: extbuf_probe <libQnnHtp.so> <context.bin> [T0|T1|T2|T3]\n");
    return 1;
  }
  if (loadBackend(argv[1]) != 0) return 1;
  auto bin = readFile(argv[2]);
  if (bin.empty()) return 1;
  std::string t = (argc > 3) ? argv[3] : "T0";

  if (t == "T0") return probe(bin, false, -1, -1, "T0 普通 createFromBinary");
  if (t == "T1") return probe(bin, true,   0, -1, "T1 DEFER + external spill-fill");
  if (t == "T2") return probe(bin, true,  -1,  0, "T2 DEFER + external weights");
  return probe(bin, true, 0, 0, "T3 DEFER + spill-fill + weights");
}
