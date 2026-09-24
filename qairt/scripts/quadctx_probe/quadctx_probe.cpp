// quadctx_probe —— 只回答一个问题：
//   **四段 transformer context 能不能同时驻留？**
//
// 背景（台账 #145）：当前 app 只让 part1a+1b 常驻，每步把 part2 两个半段各装卸一次，
// 那 6.5 s/步 × 8 = 52 秒就花在这上面。我用 MemAvailable 推出「缺口 1117 MiB」判了否决，
// 但那是**保守代理不是实测硬限** —— #141 实测峰值时 lowmemorykiller 全程报
// "device has enough memory" 753 次，另有 6.5 GB KReclaimable 与 8.6 GB zram 未计入。
// ⇒ 直接测，别再推。
//
// 两种模式：
//   plain  四段依次 createFromBinary，全部保持存活（= 方案 B 的真实形态）
//   share  DEFER + 把**同一个**外置 spill-fill buffer 注册给四个 context 再 finalize
//          依据 htp_shared_buffer_tutorial.html 第 76 行：
//            "External spill-fill buffers can also be shared between graphs of
//             multiple contexts by registering the same external spill-fill buffer
//             with multiple contexts."
//          第 74~75 行：所需尺寸 = 各图 spill-fill 的**最大者**（四段实测 289 MiB）。
//          第 81 行限制「共享者不可并行执行」—— 我们四段严格串行，天然满足。
//
// 🔴 纪律：每建完一个 context **立刻释放文件缓冲区**，因为 app 就是这么做的
//    （读 .bin 进 buffer → createFromBinary → 释放）。留着会凭空多占 6.7 GB 宿主内存，
//    测出来就不是 app 的真实形态。

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

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

typedef void* (*rpcmem_alloc_t)(int heapid, uint32_t flags, size_t size);
typedef void  (*rpcmem_free_t)(void* po);
typedef int   (*rpcmem_to_fd_t)(void* po);
static rpcmem_alloc_t rpcmem_alloc_f = nullptr;
static rpcmem_free_t  rpcmem_free_f  = nullptr;
static rpcmem_to_fd_t rpcmem_to_fd_f = nullptr;

static QNN_INTERFACE_VER_TYPE I;

static long meminfoKB(const char* key) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char line[256];
  long v = -1;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, key, strlen(key)) == 0) {
      sscanf(line + strlen(key), " %ld", &v);
      break;
    }
  }
  fclose(f);
  return v;
}

static void snap(const char* tag) {
  printf("    [mem] %-22s ION %7.0f MiB | MemFree %6.0f | MemAvail %6.0f\n", tag,
         meminfoKB("IonTotalUsed:") / 1024.0,
         meminfoKB("MemFree:") / 1024.0,
         meminfoKB("MemAvailable:") / 1024.0);
  fflush(stdout);
}

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

// 读文件到 vector。调用方负责让它尽快出作用域（见文件头纪律）。
static std::vector<char> readFile(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { printf("  打不开 %s\n", p); return {}; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> b((size_t)n);
  size_t rd = fread(b.data(), 1, (size_t)n, f);
  fclose(f);
  if (rd != (size_t)n) { printf("  读入不完整 %zu/%ld\n", rd, n); return {}; }
  return b;
}

static uint64_t maxSpillFill(Qnn_ContextHandle_t ctx) {
  QnnHtpContext_CustomProperty_t cp = QNN_HTP_CONTEXT_CUSTOM_PROPERTY_INIT;
  cp.option = QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE;
  QnnContext_Property_t qp = QNN_CONTEXT_PROPERTY_INIT;
  qp.option = QNN_CONTEXT_PROPERTY_OPTION_CUSTOM;
  qp.customProperty = &cp;
  QnnContext_Property_t* arr[] = {&qp, nullptr};
  if (I.contextGetProperty(ctx, arr) != QNN_SUCCESS) return 0;
  return cp.spillfillBufferSize;
}


// ===== enable 模式（2026-09-01 新增，台账 #86 路线 D）=====
// 只回答一件事：PD 红线按【已启用的图】算，还是按【整个 context】算？
// 用法: quadctx_probe <lib> enable:<图名|ALL> <ctx.bin> [要试取的图名 ...]
// 判定不能只看「没报错」（§7.3 通用坑）：createFromBinary 之后逐个 graphRetrieve，
// 被禁用的图必须取不到 —— 取得到就说明 ENABLE_GRAPHS 根本没生效。
static int runEnable(int argc, char** argv) {
  std::string spec = argv[2];
  size_t c = spec.find(':');
  std::string want = (c == std::string::npos) ? std::string() : spec.substr(c + 1);
  const char* path = argv[3];
  bool all = (want == "ALL" || want.empty());
  printf("\n=== enable 模式: %s ===\n", all ? "不设 ENABLE_GRAPHS (全解)" : want.c_str());

  Qnn_BackendHandle_t be = nullptr;
  Qnn_DeviceHandle_t dev = nullptr;
  if (I.backendCreate(nullptr, nullptr, &be) != QNN_SUCCESS) { printf("backendCreate 失败\n"); return 2; }
  if (I.deviceCreate(nullptr, nullptr, &dev) != QNN_SUCCESS) { printf("deviceCreate 失败\n"); return 2; }
  snap("起点(空载)");
  long ion0 = meminfoKB("IonTotalUsed:");

  long avail = meminfoKB("MemAvailable:") / 1024;
  if (avail < 700) {
    printf("安全闸: MemAvailable %ld MiB < 700 => 不装, 干净退出\n", avail);
    return 3;
  }

  Qnn_ContextHandle_t ctx = nullptr;
  Qnn_ErrorHandle_t e;
  {
    std::vector<char> bin = readFile(path);
    if (bin.empty()) return 4;
    printf("已读入 %.0f MiB\n", bin.size() / 1048576.0);
    const char* egArr[2] = { want.c_str(), nullptr };
    QnnContext_Config_t egCfg = QNN_CONTEXT_CONFIG_INIT;
    std::vector<const QnnContext_Config_t*> cfgs;
    if (!all) {
      egCfg.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS;
      egCfg.enableGraphs = egArr;
      cfgs.push_back(&egCfg);
    }
    cfgs.push_back(nullptr);
    e = I.contextCreateFromBinary(be, dev, cfgs.data(), (void*)bin.data(),
                                  bin.size(), &ctx, nullptr);
  }
  if (e != QNN_SUCCESS) {
    printf("[FAIL] createFromBinary -> 0x%x", (unsigned)e);
    if ((unsigned)e == 0x3ea) printf("   <= 这就是 #57 的 unsigned PD 容量上限");
    printf("\n");
    snap("失败后");
    return 5;
  }
  printf("createFromBinary OK\n");
  long ion1 = meminfoKB("IonTotalUsed:");
  snap("已加载");
  printf("  ION 净增 = %.0f MiB\n", (ion1 - ion0) / 1024.0);

  for (int k = 4; k < argc; ++k) {
    Qnn_GraphHandle_t g = nullptr;
    Qnn_ErrorHandle_t r = I.graphRetrieve(ctx, argv[k], &g);
    printf("  graphRetrieve(%-24s) -> %s (0x%x)\n", argv[k],
           r == QNN_SUCCESS ? "OK" : "FAIL", (unsigned)r);
  }

  I.contextFree(ctx, nullptr);
  snap("释放后");
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("用法: quadctx_probe <libQnnHtp.so> <plain|share> <ctx1.bin> [ctx2 ...]\n");
    return 1;
  }
  if (loadBackend(argv[1]) != 0) return 1;
  std::string mode = argv[2];
  if (mode.rfind("enable", 0) == 0) return runEnable(argc, argv);
  int nbin = argc - 3;
  printf("\n=== 模式 %s，共 %d 个 context ===\n", mode.c_str(), nbin);

  Qnn_BackendHandle_t be = nullptr;
  Qnn_DeviceHandle_t dev = nullptr;
  if (I.backendCreate(nullptr, nullptr, &be) != QNN_SUCCESS) { printf("backendCreate 失败\n"); return 2; }
  if (I.deviceCreate(nullptr, nullptr, &dev) != QNN_SUCCESS) { printf("deviceCreate 失败\n"); return 2; }
  snap("起点(空载)");
  long ion0 = meminfoKB("IonTotalUsed:");

  std::vector<Qnn_ContextHandle_t> ctxs;
  void* sfBuf = nullptr;         // share 模式下四个 context 共用的那一份
  int sfFd = -1;
  bool shareMode = (mode == "share");
  int failedAt = -1;

  // 🔴 安全闸：这是用户的日常手机。#55 记过「part2 在线建图曾把手机压死」。
  // 本探针要把 ION 推到比 app 平时峰值（7.2 GB）还高约 2 GB，所以每装一段之前
  // 先看 MemAvailable，低于阈值就**干净退出**（释放已占用的），不去撞那一下。
  const long GUARD_MIB = 700;

  for (int k = 0; k < nbin; ++k) {
    const char* path = argv[3 + k];
    printf("\n--- [%d/%d] %s ---\n", k + 1, nbin, path);
    long avail = meminfoKB("MemAvailable:") / 1024;
    if (avail < GUARD_MIB) {
      printf("  ⛔ 安全闸：MemAvailable %ld MiB < %ld ⇒ 不再往下装，干净退出\n",
             avail, GUARD_MIB);
      printf("     （这不是「装不下」的证据，是我主动停手；已装 %zu 段）\n", ctxs.size());
      failedAt = k;
      break;
    }
    Qnn_ContextHandle_t ctx = nullptr;
    Qnn_ErrorHandle_t e;
    {
      std::vector<char> bin = readFile(path);
      if (bin.empty()) { failedAt = k; break; }
      printf("  已读入 %.0f MiB\n", bin.size() / 1048576.0);

      std::vector<const QnnContext_Config_t*> cfgs;
      QnnContext_Config_t deferCfg = QNN_CONTEXT_CONFIG_INIT;
      if (shareMode) {
        deferCfg.option = QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT;
        deferCfg.isGraphInitDeferred = 1;
        cfgs.push_back(&deferCfg);
      }
      cfgs.push_back(nullptr);
      e = I.contextCreateFromBinary(be, dev, cfgs.data(), (void*)bin.data(),
                                    bin.size(), &ctx, nullptr);
    }   // 🔴 bin 在此析构 —— 与 app 行为一致，不让四份缓冲区叠加
    if (e != QNN_SUCCESS) {
      printf("  [FAIL] createFromBinary -> 0x%x\n", (unsigned)e);
      failedAt = k;
      break;
    }
    printf("  createFromBinary OK\n");

    if (shareMode) {
      uint64_t need = maxSpillFill(ctx);
      printf("  MAX_SPILLFILL_BUFFER_SIZE = %.1f MiB\n", need / 1048576.0);
      if (!sfBuf && need && rpcmem_alloc_f) {
        // 只分配一次；教程 74~75：尺寸取所有图里的最大者
        sfBuf = rpcmem_alloc_f(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)need);
        if (!sfBuf) { printf("  rpcmem_alloc(%.1f MiB) 失败\n", need / 1048576.0); failedAt = k; break; }
        sfFd = rpcmem_to_fd_f(sfBuf);
        printf("  已分配共享 spill-fill %.1f MiB fd=%d（后续 context 复用同一个）\n",
               need / 1048576.0, sfFd);
      }
      if (sfBuf) {
        QnnMemHtp_Descriptor_t hd{};
        hd.type = QNN_HTP_MEM_SHARED_SPILLFILL_BUFFER;
        hd.size = need;
        hd.sharedSpillfillBufferConfig.fd = sfFd;
        hd.sharedSpillfillBufferConfig.offset = 0;
        Qnn_MemDescriptor_t md = QNN_MEM_DESCRIPTOR_INIT;
        md.memShape = {0, nullptr, nullptr};
        md.dataType = QNN_DATATYPE_UNDEFINED;
        md.memType = QNN_MEM_TYPE_CUSTOM;
        md.customInfo = &hd;
        Qnn_MemHandle_t mh = nullptr;
        Qnn_ErrorHandle_t r = I.memRegister(ctx, &md, 1, &mh);
        printf("  memRegister(同一 fd) rc=0x%x %s\n", (unsigned)r,
               r == QNN_SUCCESS ? "[OK]" : "[FAIL]");
      }
      e = I.contextFinalize(ctx, nullptr);
      printf("  contextFinalize -> 0x%x %s\n", (unsigned)e,
             e == QNN_SUCCESS ? "[OK]" : "[FAIL]");
      if (e != QNN_SUCCESS) { failedAt = k; break; }
    }

    ctxs.push_back(ctx);
    char t[64];
    snprintf(t, sizeof(t), "第%d段驻留后", k + 1);
    snap(t);
  }

  long ion1 = meminfoKB("IonTotalUsed:");
  printf("\n=== 结论 ===\n");
  printf("  成功驻留 %zu / %d 段\n", ctxs.size(), nbin);
  printf("  ION 净增 %.0f MiB（%ld -> %ld kB）\n", (ion1 - ion0) / 1024.0, ion0, ion1);
  if ((int)ctxs.size() == nbin) {
    // 🔴 判据必须与"实际测了几段"绑定。原先这里无条件打印"四段可以同时驻留"，
    // 于是 nbin=1 的自检也打印了那句话 —— 一个会说谎的判据比没有判据更糟（约束 8）。
    if (nbin >= 4) {
      printf("  🟢 **%d 段可以同时驻留** ⇒ 常驻方案在内存上成立\n", nbin);
    } else {
      printf("  ✅ %d 段全部加载成功（样本量 %d，**不足以对「全部常驻」下结论**）\n",
             nbin, nbin);
    }
  } else {
    printf("  🔴 第 %d 段未能驻留 ⇒ 四段无法同时驻留（本模式下）\n", failedAt + 1);
    printf("  ⚠️ 注意区分两种失败：createFromBinary 报错 = **真的装不下**；\n");
    printf("     「安全闸」= 我主动停手，**不构成装不下的证据**（约束 6：做不了≠已排除）\n");
  }

  // 全部释放（不释放会留下 DSP 侧占用，影响用户后续使用）
  for (auto c : ctxs) I.contextFree(c, nullptr);
  if (sfBuf && rpcmem_free_f) rpcmem_free_f(sfBuf);
  I.deviceFree(dev);
  I.backendFree(be);
  snap("全部释放后");
  return ((int)ctxs.size() == nbin) ? 0 : 3;
}
