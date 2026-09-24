// 独立 benchmark：验证 ARMv8 硬件 SHA-256 的**正确性**与**吞吐**（台账 #178）。
//
// 为什么不直接改 app 再测：改 app 要走 gradle 构建 + 装 APK + 起后端，一轮十几分钟；
// 而哈希写错的后果是契约校验全失败、app 起不来。先用这个几十 KB 的可执行文件在设备上
// 把「算得对不对、快多少」这两件事**单独**确认掉，通过了再集成（约束 4：先规划后执行）。
//
// 编译（NDK，arm64）：见 scripts/build_sha_bench.py
// 运行：adb push 到 /data/local/tmp 后 `./sha_bench [要校验的文件...]`

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Sha256.hpp"
#include "Sha256Hw.hpp"

static std::string hex(const std::array<uint8_t, 32> &d) {
  static const char *H = "0123456789abcdef";
  std::string s;
  for (uint8_t b : d) {
    s += H[b >> 4];
    s += H[b & 0xf];
  }
  return s;
}

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char **argv) {
  printf("== SHA-256 硬件加速验证（台账 #178）==\n");
  printf("CPU 支持 sha2: %s\n", sha256hw::cpu_has_sha2() ? "YES" : "NO");
  printf("自检（硬件 vs 软件，5 组向量含分片边界）: %s\n",
         sha256hw::enabled() ? "PASS" : "**FAIL**");
  if (!sha256hw::enabled()) {
    printf("\n🔴 自检不过 ⇒ 硬件实现有误，**不得集成**。app 会回落软件版。\n");
    return 1;
  }

  // ---- 吞吐对比（同一块数据，先软后硬）----
  const size_t N = 256u << 20;   // 256 MiB
  std::vector<uint8_t> buf(N);
  for (size_t i = 0; i < N; ++i) buf[i] = static_cast<uint8_t>(i * 131 + 7);

  double t0 = now_s();
  Sha256 sw;
  sw.update(buf.data(), buf.size());
  const auto a = sw.finalize();
  const double t_sw = now_s() - t0;

  t0 = now_s();
  sha256hw::Hasher hw;
  hw.update(buf.data(), buf.size());
  const auto b = hw.finalize();
  const double t_hw = now_s() - t0;

  const bool same = hex(a) == hex(b);
  printf("\n-- 吞吐（%zu MiB）--\n", N >> 20);
  printf("  软件版 %7.2f s ⇒ %6.1f MiB/s   %s\n", t_sw, (N >> 20) / t_sw, hex(a).substr(0, 16).c_str());
  printf("  硬件版 %7.2f s ⇒ %6.1f MiB/s   %s\n", t_hw, (N >> 20) / t_hw, hex(b).substr(0, 16).c_str());
  printf("  哈希一致: %s   加速比: %.1f 倍\n", same ? "YES" : "**NO**", t_sw / t_hw);
  if (!same) return 1;

  // ---- 端到端：对真实 context 文件算哈希（可与契约里的值比对）----
  for (int i = 1; i < argc; ++i) {
    FILE *f = fopen(argv[i], "rb");
    if (!f) {
      printf("\n  跳过（打不开）: %s\n", argv[i]);
      continue;
    }
    std::vector<uint8_t> io(4u << 20);
    sha256hw::Hasher h;
    size_t total = 0, n;
    t0 = now_s();
    while ((n = fread(io.data(), 1, io.size(), f)) > 0) {
      h.update(io.data(), n);
      total += n;
    }
    fclose(f);
    const double dt = now_s() - t0;
    printf("\n  %s\n    %zu 字节, %.2f s ⇒ %.1f MiB/s\n    %s\n", argv[i], total, dt,
           (total >> 20) / dt, hex(h.finalize()).c_str());
  }
  printf("\n✅ 正确性与吞吐均已实测，可以集成。\n");
  return 0;
}
