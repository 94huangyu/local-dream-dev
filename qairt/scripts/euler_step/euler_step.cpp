// euler_step —— 设备侧 Flow-Match Euler 更新，只为让 D2 能脱离 USB 跑完 8 步。
//
// 为什么需要它：D2（量化后出图）此前必须由宿主逐步驱动 —— 每步把四段的输出拉回宿主、
// 做一次 Euler 更新、再推回设备。那要求 USB 全程连接 35~45 分钟。
// 有了它，整个 8 步循环可以在设备上用 nohup 跑完，用户启动后立刻可以拔线。
//
// 🔴 调度器口径**逐字抄自** scripts/t2_fp32_ref.py 的 sigmas()/Euler 更新，
//    而那份又是逐字抄自已验证的 fp32_step_runner.py。**不得自己重新推导。**
//      shift = 3.0
//      f(s)  = shift*s / (1 + (shift-1)*s)
//      s_i   = f(1 - (1 - 1/n) * i/(n-1))      i = 0..n-1 ；末尾补 0
//      dt    = s_{i+1} - s_i
//      next  = lat - dt * noise          （与 t2_fp32_ref.py:166 逐字一致）
//
// 用法: euler_step <step_idx> <steps> <lat_in.raw> <noise.raw> <lat_out.raw> <n_float>
//       euler_step --selftest            打印 8 步的 sigma / dt，供与宿主比对

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double f_shift(double s, double shift) {
  return shift * s / (1.0 + (shift - 1.0) * s);
}

// 返回 n+1 个 sigma（末位 0），与 t2_fp32_ref.sigmas() 一致
static std::vector<double> sigmas(int n, double shift) {
  std::vector<double> out;
  for (int i = 0; i < n; ++i) {
    double s = (n == 1) ? 1.0 : 1.0 - (1.0 - 1.0 / n) * (double)i / (double)(n - 1);
    out.push_back(f_shift(s, shift));
  }
  out.push_back(0.0);
  return out;
}

int main(int argc, char** argv) {
  if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
    std::vector<double> s = sigmas(8, 3.0);
    printf("i  sigma        dt\n");
    for (int i = 0; i < 8; ++i)
      printf("%d  %.10f  %.10f\n", i, s[i], s[i + 1] - s[i]);
    return 0;
  }
  if (argc != 7) {
    printf("用法: euler_step <step_idx> <steps> <lat_in> <noise> <lat_out> <n_float>\n");
    printf("      euler_step --selftest\n");
    return 1;
  }
  int idx = atoi(argv[1]), steps = atoi(argv[2]);
  const char* fin = argv[3];
  const char* fnoise = argv[4];
  const char* fout = argv[5];
  long n = atol(argv[6]);
  if (idx < 0 || idx >= steps || n <= 0) { printf("参数非法\n"); return 2; }

  std::vector<float> lat(n), noise(n);
  FILE* f = fopen(fin, "rb");
  if (!f) { printf("打不开 %s\n", fin); return 3; }
  size_t r1 = fread(lat.data(), sizeof(float), n, f); fclose(f);
  f = fopen(fnoise, "rb");
  if (!f) { printf("打不开 %s\n", fnoise); return 3; }
  size_t r2 = fread(noise.data(), sizeof(float), n, f); fclose(f);
  // 🔴 尺寸不对必须硬失败：约束 3 记过 snpe-net-run 在尺寸不符时静默缩批
  if ((long)r1 != n || (long)r2 != n) {
    printf("元素数不符: lat=%zu noise=%zu 期望=%ld\n", r1, r2, n);
    return 4;
  }

  std::vector<double> s = sigmas(steps, 3.0);
  double dt = s[idx + 1] - s[idx];
  double smin = 1e30, smax = -1e30;
  for (long i = 0; i < n; ++i) {
    // 🔴 宿主口径是 `nxt = lat - dt*noise`（t2_fp32_ref.py:166）。
    // 我第一版写成 +，被约束 8 的「先用已知样本验证操作化」当场抓出。
    double v = (double)lat[i] - dt * (double)noise[i];
    lat[i] = (float)v;
    if (v < smin) smin = v;
    if (v > smax) smax = v;
  }
  f = fopen(fout, "wb");
  if (!f) { printf("写不了 %s\n", fout); return 5; }
  fwrite(lat.data(), sizeof(float), n, f);
  fclose(f);
  printf("step %d/%d  sigma=%.6f dt=%.6f  out[min=%.4f max=%.4f] -> %s\n",
         idx, steps, s[idx], dt, smin, smax, fout);
  return 0;
}
