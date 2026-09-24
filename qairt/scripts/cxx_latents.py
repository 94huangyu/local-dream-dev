"""复现 C++ (libc++) `std::mt19937 + std::normal_distribution<float>` 产生的初始 latents。

## 为什么需要它

app 的 C++ 侧用 `std::mt19937 generator(req.seed)` + `std::normal_distribution<float>`，
PC 在环脚本用 `np.random.default_rng(seed).standard_normal()`（PCG64）。
**同一个 seed，两套 RNG 给出完全不同的初始噪声** ⇒ 两边生成的图是同一模型的
**不同采样**，像素比较无意义。2026-08-17 我据此做过一次无效对照。

要判定「C++ 实现 ≟ Python 流水线」，必须让两边吃**逐字节相同**的初始 latents。

## 实现依据（Android NDK 用 libc++）

1. `std::mt19937`：标准 MT19937，可用官方测试向量验证（第 10000 个输出 = 4123659995）。
2. `libc++ 的 uniform_real_distribution<float>(a,b)`：
   `(b-a) * __generate_canonical<float,24>(g) + a`
   而 `__generate_canonical` 在 k=1 时是 `float(g() - g.min()) / 2^32`（**用 float 运算，有精度损失**）。
3. `libc++ 的 normal_distribution<float>`：**Marsaglia 极坐标法 + 缓存第二个值**
   ```
   若 V_hot: 取缓存 V；否则循环 { u=Uni(-1,1); v=Uni(-1,1); s=u²+v² } while (s>1 || s==0)
   f = sqrt(-2*ln(s)/s);  V = v*f (缓存);  返回 u*f
   ```

⚠️ 第 2、3 条是**实现细节**，不是标准保证。**必须与真机对拍验证**
（`scripts/cxx_rng_probe.cpp`，NDK 编译后在设备上跑，比对前若干个值）。
未对拍通过前，本脚本的输出**不得使用**。

用法:
  python cxx_latents.py --check          # 只做 MT19937 测试向量自检
  python cxx_latents.py --dump N         # 打印前 N 个正态值（供与真机对拍）
  python cxx_latents.py --write <path>   # 写出 1*16*128*128 float32 latents
"""
import argparse
import math
import struct

import numpy as np


class MT19937:
    """标准 std::mt19937（32 位）。"""

    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.idx = 624

    def _generate(self):
        for i in range(624):
            y = (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
            if y % 2:
                self.mt[i] ^= 2567483615
        self.idx = 0

    def __call__(self):
        if self.idx >= 624:
            self._generate()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 2636928640
        y ^= (y << 15) & 4022730752
        y ^= y >> 18
        return y & 0xFFFFFFFF


def f32(x):
    """强制 float32 语义——libc++ 的 generate_canonical<float> 全程用 float 运算。"""
    return struct.unpack("f", struct.pack("f", x))[0]


class LibcxxNormal:
    """libc++ 的 normal_distribution<float>(0,1)：Marsaglia 极坐标 + 缓存。"""

    TWO_32 = float(2 ** 32)

    def __init__(self, gen):
        self.g = gen
        self.v = 0.0
        self.hot = False

    def _uni_m1_1(self):
        # __generate_canonical<float,24>: k=1 ⇒ float(g()) / 2^32，用 float 精度
        c = f32(f32(self.g()) / self.TWO_32)
        return f32(f32(2.0) * c + f32(-1.0))     # (b-a)*c + a, a=-1 b=1

    def __call__(self):
        if self.hot:
            self.hot = False
            return self.v
        while True:
            u = self._uni_m1_1()
            v = self._uni_m1_1()
            s = f32(f32(u * u) + f32(v * v))
            if s <= 1.0 and s != 0.0:
                break
        fp = f32(math.sqrt(-2.0 * math.log(s) / s))
        self.v = f32(v * fp)
        self.hot = True
        return f32(u * fp)


def check_engine():
    g = MT19937(5489)          # 标准默认种子
    for _ in range(9999):
        g()
    got = g()
    ok = got == 4123659995
    print(f"MT19937 官方测试向量: 第 10000 个输出 = {got}  期望 4123659995  "
          f"{'✅ 通过' if ok else '❌ 不符'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true")
    p.add_argument("--dump", type=int, default=0)
    p.add_argument("--write")
    # 多比例（台账 #86）：latent 尺寸不再固定 128x128。
    # 🔴 必须与 C++ 侧的填充顺序一致：PipelineZImage 按 [1,16,H/8,W/8] 线性填，
    #    所以这里只需要元素总数正确、顺序不变；形状本身不参与生成。
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    a = p.parse_args()

    if a.check and not check_engine():
        raise SystemExit(1)

    if a.dump:
        n = LibcxxNormal(MT19937(a.seed))
        vals = [n() for _ in range(a.dump)]
        print(f"seed={a.seed} 前 {a.dump} 个 normal(0,1):")
        for i, v in enumerate(vals):
            print(f"  [{i}] {v:.9f}")

    if a.write:
        if a.width % 8 or a.height % 8:
            raise SystemExit("宽高必须能被 8 整除（VAE 的像素比例）")
        lh, lw = a.height // 8, a.width // 8
        cnt = 1 * 16 * lh * lw
        n = LibcxxNormal(MT19937(a.seed))
        arr = np.fromiter((n() for _ in range(cnt)), dtype=np.float32, count=cnt)
        arr.tofile(a.write)
        print(f"写出 {a.write}  {a.width}x{a.height} -> latent [1,16,{lh},{lw}]  "
              f"n={arr.size}  mean={arr.mean():.6f}  std={arr.std():.6f}")


if __name__ == "__main__":
    main()
