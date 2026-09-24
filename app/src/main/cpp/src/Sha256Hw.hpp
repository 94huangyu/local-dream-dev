#ifndef SHA256HW_HPP
#define SHA256HW_HPP

// ARMv8 Crypto Extension 加速的 SHA-256（台账 #178）。
//
// 为什么要它：契约校验每次冷启动要过 **12.04 GiB**，现有 `Sha256.hpp` 是纯软件实现
// 且 `update()` **逐字节循环**，实测约 62 MiB/s ⇒ 启动开销 148~210 s，**比生图本身还长**。
// 设备 `/proc/cpuinfo` Features 实测含 `sha2 sha512 sha3 aes`（① 2026-09-19）。
//
// 🔴 三条安全设计（哈希算错 = 契约校验全失败 = app 起不来，所以不许赌）：
//   1. **运行时检测**：`getauxval(AT_HWCAP) & HWCAP_SHA2`，不支持就用软件版。
//   2. **启动自检**：用一组已知向量同时跑硬件版与软件版，**逐位相同才启用硬件版**。
//      我写错了指令序列的话，自检会当场发现并回落，app 照常工作（只是没加速）。
//   3. 硬件函数用 `__attribute__((target("+crypto")))` 单独标记 ⇒ 其余代码仍按默认
//      `-march` 编译，不会在不支持的设备上因为指令集而崩。
//
// 数值承诺：本文件**不改变任何哈希值**，只改变算得多快。任何不一致都必须回落。

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "Sha256.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_SHA2
#define HWCAP_SHA2 (1 << 6)
#endif
#define ZIMAGE_SHA256_HW_POSSIBLE 1
#endif

namespace sha256hw {

inline const uint32_t *k_table() {
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  return K;
}

#ifdef ZIMAGE_SHA256_HW_POSSIBLE

// 处理 n 个 64 字节块。state 为标准 SHA-256 的 8 个 32-bit 字（大端语义的主机序）。
__attribute__((target("+crypto"))) inline void transform_blocks(uint32_t state[8],
                                                                const uint8_t *data,
                                                                size_t blocks) {
  const uint32_t *K = k_table();
  uint32x4_t STATE0 = vld1q_u32(&state[0]);
  uint32x4_t STATE1 = vld1q_u32(&state[4]);

  while (blocks--) {
    const uint32x4_t ABEF_SAVE = STATE0;
    const uint32x4_t CDGH_SAVE = STATE1;

    uint32x4_t MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 0)));
    uint32x4_t MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 16)));
    uint32x4_t MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 32)));
    uint32x4_t MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 48)));

    uint32x4_t TMP0 = vaddq_u32(MSG0, vld1q_u32(&K[0]));
    uint32x4_t TMP1;
    uint32x4_t TMP2;

    // 轮 0~47：每组 4 轮，带消息扩展（su0/su1）
#define ZIMAGE_SHA_ROUNDS(Ma, Mb, Mc, Md, Kidx, TIn, TOut) \
  do {                                                     \
    Ma = vsha256su0q_u32(Ma, Mb);                          \
    TMP2 = STATE0;                                         \
    TOut = vaddq_u32(Mb, vld1q_u32(&K[Kidx]));             \
    STATE0 = vsha256hq_u32(STATE0, STATE1, TIn);           \
    STATE1 = vsha256h2q_u32(STATE1, TMP2, TIn);            \
    Ma = vsha256su1q_u32(Ma, Mc, Md);                      \
  } while (0)

    ZIMAGE_SHA_ROUNDS(MSG0, MSG1, MSG2, MSG3, 4, TMP0, TMP1);    // 0-3
    ZIMAGE_SHA_ROUNDS(MSG1, MSG2, MSG3, MSG0, 8, TMP1, TMP0);    // 4-7
    ZIMAGE_SHA_ROUNDS(MSG2, MSG3, MSG0, MSG1, 12, TMP0, TMP1);   // 8-11
    ZIMAGE_SHA_ROUNDS(MSG3, MSG0, MSG1, MSG2, 16, TMP1, TMP0);   // 12-15
    ZIMAGE_SHA_ROUNDS(MSG0, MSG1, MSG2, MSG3, 20, TMP0, TMP1);   // 16-19
    ZIMAGE_SHA_ROUNDS(MSG1, MSG2, MSG3, MSG0, 24, TMP1, TMP0);   // 20-23
    ZIMAGE_SHA_ROUNDS(MSG2, MSG3, MSG0, MSG1, 28, TMP0, TMP1);   // 24-27
    ZIMAGE_SHA_ROUNDS(MSG3, MSG0, MSG1, MSG2, 32, TMP1, TMP0);   // 28-31
    ZIMAGE_SHA_ROUNDS(MSG0, MSG1, MSG2, MSG3, 36, TMP0, TMP1);   // 32-35
    ZIMAGE_SHA_ROUNDS(MSG1, MSG2, MSG3, MSG0, 40, TMP1, TMP0);   // 36-39
    ZIMAGE_SHA_ROUNDS(MSG2, MSG3, MSG0, MSG1, 44, TMP0, TMP1);   // 40-43
    ZIMAGE_SHA_ROUNDS(MSG3, MSG0, MSG1, MSG2, 48, TMP1, TMP0);   // 44-47
#undef ZIMAGE_SHA_ROUNDS

    // 轮 48~51：最后一次消息扩展只做 su1 的输入已备好，这里不再扩展
    TMP2 = STATE0;
    TMP1 = vaddq_u32(MSG1, vld1q_u32(&K[52]));
    STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
    STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

    // 轮 52~55
    TMP2 = STATE0;
    TMP0 = vaddq_u32(MSG2, vld1q_u32(&K[56]));
    STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
    STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

    // 轮 56~59
    TMP2 = STATE0;
    TMP1 = vaddq_u32(MSG3, vld1q_u32(&K[60]));
    STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
    STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

    // 轮 60~63
    TMP2 = STATE0;
    STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
    STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

    STATE0 = vaddq_u32(STATE0, ABEF_SAVE);
    STATE1 = vaddq_u32(STATE1, CDGH_SAVE);
    data += 64;
  }

  vst1q_u32(&state[0], STATE0);
  vst1q_u32(&state[4], STATE1);
}

inline bool cpu_has_sha2() {
  static const bool ok = (getauxval(AT_HWCAP) & HWCAP_SHA2) != 0;
  return ok;
}

#else   // 非 aarch64：永远走软件版

inline void transform_blocks(uint32_t[8], const uint8_t *, size_t) {}
inline bool cpu_has_sha2() { return false; }

#endif

// 流式接口，语义与 `Sha256` 完全一致（同输入 ⇒ 同输出）。
class Hasher {
 public:
  Hasher() { reset(); }

  void reset() {
    total_ = 0;
    buffered_ = 0;
    state_[0] = 0x6a09e667; state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372; state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f; state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab; state_[7] = 0x5be0cd19;
  }

  // 🔴 与旧实现的关键差别：**按块处理**，不逐字节拷贝。
  void update(const uint8_t *data, size_t len) {
    total_ += len;
    if (buffered_) {
      const size_t need = 64 - buffered_;
      const size_t take = len < need ? len : need;
      std::memcpy(buf_ + buffered_, data, take);
      buffered_ += take;
      data += take;
      len -= take;
      if (buffered_ == 64) {
        transform_blocks(state_, buf_, 1);
        buffered_ = 0;
      }
    }
    const size_t blocks = len / 64;
    if (blocks) {
      transform_blocks(state_, data, blocks);
      data += blocks * 64;
      len -= blocks * 64;
    }
    if (len) {
      std::memcpy(buf_, data, len);
      buffered_ = len;
    }
  }

  std::array<uint8_t, 32> finalize() {
    const uint64_t bitlen = total_ * 8;
    uint8_t pad[128] = {0};
    pad[0] = 0x80;
    const size_t padlen = (buffered_ < 56) ? (56 - buffered_) : (120 - buffered_);
    update(pad, padlen);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[i] = static_cast<uint8_t>((bitlen >> (56 - 8 * i)) & 0xff);
    total_ -= padlen;              // 长度字段本身不计入消息长度
    update(len_be, 8);
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[i * 4 + 0] = static_cast<uint8_t>((state_[i] >> 24) & 0xff);
      out[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xff);
      out[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xff);
      out[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xff);
    }
    return out;
  }

 private:
  uint32_t state_[8];
  uint8_t buf_[64];
  size_t buffered_ = 0;
  uint64_t total_ = 0;
};

// 🔴 自检：同一组向量，硬件版与软件版必须**逐位相同**。
// 只有它通过才允许启用硬件版（见文件头第 2 条）。
inline bool self_test() {
  if (!cpu_has_sha2()) return false;
  static const char *vectors[] = {
      "", "abc",
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "The quick brown fox jumps over the lazy dog. 0123456789 "
      "0123456789 0123456789 0123456789 0123456789 0123456789"};
  for (const char *v : vectors) {
    const size_t n = std::strlen(v);
    Hasher hw;
    hw.update(reinterpret_cast<const uint8_t *>(v), n);
    const auto a = hw.finalize();
    Sha256 sw;
    sw.update(reinterpret_cast<const uint8_t *>(v), n);
    const auto b = sw.finalize();
    if (std::memcmp(a.data(), b.data(), 32) != 0) return false;
  }
  // 再用一段跨块长数据验证流式分片（分片边界最容易写错）
  std::string big(1000, 'x');
  for (size_t i = 0; i < big.size(); ++i) big[i] = static_cast<char>('a' + (i % 26));
  Hasher hw;
  for (size_t off = 0; off < big.size(); off += 37)
    hw.update(reinterpret_cast<const uint8_t *>(big.data()) + off,
              std::min<size_t>(37, big.size() - off));
  const auto a = hw.finalize();
  Sha256 sw;
  sw.update(reinterpret_cast<const uint8_t *>(big.data()), big.size());
  const auto b = sw.finalize();
  return std::memcmp(a.data(), b.data(), 32) == 0;
}

inline bool enabled() {
  static const bool ok = self_test();
  return ok;
}

}  // namespace sha256hw

#endif  // SHA256HW_HPP
