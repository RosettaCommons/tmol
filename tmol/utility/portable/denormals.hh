#pragma once

#include <cstdint>
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace tmol {
namespace utility {

// Bit 0: flush subnormal results; bit 1: treat subnormal inputs as zero.
// AArch64 controls both with FPCR.FZ. -1 reports an unsupported CPU policy.
inline int denormal_mode() {
#if defined(__SSE2__)
  const unsigned csr = _mm_getcsr();
  return int((csr >> 15) & 1) | int(((csr >> 6) & 1) << 1);
#elif defined(__aarch64__)
  std::uint64_t fpcr;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  return (fpcr & (std::uint64_t(1) << 24)) ? 3 : 0;
#else
  return -1;
#endif
}

inline bool set_denormal_mode(int mode) {
  if (mode < 0 || mode > 3) return false;
#if defined(__SSE2__)
  unsigned csr = _mm_getcsr();
  csr &= ~((1u << 15) | (1u << 6));
  csr |= (unsigned(mode & 1) << 15) | (unsigned((mode >> 1) & 1) << 6);
  _mm_setcsr(csr);
  return true;
#elif defined(__aarch64__)
  if (mode != 0 && mode != 3) return false;
  std::uint64_t fpcr;
  asm volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr &= ~(std::uint64_t(1) << 24);
  if (mode) fpcr |= std::uint64_t(1) << 24;
  asm volatile("msr fpcr, %0" : : "r"(fpcr) : "memory");
  return true;
#else
  return false;
#endif
}

}  // namespace utility
}  // namespace tmol
