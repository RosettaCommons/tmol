#pragma once
#include <cstdint>
#include <stdexcept>

namespace tmol::pack::compiled {
// PR626 deliberately retains different CPU and GPU rounding for counts that
// are not divisible by four. This metadata is shared by native bindings.
struct AnnealerTrajectoryCounts {
  static constexpr int default_high = 128, min_cpu_threads = 8;
  int high, low, final;
  explicit AnnealerTrajectoryCounts(int n = default_high, bool cpu = false)
      : high(n) {
    if (n < 4)
      throw std::invalid_argument("annealer requires at least 4 trajectories");
    int64_t l = cpu ? int64_t(n) * 10 / 4 : int64_t(n / 4) * 10;
    if (l > INT32_MAX)
      throw std::invalid_argument("annealer trajectory count exceeds int32");
    low = static_cast<int>(l);
    final = low / 4;
  }
};
}  // namespace tmol::pack::compiled
