#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tmol {
namespace portable {

// Output extent, recursive carry storage and load-balance boundaries for the
// canonical segmented-scan kernel. Integer ceilings keep capacities exact even
// beyond the range where float can distinguish adjacent work items.
template <typename Int = int64_t>
std::array<Int, 3> scan_workspace_sizes(
    int64_t scan_size, int64_t threads, int64_t values_per_thread) {
  static_assert(std::is_integral_v<Int> && std::is_signed_v<Int>);
  if (scan_size < 0 || threads < 2 || values_per_thread < 1)
    throw std::invalid_argument("Invalid segmented-scan workspace dimensions");
  if (scan_size > std::numeric_limits<Int>::max()
      || threads > std::numeric_limits<int64_t>::max() / values_per_thread)
    throw std::overflow_error(
        "Segmented-scan workspace exceeds index capacity");
  auto ceil_divide = [](int64_t count, int64_t width) {
    return count / width + (count % width != 0);
  };
  int64_t remaining = ceil_divide(scan_size, threads * values_per_thread);
  int64_t boundaries = remaining + 1;
  int64_t carry = remaining;
  while (remaining > 1) {
    remaining = ceil_divide(remaining, threads);
    if (carry > std::numeric_limits<Int>::max() - remaining)
      throw std::overflow_error(
          "Segmented-scan carry storage exceeds capacity");
    carry += remaining;
  }
  if (boundaries > std::numeric_limits<Int>::max())
    throw std::overflow_error(
        "Segmented-scan boundary storage exceeds capacity");
  return {Int(scan_size), Int(carry), Int(boundaries)};
}

}  // namespace portable
}  // namespace tmol
