#pragma once
#include <cstdint>
#ifdef __NVCC__
#include <moderngpu/kernel_load_balance.hxx>
#endif

namespace tmol::score::common {
// Map each work item to its generator using exclusive offsets. Callers own
// storage and validate the nondecreasing offsets starting at zero. Empty
// generators are permitted anywhere. No framework allocation is performed.
template <typename Offset, typename Int>
void load_balancing_search_into_cpu(
    int n_work_units_total,
    Offset const* offsets,
    int n_generators,
    Int* output) {
  for (int i = 0; i < n_generators; ++i) {
    int64_t i_offset = offsets[i];
    int64_t i_n_work_units =
        (i + 1 == n_generators ? n_work_units_total : offsets[i + 1])
        - i_offset;
    for (int64_t j = 0; j < i_n_work_units; ++j) {
      output[i_offset + j] = i;
    }
  }
}
#ifdef __NVCC__
template <typename Offset, typename Int>
void load_balancing_search_into_cuda(
    int n_work_units_total,
    Offset const* offsets,
    int n_generators,
    Int* output,
    mgpu::context_t& context) {
  if (n_work_units_total == 0) return;
  mgpu::load_balance_search(
      n_work_units_total, offsets, n_generators, output, context);
}
#endif
}  // namespace tmol::score::common
