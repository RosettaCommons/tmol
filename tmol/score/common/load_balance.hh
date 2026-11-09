#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
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

// Number of Offset elements, not bytes. The canonical ModernGPU launch policy
// determines the partition spacing; no framework-specific tuning constants.
// Merge-path dispatch uses a signed-int combined extent even for int64 offsets.
template <typename Offset>
std::size_t load_balancing_search_workspace_size(
    int total, int generators, int ptx_version) {
  static_assert(
      std::is_same_v<Offset, int32_t> || std::is_same_v<Offset, int64_t>);
  if (total < 0 || generators < 0 || (total > 0 && generators == 0))
    throw std::invalid_argument("invalid load-balancing extent");
  if (total == 0) return 0;
  int64_t const combined = int64_t(total) + generators;
  if (combined > std::numeric_limits<int>::max())
    throw std::overflow_error("load-balancing merged extent exceeds int32");
  int const spacing =
      mgpu::default_load_balance_launch_t::cta_dim(ptx_version).nv();
  if (spacing <= 0)
    throw std::invalid_argument("unsupported load-balancing PTX version");
  return static_cast<std::size_t>((combined + spacing - 1) / spacing + 1);
}

namespace detail {
// Supply the canonical algorithm's single temporary allocation from caller
// storage. Neither allocation nor synchronization reaches the parent context.
// Storage lifetime must extend until all work on parent.stream() completes.
class LoadBalanceWorkspaceContext final : public mgpu::context_t {
  mgpu::context_t& parent_;
  void* storage_;
  std::size_t bytes_;
  bool active_ = false;
  bool used_ = false;

 public:
  LoadBalanceWorkspaceContext(
      mgpu::context_t& parent, void* storage, std::size_t bytes)
      : parent_(parent), storage_(storage), bytes_(bytes) {}
  const cudaDeviceProp& props() const override { return parent_.props(); }
  int ptx_version() const override { return parent_.ptx_version(); }
  cudaStream_t stream() override { return parent_.stream(); }
  void* alloc(std::size_t bytes, mgpu::memory_space_t space) override {
    if (used_ || space != mgpu::memory_space_device || bytes > bytes_
        || !storage_)
      throw std::invalid_argument(
          "insufficient or invalid load-balancing workspace");
    active_ = true;
    used_ = true;
    return storage_;
  }
  void free(void* pointer, mgpu::memory_space_t space) override {
    if (pointer != storage_ || space != mgpu::memory_space_device || !active_)
      throw std::logic_error("unexpected load-balancing workspace release");
    active_ = false;
  }
  void synchronize() override {
    throw std::logic_error("unexpected load-balancing synchronization");
  }
  cudaEvent_t event() override {
    throw std::logic_error("unexpected load-balancing event");
  }
  void timer_begin() override {
    throw std::logic_error("unexpected load-balancing timer");
  }
  double timer_end() override {
    throw std::logic_error("unexpected load-balancing timer");
  }
};
}  // namespace detail

template <typename Offset, typename Int>
void load_balancing_search_into_cuda(
    int total,
    Offset const* offsets,
    int generators,
    Int* output,
    Offset* partitions,
    std::size_t partition_capacity,
    mgpu::context_t& context) {
  auto const required = load_balancing_search_workspace_size<Offset>(
      total, generators, context.ptx_version());
  if (required == 0) return;
  if (!partitions || partition_capacity < required)
    throw std::invalid_argument(
        "insufficient load-balancing partition capacity");
  // Limit the allocator to the exact requested workspace even when the caller
  // supplies a larger reusable buffer; an algorithm change cannot hide a second
  // temporary allocation behind excess capacity.
  detail::LoadBalanceWorkspaceContext scratch(
      context, partitions, required * sizeof(Offset));
  mgpu::load_balance_search(total, offsets, generators, output, scratch);
}
#endif
}  // namespace tmol::score::common
