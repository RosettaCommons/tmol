#pragma once
#include <tmol/score/common/segmented_scan_workspace.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/kinematics/compiled/kernel_segscan.cuh>

namespace tmol::score::common {
constexpr mgpu::scan_type_t mgpu_scan_type(tmol::portable::scan_type t) {
  return t == tmol::portable::scan_type::inclusive ? mgpu::scan_type_inc
                                                   : mgpu::scan_type_exc;
}

// Canonical segmented scan with caller-owned CUDA scratch and stream context.
// No allocations or host synchronization occur in this function.
template <portable::scan_type Scan, typename T, typename Int, typename Op>
void segmented_scan_into_cuda(
    mgpu::context_t& context,
    T* source,
    Int* starts,
    T* destination,
    int count,
    int segments,
    Op op,
    T identity,
    SegmentedScanWorkspace<T, Device::CUDA> workspace) {
  if (count < 0 || segments < 0)
    throw std::invalid_argument("Negative segmented-scan extent");
  if (count == 0) return;
  using Launch = mgpu::launch_params_t<128, 2>;
  auto sizes = portable::scan_workspace_sizes<int>(
      int64_t(count) + segments, Launch::nt, Launch::vt);
  if (workspace.carry.size(0) < sizes[1] || workspace.codes.size(0) < sizes[1]
      || workspace.boundaries.size(0) < sizes[2])
    throw std::invalid_argument("Insufficient segmented-scan workspace");
  auto indexing = [=] MGPU_DEVICE(int i, int, int) { return source[i]; };
  tmol::kinematics::kernel_segscan<Launch, mgpu_scan_type(Scan)>(
      indexing,
      count,
      starts,
      segments,
      destination,
      workspace.carry.data(),
      workspace.codes.data(),
      workspace.boundaries.data(),
      op,
      identity,
      context);
}
}  // namespace tmol::score::common
