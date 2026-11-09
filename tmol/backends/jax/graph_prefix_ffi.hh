// Canonical exclusive scan with XLA-owned int64 output and native scratch.
#pragma once
#include "workspace_ffi.hh"
#include "graph_topology_ffi.hh"
#include <tmol/score/common/segmented_scan_workspace.hh>
#include <tmol/utility/portable/reduce_ops.hh>

namespace tmol::jax {
template <Device D>
struct GraphPrefixWorkspace {
  ByteWorkspace storage;
  score::common::SegmentedScanWorkspace<int64_t, D> scan;
  GraphPrefixWorkspace(int64_t count, void* data = nullptr) : storage(data) {
    auto sizes = portable::scan_workspace_sizes<int>(count + 1, 128, 2);
    scan = {
        contiguous_view<int64_t, 1, D>(
            storage.take<int64_t>(sizes[1]), {sizes[1]}),
        contiguous_view<int, 1, D>(storage.take<int>(sizes[1]), {sizes[1]}),
        contiguous_view<int, 1, D>(storage.take<int>(sizes[2]), {sizes[2]})};
  }
};
template <bool Maximum>
struct GraphPrefixKernelT {
  static constexpr bool zero_energy = false, zero_workspace = false;
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer out,
      const ffi::AnyBuffer* derivative) {
    if (a.size() != 2 || derivative)
      throw std::invalid_argument("invalid graph prefix operands");
    graph_wide(a[0]);
    require_shape(a[1], ffi::DataType::S32, {1});
    graph_wide(out, a[0].dimensions()[0] + 1);
  }
  static size_t workspace_bytes(const std::vector<ffi::AnyBuffer>& a, bool) {
    return GraphPrefixWorkspace<Device::CPU>(a[0].dimensions()[0])
        .storage.size_bytes();
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer out,
      const ffi::AnyBuffer*,
      ffi::AnyBuffer workspace) {
    auto source = buffer_view<int64_t, 1, D>(a[0]);
    auto result = buffer_view<int64_t, 1, D>(out);
    auto starts = buffer_view<int32_t, 1, D>(a[1]);
    int n =
        score::common::checked_dispatch_size(source.size(0), "graph prefix");
    GraphPrefixWorkspace<D> work(n, workspace.untyped_data());
    ContextManager context;
    using Op = std::conditional_t<
        Maximum,
        portable::maximum_t<int64_t>,
        portable::plus_t<int64_t>>;
    Ops<D>::template segmented_scan_into<portable::scan_type::exclusive>(
        context,
        source.data(),
        starts.data(),
        result.data(),
        n,
        1,
        Op(),
        int64_t(0),
        work.scan);
    LAUNCH_BOX_32;
    Ops<D>::template forall<launch_t>(context, 1, [=] TMOL_DEVICE_FUNC(int) {
      result[n] = n ? Op()(result[n - 1], source[n - 1]) : 0;
    });
  }
};
using GraphPrefixKernel = GraphPrefixKernelT<false>;
using GraphMaximumKernel = GraphPrefixKernelT<true>;
}  // namespace tmol::jax
