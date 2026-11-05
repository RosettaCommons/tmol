#pragma once
#include <tmol/utility/portable/scan_workspace.hh>
#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol::score::common {
// Only CUDA consumes these buffers. Host scans need no auxiliary storage.
template <typename T, Device D>
struct SegmentedScanWorkspace {
  TView<T, 1, D> carry;
  TView<int, 1, D> codes;
  TView<int, 1, D> boundaries;
};

// Eager adapters select their allocator; staged bindings supply the views
// above. The native scan algorithm itself never owns or allocates these
// buffers.
template <typename T, Device D, template <typename, size_t, Device> class Pack>
struct OwnedSegmentedScanWorkspace {
  Pack<T, 1, D> carry;
  Pack<int, 1, D> codes;
  Pack<int, 1, D> boundaries;

  OwnedSegmentedScanWorkspace(int count, int segments) {
    if constexpr (D == Device::CUDA) {
      auto sizes = portable::scan_workspace_sizes<int>(
          int64_t(count) + segments, 128, 2);
      carry = Pack<T, 1, D>::empty({sizes[1]});
      codes = Pack<int, 1, D>::empty({sizes[1]});
      boundaries = Pack<int, 1, D>::empty({sizes[2]});
    }
  }
  SegmentedScanWorkspace<T, D> views() const {
    return {carry.view, codes.view, boundaries.view};
  }
};
}  // namespace tmol::score::common
