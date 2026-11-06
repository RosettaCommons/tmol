// Device load balancing with caller-owned output and canonical scratch.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/score/common/load_balance.hh>

namespace tmol::jax {
struct LoadBalanceStage {
  static int fill(int64_t, size_t) { return -1; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t total,
      int64_t) {
    if (operation != 0 || a.size() != 1 || out.size() != 2)
      throw std::invalid_argument("invalid load-balance operands");
    graph_wide(a[0]);
    auto generators = a[0].dimensions()[0];
    if (total < 0 || generators > INT32_MAX || total > INT32_MAX - generators
        || (total && !generators))
      throw std::invalid_argument(
          "load-balance extent exceeds native capacity");
    require_shape(out[0], ffi::DataType::S32, {total});
    graph_wide(out[1]);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t,
      int64_t total,
      int64_t) {
    Ops<D>::load_balance(
        total,
        static_cast<const int64_t*>(a[0].untyped_data()),
        a[0].dimensions()[0],
        static_cast<int32_t*>(out[0].untyped_data()),
        static_cast<int64_t*>(out[1].untyped_data()),
        out[1].dimensions()[0]);
  }
};
}  // namespace tmol::jax
