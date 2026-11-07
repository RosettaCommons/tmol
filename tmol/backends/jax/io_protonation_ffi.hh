// JAX array contract for the common canonical protonation selector.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/io/details/compiled/protonation.hh>

namespace tmol::jax {
struct ProtonationStage {
  static int fill(int64_t, size_t) { return -1; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t deprotonated,
      int64_t) {
    if (op || a.size() != 11 || out.size() != 2 || deprotonated < 0
        || deprotonated > INT32_MAX)
      throw std::invalid_argument("invalid protonation selection arguments");
    auto i = ffi::DataType::S32, b = ffi::DataType::PRED, w = ffi::DataType::U8;
    require_shape(a[0], i, {-1, -1});
    auto p = a[0].dimensions()[0], r = a[0].dimensions()[1];
    require_shape(a[1], i, {p, r});
    require_shape(a[2], b, {p, r, -1});
    auto atoms = a[2].dimensions()[2];
    require_shape(a[3], b, {-1, atoms});
    auto types = a[3].dimensions()[0];
    for (int k = 4; k < 7; ++k) require_shape(a[k], b, {types, atoms});
    require_shape(a[7], b, {types, -1});
    require_shape(a[8], b, {types, a[7].dimensions()[1], atoms});
    require_shape(a[9], w, {-1, 5, 8});
    require_shape(a[10], w, {-1, 3, 8});
    require_shape(out[0], i, {p, r});
    require_shape(out[1], b, {p, r});
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t,
      int64_t deprotonated,
      int64_t) {
    io::details::compiled::ProtonationContext<int32_t, D> c{
        buffer_view<int32_t, 2, D>(a[0]),
        buffer_view<int32_t, 2, D>(a[1]),
        buffer_view<bool, 3, D>(a[2]),
        buffer_view<bool, 2, D>(a[3]),
        buffer_view<bool, 2, D>(a[4]),
        buffer_view<bool, 2, D>(a[5]),
        buffer_view<bool, 2, D>(a[6]),
        buffer_view<bool, 2, D>(a[7]),
        buffer_view<bool, 3, D>(a[8]),
        buffer_view<int64_t, 2, D>(a[9]),
        buffer_view<int64_t, 2, D>(a[10]),
        int32_t(deprotonated),
        buffer_view<int32_t, 2, D>(out[0]),
        buffer_view<bool, 2, D>(out[1])};
    ContextManager mgr;
    io::details::compiled::launch_protonation<Ops, D>(mgr, c);
  }
};
}  // namespace tmol::jax
