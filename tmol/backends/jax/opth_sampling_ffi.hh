// Typed, caller-owned buffers for the common OptH sample enumeration.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/rotamer/opth_sampling.hh>

namespace tmol::jax {
struct OptHSamplingStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t flip) {
    if (op < 0 || op > 2 || (flip != 0 && flip != 1)
        || a.size()
               != (op == 0   ? 15
                   : op == 1 ? 16
                             : 18)
        || out.size()
               != (op == 0   ? 2
                   : op == 1 ? 1
                             : 3))
      throw std::invalid_argument("invalid OptH stage operands");
    auto wide = ffi::DataType::U8, i = ffi::DataType::S32;
    auto f = ffi::DataType::F32, flag = ffi::DataType::PRED;
    graph_wide(a[0]);
    auto n = a[0].dimensions()[0];
    graph_wide(a[1], n);
    graph_wide(a[2], n);
    require_shape(a[3], wide, {-1, -1, 8});
    auto p = a[3].dimensions()[0], b = a[3].dimensions()[1];
    require_shape(a[4], flag, {n});
    require_shape(a[5], flag, {p, b});
    require_shape(a[7], i, {-1});
    auto t = a[7].dimensions()[0];
    require_shape(a[6], i, {2, t, t});
    require_shape(a[8], i, {t});
    require_shape(a[9], flag, {t});
    require_shape(a[10], flag, {t});
    require_shape(a[11], i, {t, -1});
    auto c = a[11].dimensions()[1];
    require_shape(a[12], i, {t, c});
    require_shape(a[13], f, {t, c, -1});
    require_shape(a[14], f, {p, b});
    score::common::checked_dispatch_size(n, "OptH candidate count");
    if (op == 0) {
      graph_wide(out[0], n);
      require_shape(out[1], flag, {n});
    } else if (op == 1) {
      graph_wide(a[15], n);
      graph_wide(out[0], n);
    } else {
      graph_wide(a[15], n);
      graph_wide(a[16], n);
      require_shape(a[17], i, {-1});
      auto r = a[17].dimensions()[0];
      score::common::checked_dispatch_size(r, "OptH sample count");
      require_shape(out[0], i, {r, c});
      require_shape(out[1], f, {r, c});
      require_shape(out[2], wide, {r, c, 8});
    }
    for (int j : {0, 1, 2, 3})
      if (reinterpret_cast<uintptr_t>(a[j].untyped_data()) % alignof(int64_t))
        throw std::invalid_argument("unaligned OptH index buffer");
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t flip) {
    using namespace pack::rotamer;
    OptHSamplingContext<D> c{
        buffer_view<int64_t, 1, D>(a[0]),
        buffer_view<int64_t, 1, D>(a[1]),
        buffer_view<int64_t, 1, D>(a[2]),
        buffer_view<int64_t, 2, D>(a[3]),
        buffer_view<bool, 1, D>(a[4]),
        buffer_view<bool, 2, D>(a[5]),
        buffer_view<int32_t, 3, D>(a[6]),
        buffer_view<int32_t, 1, D>(a[7]),
        buffer_view<int32_t, 1, D>(a[8]),
        buffer_view<bool, 1, D>(a[9]),
        {buffer_view<bool, 1, D>(a[10]),
         buffer_view<int32_t, 2, D>(a[11]),
         buffer_view<int32_t, 2, D>(a[12]),
         buffer_view<float, 3, D>(a[13])},
        buffer_view<float, 2, D>(a[14])};
    ContextManager mgr;
    if (op == 0)
      opth_count_native<Ops, D>(
          mgr,
          c,
          flip,
          buffer_view<int64_t, 1, D>(out[0]),
          buffer_view<bool, 1, D>(out[1]));
    else if (op == 1)
      opth_flip_counts_native<Ops, D>(
          mgr,
          c,
          buffer_view<int64_t, 1, D>(a[15]),
          buffer_view<int64_t, 1, D>(out[0]));
    else
      opth_samples_native<Ops, D>(
          mgr,
          c,
          buffer_view<int64_t, 1, D>(a[15]),
          buffer_view<int64_t, 1, D>(a[16]),
          buffer_view<int32_t, 1, D>(a[17]),
          buffer_view<int32_t, 2, D>(out[0]),
          buffer_view<float, 2, D>(out[1]),
          buffer_view<int64_t, 2, D>(out[2]));
  }
};
}  // namespace tmol::jax
