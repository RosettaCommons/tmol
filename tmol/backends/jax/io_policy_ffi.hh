// Framework-owned buffers for shared canonical variant, cycle and port
// policies.
#pragma once
#include <cmath>
#include <cstring>
#include "graph_topology_ffi.hh"
#include <tmol/io/details/compiled/canonical_policies.hh>

namespace tmol::jax {
struct CanonicalPolicyStage {
  static double cycle_cutoff(int64_t encoded) {
    double value;
    std::memcpy(&value, &encoded, sizeof(value));
    return value;
  }
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    if (op < 0 || op > 2
        || a.size()
               != (op == 0   ? 4
                   : op == 1 ? 6
                             : 11)
        || out.size() != (op == 2 ? 2 : 1))
      throw std::invalid_argument("invalid canonical policy arguments");
    auto i = ffi::DataType::S32, b = ffi::DataType::PRED;
    require_shape(a[0], i, {-1, -1});
    auto p = a[0].dimensions()[0], r = a[0].dimensions()[1];
    if (op != 2) require_shape(out[0], i, {p, r});
    if (op == 0) {
      require_shape(a[1], i, {p, r});
      require_shape(a[2], i, {p, r});
      require_shape(a[3], b, {-1, 4, -1, -1});
      if (extent < 0)
        throw std::invalid_argument("negative neutral variant base");
    } else if (op == 1) {
      require_shape(a[1], i, {p, r});
      require_shape(a[2], i, {-1});
      require_shape(a[3], i, {a[2].dimensions()[0]});
      auto real = a[4].element_type();
      if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
        throw std::invalid_argument(
            "cycle coordinates require float32 or float64");
      require_shape(a[4], real, {p, r, -1, 3});
      require_shape(a[5], i, {-1, 3});
      if (!(cycle_cutoff(extent) > 0) || !std::isfinite(cycle_cutoff(extent)))
        throw std::invalid_argument("cycle cutoff must be finite and positive");
    } else {
      for (int k = 1; k < 4; ++k) require_shape(a[k], b, {p, r});
      require_shape(a[4], b, {p, r, 2});
      require_shape(a[5], i, {-1});
      auto t = a[5].dimensions()[0];
      require_shape(a[6], i, {t});
      require_shape(a[7], i, {t});
      require_shape(a[8], i, {-1, 3});
      require_shape(a[9], i, {-1, 3});
      require_shape(a[10], i, {-1, 5});
      if (width < 0) throw std::invalid_argument("negative connection count");
      require_shape(out[0], i, {p, r, width, 2});
      require_shape(out[1], i, {p});
    }
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void cycles(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer out,
      int64_t cutoff) {
    io::details::compiled::CanonicalCycleContext<Real, int32_t, D> c{
        buffer_view<int32_t, 2, D>(a[0]),
        buffer_view<int32_t, 2, D>(a[1]),
        buffer_view<int32_t, 1, D>(a[2]),
        buffer_view<int32_t, 1, D>(a[3]),
        buffer_view<portable::Vec<Real, 3>, 3, D>(a[4]),
        buffer_view<int32_t, 2, D>(a[5]),
        Real(cycle_cutoff(cutoff)),
        buffer_view<int32_t, 2, D>(out)};
    ContextManager manager;
    io::details::compiled::launch_canonical_cycles<Ops, D, Real, int32_t>(
        manager, c);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t) {
    ContextManager manager;
    if (op == 0) {
      io::details::compiled::CanonicalVariantContext<int32_t, D> c{
          buffer_view<int32_t, 2, D>(a[0]),
          buffer_view<int32_t, 2, D>(a[1]),
          buffer_view<int32_t, 2, D>(a[2]),
          buffer_view<bool, 4, D>(a[3]),
          extent,
          buffer_view<int32_t, 2, D>(out[0])};
      io::details::compiled::launch_canonical_variants<Ops, D, int32_t>(
          manager, c);
    } else if (op == 1) {
      if (a[4].element_type() == ffi::DataType::F32)
        cycles<Ops, D, float>(a, out[0], extent);
      else
        cycles<Ops, D, double>(a, out[0], extent);
    } else {
      io::details::compiled::CanonicalWiringContext<int32_t, D> c{
          buffer_view<int32_t, 2, D>(a[0]),
          buffer_view<bool, 2, D>(a[1]),
          buffer_view<bool, 2, D>(a[2]),
          buffer_view<bool, 2, D>(a[3]),
          buffer_view<bool, 3, D>(a[4]),
          buffer_view<int32_t, 1, D>(a[5]),
          buffer_view<int32_t, 1, D>(a[6]),
          buffer_view<int32_t, 1, D>(a[7]),
          buffer_view<int32_t, 2, D>(a[8]),
          buffer_view<int32_t, 2, D>(a[9]),
          buffer_view<int32_t, 2, D>(a[10]),
          buffer_view<portable::Vec<int32_t, 2>, 3, D>(out[0]),
          buffer_view<int32_t, 1, D>(out[1])};
      io::details::compiled::launch_canonical_wiring<Ops, D, int32_t>(
          manager, c);
    }
  }
};
}  // namespace tmol::jax
