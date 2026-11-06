// Shape and dtype contracts for shared hydrogen reconstruction and input
// policy.
#pragma once
#include "graph_topology_ffi.hh"
#include <cmath>
#include <cstring>
#include <tmol/io/details/compiled/hydrogen_completion.hh>

namespace tmol::jax {
struct HydrogenStage {
  // Every output is initialized by the shared native launcher.
  static int fill(int64_t, size_t) { return -1; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    constexpr size_t inputs[] = {13, 14, 16, 5};
    constexpr size_t outputs[] = {1, 2, 3, 1};
    if (op < 0 || op > 3 || a.size() != inputs[op] || out.size() != outputs[op])
      throw std::invalid_argument("invalid hydrogen execution stage");
    auto real = a[0].element_type(), i = ffi::DataType::S32,
         b = ffi::DataType::PRED, w = ffi::DataType::U8;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument(
          "hydrogen coordinates require float32 or float64");
    if (op == 3) {
      require_shape(a[0], real, {-1, -1, -1, 3});
      auto p = a[0].dimensions()[0], r = a[0].dimensions()[1],
           n = a[0].dimensions()[2];
      require_shape(a[1], i, {p, r});
      require_shape(a[2], b, {p, r, n});
      require_shape(a[3], w, {-1, n, 8});
      require_shape(a[4], real, {a[3].dimensions()[0], n});
      require_shape(out[0], b, {p, r, n});
      return;
    }
    require_shape(a[0], real, {-1, -1, 3});
    auto p = a[0].dimensions()[0], n = a[0].dimensions()[1];
    require_shape(a[1], i, {p, -1});
    auto r = a[1].dimensions()[1];
    require_shape(a[2], i, {p, r});
    require_shape(a[3], i, {p, r, -1, 2});
    require_shape(a[4], i, {-1, a[3].dimensions()[2]});
    auto t = a[4].dimensions()[0];
    require_shape(a[5], b, {p, r, -1});
    auto width = a[5].dimensions()[2];
    require_shape(a[6], b, {t, width});
    for (int k : {7, 9, 12}) require_shape(a[k], w, {t, width, 8});
    for (int k : {8, 11}) require_shape(a[k], w, {t, width, 3, 8});
    require_shape(a[10], real, {t, width});
    require_shape(out[0], real, {p, n, 3});
    if (op) {
      require_shape(a[13], real, {p, n, 3});
      require_shape(out[1], real, {t, width});
    }
    if (op == 2) {
      require_shape(a[14], real, {p, n, 3});
      require_shape(a[15], real, {t, width});
      require_shape(out[2], real, {p, n, 3});
    }
  }
  template <template <Device> class Ops, Device D, class Real, int Mode>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t extent) {
    using V = portable::Vec<Real, 3>;
    using C = portable::Vec<int32_t, 2>;
    using namespace io::details::compiled;
    ContextManager mgr;
    if constexpr (Mode == 3) {
      double tolerance;
      std::memcpy(&tolerance, &extent, sizeof(tolerance));
      if (!std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument(
            "invalid hydrogen parent distance tolerance");
      HydrogenInputContext<Real, int32_t, D> c{
          buffer_view<int32_t, 2, D>(a[1]),
          buffer_view<V, 3, D>(a[0]),
          buffer_view<bool, 3, D>(a[2]),
          buffer_view<int64_t, 2, D>(a[3]),
          buffer_view<Real, 2, D>(a[4]),
          Real(tolerance),
          buffer_view<bool, 3, D>(out[0])};
      launch_hydrogen_input<Ops, D>(mgr, c);
    } else {
      HydrogenCompletionContext<Real, int32_t, D> c{
          buffer_view<V, 2, D>(a[0]),
          buffer_view<int32_t, 2, D>(a[1]),
          buffer_view<int32_t, 2, D>(a[2]),
          buffer_view<C, 3, D>(a[3]),
          buffer_view<int32_t, 2, D>(a[4]),
          buffer_view<bool, 3, D>(a[5]),
          buffer_view<bool, 2, D>(a[6]),
          buffer_view<int64_t, 2, D>(a[7]),
          buffer_view<int64_t, 3, D>(a[8]),
          buffer_view<int64_t, 2, D>(a[9]),
          buffer_view<Real, 2, D>(a[10]),
          buffer_view<int64_t, 3, D>(a[11]),
          buffer_view<int64_t, 2, D>(a[12])};
      TView<V, 2, D> weights, direction, weight_out;
      TView<Real, 2, D> distance_direction, distance_out;
      if constexpr (Mode > 0) {
        weights = buffer_view<V, 2, D>(a[13]);
        distance_out = buffer_view<Real, 2, D>(out[1]);
      }
      if constexpr (Mode == 2) {
        direction = buffer_view<V, 2, D>(a[14]);
        distance_direction = buffer_view<Real, 2, D>(a[15]);
        weight_out = buffer_view<V, 2, D>(out[2]);
      }
      launch_hydrogen_completion<Mode, Ops, D>(
          mgr,
          c,
          weights,
          direction,
          distance_direction,
          buffer_view<V, 2, D>(out[0]),
          distance_out,
          weight_out);
    }
  }
  template <template <Device> class Ops, Device D, class Real>
  static void dispatch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent) {
    switch (op) {
      case 0:
        launch<Ops, D, Real, 0>(a, out, extent);
        break;
      case 1:
        launch<Ops, D, Real, 1>(a, out, extent);
        break;
      case 2:
        launch<Ops, D, Real, 2>(a, out, extent);
        break;
      case 3:
        launch<Ops, D, Real, 3>(a, out, extent);
        break;
    }
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t) {
    if (a[0].element_type() == ffi::DataType::F32)
      dispatch<Ops, D, float>(a, out, op, extent);
    else
      dispatch<Ops, D, double>(a, out, op, extent);
  }
};
}  // namespace tmol::jax
