// Typed adapters for common backbone candidate and physics kernels.
#pragma once
#include "graph_topology_ffi.hh"
#include "backbone_torsion_ffi.hh"
#include <tmol/score/backbone_torsion/potentials/backbone_rotamer_kernels.hh>
namespace tmol::jax {
struct BackboneRotamerTopologyStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if ((op != 0 && op != 1) || a.size() != (op ? 8 : 5) || out.size() != 1)
      throw std::invalid_argument("invalid backbone candidate operands");
    auto i = ffi::DataType::S32;
    require_shape(a[0], i, {-1, -1});
    auto p = a[0].dimensions()[0], b = a[0].dimensions()[1];
    require_shape(a[1], i, {p, b});
    require_shape(a.back(), i, {p, b});
    require_shape(a[2], i, {p, b, -1, 2});
    require_shape(a[3], i, {-1});
    if (op) {
      require_shape(a[4], i, {p, b});
      require_shape(a[5], ffi::DataType::U8, {p, b, 8});
      require_shape(a[6], i, {-1});
      require_shape(out[0], i, {3, a[6].dimensions()[0]});
    } else
      require_shape(out[0], ffi::DataType::U8, {p, b, 8});
    score::common::checked_dispatch_product(p, b, "backbone candidate cells");
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    using namespace score::backbone_torsion::potentials;
    using I2 = Vec<int32_t, 2>;
    ContextManager manager;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
    if (op)
      backbone_rotamer_fill_native<Ops, D, float, int32_t>(
          manager,
          IN(int32_t, 2, 4),
          IN(int32_t, 2, 0),
          IN(int32_t, 2, 1),
          IN(I2, 3, 2),
          IN(int32_t, 1, 3),
          IN(int64_t, 2, 5),
          IN(int32_t, 1, 6),
          buffer_view<int32_t, 2, D>(out[0]),
          buffer_view<int32_t, 2, D>(a.back()));
    else
      backbone_rotamer_count_native<Ops, D, float, int32_t>(
          manager,
          IN(int32_t, 2, 0),
          IN(int32_t, 2, 1),
          IN(I2, 3, 2),
          IN(int32_t, 1, 3),
          buffer_view<int64_t, 2, D>(out[0]),
          buffer_view<int32_t, 2, D>(a.back()));
#undef IN
  }
};

struct RotamerBackboneStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double) {
    if ((op != 0 && op != 1) || a.size() != 20 + op || out.size() != 1)
      throw std::invalid_argument("invalid sparse backbone operands");
    auto real = a[0].element_type(), i = ffi::DataType::S32;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("backbone coordinates require float32/64");
    require_shape(a[0], real, {-1, 3});
    require_shape(a[1], i, {-1});
    auto r = a[1].dimensions()[0];
    require_shape(a[2], i, {-1, -1});
    auto p = a[2].dimensions()[0], b = a[2].dimensions()[1];
    require_shape(a[3], i, {p, b});
    require_shape(a[4], i, {p, b, -1, 2});
    validate_backbone_tables(a, real, a[4].dimensions()[2]);
    for (int k : {16, 17}) require_shape(a[k], i, {r});
    require_shape(a[18], i, {p, b});
    require_shape(a[19], i, {3, -1});
    auto entries = a[19].dimensions()[1];
    if (op) {
      require_shape(a[20], real, {2, entries});
      require_shape(out[0], real, {a[0].dimensions()[0], 3});
    } else
      require_shape(out[0], real, {2, entries});
  }
  template <bool Backward, template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out) {
    ContextManager manager;
    auto e = contiguous_view<Real, 2, D>(
        Backward ? nullptr : out[0].untyped_data(),
        {2, Backward ? 0 : a[19].dimensions()[1]});
    int64_t sizes[] = {2, Backward ? a[0].dimensions()[0] : 0},
            strides[] = {0, 1};
    TView<Vec<Real, 3>, 2, D> g(
        Backward ? static_cast<Vec<Real, 3>*>(out[0].untyped_data()) : nullptr,
        sizes,
        strides);
    auto weights = contiguous_view<Real, 2, D>(
        Backward ? a[20].untyped_data() : nullptr,
        {2, Backward ? a[19].dimensions()[1] : 0});
    std::apply(
        [&](auto... tables) {
          score::backbone_torsion::potentials::
              backbone_rotamer_score_native<Backward, Ops, D, Real, int32_t>(
                  manager,
                  buffer_view<Vec<Real, 3>, 1, D>(a[0]),
                  buffer_view<int32_t, 1, D>(a[1]),
                  buffer_view<int32_t, 2, D>(a[2]),
                  buffer_view<int32_t, 2, D>(a[3]),
                  buffer_view<int32_t, 1, D>(a[16]),
                  buffer_view<int32_t, 1, D>(a[17]),
                  buffer_view<Vec<int32_t, 2>, 3, D>(a[4]),
                  tables...,
                  buffer_view<int32_t, 2, D>(a[19]),
                  e,
                  g,
                  weights,
                  true,
                  Backward,
                  buffer_view<int32_t, 2, D>(a[18]));
        },
        backbone_tables<true, D, Real>(a));
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t,
      double) {
    if (a[0].element_type() == ffi::DataType::F32) {
      if (op)
        launch<true, Ops, D, float>(a, out);
      else
        launch<false, Ops, D, float>(a, out);
    } else {
      if (op)
        launch<true, Ops, D, double>(a, out);
      else
        launch<false, Ops, D, double>(a, out);
    }
  }
};
}  // namespace tmol::jax
