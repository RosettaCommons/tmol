// Typed sparse generic-bonded execution; chemistry and dispatch live in core.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/score/genbonded/potentials/genbonded_rotamer_kernels.hh>

namespace tmol::jax {
struct ConnectedRotamerStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t) {
    if ((operation != 0 && operation != 1) || a.size() != (operation ? 9 : 6)
        || out.size() != 1)
      throw std::invalid_argument("invalid connected conformer operands");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], integer, {-1});
    auto rots = a[0].dimensions()[0];
    for (int i : {1, 2}) require_shape(a[i], integer, {rots});
    require_shape(a[3], integer, {-1, -1});
    auto poses = a[3].dimensions()[0], blocks = a[3].dimensions()[1];
    require_shape(a[4], integer, {poses, blocks});
    require_shape(a[5], integer, {poses, blocks, -1, 2});
    int count = score::common::checked_dispatch_product(
        rots, a[5].dimensions()[2] + 1, "connected conformer count");
    if (operation) {
      require_shape(a[6], integer, {poses, blocks});
      graph_wide(a[7], count);
      require_shape(a[8], integer, {-1});
      require_shape(out[0], integer, {3, a[8].dimensions()[0]});
    } else
      graph_wide(out[0], count);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t) {
    using namespace score::genbonded::potentials;
    ContextManager context;
    using I2 = Vec<int32_t, 2>;
#define IN(T, R, I) buffer_view<T, R, D>(a[I])
    if (operation)
      genbonded_rotamer_fill_native<Ops, D, float, int32_t>(
          context,
          IN(int32_t, 2, 6),
          IN(int32_t, 1, 0),
          IN(int32_t, 1, 1),
          IN(int32_t, 2, 4),
          IN(I2, 3, 5),
          IN(int64_t, 1, 7),
          IN(int32_t, 1, 8),
          buffer_view<int32_t, 2, D>(out[0]));
    else
      genbonded_rotamer_count_native<Ops, D, float, int32_t>(
          context,
          IN(int32_t, 1, 0),
          IN(int32_t, 1, 1),
          IN(int32_t, 1, 2),
          IN(int32_t, 2, 3),
          IN(int32_t, 2, 4),
          IN(I2, 3, 5),
          buffer_view<int64_t, 1, D>(out[0]));
#undef IN
  }
};

struct RotamerGenBondedStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t,
      double) {
    if ((operation != 0 && operation != 1) || a.size() != 22 + operation
        || out.size() != 1)
      throw std::invalid_argument("invalid sparse generic-bonded operands");
    auto real = a[0].element_type(), integer = ffi::DataType::S32;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("sparse coordinates require float32/64");
    require_shape(a[0], real, {-1, 3});
    require_shape(a[1], integer, {-1});
    auto rots = a[1].dimensions()[0], atoms = a[0].dimensions()[0];
    for (int i : {2, 3, 19}) require_shape(a[i], integer, {rots});
    require_shape(a[4], integer, {-1, -1, -1, 2});
    validate_genbonded_tables(a, real, a[4].dimensions()[2]);
    require_shape(a[20], integer, {-1});
    auto entries = a[20].dimensions()[0];
    require_shape(a[21], integer, {3, entries});
    if (operation) {
      require_shape(a[22], real, {1, entries});
      require_shape(out[0], real, {atoms, 3});
    } else
      require_shape(out[0], real, {1, entries});
  }
  template <bool Backward, template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out) {
    ContextManager context;
    auto energy = contiguous_view<Real, 2, D>(
        Backward ? nullptr : out[0].untyped_data(),
        {1, Backward ? 0 : a[20].dimensions()[0]});
    auto gradient = contiguous_view<Vec<Real, 3>, 2, D>(
        Backward ? out[0].untyped_data() : nullptr,
        {1, Backward ? a[0].dimensions()[0] : 0});
    auto weights = contiguous_view<Real, 2, D>(
        Backward ? a[22].untyped_data() : nullptr,
        {1, Backward ? a[20].dimensions()[0] : 0});
    std::apply(
        [&](auto... tables) {
          score::genbonded::potentials::
              genbonded_rotamer_score_native<Backward, Ops, D, Real, int32_t>(
                  context,
                  buffer_view<Vec<Real, 3>, 1, D>(a[0]),
                  buffer_view<int32_t, 1, D>(a[1]),
                  buffer_view<int32_t, 1, D>(a[2]),
                  buffer_view<int32_t, 1, D>(a[3]),
                  buffer_view<int32_t, 1, D>(a[19]),
                  buffer_view<Vec<int32_t, 2>, 3, D>(a[4]),
                  tables...,
                  buffer_view<int32_t, 1, D>(a[20]),
                  buffer_view<int32_t, 2, D>(a[21]),
                  energy,
                  gradient,
                  weights,
                  true,
                  Backward);
        },
        genbonded_tables<D, Real>(a));
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t,
      double) {
    if (a[0].element_type() == ffi::DataType::F32) {
      if (operation)
        launch<true, Ops, D, float>(a, out);
      else
        launch<false, Ops, D, float>(a, out);
    } else {
      if (operation)
        launch<true, Ops, D, double>(a, out);
      else
        launch<false, Ops, D, double>(a, out);
    }
  }
};
}  // namespace tmol::jax
