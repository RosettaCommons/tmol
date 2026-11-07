// Typed sparse generic-bonded execution; chemistry and dispatch live in core.
#pragma once
#include "graph_topology_ffi.hh"
#include "cartbonded_ffi.hh"
#include <tmol/score/common/connected_rotamer_dispatch.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_rotamer_kernels.hh>
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
    using namespace score::common;
    ContextManager context;
    using I2 = Vec<int32_t, 2>;
#define IN(T, R, I) buffer_view<T, R, D>(a[I])
    if (operation)
      connected_rotamer_fill_native<Ops, D, float, int32_t>(
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
      connected_rotamer_count_native<Ops, D, float, int32_t>(
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

template <bool Cartesian>
struct RotamerBondedStage {
  static constexpr int channels = Cartesian ? 5 : 1;
  static constexpr int types_slot = Cartesian ? 21 : 19;
  static constexpr int owner_slot = types_slot + 1;
  static constexpr int index_slot = types_slot + 2;
  static constexpr int weight_slot = types_slot + 3;
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t operation,
      int64_t,
      int64_t,
      double) {
    if ((operation != 0 && operation != 1)
        || a.size() != weight_slot + operation || out.size() != 1)
      throw std::invalid_argument("invalid sparse generic-bonded operands");
    auto real = a[0].element_type(), integer = ffi::DataType::S32;
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("sparse coordinates require float32/64");
    require_shape(a[0], real, {-1, 3});
    require_shape(a[1], integer, {-1});
    auto rots = a[1].dimensions()[0], atoms = a[0].dimensions()[0];
    for (int i : {2, 3, types_slot}) require_shape(a[i], integer, {rots});
    require_shape(a[4], integer, {-1, -1, -1, 2});
    if constexpr (Cartesian)
      validate_cartbonded_tables(a, real, a[4].dimensions()[2]);
    else
      validate_genbonded_tables(a, real, a[4].dimensions()[2]);
    require_shape(a[owner_slot], integer, {-1});
    auto entries = a[owner_slot].dimensions()[0];
    require_shape(a[index_slot], integer, {3, entries});
    if (operation) {
      require_shape(a[weight_slot], real, {channels, entries});
      require_shape(out[0], real, {atoms, 3});
    } else
      require_shape(out[0], real, {channels, entries});
  }
  template <bool Backward, template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out) {
    ContextManager context;
    auto energy = contiguous_view<Real, 2, D>(
        Backward ? nullptr : out[0].untyped_data(),
        {channels, Backward ? 0 : a[owner_slot].dimensions()[0]});
    // A VJP accumulates all weighted channels into one coordinate buffer.
    // The common kernels use atomic additions and never clear individual rows.
    int64_t gradient_shape[] = {channels, Backward ? a[0].dimensions()[0] : 0};
    int64_t gradient_strides[] = {0, 1};
    TView<Vec<Real, 3>, 2, D> gradient(
        Backward ? static_cast<Vec<Real, 3>*>(out[0].untyped_data()) : nullptr,
        gradient_shape,
        gradient_strides);
    auto weights = contiguous_view<Real, 2, D>(
        Backward ? a[weight_slot].untyped_data() : nullptr,
        {channels, Backward ? a[owner_slot].dimensions()[0] : 0});
    if constexpr (Cartesian) {
      std::apply(
          [&](auto... tables) {
            score::cartbonded::potentials::cartbonded_rotamer_score_native<
                Backward,
                Ops,
                D,
                Real,
                int32_t>(
                context,
                buffer_view<Vec<Real, 3>, 1, D>(a[0]),
                buffer_view<int32_t, 1, D>(a[1]),
                buffer_view<int32_t, 1, D>(a[2]),
                buffer_view<int32_t, 1, D>(a[3]),
                buffer_view<int32_t, 1, D>(a[types_slot]),
                buffer_view<Vec<int32_t, 2>, 3, D>(a[4]),
                tables...,
                buffer_view<int32_t, 1, D>(a[20]),
                buffer_view<int32_t, 1, D>(a[owner_slot]),
                buffer_view<int32_t, 2, D>(a[index_slot]),
                energy,
                gradient,
                weights,
                true,
                Backward);
          },
          cartbonded_tables<D, Real>(a));
    } else {
      std::apply(
          [&](auto... tables) {
            score::genbonded::potentials::
                genbonded_rotamer_score_native<Backward, Ops, D, Real, int32_t>(
                    context,
                    buffer_view<Vec<Real, 3>, 1, D>(a[0]),
                    buffer_view<int32_t, 1, D>(a[1]),
                    buffer_view<int32_t, 1, D>(a[2]),
                    buffer_view<int32_t, 1, D>(a[3]),
                    buffer_view<int32_t, 1, D>(a[types_slot]),
                    buffer_view<Vec<int32_t, 2>, 3, D>(a[4]),
                    tables...,
                    buffer_view<int32_t, 1, D>(a[owner_slot]),
                    buffer_view<int32_t, 2, D>(a[index_slot]),
                    energy,
                    gradient,
                    weights,
                    true,
                    Backward);
          },
          genbonded_tables<D, Real>(a));
    }
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
using RotamerGenBondedStage = RotamerBondedStage<false>;
using RotamerCartBondedStage = RotamerBondedStage<true>;
}  // namespace tmol::jax
