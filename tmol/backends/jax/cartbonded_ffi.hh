// Typed XLA buffers for canonical whole-pose Cartesian bonded physics.
#pragma once
#include "workspace_ffi.hh"
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>

namespace tmol::jax {
inline void validate_cartbonded_tables(
    const std::vector<ffi::AnyBuffer>& a, ffi::DataType real, int64_t ports) {
  auto integer = ffi::DataType::S32;
  require_shape(a[17], integer, {-1});
  auto types = a[17].dimensions()[0];
  require_shape(a[5], integer, {types, ports, 13, 3});
  require_shape(a[6], integer, {types, -1});
  auto type_atoms = a[6].dimensions()[1];
  for (int index : {7, 8, 10})
    require_shape(a[index], integer, {types, type_atoms});
  require_shape(a[9], integer, {types});
  require_shape(a[11], integer, {-1, 5});
  require_shape(a[12], integer, {-1, 2});
  require_shape(a[13], integer, {-1, 5});
  require_shape(a[14], integer, {-1, 5});
  require_shape(a[15], real, {-1, 7});
  require_shape(a[16], integer, {-1, 4});
  require_shape(a[18], integer, {types, 3});
  require_shape(a[19], integer, {types, 3});
  require_shape(a[20], integer, {a[16].dimensions()[0]});
}
template <Device D, typename Real>
auto cartbonded_tables(const std::vector<ffi::AnyBuffer>& a) {
  using I = int32_t;
  return std::make_tuple(
      buffer_view<Vec<I, 3>, 3, D>(a[5]),
      buffer_view<I, 2, D>(a[6]),
      buffer_view<I, 2, D>(a[7]),
      buffer_view<I, 2, D>(a[8]),
      buffer_view<I, 1, D>(a[9]),
      buffer_view<I, 2, D>(a[10]),
      buffer_view<Vec<I, 5>, 1, D>(a[11]),
      buffer_view<Vec<I, 2>, 1, D>(a[12]),
      buffer_view<Vec<I, 5>, 1, D>(a[13]),
      buffer_view<Vec<I, 5>, 1, D>(a[14]),
      buffer_view<Vec<Real, 7>, 1, D>(a[15]),
      buffer_view<Vec<I, 4>, 1, D>(a[16]),
      buffer_view<I, 1, D>(a[17]));
}
struct CartBondedKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 21)
      throw std::invalid_argument("Cartesian bonded FFI expects 21 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument(
          "Cartesian bonded coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[3], integer, {poses, -1});
    auto blocks = a[3].dimensions()[1];
    require_shape(a[1], integer, {poses * blocks});
    require_shape(a[2], integer, {poses, blocks});
    require_shape(a[4], integer, {poses, blocks, -1, 2});
    auto ports = a[4].dimensions()[2];
    validate_cartbonded_tables(a, real, ports);
    require_shape(energy, real, {5, poses});
    if (gradient) require_shape(*gradient, real, {5, poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("Cartesian bonded atom count exceeds int32");
  }

  template <
      template <Device> class Ops,
      Device D,
      typename Real,
      typename AccumReal = Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a, void* energy, void* gradient) {
    using I = int32_t;
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto xyz = contiguous_view<Vec<Real, 3>, 1, D>(
        a[0].untyped_data(), {poses * atoms});
    auto e = contiguous_view<AccumReal, 4, D>(energy, {5, poses, 1, 1});
    auto g = contiguous_view<Vec<AccumReal, 3>, 2, D>(
        gradient, {5, gradient ? poses * atoms : 0});
    ContextManager context;
    std::apply(
        [&](auto... tables) {
          score::cartbonded::potentials::
              cartbonded_pose_forward<Ops, D, Real, I>(
                  context,
                  xyz,
                  buffer_view<I, 1, D>(a[1]),
                  buffer_view<I, 2, D>(a[2]),
                  buffer_view<I, 2, D>(a[3]),
                  buffer_view<Vec<I, 2>, 3, D>(a[4]),
                  tables...,
                  buffer_view<Vec<I, 3>, 1, D>(a[18]),
                  buffer_view<Vec<I, 3>, 1, D>(a[19]),
                  buffer_view<I, 1, D>(a[20]),
                  false,
                  gradient != nullptr,
                  e,
                  g);
        },
        cartbonded_tables<D, Real>(a));
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    launch<Ops, D, Real>(
        a,
        energy.untyped_data(),
        gradient ? gradient->untyped_data() : nullptr);
  }
};

struct CartBondedPreciseKernel : CartBondedKernel {
  static size_t values(size_t poses, size_t atoms, bool derivative) {
    auto energy = score::common::checked_dispatch_product(
        poses, 5, "Cartesian energy values");
    auto gradients = derivative ? score::common::checked_dispatch_product(
                                      energy,
                                      score::common::checked_dispatch_product(
                                          atoms, 3, "Cartesian atoms"),
                                      "Cartesian gradient values")
                                : 0;
    if (energy > INT32_MAX - gradients)
      throw std::length_error("Cartesian accumulator dispatch exceeds int32");
    return energy + gradients;
  }
  static size_t workspace_bytes(
      const std::vector<ffi::AnyBuffer>& a, bool derivative) {
    return a[0].element_type() == ffi::DataType::F32
               ? values(a[0].dimensions()[0], a[0].dimensions()[1], derivative)
                     * sizeof(double)
               : 0;
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      ffi::AnyBuffer workspace) {
    if constexpr (!std::is_same_v<Real, float>) {
      CartBondedKernel::execute<Ops, D, Real>(a, energy, gradient);
    } else {
      auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
      auto nenergy = 5 * poses;
      auto total = values(poses, atoms, gradient != nullptr);
      auto* wide = static_cast<double*>(workspace.untyped_data());
      launch<Ops, D, Real, double>(
          a, wide, gradient ? wide + nenergy : nullptr);
      auto* e = static_cast<Real*>(energy.untyped_data());
      auto* g =
          gradient ? static_cast<Real*>(gradient->untyped_data()) : nullptr;
      ContextManager context;
      LAUNCH_BOX_32;
      Ops<D>::template forall<launch_t>(
          context, total, [=] TMOL_DEVICE_FUNC(int i) {
            if (i < nenergy)
              e[i] = Real(wide[i]);
            else
              g[i - nenergy] = Real(wide[i]);
          });
    }
  }
};
}  // namespace tmol::jax
