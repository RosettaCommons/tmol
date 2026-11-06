// Typed runtime binding for canonical disulfide whole-pose physics.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/score/disulfide/potentials/disulfide_pose_kernels.hh>

namespace tmol::jax {
inline void validate_disulfide_tables(
    const std::vector<ffi::AnyBuffer>& a, ffi::DataType real, int64_t ports) {
  auto integer = ffi::DataType::S32;
  require_shape(a[5], ffi::DataType::PRED, {-1, ports});
  auto types = a[5].dimensions()[0];
  require_shape(a[6], integer, {types, ports, -1});
  if (a[6].dimensions()[2] < 3)
    throw std::invalid_argument("disulfide requires three downstream atoms");
  require_shape(a[7], real, {types, 33});
}
template <Device D, typename Real>
auto disulfide_tables(const std::vector<ffi::AnyBuffer>& a) {
  using I = int32_t;
  using Params = score::disulfide::potentials::DisulfideGlobalParams<Real>;
  static_assert(sizeof(Params) == 33 * sizeof(Real));
  return std::make_tuple(
      buffer_view<bool, 2, D>(a[5]),
      buffer_view<I, 3, D>(a[6]),
      buffer_view<Params, 1, D>(a[7]));
}

struct DisulfideKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 8)
      throw std::invalid_argument("disulfide FFI expects 8 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("disulfide coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[3], integer, {poses, -1});
    auto blocks = a[3].dimensions()[1];
    require_shape(a[1], integer, {poses * blocks});
    require_shape(a[2], integer, {poses, blocks});
    require_shape(a[4], integer, {poses, blocks, -1, 2});
    auto ports = a[4].dimensions()[2];
    validate_disulfide_tables(a, real, ports);
    require_shape(energy, real, {poses});
    if (!gradient)
      throw std::invalid_argument(
          "disulfide kernel requires a derivative output");
    require_shape(*gradient, real, {poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("disulfide atom count exceeds int32");
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    using I = int32_t;
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto xyz = contiguous_view<Vec<Real, 3>, 1, D>(
        a[0].untyped_data(), {poses * atoms});
    auto e =
        contiguous_view<Real, 4, D>(energy.untyped_data(), {1, poses, 1, 1});
    auto g = contiguous_view<Vec<Real, 3>, 2, D>(
        gradient->untyped_data(), {1, poses * atoms});
    ContextManager context;
    std::apply(
        [&](auto... tables) {
          score::disulfide::potentials::disulfide_pose_forward<Ops, D, Real, I>(
              context,
              xyz,
              buffer_view<I, 1, D>(a[1]),
              buffer_view<I, 2, D>(a[2]),
              buffer_view<I, 2, D>(a[3]),
              buffer_view<Vec<I, 2>, 3, D>(a[4]),
              tables...,
              false,
              e,
              g);
        },
        disulfide_tables<D, Real>(a));
  }
};
}  // namespace tmol::jax
