// Runtime buffer contract for the common whole-pose backbone kernels.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/score/backbone_torsion/potentials/backbone_torsion_pose_kernels.hh>

namespace tmol::jax {
inline void validate_backbone_tables(
    const std::vector<ffi::AnyBuffer>& a, ffi::DataType real, int64_t ports) {
  auto integer = ffi::DataType::S32;
  require_shape(a[5], integer, {-1, ports, -1});
  auto types = a[5].dimensions()[0];
  for (int index : {6, 7}) require_shape(a[index], integer, {types, 2});
  for (int index : {8, 9, 10}) require_shape(a[index], integer, {types});
  require_shape(a[11], integer, {types, 12, 3});
  require_shape(a[12], real, {-1, -1, -1});
  require_shape(a[13], real, {a[12].dimensions()[0], 4});
  require_shape(a[14], real, {-1, 2, -1, -1});
  require_shape(a[15], real, {a[14].dimensions()[0], 4});
}
template <bool Sparse, Device D, typename Real>
auto backbone_tables(const std::vector<ffi::AnyBuffer>& a) {
  using I = int32_t;
  using Params = score::backbone_torsion::potentials::RamaTableParams<Real>;
  static_assert(sizeof(Params) == 4 * sizeof(Real));
  static_assert(sizeof(UnresolvedAtomID<I>) == 3 * sizeof(I));
  auto lower = [&] {
    if constexpr (Sparse)
      return std::tuple<>();
    else
      return std::make_tuple(buffer_view<I, 1, D>(a[8]));
  }();
  return std::tuple_cat(
      std::make_tuple(
          buffer_view<I, 3, D>(a[5]),
          buffer_view<I, 2, D>(a[6]),
          buffer_view<I, 2, D>(a[7])),
      lower,
      std::make_tuple(
          buffer_view<I, 1, D>(a[9]),
          buffer_view<I, 1, D>(a[10]),
          buffer_view<UnresolvedAtomID<I>, 2, D>(a[11]),
          buffer_view<Real, 3, D>(a[12]),
          buffer_view<Params, 1, D>(a[13]),
          buffer_view<Real, 4, D>(a[14]),
          buffer_view<Params, 1, D>(a[15])));
}

struct BackboneTorsionKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 16)
      throw std::invalid_argument("backbone torsion FFI expects 16 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("backbone coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[3], integer, {poses, -1});
    auto blocks = a[3].dimensions()[1];
    require_shape(a[1], integer, {poses * blocks});
    require_shape(a[2], integer, {poses, blocks});
    require_shape(a[4], integer, {poses, blocks, -1, 2});
    auto ports = a[4].dimensions()[2];
    validate_backbone_tables(a, real, ports);
    require_shape(energy, real, {2, poses});
    if (gradient) require_shape(*gradient, real, {2, poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("backbone atom count exceeds int32");
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
        contiguous_view<Real, 4, D>(energy.untyped_data(), {2, poses, 1, 1});
    auto g = contiguous_view<Vec<Real, 3>, 2, D>(
        gradient ? gradient->untyped_data() : nullptr,
        {2, gradient ? poses * atoms : 0});
    ContextManager context;
    std::apply(
        [&](auto... tables) {
          score::backbone_torsion::potentials::
              backbone_torsion_pose_forward<Ops, D, Real, I>(
                  context,
                  xyz,
                  buffer_view<I, 1, D>(a[1]),
                  buffer_view<I, 2, D>(a[2]),
                  buffer_view<I, 2, D>(a[3]),
                  buffer_view<Vec<I, 2>, 3, D>(a[4]),
                  tables...,
                  false,
                  gradient != nullptr,
                  e,
                  g);
        },
        backbone_tables<false, D, Real>(a));
  }
};
}  // namespace tmol::jax
