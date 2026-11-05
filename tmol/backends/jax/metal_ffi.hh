// Buffer validation and execution binding for shared metal coordination.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/score/metal/potentials/metal_coordination_pose_kernels.hh>

namespace tmol::jax {
struct MetalCoordinationKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 15)
      throw std::invalid_argument("metal coordination FFI expects 15 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("metal coordinates must be float32/64");
    auto integer = ffi::DataType::S32;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[3], integer, {poses, -1});
    auto blocks = a[3].dimensions()[1];
    require_shape(a[1], integer, {poses * blocks});
    require_shape(a[2], integer, {poses, blocks});
    require_shape(a[4], integer, {poses, blocks, -1, 2});
    auto ports = a[4].dimensions()[2];
    require_shape(a[5], integer, {-1, ports});
    auto types = a[5].dimensions()[0];
    for (int index : {6, 7, 8})
      require_shape(a[index], integer, {types, ports});
    require_shape(a[9], real, {types, ports, -1, 4});
    auto keys = a[9].dimensions()[2];
    require_shape(a[10], integer, {types, -1, 2});
    require_shape(a[11], real, {types, a[10].dimensions()[1], 2});
    require_shape(a[12], integer, {types, ports, -1});
    require_shape(a[13], real, {types, ports, a[12].dimensions()[2]});
    require_shape(a[14], real, {keys, 2});
    require_shape(energy, real, {poses});
    if (gradient) require_shape(*gradient, real, {poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("metal atom count exceeds int32");
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    using I = int32_t;
    using Site = score::metal::potentials::MetalSiteParams<Real>;
    using Fan = score::metal::potentials::MetalFanParams<Real>;
    using Bridge = score::metal::potentials::MetalBridgeParams<Real>;
    static_assert(sizeof(Site) == 4 * sizeof(Real));
    static_assert(sizeof(Fan) == 2 * sizeof(Real));
    static_assert(sizeof(Bridge) == 2 * sizeof(Real));
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    auto xyz = contiguous_view<Vec<Real, 3>, 1, D>(
        a[0].untyped_data(), {poses * atoms});
    auto e =
        contiguous_view<Real, 4, D>(energy.untyped_data(), {1, poses, 1, 1});
    auto g = contiguous_view<Vec<Real, 3>, 2, D>(
        gradient ? gradient->untyped_data() : nullptr,
        {1, gradient ? poses * atoms : 0});
    ContextManager context;
    score::metal::potentials::metal_coordination_pose_forward<Ops, D, Real, I>(
        context,
        xyz,
        buffer_view<I, 1, D>(a[1]),
        buffer_view<I, 2, D>(a[2]),
        buffer_view<I, 2, D>(a[3]),
        buffer_view<Vec<I, 2>, 3, D>(a[4]),
        buffer_view<I, 2, D>(a[5]),
        buffer_view<I, 2, D>(a[6]),
        buffer_view<I, 2, D>(a[7]),
        buffer_view<I, 2, D>(a[8]),
        buffer_view<Site, 3, D>(a[9]),
        buffer_view<Vec<I, 2>, 2, D>(a[10]),
        buffer_view<Fan, 2, D>(a[11]),
        buffer_view<I, 3, D>(a[12]),
        buffer_view<Real, 3, D>(a[13]),
        buffer_view<Bridge, 1, D>(a[14]),
        false,
        gradient != nullptr,
        e,
        g);
  }
};
}  // namespace tmol::jax
