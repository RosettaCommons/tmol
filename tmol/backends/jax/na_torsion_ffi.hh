// Typed JAX binding for shared nucleic-acid coordinate kernels.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>

namespace tmol::jax {
struct NaTorsionKernel {
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient) {
    if (a.size() != 23)
      throw std::invalid_argument("NA FFI expects 23 operands");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("NA coordinates must be float32/64");
    auto integer = ffi::DataType::S32, boolean = ffi::DataType::PRED;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[1], integer, {poses, -1});
    auto blocks = a[1].dimensions()[1];
    require_shape(a[2], boolean, {poses, blocks});
    require_shape(a[3], integer, {poses, blocks, 10, 4});
    require_shape(a[4], boolean, {poses, blocks, 10});
    require_shape(a[5], integer, {poses, blocks, 5});
    require_shape(a[6], boolean, {poses, blocks, 5});
    require_shape(a[7], integer, {poses, blocks});
    require_shape(a[8], real, {2, 6, 3});
    require_shape(a[9], real, {2, 6});
    require_shape(a[10], real, {2, 10, 4});
    require_shape(a[11], real, {10, 10});
    for (int index : {12, 13, 20, 21, 22}) require_shape(a[index], real, {2});
    require_shape(a[14], real, {2, 10});
    require_shape(a[15], real, {2, 3, 3});
    require_shape(a[16], real, {2, 2, 2});
    require_shape(a[17], real, {2, 3, 2});
    require_shape(a[18], real, {2, 10, 10});
    require_shape(a[19], boolean, {10});
    require_shape(energy, real, {2, poses});
    if (gradient) require_shape(*gradient, real, {2, poses, atoms, 3});
    if (poses * atoms > std::numeric_limits<int32_t>::max()
        || poses * blocks > std::numeric_limits<int32_t>::max())
      throw std::invalid_argument("NA flattened layout exceeds int32");
  }

  template <
      template <Device> class Ops,
      Device D,
      typename Real,
      bool Derivative>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      double pucker_temperature,
      double bin_blend_sdev) {
    const auto* coords = static_cast<const Real*>(a[0].untyped_data());
    const auto* base = static_cast<const int32_t*>(a[1].untyped_data());
    const auto* is_na = static_cast<const bool*>(a[2].untyped_data());
    const auto* torsion_indices =
        static_cast<const int32_t*>(a[3].untyped_data());
    const auto* torsion_ok = static_cast<const bool*>(a[4].untyped_data());
    const auto* ring_indices = static_cast<const int32_t*>(a[5].untyped_data());
    const auto* ring_ok = static_cast<const bool*>(a[6].untyped_data());
    const auto* prev = static_cast<const int32_t*>(a[7].untyped_data());
    const auto* backbone_means = static_cast<const Real*>(a[8].untyped_data());
    const auto* backbone_sdev = static_cast<const Real*>(a[9].untyped_data());
    const auto* sugar_means = static_cast<const Real*>(a[10].untyped_data());
    const auto* chi_means = static_cast<const Real*>(a[11].untyped_data());
    const auto* sdev_sugar = static_cast<const Real*>(a[12].untyped_data());
    const auto* sdev_chi = static_cast<const Real*>(a[13].untyped_data());
    const auto* well_pucker = static_cast<const Real*>(a[14].untyped_data());
    const auto* well_alpha_gamma =
        static_cast<const Real*>(a[15].untyped_data());
    const auto* well_bibii_pucker =
        static_cast<const Real*>(a[16].untyped_data());
    const auto* well_alphanext_bibii =
        static_cast<const Real*>(a[17].untyped_data());
    const auto* well_chi_syn = static_cast<const Real*>(a[18].untyped_data());
    const auto* weight_bb = static_cast<const Real*>(a[20].untyped_data());
    const auto* weight_chi = static_cast<const Real*>(a[21].untyped_data());
    const auto* weight_sugar = static_cast<const Real*>(a[22].untyped_data());
    const auto* is_north = static_cast<const bool*>(a[19].untyped_data());
    auto* output = static_cast<Real*>(energy.untyped_data());
    auto* derivatives =
        gradient ? static_cast<Real*>(gradient->untyped_data()) : nullptr;
    int poses = a[0].dimensions()[0], blocks = a[1].dimensions()[1];
    int atoms = poses * a[0].dimensions()[1], bases = a[11].dimensions()[0];
    ContextManager context;
    LAUNCH_BOX_32;
    auto work = [=] TMOL_DEVICE_FUNC(int block) {
      score::na_torsion::potentials::
          na_torsion_block<Real, D, Derivative, int32_t>(
              block,
              coords,
              base,
              is_na,
              torsion_indices,
              torsion_ok,
              ring_indices,
              ring_ok,
              prev,
              backbone_means,
              backbone_sdev,
              sugar_means,
              chi_means,
              sdev_sugar,
              sdev_chi,
              well_pucker,
              well_alpha_gamma,
              well_bibii_pucker,
              well_alphanext_bibii,
              well_chi_syn,
              is_north,
              weight_bb,
              weight_chi,
              weight_sugar,
              Real(pucker_temperature),
              Real(bin_blend_sdev),
              poses,
              blocks,
              atoms,
              bases,
              derivatives,
              output);
    };
    Ops<D>::template forall_grouped<launch_t>(context, poses, blocks, work);
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      ffi::AnyBuffer energy,
      const ffi::AnyBuffer* gradient,
      double pucker_temperature,
      double bin_blend_sdev) {
    if (!std::isfinite(pucker_temperature) || pucker_temperature <= 0
        || !std::isfinite(bin_blend_sdev) || bin_blend_sdev <= 0)
      throw std::invalid_argument(
          "NA smoothing widths must be finite and positive");
    if (gradient)
      launch<Ops, D, Real, true>(
          a, energy, gradient, pucker_temperature, bin_blend_sdev);
    else
      launch<Ops, D, Real, false>(
          a, energy, nullptr, pucker_temperature, bin_blend_sdev);
  }
};
}  // namespace tmol::jax
