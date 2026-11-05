#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {
std::tuple<tmol::native::Tensor, tmol::native::Tensor>
na_torsion_pose_score_cpu(
    tmol::native::Tensor coords,
    tmol::native::Tensor base,
    tmol::native::Tensor is_na,
    tmol::native::Tensor torsion_indices,
    tmol::native::Tensor torsion_ok,
    tmol::native::Tensor ring_indices,
    tmol::native::Tensor ring_ok,
    tmol::native::Tensor prev,
    tmol::native::Tensor backbone_means,
    tmol::native::Tensor backbone_sdev,
    tmol::native::Tensor sugar_means,
    tmol::native::Tensor chi_means,
    tmol::native::Tensor sdev_sugar,
    tmol::native::Tensor sdev_chi,
    tmol::native::Tensor well_pucker,
    tmol::native::Tensor well_alpha_gamma,
    tmol::native::Tensor well_bibii_pucker,
    tmol::native::Tensor well_alphanext_bibii,
    tmol::native::Tensor well_chi_syn,
    tmol::native::Tensor is_north,
    tmol::native::Tensor weight_bb,
    tmol::native::Tensor weight_chi,
    tmol::native::Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev,
    bool compute_derivs) {
  TORCH_CHECK(
      coords.device().is_cpu(), "na_torsion_pose_score requires CPU tensors");
  TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
  int n_bases = int(chi_means.size(0));
  TORCH_CHECK(
      n_bases > 0 && n_bases % 2 == 0,
      "chi tables must have equal DNA and RNA base counts");
  TORCH_CHECK(
      chi_means.size(1) == N_PUCKER && well_chi_syn.size(2) == n_bases,
      "inconsistent nucleic-acid base table shapes");
  int n_poses = int(base.size(0));
  int max_n_blocks = int(base.size(1));
  auto output = tmol::native::zeros({2, n_poses}, coords.options());
  auto derivatives =
      compute_derivs
          ? tmol::native::zeros({2, coords.size(0), 3}, coords.options())
          : tmol::native::empty({0}, coords.options());
  AT_DISPATCH_FLOATING_TYPES(
      coords.scalar_type(), "na_torsion_pose_score_cpu", [&] {
        if (!compute_derivs) {
          for (int block = 0; block < n_poses * max_n_blocks; ++block)
            na_torsion_block<scalar_t, tmol::Device::CPU, false>(
                block,
                coords.const_data_ptr<scalar_t>(),
                base.const_data_ptr<int64_t>(),
                is_na.const_data_ptr<bool>(),
                torsion_indices.const_data_ptr<int64_t>(),
                torsion_ok.const_data_ptr<bool>(),
                ring_indices.const_data_ptr<int64_t>(),
                ring_ok.const_data_ptr<bool>(),
                prev.const_data_ptr<int64_t>(),
                backbone_means.const_data_ptr<scalar_t>(),
                backbone_sdev.const_data_ptr<scalar_t>(),
                sugar_means.const_data_ptr<scalar_t>(),
                chi_means.const_data_ptr<scalar_t>(),
                sdev_sugar.const_data_ptr<scalar_t>(),
                sdev_chi.const_data_ptr<scalar_t>(),
                well_pucker.const_data_ptr<scalar_t>(),
                well_alpha_gamma.const_data_ptr<scalar_t>(),
                well_bibii_pucker.const_data_ptr<scalar_t>(),
                well_alphanext_bibii.const_data_ptr<scalar_t>(),
                well_chi_syn.const_data_ptr<scalar_t>(),
                is_north.const_data_ptr<bool>(),
                weight_bb.const_data_ptr<scalar_t>(),
                weight_chi.const_data_ptr<scalar_t>(),
                weight_sugar.const_data_ptr<scalar_t>(),
                scalar_t(pucker_temperature),
                scalar_t(bin_blend_sdev),
                n_poses,
                max_n_blocks,
                int(coords.size(0)),
                n_bases,
                nullptr,
                output.data_ptr<scalar_t>());
        } else {
          for (int block = 0; block < n_poses * max_n_blocks; ++block)
            na_torsion_block<scalar_t, tmol::Device::CPU, true>(
                block,
                coords.const_data_ptr<scalar_t>(),
                base.const_data_ptr<int64_t>(),
                is_na.const_data_ptr<bool>(),
                torsion_indices.const_data_ptr<int64_t>(),
                torsion_ok.const_data_ptr<bool>(),
                ring_indices.const_data_ptr<int64_t>(),
                ring_ok.const_data_ptr<bool>(),
                prev.const_data_ptr<int64_t>(),
                backbone_means.const_data_ptr<scalar_t>(),
                backbone_sdev.const_data_ptr<scalar_t>(),
                sugar_means.const_data_ptr<scalar_t>(),
                chi_means.const_data_ptr<scalar_t>(),
                sdev_sugar.const_data_ptr<scalar_t>(),
                sdev_chi.const_data_ptr<scalar_t>(),
                well_pucker.const_data_ptr<scalar_t>(),
                well_alpha_gamma.const_data_ptr<scalar_t>(),
                well_bibii_pucker.const_data_ptr<scalar_t>(),
                well_alphanext_bibii.const_data_ptr<scalar_t>(),
                well_chi_syn.const_data_ptr<scalar_t>(),
                is_north.const_data_ptr<bool>(),
                weight_bb.const_data_ptr<scalar_t>(),
                weight_chi.const_data_ptr<scalar_t>(),
                weight_sugar.const_data_ptr<scalar_t>(),
                scalar_t(pucker_temperature),
                scalar_t(bin_blend_sdev),
                n_poses,
                max_n_blocks,
                int(coords.size(0)),
                n_bases,
                derivatives.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>());
        }
      });
  return {output, derivatives};
}

tmol::native::Tensor na_torsion_pose_score_backward_cpu(
    tmol::native::Tensor derivatives, tmol::native::Tensor grad_output) {
  TORCH_CHECK(
      derivatives.device().is_cpu(), "NA torsion derivatives must be CPU");
  TORCH_CHECK(
      grad_output.device().is_cpu(), "NA torsion output gradients must be CPU");
  TORCH_CHECK(
      derivatives.dim() == 3 && derivatives.size(0) == 2,
      "expected derivatives shaped [2, n_atoms, 3]");
  TORCH_CHECK(
      grad_output.dim() == 2 && grad_output.size(0) == 2,
      "expected output gradients shaped [2, n_poses]");
  int n_poses = int(grad_output.size(1));
  int n_atoms = int(derivatives.size(1));
  TORCH_CHECK(n_atoms % n_poses == 0, "atoms must divide evenly among poses");
  auto grad_coords = tmol::native::empty({n_atoms, 3}, derivatives.options());
  AT_DISPATCH_FLOATING_TYPES(
      derivatives.scalar_type(), "na_torsion_pose_score_backward_cpu", [&] {
        for (int index = 0; index < n_atoms * 3; ++index)
          na_torsion_weighted_derivative<scalar_t>(
              index,
              derivatives.const_data_ptr<scalar_t>(),
              grad_output.const_data_ptr<scalar_t>(),
              grad_output.stride(0),
              grad_output.stride(1),
              n_poses,
              n_atoms,
              grad_coords.data_ptr<scalar_t>());
      });
  return grad_coords;
}

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
