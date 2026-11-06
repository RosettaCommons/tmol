#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/na_torsion/potentials/na_rotamer_topology.impl.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_score.impl.hh>
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
    bool compute_derivs,
    tmol::native::Tensor entry_weights) {
  TORCH_CHECK(coords.device().is_cpu(), "NA execution device mismatch");
  static ContextManager mgr;
  return na_torsion_score_dispatch<
      tmol::score::common::DeviceOperations,
      tmol::Device::CPU>(
      mgr,
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
      pucker_temperature,
      bin_blend_sdev,
      compute_derivs,
      entry_weights);
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

NARotamerTopologyOutput na_rotamer_topology_cpu(
    ContextManager& mgr,
    tmol::native::Tensor types,
    tmol::native::Tensor poses,
    tmol::native::Tensor blocks,
    tmol::native::Tensor offsets,
    tmol::native::Tensor first,
    tmol::native::Tensor first_type,
    tmol::native::Tensor connections,
    tmol::native::Tensor downstream,
    tmol::native::Tensor bt_base,
    tmol::native::Tensor bt_uaids,
    tmol::native::Tensor bt_ring,
    tmol::native::Tensor bt_down) {
  return na_rotamer_topology_dispatch<
      tmol::score::common::DeviceOperations,
      tmol::Device::CPU>(
      mgr,
      types,
      poses,
      blocks,
      offsets,
      first,
      first_type,
      connections,
      downstream,
      bt_base,
      bt_uaids,
      bt_ring,
      bt_down);
}
}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
