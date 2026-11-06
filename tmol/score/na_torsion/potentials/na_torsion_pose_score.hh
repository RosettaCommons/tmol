#include <tmol/utility/tensor/native.hh>
#pragma once
#include <tmol/utility/tensor/context_manager.hh>

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
    tmol::native::Tensor entry_weights);

tmol::native::Tensor na_torsion_pose_score_backward_cpu(
    tmol::native::Tensor derivatives, tmol::native::Tensor grad_output);

std::tuple<tmol::native::Tensor, tmol::native::Tensor>
na_torsion_pose_score_cuda(
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
    tmol::native::Tensor entry_weights);

tmol::native::Tensor na_torsion_pose_score_backward_cuda(
    tmol::native::Tensor derivatives, tmol::native::Tensor grad_output);

using NARotamerTopologyOutput = std::tuple<
    tmol::native::Tensor,
    tmol::native::Tensor,
    tmol::native::Tensor,
    tmol::native::Tensor,
    tmol::native::Tensor,
    tmol::native::Tensor,
    tmol::native::Tensor,
    tmol::native::Tensor>;

NARotamerTopologyOutput na_rotamer_topology_cpu(
    tmol::ContextManager& mgr,
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
    tmol::native::Tensor bt_down);
NARotamerTopologyOutput na_rotamer_topology_cuda(
    tmol::ContextManager& mgr,
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
    tmol::native::Tensor bt_down);
}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
