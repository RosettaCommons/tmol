#include <tmol/utility/tensor/native_library.hh>

#include "na_torsion_pose_score.hh"

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {

using tmol::native::Tensor;
using tmol::native::autograd::AutogradContext;
using tmol::native::autograd::Function;
using tmol::native::autograd::tensor_list;

class NaTorsionPoseScoreOp
    : public tmol::native::autograd::Function<NaTorsionPoseScoreOp> {
 public:
  static constexpr bool capture_pose_gradient = true;
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor base,
      Tensor is_na,
      Tensor torsion_indices,
      Tensor torsion_ok,
      Tensor ring_indices,
      Tensor ring_ok,
      Tensor prev,
      Tensor backbone_means,
      Tensor backbone_sdev,
      Tensor sugar_means,
      Tensor chi_means,
      Tensor sdev_sugar,
      Tensor sdev_chi,
      Tensor well_pucker,
      Tensor well_alpha_gamma,
      Tensor well_bibii_pucker,
      Tensor well_alphanext_bibii,
      Tensor well_chi_syn,
      Tensor is_north,
      Tensor weight_bb,
      Tensor weight_chi,
      Tensor weight_sugar,
      double pucker_temperature,
      double bin_blend_sdev) {
    auto score_kernel = &na_torsion_pose_score_cpu;
#ifdef WITH_CUDA
    if (coords.is_cuda()) score_kernel = &na_torsion_pose_score_cuda;
#endif
    auto result = score_kernel(
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
        coords.requires_grad(),
        Tensor());
    auto score = std::get<0>(result);
    auto derivatives = std::get<1>(result);
    ctx->save_for_backward({derivatives});
    ctx->mark_non_differentiable({derivatives});
    return {score, derivatives};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto backward_kernel = &na_torsion_pose_score_backward_cpu;
#ifdef WITH_CUDA
    if (saved[0].is_cuda())
      backward_kernel = &na_torsion_pose_score_backward_cuda;
#endif
    auto grad_coords = backward_kernel(saved[0], grad_outputs[0]);
    return {
        grad_coords, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        Tensor(),    Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        Tensor(),    Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        Tensor(),    Tensor(), Tensor(), Tensor(),
    };
  }
};

std::tuple<Tensor, Tensor> na_torsion_pose_score(
    Tensor coords,
    Tensor base,
    Tensor is_na,
    Tensor torsion_indices,
    Tensor torsion_ok,
    Tensor ring_indices,
    Tensor ring_ok,
    Tensor prev,
    Tensor backbone_means,
    Tensor backbone_sdev,
    Tensor sugar_means,
    Tensor chi_means,
    Tensor sdev_sugar,
    Tensor sdev_chi,
    Tensor well_pucker,
    Tensor well_alpha_gamma,
    Tensor well_bibii_pucker,
    Tensor well_alphanext_bibii,
    Tensor well_chi_syn,
    Tensor is_north,
    Tensor weight_bb,
    Tensor weight_chi,
    Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev) {
  auto result = NaTorsionPoseScoreOp::apply(
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
      bin_blend_sdev);
  return {result[0], result[1]};
}

std::tuple<Tensor, Tensor> na_torsion_native_score(
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
    Tensor entry_weights,
    bool compute_derivs) {
  auto kernel = &na_torsion_pose_score_cpu;
#ifdef WITH_CUDA
  if (coords.is_cuda()) kernel = &na_torsion_pose_score_cuda;
#endif
  return kernel(
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
      compute_derivs ? entry_weights : Tensor());
}

NARotamerTopologyOutput na_rotamer_topology(
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
  static ContextManager mgr;
#ifdef WITH_CUDA
  if (types.is_cuda())
    return na_rotamer_topology_cuda(
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
#endif
  TORCH_CHECK(
      types.device().is_cpu(),
      "NA topology requires CPU or CUDA input buffers");
  return na_rotamer_topology_cpu(
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

TMOL_LIBRARY(tmol_na_torsion, m) {
  m.def("na_rotamer_topology", TMOL_FN(&na_rotamer_topology));
  m.def("na_torsion_native_score", TMOL_FN(&na_torsion_native_score));
  m.def("na_torsion_pose_score", TMOL_FN(&na_torsion_pose_score));
}

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
