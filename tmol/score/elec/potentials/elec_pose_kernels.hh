#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/common/sphere_overlap_dispatch.hh>
#include <tmol/score/elec/potentials/elec_kernels.hh>

// Independent electrostatics needs only electrostatic parameters. Bindings
// own zeroed outputs/derivatives and current-coordinate neighbor preparation.
namespace tmol {
namespace score {
namespace elec {
namespace potentials {
template <typename Real, typename Int, tmol::Device D, int OutRank = 4>
using ElecCtx = ElecCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    OutRank>;
using tmol::portable::Vec;
template <
    int Mode,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int,
    typename Launch>
struct ElecPoseEvaluator {
  ElecCtx<Real, Int, D> ctx;
  TMOL_DEVICE_FUNC void operator()(int cta) const {
    SHARED_MEMORY ElecSharedMem<Real, elec_tile_size, elec_max_n_conn, Launch>
        shared;
    if constexpr (Mode == 0)
      elec_eval_energies_by_block_2<Ops, D, Real, Int, Launch>(
          ctx, shared, cta);
    else if constexpr (Mode == 1)
      elec_eval_energies<Ops, D, Real, Int, Launch>(ctx, shared, cta);
    else
      elec_eval_derivs_2<Ops, D, Real, Int, Launch>(ctx, shared, cta);
  }
};

template <
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int>
void elec_pose_forward_native(
    ContextManager& mgr,
    ElecCtx<Real, Int, D> ctx,
    TView<Int, 1, D> neighbors,
    int n_poses,
    bool compute_derivs) {
  LAUNCH_BOX_32_OCC(32);
  ElecPoseEvaluator<0, Ops, D, Real, Int, launch_t> energy{ctx};
  ElecPoseEvaluator<1, Ops, D, Real, Int, launch_t> deriv{ctx};
  bool const gradients = compute_derivs && !ctx.output_block_pair_energies;
  if (neighbors.size(0) && !ctx.output_block_pair_energies) {
    if (gradients)
      common::sphere_overlap::
          launch_precomputed_block_neighbors<Ops, D, launch_t, Int>(
              mgr, neighbors, n_poses, ctx.max_n_blocks, deriv);
    else
      common::sphere_overlap::
          launch_precomputed_block_neighbors<Ops, D, launch_t, Int>(
              mgr, neighbors, n_poses, ctx.max_n_blocks, energy);
  } else if (gradients) {
    Ops<D>::template foreach_pose_workgroup<launch_t>(
        mgr, n_poses, ctx.max_n_upper_triangle_inds, deriv);
  } else {
    Ops<D>::template foreach_pose_workgroup<launch_t>(
        mgr, n_poses, ctx.max_n_upper_triangle_inds, energy);
  }
}

template <
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int>
void elec_pose_backward_native(
    ContextManager& mgr, ElecCtx<Real, Int, D> ctx, int n_poses) {
  LAUNCH_BOX_32;
  ElecPoseEvaluator<2, Ops, D, Real, Int, launch_t> deriv{ctx};
  Ops<D>::template foreach_pose_workgroup<launch_t>(
      mgr, n_poses, ctx.max_n_upper_triangle_inds, deriv);
}
}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
