#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/common/sphere_overlap_dispatch.hh>
#include <tmol/score/ljlk/potentials/ljlk_kernels.hh>

// Independent LJ/LK scoring shares the complete native physics without
// requiring electrostatic parameters. Bindings supply zeroed
// outputs/derivatives and, on CPU, zeroed double scratch of size [n_poses, 3]
// for reproducible accumulation.
namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {
template <typename Real, typename Int, tmol::Device D, int OutRank = 4>
using LJLKCtx = LJLKCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    OutRank>;

template <
    bool Grad,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int,
    typename Launch>
struct LJLKPoseEvaluator {
  LJLKCtx<Real, Int, D> ctx;
  double* cpu_pose_accum;
  TMOL_DEVICE_FUNC void operator()(
      int cta, int shard = 0, int partitions = 1) const {
    SHARED_MEMORY LJLKSharedMem<Real, ljlk_tile_size, ljlk_max_n_conn, Launch>
        shared;
    if constexpr (Grad)
      ljlk_eval_energies<Ops, D, Real, Int, Launch>(
          ctx, shared, cta, cpu_pose_accum, {}, shard, partitions);
    else
      ljlk_eval_energies_by_block<Ops, D, Real, Int, Launch>(
          ctx, shared, cta, cpu_pose_accum, {}, shard, partitions);
  }
};

template <
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int>
void ljlk_pose_forward_native(
    ContextManager& mgr,
    LJLKCtx<Real, Int, D> ctx,
    TView<Int, 1, D> neighbors,
    int n_poses,
    double* cpu_pose_accum) {
  LAUNCH_BOX_32;
  LAUNCH_BOX_32_OCC_AS(launch_high, 32);
  bool const implicit_self = ctx.max_n_blocks == 1 && neighbors.size(0) == 0;
  if (implicit_self) ctx.use_shared_compact_block_neighbors = true;
  LJLKPoseEvaluator<false, Ops, D, Real, Int, launch_t> energy{
      ctx, cpu_pose_accum};
  LJLKPoseEvaluator<true, Ops, D, Real, Int, launch_t> deriv{
      ctx, cpu_pose_accum};
  bool const gradients =
      ctx.require_gradient && !ctx.output_block_pair_energies;
  if constexpr (D == tmol::Device::CUDA) {
    if (!ctx.output_block_pair_energies) {
      int capacity = implicit_self ? n_poses : neighbors.size(0) - 1;
      int partitions = common::sphere_overlap::small_pair_partitions(
          capacity, ctx.block_type_atom_types.size(1));
      if (partitions > 1 || implicit_self) {
        bool handled =
            gradients
                ? common::sphere_overlap::
                      launch_partitioned_block_neighbors<Ops, D, launch_t>(
                          mgr,
                          neighbors,
                          n_poses,
                          ctx.max_n_blocks,
                          partitions,
                          deriv)
                : common::sphere_overlap::
                      launch_partitioned_block_neighbors<Ops, D, launch_t>(
                          mgr,
                          neighbors,
                          n_poses,
                          ctx.max_n_blocks,
                          partitions,
                          energy);
        if (handled) return;
      }
    }
  }
  if (neighbors.size(0) && !ctx.output_block_pair_energies) {
    if (gradients)
      common::sphere_overlap::
          launch_precomputed_block_neighbors<Ops, D, launch_high, Int>(
              mgr, neighbors, n_poses, ctx.max_n_blocks, deriv);
    else
      common::sphere_overlap::
          launch_precomputed_block_neighbors<Ops, D, launch_high, Int>(
              mgr, neighbors, n_poses, ctx.max_n_blocks, energy);
  } else {
    int const candidates = common::checked_dispatch_product(
        n_poses, ctx.max_n_upper_triangle_inds, "LJ/LK block-pair dispatch");
    bool const high =
        !ctx.output_block_pair_energies && candidates >= (1 << 14);
    if (gradients && high)
      Ops<D>::template foreach_pose_workgroup<launch_high>(
          mgr, n_poses, ctx.max_n_upper_triangle_inds, deriv);
    else if (gradients)
      Ops<D>::template foreach_pose_workgroup<launch_t>(
          mgr, n_poses, ctx.max_n_upper_triangle_inds, deriv);
    else if (high)
      Ops<D>::template foreach_pose_workgroup<launch_high>(
          mgr, n_poses, ctx.max_n_upper_triangle_inds, energy);
    else
      Ops<D>::template foreach_pose_workgroup<launch_t>(
          mgr, n_poses, ctx.max_n_upper_triangle_inds, energy);
  }
  if constexpr (D == tmol::Device::CPU) {
    if (!ctx.output_block_pair_energies)
      for (int p = 0; p < n_poses; ++p)
        for (int term = 0; term < 3; ++term)
          ctx.output[term][p][0][0] = cpu_pose_accum[3 * p + term];
  }
}
// ctx.dV_dcoords is zeroed [1, n_atoms]; ctx.dTdV contains arbitrary
// per-term upper-triangle block-pair weights, or [3, n_pairs] for rotamers.
template <
    bool Rotamer,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int,
    typename Launch>
struct LJLKBackwardEvaluator {
  LJLKCtx<Real, Int, D, Rotamer ? 2 : 4> ctx;
  TMOL_DEVICE_FUNC void operator()(int cta) const {
    SHARED_MEMORY LJLKSharedMem<Real, ljlk_tile_size, ljlk_max_n_conn, Launch>
        shared;
    ljlk_evaluate_pair<LJLKMode::Backward, Rotamer, Ops, D, Real, Int, Launch>(
        ctx, shared, cta, static_cast<double*>(nullptr));
  }
};

template <
    bool Rotamer,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int>
void ljlk_backward_native(
    ContextManager& mgr,
    LJLKCtx<Real, Int, D, Rotamer ? 2 : 4> ctx,
    int n_poses) {
  LAUNCH_BOX_32;
  LJLKBackwardEvaluator<Rotamer, Ops, D, Real, Int, launch_t> evaluate{ctx};
  if constexpr (Rotamer)
    Ops<D>::template foreach_workgroup<launch_t>(
        mgr, ctx.dispatch_indices.size(1), evaluate);
  else
    Ops<D>::template foreach_pose_workgroup<launch_t>(
        mgr, n_poses, ctx.max_n_upper_triangle_inds, evaluate);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
