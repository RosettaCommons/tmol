#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/common/sphere_overlap_dispatch.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_kernels.hh>
#include <type_traits>

// Complete pose execution of the common LK-ball physics. Bindings supply
// current-coordinate neighbors, zeroed outputs and coordinate/water gradients.
namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {
template <
    typename Real,
    typename Int,
    tmol::Device D,
    int OutRank = 4,
    typename GradReal = Real,
    typename EnergyReal = Real>
using LKBallCtx = LKBallCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    OutRank,
    GradReal,
    EnergyReal>;

using tmol::portable::Vec;

template <common::TilePairMode Mode, common::TilePairMode IndexMode = Mode>
struct TilePairModeTag : std::integral_constant<common::TilePairMode, Mode> {
  static constexpr auto index_mode = IndexMode;
};

// interres_count_pair_separation and lk_ball_pose_pair_indices moved to
// lk_ball_kernels.hh, which a Metal translation unit can include.

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Launch,
    typename Eval>
void launch_lk_ball_pose_pair_workgroups(
    ContextManager& mgr, int n_poses, int max_n_blocks, Eval eval) {
  int const n_pairs = score::common::checked_triangular_size(
      max_n_blocks, true, "LK-ball block-pair dispatch");
  int const n_candidate_pairs = score::common::checked_dispatch_product(
      n_poses, n_pairs, "LK-ball block-pair dispatch");
#ifdef __NVCC__
  auto eval_all = ([=] TMOL_DEVICE_FUNC(int cta) {
    eval(cta, TilePairModeTag<common::TilePairMode::InterAndIntra>{});
  });
  // The specialized kernels remove the cold intra/inter instruction path.
  // Use them only once their throughput gain exceeds the extra launch cost.
  constexpr int min_split_workgroups = 1 << 15;
  if (max_n_blocks > 1 && n_candidate_pairs >= min_split_workgroups) {
    auto eval_interres = ([=] TMOL_DEVICE_FUNC(int cta) {
      eval(cta, TilePairModeTag<common::TilePairMode::Inter>{});
    });
    auto eval_intrares = ([=] TMOL_DEVICE_FUNC(int cta) {
      eval(cta, TilePairModeTag<common::TilePairMode::Intra>{});
    });
    DeviceDispatch<Dev>::template foreach_pose_workgroup<Launch>(
        mgr, n_poses, n_pairs - max_n_blocks, eval_interres);
    DeviceDispatch<Dev>::template foreach_pose_workgroup<Launch>(
        mgr, n_poses, max_n_blocks, eval_intrares);
    return;
  }
  DeviceDispatch<Dev>::template foreach_pose_workgroup<Launch>(
      mgr, n_poses, n_pairs, eval_all);
#else
  DeviceDispatch<Dev>::template foreach_pose_workgroup<Launch>(
      mgr, n_poses, n_pairs, eval);
#endif
}

#ifdef __NVCC__
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Launch,
    typename Int,
    typename Eval>
void launch_lk_ball_compact_pose_pair_workgroups(
    ContextManager& mgr,
    TView<Int, 1, Dev> neighbors,
    int n_poses,
    int max_n_blocks,
    Eval eval,
    bool allow_split_pairs = true) {
  using common::TilePairMode;
  constexpr int min_split_workgroups = 1 << 15;
  if (allow_split_pairs && max_n_blocks > 1
      && neighbors.size(0) - 1 >= min_split_workgroups) {
    // Keep both passes on the original list, including caller-provided
    // subsets. Specialize the scoring body while retaining its triangular
    // candidate IDs, so each pass excludes the other kind of pair.
    auto eval_inter = ([=] TMOL_DEVICE_FUNC(int cta) {
      eval(
          cta,
          TilePairModeTag<TilePairMode::Inter, TilePairMode::InterAndIntra>{});
    });
    auto eval_intra = ([=] TMOL_DEVICE_FUNC(int cta) {
      eval(
          cta,
          TilePairModeTag<TilePairMode::Intra, TilePairMode::InterAndIntra>{});
    });
    common::sphere_overlap::
        launch_precomputed_block_neighbors<DeviceDispatch, Dev, Launch, Int>(
            mgr, neighbors, n_poses, max_n_blocks, eval_inter);
    common::sphere_overlap::
        launch_precomputed_block_neighbors<DeviceDispatch, Dev, Launch, Int>(
            mgr, neighbors, n_poses, max_n_blocks, eval_intra);
  } else {
    auto eval_all = ([=] TMOL_DEVICE_FUNC(int cta) {
      eval(cta, TilePairModeTag<TilePairMode::InterAndIntra>{});
    });
    common::sphere_overlap::
        launch_precomputed_block_neighbors<DeviceDispatch, Dev, Launch, Int>(
            mgr, neighbors, n_poses, max_n_blocks, eval_all);
  }
}
#endif

template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename Launch,
    typename GradReal = Real,
    typename EnergyReal = Real>
struct LKBallPoseEvaluator {
  LKBallCtx<Real, Int, Dev, 4, GradReal, EnergyReal> ctx;
  using Shared = LKBallSharedMem<
      Real,
      lk_tile_size,
      lk_max_n_water,
      lk_max_n_conn,
      Launch>;
#ifdef __NVCC__
  template <typename ModeTag>
  TMOL_DEVICE_FUNC void operator()(int cta, ModeTag pair_mode_tag) const {
    constexpr auto pair_mode = decltype(pair_mode_tag)::value;
    constexpr auto index_mode = decltype(pair_mode_tag)::index_mode;
#else
  void operator()(int cta) const {
    constexpr auto pair_mode = common::TilePairMode::InterAndIntra;
    constexpr auto index_mode = pair_mode;
#endif
    SHARED_MEMORY Shared shared;
    if constexpr (Backward) {
      lk_ball_eval_derivs<
          DeviceDispatch,
          Dev,
          Real,
          Int,
          Launch,
          decltype(ctx),
          Shared,
          pair_mode,
          index_mode>(ctx, shared, cta);
    } else {
      lk_ball_eval_energies_by_block<
          DeviceDispatch,
          Dev,
          Real,
          Int,
          Launch,
          decltype(ctx),
          Shared,
          pair_mode,
          index_mode>(ctx, shared, cta);
    }
  }
};

template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename Launch,
    typename GradReal = Real,
    typename EnergyReal = Real>
LKBallPoseEvaluator<
    Backward,
    DeviceDispatch,
    Dev,
    Real,
    Int,
    Launch,
    GradReal,
    EnergyReal>
lk_ball_pose_evaluator(LKBallCtx<Real, Int, Dev, 4, GradReal, EnergyReal> ctx) {
  return {ctx};
}

template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename GradReal = Real,
    typename EnergyReal = Real>
void lk_ball_pose_native(
    ContextManager& mgr,
    LKBallCtx<Real, Int, Dev, 4, GradReal, EnergyReal> ctx,
    TView<Int, 1, Dev> compact_neighbors,
    int n_poses,
    bool allow_split_pairs = true) {
  LAUNCH_BOX_32_OCC(16);
  auto eval = lk_ball_pose_evaluator<
      Backward,
      DeviceDispatch,
      Dev,
      Real,
      Int,
      launch_t>(ctx);
  bool const use_compact = compact_neighbors.size(0)
                           && (Backward || !ctx.output_block_pair_energies);
#ifdef __NVCC__
  if (use_compact) {
    launch_lk_ball_compact_pose_pair_workgroups<
        DeviceDispatch,
        Dev,
        launch_t,
        Int>(
        mgr,
        compact_neighbors,
        n_poses,
        ctx.max_n_blocks,
        eval,
        allow_split_pairs);
  } else if constexpr (Backward) {
    int const n_candidates = common::checked_dispatch_product(
        n_poses, ctx.n_block_pairs, "LK-ball derivative block-pair dispatch");
    constexpr int max_workgroups = 1 << 14;
    int const n_workgroups =
        n_candidates < max_workgroups ? n_candidates : max_workgroups;
    auto strided = [=] TMOL_DEVICE_FUNC(int cta) {
      for (int i = cta; i < n_candidates; i += n_workgroups)
        eval(i, TilePairModeTag<common::TilePairMode::InterAndIntra>{});
    };
    DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
        mgr, n_workgroups, strided);
  } else {
    launch_lk_ball_pose_pair_workgroups<DeviceDispatch, Dev, launch_t>(
        mgr, n_poses, ctx.max_n_blocks, eval);
  }
#else
  if (use_compact) {
    common::sphere_overlap::
        launch_precomputed_block_neighbors<DeviceDispatch, Dev, launch_t, Int>(
            mgr, compact_neighbors, n_poses, ctx.max_n_blocks, eval);
  } else {
    launch_lk_ball_pose_pair_workgroups<DeviceDispatch, Dev, launch_t>(
        mgr, n_poses, ctx.max_n_blocks, eval);
  }
#endif
}

// Sparse dispatch consumes prepared candidate/water coordinates. Both outputs
// start at zero; weights view is [4,n_pairs,1,1] over rank-2 upstream storage.
// Keep the mature sparse CUDA derivative accumulation policy independent of
// the whole-pose shared-scratch optimization.
template <
    bool Backward,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int,
    typename GradReal = Real,
    typename EnergyReal = Real>
void lk_ball_rotamer_native(
    ContextManager& mgr, LKBallCtx<Real, Int, D, 2, GradReal, EnergyReal> ctx) {
  LAUNCH_BOX_32;
  auto evaluate = [ctx] TMOL_DEVICE_FUNC(int cta) {
    using Shared = LKBallSharedMem<
        Real,
        lk_tile_size,
        lk_max_n_water,
        lk_max_n_conn,
        launch_t>;
    SHARED_MEMORY Shared shared;
    if constexpr (Backward)
      lk_ball_eval_derivs<
          Ops,
          D,
          Real,
          Int,
          launch_t,
          decltype(ctx),
          Shared,
          common::TilePairMode::InterAndIntra,
          common::TilePairMode::InterAndIntra,
          true>(ctx, shared, cta);
    else
      lk_ball_eval_rotamer_energies_by_block<Ops, D, Real, Int, launch_t>(
          ctx, shared, cta);
  };
  if constexpr (Backward)
    Ops<D>::template foreach_workgroup<launch_t>(
        mgr, ctx.dispatch_indices.size(1), evaluate);
  else
    Ops<D>::template foreach_independent_workgroup<launch_t>(
        mgr, ctx.dispatch_indices.size(1), evaluate);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
