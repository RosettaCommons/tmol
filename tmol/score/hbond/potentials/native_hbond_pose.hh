#pragma once
#include <type_traits>
#include <tmol/score/hbond/potentials/hbond_ctx_tensor.hh>
#include <tmol/score/hbond/potentials/hbond_kernels.hh>

namespace tmol::score::hbond::potentials {

inline constexpr double hbond_neighbor_reach() { return 5.5; }
// Named callable keeps CUDA instantiation independent of a runtime callback's
// local lambda type, while retaining the canonical workgroup body.
template <
    template <Device> class DeviceDispatch,
    Device Dev,
    typename Real,
    typename Int,
    typename Ctx,
    common::TilePairMode Mode,
    typename DerivTag>
struct HBondPoseEvaluator {
  Ctx ctx;
  DerivTag derivative;
  TMOL_DEVICE_FUNC void operator()(int cta) const {
    LAUNCH_BOX_32;
    SHARED_MEMORY HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared;
    hbond_pose_eval_energies<DeviceDispatch, Dev, Real, Int, launch_t, Mode>(
        ctx, shared, cta, derivative);
  }
};

// Allocation-free launch policy shared by framework bindings. launch_candidates
// receives the canonical workgroup functor and owns only candidate execution.
template <
    template <Device> class DeviceDispatch,
    Device Dev,
    typename Real,
    typename Int,
    typename Ctx,
    typename LaunchCandidates>
void launch_hbond_pose(
    Ctx ctx,
    int n_poses,
    int n_rots,
    int compact_neighbor_capacity,
    bool allow_split_pairs,
    LaunchCandidates launch_energies) {
  using PairMode = common::TilePairMode;
  bool const accumulate_derivs = ctx.accumulate_derivs;
  auto launch = [&](auto derivative, auto mode) {
    using Eval = HBondPoseEvaluator<
        DeviceDispatch,
        Dev,
        Real,
        Int,
        Ctx,
        decltype(mode)::value,
        decltype(derivative)>;
    launch_energies(Eval{ctx, derivative});
  };
  using Both = std::integral_constant<PairMode, PairMode::InterAndIntra>;
#ifdef __NVCC__
  bool const split_pairs = allow_split_pairs && !ctx.output_block_pair_energies
                           && ctx.use_shared_compact_block_neighbors
                           && ctx.max_n_blocks > 1
                           && int64_t(n_poses) * ctx.max_n_blocks >= 4096
                           && compact_neighbor_capacity >= (1 << 15);
  // Preserve the existing combined small-stack path and large score-only
  // specialization; splitting only pays off for large derivative workloads.
  if (accumulate_derivs || n_rots < 512) {
    if (accumulate_derivs && split_pairs) {
      launch(
          std::true_type{},
          std::integral_constant<PairMode, PairMode::Inter>{});
      launch(
          std::true_type{},
          std::integral_constant<PairMode, PairMode::Intra>{});
    } else {
      launch(accumulate_derivs, Both{});
    }
  } else {
    launch(std::false_type{}, Both{});
  }
#else
  launch(accumulate_derivs, Both{});
#endif
}
// Sparse candidates and prepared donor/acceptor base geometry are caller-owned.
// Preserve compute_derivs launch policy independently of whether the forward
// call accumulates derivatives: block-pair outputs use a separate backward.
template <
    bool Backward,
    template <Device> class DeviceDispatch,
    Device Dev,
    typename Real,
    typename Int,
    typename Context,
    typename Ctx>
void launch_hbond_rotamer(Context& mgr, Ctx ctx, bool compute_derivs = false) {
  LAUNCH_BOX_32;
  auto evaluate = [ctx] TMOL_DEVICE_FUNC(int cta) {
    SHARED_MEMORY HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared;
    if constexpr (Backward) {
      hbond_rotamer_eval_derivs<DeviceDispatch, Dev, Real, Int, launch_t>(
          ctx, shared, cta, WorkgroupLane<Dev>{});
    } else {
      if (ctx.accumulate_derivs)
        hbond_rotamer_eval_energies<
            DeviceDispatch,
            Dev,
            Real,
            Int,
            launch_t,
            true>(ctx, shared, cta);
      else
        hbond_rotamer_eval_energies<
            DeviceDispatch,
            Dev,
            Real,
            Int,
            launch_t,
            false>(ctx, shared, cta);
    }
  };
  if (Backward || compute_derivs)
    DeviceDispatch<Dev>::template foreach_workgroup<launch_t>(
        mgr, ctx.dispatch_indices.size(1), evaluate);
  else
    DeviceDispatch<Dev>::template foreach_independent_workgroup<launch_t>(
        mgr, ctx.dispatch_indices.size(1), evaluate);
}

}  // namespace tmol::score::hbond::potentials
