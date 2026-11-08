#pragma once

#include <tmol/score/lk_ball/potentials/lk_ball_pose_kernels.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_native.hh>

namespace tmol::score::lk_ball::potentials {

// Caller owns and zeros both derivative buffers. Keep the water cotangent and
// atom sum in Accum until the complete chain rule has been applied; only the
// execution boundary converts the resulting coordinate gradient to Real.
template <
    bool Sparse,
    template <tmol::Device> class Ops,
    tmol::Device Dev,
    typename Real,
    typename Int,
    int OutRank,
    typename Accum,
    typename EnergyReal>
void lk_ball_score_water_backward(
    ContextManager& mgr,
    LKBallCtx<Real, Int, Dev, OutRank, Accum, EnergyReal> score,
    WaterGenCtx<Real, Int, Dev> water,
    TView<Int, 1, Dev> neighbors,
    int n_poses,
    bool allow_split_pairs = true) {
  if constexpr (Sparse)
    lk_ball_rotamer_native<true, Ops, Dev, Real, Int>(mgr, score);
  else
    lk_ball_pose_native<true, Ops, Dev, Real, Int>(
        mgr, score, neighbors, n_poses, allow_split_pairs);
  water_gen_backward<Ops, Dev, Real, Int>(
      mgr, water, score.dV_d_water_coords, score.dV_d_pose_coords);
}

}  // namespace tmol::score::lk_ball::potentials
