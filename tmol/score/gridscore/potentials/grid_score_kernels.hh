#pragma once

// Body of the per-atom grid-scoring kernel.
//
// Separate from grid_score.impl.hh -- which holds the host launchers and so
// reaches the torch tensor layer -- so that a Metal translation unit can
// include just the kernel. The body takes its state as a context, so the CPU
// and CUDA builds (TView-backed) and the Metal entry (MetalView-backed) run
// the same source.

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/gridscore/potentials/grid_interp_point.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

// Weighted fa_atr / fa_rep / fa_sol / fa_elec (+ the heavy-atom part of the
// gridded LK-ball total) and the out-of-bounds penalty of one grid-scored
// atom of one pose, with its Cartesian gradient. ``work`` indexes
// (pose, grid atom); every work item owns its outputs.
template <tmol::Device D, typename Real, typename Int, typename Ctx>
TMOL_HOST_DEVICE_FUNC void grid_score_atom_body(
    TMOL_THREAD Ctx const& ctx, int work) {
  using Real3 = tmol::portable::Vec<Real, 3>;
  namespace math = tmol::portable::math;

  int const pose = work / ctx.n_grid_atoms;
  int const g = work % ctx.n_grid_atoms;
  Real const voxel_spacing = ctx.voxel_spacing;
  Real const scale = Real(1) / voxel_spacing;
  Int const atom = ctx.grid_atoms[g];
  Int const type = ctx.atom_types[g];
  int const dims[3] = {
      int(ctx.atr.size(1)), int(ctx.atr.size(2)), int(ctx.atr.size(3))};

  Real3 const xyz = ctx.coords[pose][atom];
  Real position[3];
  for (int axis = 0; axis < 3; ++axis) {
    position[axis] = (xyz[axis] - ctx.origin[axis]) / voxel_spacing;
  }
  GridStencil<Real> const stencil =
      make_grid_stencil<Real>(position, dims, ctx.out_of_bounds_penalty);
  if (!stencil.valid || type < 0 || type >= ctx.n_types) {
    Real const nan = math::quiet_nan<Real>();
    ctx.energy[pose][g] = nan;
    Real3 bad;
    bad[0] = nan;
    bad[1] = nan;
    bad[2] = nan;
    ctx.gradient[pose][atom] = bad;
    return;
  }
  Real v_atr, v_rep, v_sol, v_elec, v_lk = 0;
  Real d_atr[3], d_rep[3], d_sol[3], d_elec[3], d_lk[3] = {0, 0, 0};
  contract_grid<D, Real>(ctx.atr, int(type), stencil, v_atr, d_atr);
  contract_grid<D, Real>(ctx.rep, int(type), stencil, v_rep, d_rep);
  contract_grid<D, Real>(ctx.sol, int(type), stencil, v_sol, d_sol);
  contract_grid<D, Real>(ctx.elec, 0, stencil, v_elec, d_elec);
  if (ctx.with_lk_ball) {
    contract_grid<D, Real>(ctx.lk_ball, int(type), stencil, v_lk, d_lk);
  }
  Real const charge = ctx.charges[g];
  Real const w0 = ctx.weights[0];
  Real const w1 = ctx.weights[1];
  Real const w2 = ctx.weights[2];
  Real const w3 = ctx.weights[3];
  // fa_rep is stored transformed (rep^-4 - 1): chain rule through it.
  Real const rep_slope = Real(-4) * math::pow(v_rep, Real(-5));
  ctx.energy[pose][g] = w0 * v_atr + w1 * (math::pow(v_rep, Real(-4)) - Real(1))
                        + w2 * v_sol + w3 * charge * v_elec + v_lk
                        + stencil.penalty;
  Real3 grad;
  for (int axis = 0; axis < 3; ++axis) {
    grad[axis] = scale
                 * (w0 * d_atr[axis] + w1 * rep_slope * d_rep[axis]
                    + w2 * d_sol[axis] + w3 * charge * d_elec[axis] + d_lk[axis]
                    + stencil.penalty_gradient[axis]);
  }
  ctx.gradient[pose][atom] = grad;
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
