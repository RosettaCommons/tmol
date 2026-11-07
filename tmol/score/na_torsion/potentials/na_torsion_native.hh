#pragma once
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
namespace tmol::score::na_torsion::potentials {
// Caller-owned contiguous structural/parameter arrays. Atom indices use Int;
// cotangent strides count Real elements. Outputs are zero initialized by
// caller. Weighted=true requires [2,n_blocks] cotangents and a single
// [n_atoms,3] output. CPU launch is serial because entries can share background
// atoms.
template <typename Real, typename Int = int64_t>
struct NATorsionContext {
  Real const* coords;
  Int const* base;
  bool const* is_na;
  Int const* torsion_indices;
  bool const* torsion_ok;
  Int const* ring_indices;
  bool const* ring_ok;
  Int const* prev;
  Real const* backbone_means;
  Real const* backbone_sdev;
  Real const* sugar_means;
  Real const* chi_means;
  Real const* sdev_sugar;
  Real const* sdev_chi;
  Real const* well_pucker;
  Real const* well_alpha_gamma;
  Real const* well_bibii_pucker;
  Real const* well_alphanext_bibii;
  Real const* well_chi_syn;
  bool const* is_north;
  Real const* weight_bb;
  Real const* weight_chi;
  Real const* weight_sugar;
  Real pucker_temperature;
  Real bin_blend_sdev;
  int n_poses;
  int max_n_blocks;
  int n_atoms;
  int n_bases;
  Real* derivatives;
  Real* output;
  Real const* entry_weights;
  int64_t weight_stride_score;
  int64_t weight_stride_entry;
};
template <
    bool Derivatives,
    bool Weighted,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Real,
    typename Int>
void launch_na_torsion_native(
    ContextManager& mgr, NATorsionContext<Real, Int> ctx) {
  LAUNCH_BOX_128;
  auto eval = [ctx] TMOL_DEVICE_FUNC(int block) {
    na_torsion_block<Real, D, Derivatives, Int, Weighted>(
        block,
        ctx.coords,
        ctx.base,
        ctx.is_na,
        ctx.torsion_indices,
        ctx.torsion_ok,
        ctx.ring_indices,
        ctx.ring_ok,
        ctx.prev,
        ctx.backbone_means,
        ctx.backbone_sdev,
        ctx.sugar_means,
        ctx.chi_means,
        ctx.sdev_sugar,
        ctx.sdev_chi,
        ctx.well_pucker,
        ctx.well_alpha_gamma,
        ctx.well_bibii_pucker,
        ctx.well_alphanext_bibii,
        ctx.well_chi_syn,
        ctx.is_north,
        ctx.weight_bb,
        ctx.weight_chi,
        ctx.weight_sugar,
        ctx.pucker_temperature,
        ctx.bin_blend_sdev,
        ctx.n_poses,
        ctx.max_n_blocks,
        ctx.n_atoms,
        ctx.n_bases,
        ctx.derivatives,
        ctx.output,
        ctx.entry_weights,
        ctx.weight_stride_score,
        ctx.weight_stride_entry);
  };
  Ops<D>::template forall<launch_t>(mgr, ctx.n_poses * ctx.max_n_blocks, eval);
}
}  // namespace tmol::score::na_torsion::potentials
