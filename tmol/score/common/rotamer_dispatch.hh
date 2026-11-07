#pragma once

// Caller-owned stages for canonical sparse rotamer-pair dispatch. The same
// contexts and scientific bodies serve Torch, JAX FFI and Metal. Execution
// adapters own allocation, exclusive scans and the explicit overflow/total
// host barrier before allocating [3, total] indices. Counts/overflow start at
// zero; CPU sphere buffers start at zero for absent residue types.
#include <tmol/score/common/sphere_overlap_ctx.hh>
#include <tmol/score/common/sphere_overlap_kernels.hh>
#include <tmol/score/common/counting.hh>
#include <stdexcept>

namespace tmol::score::common::sphere_overlap {
struct RotamerPairWindow {
  int max_n_blocks, block_pairs_per_pose, candidate_begin, n_candidates;
};
inline RotamerPairWindow rotamer_pair_window(
    int n_poses, int max_n_blocks, int64_t begin = 0, int64_t count = -1) {
  int const per_pose = checked_triangular_size(
      max_n_blocks, true, "rotamer block-pair dispatch candidates per pose");
  int const total = checked_dispatch_product(
      n_poses, per_pose, "rotamer block-pair dispatch candidates");
  if (count < 0) count = int64_t(total) - begin;
  // Subtraction avoids overflow when an untrusted window has huge bounds.
  if (begin < 0 || begin > total || count < 0 || count > total - begin)
    throw std::out_of_range("rotamer block-pair dispatch window is invalid");
  return {
      max_n_blocks,
      per_pose,
      int(begin),
      checked_dispatch_size(count, "rotamer block-pair dispatch window")};
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Int,
    typename Context,
    typename Ctx>
void launch_rotamer_spheres(Context& mgr, Ctx const& ctx) {
  LAUNCH_BOX_32;
  int const n = checked_dispatch_size(
      ctx.rot_coord_offset.size(0), "rotamer dispatch stage");
  auto run = ([=] TMOL_DEVICE_FUNC(int cta) {
    compute_rot_spheres_body<Ops, D, Real, Int, launch_t>(ctx, cta);
  });
  Ops<D>::template foreach_independent_workgroup<launch_t>(mgr, n, run);
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Int,
    typename Context,
    typename Ctx>
void launch_blocks_from_rotamer_spheres(Context& mgr, Ctx const& ctx) {
  LAUNCH_BOX_32;
  int const n = checked_dispatch_size(
      checked_dispatch_product(
          ctx.n_rots_for_block.size(0),
          ctx.n_rots_for_block.size(1),
          "rotamer block-sphere dispatch"),
      "rotamer dispatch stage");
  auto run = ([=] TMOL_DEVICE_FUNC(int ind) {
    compute_block_spheres_from_rot_body<D, Real, Int>(ctx, ind);
  });
  Ops<D>::template forall_independent<launch_t>(mgr, n, run);
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Int,
    typename Context,
    typename Ctx>
void launch_rotamer_pair_counts(Context& mgr, Ctx const& ctx) {
  LAUNCH_BOX_32;
  int const n =
      checked_dispatch_size(ctx.pair_counts.size(0), "rotamer dispatch stage");
  auto run = ([=] TMOL_DEVICE_FUNC(int candidate) {
    rot_pair_counts_body<D, Real, Int>(ctx, candidate);
  });
  Ops<D>::template forall_independent<launch_t>(mgr, n, run);
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Int,
    typename Context,
    typename Ctx>
void launch_rotamer_pair_fill(Context& mgr, Ctx const& ctx) {
  LAUNCH_BOX_32;
  int const n =
      checked_dispatch_size(ctx.pair_counts.size(0), "rotamer dispatch stage");
  auto run = ([=] TMOL_DEVICE_FUNC(int candidate) {
    rot_pair_fill_body<D, Real, Int>(ctx, candidate);
  });
  Ops<D>::template forall_independent<launch_t>(mgr, n, run);
}

}  // namespace tmol::score::common::sphere_overlap
