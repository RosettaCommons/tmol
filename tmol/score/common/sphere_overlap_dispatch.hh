#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Launch,
    typename Int,
    typename Eval>
void launch_precomputed_block_neighbors(
    ContextManager& mgr,
    TView<Int, 1, D> neighbor_indices,
    int n_poses,
    int max_n_blocks,
    Eval eval) {
  int const n_candidates = neighbor_indices.size(0) - 1;
  if (n_candidates <= 0) return;
#ifdef __NVCC__
  constexpr int normal_workgroups = 1 << 14;
  int const n_workgroups =
      n_candidates < normal_workgroups ? n_candidates : normal_workgroups;
  auto eval_compact = ([=] TMOL_DEVICE_FUNC(int cta) {
    for (int index = cta; index < n_candidates; index += n_workgroups) {
      if (index >= neighbor_indices[0]) return;
      eval(neighbor_indices[index + 1]);
    }
  });
  DeviceDispatch<D>::template foreach_workgroup<Launch>(
      mgr, n_workgroups, eval_compact);
#else
  int const n_neighbors = neighbor_indices[0];
  bool ordered = n_poses > 1;
  for (int index = 2; ordered && index <= n_neighbors; ++index) {
    ordered = neighbor_indices[index - 1] <= neighbor_indices[index];
  }
  if (ordered) {
    int const pairs_per_pose = common::checked_triangular_size(
        max_n_blocks, true, "compact CPU block-pair dispatch");
    common::checked_dispatch_product(
        n_poses, pairs_per_pose, "compact CPU block-pair dispatch");
    // CPU neighbor construction retains ascending candidate IDs. Partition
    // this list by pose: coordinate/energy outputs are disjoint across poses,
    // while each pose keeps the original pair accumulation order. Explicit
    // caller-provided lists with another order retain the serial fallback.
    auto lower_bound = ([=](int candidate) {
      int first = 1;
      int last = n_neighbors + 1;
      while (first < last) {
        int const middle = first + (last - first) / 2;
        if (neighbor_indices[middle] < candidate) {
          first = middle + 1;
        } else {
          last = middle;
        }
      }
      return first;
    });
    auto eval_pose = ([=](int pose) {
      int const first = lower_bound(pose * pairs_per_pose);
      int const last = lower_bound((pose + 1) * pairs_per_pose);
      for (int index = first; index < last; ++index) {
        eval(neighbor_indices[index]);
      }
    });
    DeviceDispatch<D>::template foreach_pose_workgroup<Launch>(
        mgr, n_poses, 1, eval_pose);
  } else {
    auto eval_compact = ([=] TMOL_DEVICE_FUNC(int index) {
      eval(neighbor_indices[index + 1]);
    });
    DeviceDispatch<D>::template foreach_workgroup<Launch>(
        mgr, n_neighbors, eval_compact);
  }
#endif
}

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
