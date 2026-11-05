#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {
inline int small_pair_partitions(int candidates, int max_n_atoms) {
  if (candidates <= 0) return 1;
  if (candidates == 1 && max_n_atoms >= 32) return 32;
  return candidates <= 64 ? 16 : candidates <= 1024 ? 4 : 1;
}

template <typename Int, tmol::Device D, typename Eval>
struct PartitionedNeighborEvaluator {
  TView<Int, 1, D> neighbors;
  Eval evaluate;
  int tasks, workgroups, partitions;
  bool implicit_self;
  TMOL_DEVICE_FUNC void operator()(int group) const {
    for (int task = group; task < tasks; task += workgroups) {
      int index = task / partitions;
      int candidate = index;
      if (!implicit_self) {
        if (index >= neighbors[0]) return;
        candidate = neighbors[index + 1];
      }
      evaluate(candidate, task % partitions, partitions);
    }
  }
};

// Partition atom-pair lanes while retaining one canonical candidate list.
// An empty view denotes implicit self pairs only for a single-block layout;
// an explicit list with count zero continues to mean no interactions.
template <
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Launch,
    typename Int,
    typename Eval>
bool launch_partitioned_block_neighbors(
    ContextManager& mgr,
    TView<Int, 1, D> neighbors,
    int n_poses,
    int max_n_blocks,
    int partitions,
    Eval evaluate) {
  bool implicit = max_n_blocks == 1 && neighbors.size(0) == 0;
  if (!implicit && neighbors.size(0) == 0) return false;
  int capacity = implicit ? n_poses : neighbors.size(0) - 1;
  if (capacity <= 0) return true;
  int tasks = checked_dispatch_product(
      capacity, partitions, "partitioned atom-pair dispatch");
  int groups = tasks < (1 << 14) ? tasks : (1 << 14);
  PartitionedNeighborEvaluator<Int, D, Eval> work{
      neighbors, evaluate, tasks, groups, partitions, implicit};
  Ops<D>::template foreach_workgroup<Launch>(mgr, groups, work);
  return true;
}

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
