#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/native.hh>
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap_ctx.hh>
#include <tmol/score/common/sphere_overlap_dispatch.hh>
#include <tmol/score/common/sphere_overlap_preparation.hh>
#include <tmol/score/common/sphere_overlap_kernels.hh>
#include <tmol/score/common/rotamer_dispatch.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/backends/metal/score/common/mps_launch.hh>
#ifdef WITH_MPS
#include <tmol/backends/metal/score/common/sphere_overlap_mps_kernels_source.h>
#endif
#include <tmol/score/common/upper_triangle_indices.hh>

#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

using tmol::portable::Vec;

inline bool should_compact_block_neighbors(
    int n_poses, int max_n_blocks, bool computes_derivatives) {
  // Compaction pays for one large inference pose or a wide batch of smaller
  // poses. Derivative kernels carry more useful work per CTA, so require both
  // a large pose and a genuinely wide workload.
  constexpr int min_blocks_per_pose = 256;
  constexpr int min_inference_candidates = 1 << 16;
  constexpr int min_batched_blocks_for_derivatives = 1000;
  int64_t const pairs_per_pose =
      (int64_t(max_n_blocks) * (int64_t(max_n_blocks) + 1)) / 2;
  bool const enough_inference_candidates =
      n_poses > 0
      && pairs_per_pose
             >= (min_inference_candidates + int64_t(n_poses) - 1) / n_poses;
  bool const large_inference_workload =
      max_n_blocks >= min_blocks_per_pose || enough_inference_candidates;
  if (!large_inference_workload) return false;
  return !computes_derivatives
         || (max_n_blocks >= min_blocks_per_pose
             && int64_t(n_poses) * max_n_blocks
                    >= min_batched_blocks_for_derivatives);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_rot_spheres {
  // Templated on the coordinate view so terms can move from Eigen to
  // tmol::portable::Vec independently; both spell a 3-vector of Real.
  template <typename CoordTView>
  static void f(
      ContextManager& mgr,
      CoordTView rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 2, D> rot_spheres) {
    // One context, three backends: the generic launch captures it in a device
    // lambda, the Metal launch binds its views as buffers, and both reach the
    // same body in sphere_overlap_kernels.hh.
    RotSpheresCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.rot_coords = rot_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.rot_spheres = rot_spheres;

    if (D == tmol::Device::MPS) {
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kSphereOverlapMetalShaderSource),
          "compute_rot_spheres_f32",
          ctx,
          rot_coord_offset.size(0),
          true);
    } else {
      launch_rotamer_spheres<DeviceDispatch, D, Real, Int>(mgr, ctx);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    bool ResetCount = false>
struct compute_block_spheres {
  // Templated on the coordinate view so terms can move from Eigen to
  // tmol::portable::Vec independently; both spell a 3-vector of Real.
  template <typename CoordTView>
  static void f(
      ContextManager& mgr,
      CoordTView rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Real, 3, D> block_spheres,
      Int* count_to_reset = nullptr) {
    if (D == tmol::Device::MPS) {
      // Metal cannot launch a device lambda, so the same body is reached
      // through a named entry point instead of through compute_spheres.
      BlockSpheresCtxT<
          tmol::score::common::TViewBinder<D>::template View,
          Real,
          Int>
          ctx;
      ctx.rot_coords = rot_coords;
      ctx.rot_coord_offset = rot_coord_offset;
      ctx.block_ind_for_rot = block_ind_for_rot;
      ctx.pose_ind_for_rot = pose_ind_for_rot;
      ctx.block_type_ind_for_rot = block_type_ind_for_rot;
      ctx.block_type_n_atoms = block_type_n_atoms;
      ctx.block_spheres = block_spheres;
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kSphereOverlapMetalShaderSource),
          "compute_block_spheres_f32",
          ctx,
          block_ind_for_rot.size(0),
          true);
    } else {
      compute_block_spheres_native<DeviceDispatch, D, Real, Int, ResetCount>(
          mgr,
          rot_coords,
          rot_coord_offset,
          block_ind_for_rot,
          pose_ind_for_rot,
          block_type_ind_for_rot,
          block_type_n_atoms,
          block_spheres,
          count_to_reset);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_rot_neighbors {
  static void f(
      ContextManager& mgr,
      Int max_n_rots_per_pose,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Real, 2, D> rot_spheres,
      TView<Int, 3, D> rot_neighbors,
      Real reach) {
    LAUNCH_BOX_32;

    int const n_poses = n_rots_for_block.size(0);
    int const max_n_rots = max_n_rots_per_pose;
    int const rot_pairs_per_pose = common::checked_dispatch_product(
        max_n_rots, max_n_rots, "rotamer-neighbor candidates per pose");
    int const n_rot_pairs = common::checked_dispatch_product(
        n_poses, rot_pairs_per_pose, "rotamer-neighbor candidates");

    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const n_block_types = block_type_n_atoms.size(0);

      int const pose_ind = ind / rot_pairs_per_pose;
      int const rot_pair_ind = ind % rot_pairs_per_pose;
      int const rot_ind1 = rot_pair_ind / max_n_rots;
      int const rot_ind2 = rot_pair_ind % max_n_rots;

      if (rot_ind1 >= n_rots_for_pose[pose_ind]
          || rot_ind2 >= n_rots_for_pose[pose_ind]) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      if (rot_ind1 > rot_ind2) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      int const global_rot_ind1 = rot_ind1 + rot_offset_for_pose[pose_ind];
      int const global_rot_ind2 = rot_ind2 + rot_offset_for_pose[pose_ind];

      int const block_ind1 = block_ind_for_rot[global_rot_ind1];
      int const block_ind2 = block_ind_for_rot[global_rot_ind2];

      bool same_rot = rot_ind1 == rot_ind2;
      bool same_block = block_ind1 == block_ind2;
      if (same_block && !same_rot) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      int const block_type1 = block_type_ind_for_rot[global_rot_ind1];
      if (block_type1 < 0) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }
      int const block_type2 = block_type_ind_for_rot[global_rot_ind2];
      if (block_type2 < 0) {
        if (D == Device::CUDA) {
          rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
        }
        return;
      }

      Vec<Real, 4> sphere1{0, 0, 0, 0};
      Vec<Real, 4> sphere2{0, 0, 0, 0};

      for (int i = 0; i < 4; ++i) {
        sphere1[i] = rot_spheres[global_rot_ind1][i];
        sphere2[i] = rot_spheres[global_rot_ind2][i];
      }

      Real d2 =
          ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
           + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
           + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));

      Real d_threshold = sphere1[3] + sphere2[3] + reach;

      // CUDA writes every cell so its caller can avoid a separate fill. CPU
      // retains zero-initialized scratch because skipping non-neighbor stores
      // is faster there.
      if (d2 < d_threshold * d_threshold) {
        rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 1;
      } else if (D == Device::CUDA) {
        rot_neighbors[pose_ind][rot_ind1][rot_ind2] = 0;
      }
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_rot_pairs, detect_neighbors);
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_block_neighbors {
  static void f(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 3, D> block_neighbors,
      Real reach) {
    LAUNCH_BOX_32;

    int const n_poses = pose_stack_block_type.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const block_pairs_per_pose = common::checked_dispatch_product(
        max_n_blocks, max_n_blocks, "block-neighbor candidates per pose");
    int const n_block_pairs = common::checked_dispatch_product(
        n_poses, block_pairs_per_pose, "block-neighbor candidates");
    auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int index) {
      int const pose_ind = index / block_pairs_per_pose;
      int const pair = index % block_pairs_per_pose;
      int const block_ind1 = pair / max_n_blocks;
      int const block_ind2 = pair % max_n_blocks;

      if (block_ind1 > block_ind2) {
        if (D == Device::CUDA) {
          block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
        }
        return;
      }

      int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
      if (block_type1 < 0) {
        if (D == Device::CUDA) {
          block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
        }
        return;
      }
      int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
      if (block_type2 < 0) {
        if (D == Device::CUDA) {
          block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
        }
        return;
      }

      Vec<Real, 4> sphere1{0, 0, 0, 0};
      Vec<Real, 4> sphere2{0, 0, 0, 0};

      for (int i = 0; i < 4; ++i) {
        sphere1[i] = block_spheres[pose_ind][block_ind1][i];
        sphere2[i] = block_spheres[pose_ind][block_ind2][i];
      }

      Real d2 =
          ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
           + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
           + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));

      Real d_threshold = sphere1[3] + sphere2[3] + reach;

      // CUDA writes every cell so its caller can avoid a separate fill. CPU
      // retains zero-initialized scratch because skipping non-neighbor stores
      // is faster there.
      if (d2 < d_threshold * d_threshold) {
        block_neighbors[pose_ind][block_ind1][block_ind2] = 1;
      } else if (D == Device::CUDA) {
        block_neighbors[pose_ind][block_ind1][block_ind2] = 0;
      }
    });
    if (D == tmol::Device::MPS) {
      BlockNeighborsCtxT<
          tmol::score::common::TViewBinder<D>::template View,
          Real,
          Int>
          ctx;
      ctx.pose_stack_block_type = pose_stack_block_type;
      ctx.block_spheres = block_spheres;
      ctx.block_neighbors = block_neighbors;
      ctx.max_n_blocks = max_n_blocks;
      ctx.block_pairs_per_pose = block_pairs_per_pose;
      ctx.reach = reach;
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kSphereOverlapMetalShaderSource),
          "detect_block_neighbors_f32",
          ctx,
          n_block_pairs,
          false);
    } else {
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_block_pairs, detect_neighbors);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct compact_block_neighbors {
  static void f(
      ContextManager& mgr,
      TView<Int, 3, D> block_neighbors,
      TView<Int, 1, D> neighbor_indices,
      TView<Int, 1, D> n_neighbors) {
    LAUNCH_BOX_32;

    int const n_poses = block_neighbors.size(0);
    int const max_n_blocks = block_neighbors.size(1);
    int const n_pairs = common::checked_dispatch_size(
        common::checked_count_product(
            max_n_blocks,
            int64_t(max_n_blocks) + 1,
            "compact block-neighbor candidates per pose")
            / 2,
        "compact block-neighbor candidates per pose");
    int const n_candidates = common::checked_dispatch_product(
        n_poses, n_pairs, "compact block-neighbor candidates");
    if constexpr (D == tmol::Device::CPU) {
      auto compact = ([=] TMOL_DEVICE_FUNC(int candidate) {
        neighbor_indices[candidate] = -1;
        int const pose = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block1 = common::get<0>(pair);
        int const block2 = common::get<1>(pair) - 1;
        if (block_neighbors[pose][block1][block2] == 0) return;
        neighbor_indices[candidate] = candidate;
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, compact);
      Int count = 0;
      for (int candidate = 0; candidate < n_candidates; ++candidate) {
        Int const value = neighbor_indices[candidate];
        if (value >= 0) neighbor_indices[count++] = value;
      }
      n_neighbors[0] = count;
    } else {
      auto compact = ([=] TMOL_DEVICE_FUNC(int candidate) {
        int const pose = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block1 = common::get<0>(pair);
        int const block2 = common::get<1>(pair) - 1;
        if (block_neighbors[pose][block1][block2] == 0) return;
        int const output = accumulate<D, Int>::add(n_neighbors[0], Int(1));
        neighbor_indices[output] = candidate;
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, compact);
    }
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Launch,
    typename Int,
    typename Eval>
void launch_compact_block_neighbors(
    ContextManager& mgr, TView<Int, 3, D> block_neighbors, Eval eval) {
  // Keep the compact count on device so this remains capture-safe. A bounded
  // grid then strides over only neighboring pairs without a host sync.
  int const n_poses = block_neighbors.size(0);
  int const max_n_blocks = block_neighbors.size(1);
  int const n_pairs = common::checked_dispatch_size(
      common::checked_count_product(
          max_n_blocks,
          int64_t(max_n_blocks) + 1,
          "compact block-neighbor candidates per pose")
          / 2,
      "compact block-neighbor candidates per pose");
  int const n_candidates = common::checked_dispatch_product(
      n_poses, n_pairs, "compact block-neighbor candidates");
  if (n_candidates == 0) return;
  auto neighbor_indices_t = TPack<Int, 1, D>::empty({n_candidates});
  auto neighbor_indices = neighbor_indices_t.view;
  auto n_neighbors_t = TPack<Int, 1, D>::zeros({1});
  auto n_neighbors = n_neighbors_t.view;
  compact_block_neighbors<DeviceDispatch, D, Int>::f(
      mgr, block_neighbors, neighbor_indices, n_neighbors);

  constexpr int wide_batch_candidates = 1 << 20;
  constexpr int normal_workgroups = 1 << 14;
  constexpr int wide_batch_workgroups = 1 << 15;
  int const max_workgroups = n_candidates >= wide_batch_candidates
                                 ? wide_batch_workgroups
                                 : normal_workgroups;
  int const n_workgroups =
      n_candidates < max_workgroups ? n_candidates : max_workgroups;
  auto eval_compact = ([=] TMOL_DEVICE_FUNC(int cta) {
    for (int index = cta; index < n_candidates; index += n_workgroups) {
      if (index >= n_neighbors[0]) return;
      eval(neighbor_indices[index]);
    }
  });
  DeviceDispatch<D>::template foreach_workgroup<Launch>(
      mgr, n_workgroups, eval_compact);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct rot_neighbor_indices {
  static auto f(
      ContextManager& mgr,
      TView<Int, 3, D> rot_neighbors,
      TView<Int, 1, D> rot_offset_for_pose) -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int n_pose = rot_neighbors.size(0);
    int n_rot = rot_neighbors.size(1);

    int const pose_rot_cells = common::checked_dispatch_product(
        n_pose, n_rot, "rotamer-neighbor dispatch candidates");
    int const n_cells = common::checked_dispatch_product(
        pose_rot_cells, n_rot, "rotamer-neighbor dispatch candidates");
    auto count_for_cell_tp =
        TPack<int64_t, 3, D>::zeros({n_pose, n_rot, n_rot});
    auto count_for_cell = count_for_cell_tp.view;
    auto copy_counts = ([=] TMOL_DEVICE_FUNC(int ind) {
      count_for_cell.data()[ind] = rot_neighbors.data()[ind];
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, copy_counts);

    auto offset_for_cell_tp =
        TPack<int64_t, 3, D>::empty({n_pose, n_rot, n_rot});
    auto offset_for_cell = offset_for_cell_tp.view;

    int64_t const n_dispatch_total_64 =
        DeviceDispatch<D>::template scan_and_return_total<
            tmol::portable::scan_type::exclusive>(
            mgr,
            count_for_cell.data(),
            offset_for_cell.data(),
            n_cells,
            tmol::portable::plus_t<int64_t>());
    int const n_dispatch_total = common::checked_dispatch_size(
        n_dispatch_total_64, "rotamer-neighbor dispatch");

    auto rot_neighbor_indices =
        TPack<Int, 2, D>::full({3, n_dispatch_total}, -1);
    auto rot_neighbor_indices_v = rot_neighbor_indices.view;

    auto fill_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
      int pose = ind / (n_rot * n_rot);
      ind = ind % (n_rot * n_rot);
      int rot1 = ind / n_rot;
      int rot2 = ind % n_rot;

      if (rot_neighbors[pose][rot1][rot2]) {
        int offset = offset_for_cell[pose][rot1][rot2];
        rot_neighbor_indices_v[0][offset] = pose;
        rot_neighbor_indices_v[1][offset] = rot1 + rot_offset_for_pose[pose];
        rot_neighbor_indices_v[2][offset] = rot2 + rot_offset_for_pose[pose];
      }
    });

    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, fill_indices);

    return rot_neighbor_indices;
  }
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct block_neighbor_indices {
  static auto f(ContextManager& mgr, TView<Int, 3, D> block_neighbors)
      -> TPack<Int, 2, D> {
    LAUNCH_BOX_32;

    int n_pose = block_neighbors.size(0);
    int n_res = block_neighbors.size(1);

    int const pose_res_cells = common::checked_dispatch_product(
        n_pose, n_res, "block-neighbor dispatch candidates");
    int const n_cells = common::checked_dispatch_product(
        pose_res_cells, n_res, "block-neighbor dispatch candidates");
    auto count_for_cell_tp =
        TPack<int64_t, 3, D>::zeros({n_pose, n_res, n_res});
    auto count_for_cell = count_for_cell_tp.view;
    auto copy_counts = ([=] TMOL_DEVICE_FUNC(int ind) {
      count_for_cell.data()[ind] = block_neighbors.data()[ind];
    });
    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, copy_counts);

    auto offset_for_cell_tp =
        TPack<int64_t, 3, D>::empty({n_pose, n_res, n_res});
    auto offset_for_cell = offset_for_cell_tp.view;

    int64_t const n_dispatch_total_64 =
        DeviceDispatch<D>::template scan_and_return_total<
            tmol::portable::scan_type::exclusive>(
            mgr,
            count_for_cell.data(),
            offset_for_cell.data(),
            n_cells,
            tmol::portable::plus_t<int64_t>());
    int const n_dispatch_total = common::checked_dispatch_size(
        n_dispatch_total_64, "block-neighbor dispatch");

    auto block_neighbor_indices =
        TPack<Int, 2, D>::full({3, n_dispatch_total}, -1);
    auto block_neighbor_indices_v = block_neighbor_indices.view;

    auto fill_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
      int pose = ind / (n_res * n_res);
      ind = ind % (n_res * n_res);
      int res1 = ind / n_res;
      int res2 = ind % n_res;

      if (block_neighbors[pose][res1][res2]) {
        int offset = offset_for_cell[pose][res1][res2];
        block_neighbor_indices_v[0][offset] = pose;
        block_neighbor_indices_v[1][offset] = res1;
        block_neighbor_indices_v[2][offset] = res2;
      }
    });

    DeviceDispatch<D>::template forall_independent<launch_t>(
        mgr, n_cells, fill_indices);

    return block_neighbor_indices;
  }
};

// Compute block-level bounding spheres that enclose all rotamers for each
// block, using pre-computed per-rotamer spheres.  One thread per block; serial
// loop over n_rots_for_block[pose][block] rotamers.  Safe for packing (no
// write races) unlike compute_block_spheres which launches per rotamer.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct compute_block_spheres_from_rot_spheres {
  static void f(
      ContextManager& mgr,
      TView<Real, 2, D> rot_spheres,          // [n_rots_global, 4]
      TView<Int, 2, D> n_rots_for_block,      // [n_poses, max_n_blocks]
      TView<Int, 2, D> rot_offset_for_block,  // [n_poses, max_n_blocks] global
      TView<Real, 3, D> block_spheres         // [n_poses, max_n_blocks, 4] out
  ) {
    int const n_poses = n_rots_for_block.size(0);
    int const max_n_blocks = n_rots_for_block.size(1);
    int const n_pose_blocks = common::checked_dispatch_product(
        n_poses, max_n_blocks, "rotamer block-sphere dispatch");

    BlockSpheresFromRotCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.rot_spheres = rot_spheres;
    ctx.n_rots_for_block = n_rots_for_block;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.block_spheres = block_spheres;
    ctx.max_n_blocks = max_n_blocks;

    if (D == tmol::Device::MPS) {
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kSphereOverlapMetalShaderSource),
          "compute_block_spheres_from_rot_f32",
          ctx,
          n_pose_blocks,
          false);
    } else {
      launch_blocks_from_rotamer_spheres<DeviceDispatch, D, Real, Int>(
          mgr, ctx);
    }
  }
};

// Convert block and rotamer bounding spheres into rotamer-pair dispatch indices
// (the same [3, n_pairs] format as rot_neighbor_indices). Canonical packed
// upper-triangle block pairs avoid all square block-pair scratch while
// retaining the original pose-major, b1-major, b2-major ordering.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct rot_neighbor_indices_from_block_neighbors {
  // Exclusive scan of the per-candidate counts.
  //
  // DeviceDispatch::scan takes raw pointers, which Metal cannot honour: an
  // MPS tensor's data pointer is an MTLBuffer handle, not an address a kernel
  // can be handed, and the stable ABI exposes no cumsum. The scan runs on the
  // host there instead -- which costs no extra synchronization, because the
  // caller has already blocked on the overflow flag and the count sum
  // immediately above.
  static auto exclusive_scan_counts(
      ContextManager& mgr, TPack<Int, 1, D>& counts_t, int n_candidates)
      -> TPack<Int, 1, D> {
    if (D == tmol::Device::MPS) {
      auto host_counts =
          counts_t.tensor.to(tmol::native::Device(tmol::native::kCPU));
      auto host_offsets_t =
          TPack<Int, 1, tmol::Device::CPU>::empty({int64_t(n_candidates)});
      Int const* counts = host_counts.template data_ptr<Int>();
      Int* offsets = host_offsets_t.tensor.template data_ptr<Int>();
      Int running = 0;
      for (int i = 0; i < n_candidates; ++i) {
        offsets[i] = running;
        running += counts[i];
      }
      return TPack<Int, 1, D>(
          host_offsets_t.tensor.to(counts_t.tensor.device()));
    }
    auto offsets_t = TPack<Int, 1, D>::zeros({n_candidates});
    DeviceDispatch<D>::template scan<tmol::portable::scan_type::exclusive>(
        mgr,
        counts_t.view.data(),
        offsets_t.view.data(),
        n_candidates,
        tmol::portable::plus_t<Int>());
    return offsets_t;
  }

  static auto f(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 2, D> n_rots_for_block,      // [n_poses, max_n_blocks]
      TView<Int, 2, D> rot_offset_for_block,  // [n_poses, max_n_blocks] global
      TView<Real, 2, D> rot_spheres,          // [n_rots_global, 4]
      TView<Int, 2, D> lockstep_group_for_block,
      Real reach,
      int64_t candidate_begin = 0,
      int64_t candidate_count = -1) -> TPack<Int, 2, D> {
    int const n_poses = pose_stack_block_type.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);
    auto const window = rotamer_pair_window(
        n_poses, max_n_blocks, candidate_begin, candidate_count);
    int const block_pairs_per_pose = window.block_pairs_per_pose;
    int const n_candidates = window.n_candidates;
    if (n_candidates == 0) {
      return TPack<Int, 2, D>::empty({3, 0});
    }

    // Step 1: per-block-pair rotamer pair counts.
    // For diagonal (b1==b2): only self-pairs (r,r), count = n_rots[b1].
    // For off-diagonal (b1<b2): count the overlapping rotamer spheres.
    auto pair_counts_t = TPack<Int, 1, D>::zeros({n_candidates});
    auto pair_count_overflow_t = TPack<int64_t, 1, D>::zeros({1});

    RotPairListCtxT<
        tmol::score::common::TViewBinder<D>::template View,
        Real,
        Int>
        ctx;
    ctx.pose_stack_block_type = pose_stack_block_type;
    ctx.block_spheres = block_spheres;
    ctx.n_rots_for_block = n_rots_for_block;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.rot_spheres = rot_spheres;
    ctx.lockstep_group_for_block = lockstep_group_for_block;
    ctx.pair_counts = pair_counts_t.view;
    ctx.pair_count_overflow = pair_count_overflow_t.view;
    ctx.max_n_blocks = max_n_blocks;
    ctx.block_pairs_per_pose = block_pairs_per_pose;
    ctx.candidate_begin = int(candidate_begin);
    ctx.reach = reach;
    // The fill pass reuses this context; its two remaining views are bound
    // once the scan and the output allocation exist. They are bound to the
    // counts for now so the Metal collector never sees an undefined view.
    ctx.pair_offsets = pair_counts_t.view;
    auto indices_placeholder_t = TPack<Int, 2, D>::empty({3, 1});
    ctx.indices = indices_placeholder_t.view;

    if (D == tmol::Device::MPS) {
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kSphereOverlapMetalShaderSource),
          "rot_pair_counts_f32",
          ctx,
          n_candidates,
          false);
    } else {
      launch_rotamer_pair_counts<DeviceDispatch, D, Real, Int>(mgr, ctx);
    }

    // Step 2: check an int64 reduction before scanning int32 offsets. This
    // preserves the signed-32-bit dispatch limit without two int64 pair arrays.
    if (pair_count_overflow_t.tensor.template item<int64_t>() != 0) {
      throw std::overflow_error(
          "rotamer block-pair dispatch exceeds the signed 32-bit native "
          "dispatch limit (2147483647)");
    }
    int64_t const total_64 =
        pair_counts_t.tensor.sum(tmol::native::ScalarType::Long)
            .template item<int64_t>();
    int const total =
        common::checked_dispatch_size(total_64, "rotamer block-pair dispatch");

    auto pair_offsets_t =
        exclusive_scan_counts(mgr, pair_counts_t, n_candidates);

    // Step 3: allocate output [3, total]
    auto indices_t = TPack<Int, 2, D>::full({3, total}, -1);
    ctx.pair_offsets = pair_offsets_t.view;
    ctx.indices = indices_t.view;

    // Step 4: fill -- one thread per block pair, serial loop over rot pairs.
    // Diagonal (b1==b2): only (r,r) self-pairs (intrares scoring).
    // Same lockstep group: matching states; otherwise overlapping spheres.
    if (D == tmol::Device::MPS) {
      tmol::score::common::mps_launch_if<D>(
          TMOL_MPS_SOURCE(kSphereOverlapMetalShaderSource),
          "rot_pair_fill_f32",
          ctx,
          n_candidates,
          false);
    } else {
      launch_rotamer_pair_fill<DeviceDispatch, D, Real, Int>(mgr, ctx);
    }

    return indices_t;
  }
};

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
