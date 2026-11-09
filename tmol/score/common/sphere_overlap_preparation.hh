#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/upper_triangle_indices.hh>

// Recompute from current coordinates; callers own sphere and neighbor buffers.
// Neighbors have capacity 1 + poses * blocks * (blocks + 1) / 2. Entry zero
// is the active count, followed by canonical triangular candidate indices.
namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {
using tmol::portable::Vec;

// A pose with one block has only its self pair. Its candidate list is
// independent of coordinates; the consuming kernel must still reject absent
// rotamers/block types and apply atom-distance cutoffs. This list deliberately
// includes padded poses, unlike the compact geometric neighbor detector.
// Caller provides n_poses + 1 entries and requires max_n_blocks == 1.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
void single_block_neighbor_candidates_native(
    ContextManager& mgr, int n_poses, TView<Int, 1, D> neighbor_indices) {
  LAUNCH_BOX_32;
  auto fill = ([=] TMOL_DEVICE_FUNC(int index) {
    neighbor_indices[index] = index == 0 ? n_poses : index - 1;
  });
  DeviceDispatch<D>::template forall<launch_t>(mgr, n_poses + 1, fill);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    bool ResetCount = false,
    typename CoordTView>
void compute_block_spheres_native(
    ContextManager& mgr,
    CoordTView rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Real, 3, D> block_spheres,
    Int* count_to_reset = nullptr) {
  LAUNCH_BOX_32;

  auto compute_spheres = ([=] TMOL_DEVICE_FUNC(int cta) {
    CTA_LAUNCH_T_PARAMS;

    // Define this outside the constexpr branch so NVCC captures the pointer
    // in an ordinary extended-lambda context.
    auto reset_count = ([=] TMOL_DEVICE_FUNC(int tid) {
      if (cta == 0 && tid == 0) count_to_reset[0] = 0;
    });
    if constexpr (ResetCount) {
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(reset_count);
    }

    int const pose_ind = pose_ind_for_rot[cta];
    int const block_ind = block_ind_for_rot[cta];
    int const block_type = block_type_ind_for_rot[cta];
    int const coord_offset = rot_coord_offset[cta];

    if (block_type < 0) return;
    int const n_atoms = block_type_n_atoms[block_type];
    Vec<Real, 3> local_coords{0, 0, 0};

    auto per_thread_com = ([&] TMOL_DEVICE_FUNC(int tid) {
      for (int i = tid; i < n_atoms; i += nt) {
        auto ci = rot_coords[coord_offset + i];
        for (int j = 0; j < 3; ++j) {
          local_coords[j] += ci[j];
        }
      }
      for (int j = 0; j < 3; ++j) {
        local_coords[j] /= n_atoms;
      }
    });

    DeviceDispatch<D>::template for_each_in_workgroup<nt>(per_thread_com);

    // CPU's scalar workgroup implementation is faster with the square root
    // before reduction; CUDA reduces squared distances to execute one sqrt.
    Real dmax(0);

    DeviceDispatch<D>::synchronize_workgroup();
    Vec<Real, 3> com =
        DeviceDispatch<D>::template shuffle_reduce_and_broadcast_in_workgroup<
            nt>(local_coords, tmol::portable::plus_t<Real>());

    Real d2max = 0;
    // Now find maximum distance
    auto per_thread_dist_to_com = ([&] TMOL_DEVICE_FUNC(int tid) {
      for (int i = tid; i < n_atoms; i += nt) {
        auto ci = rot_coords[coord_offset + i];
        Real d2 =
            ((ci[0] - com[0]) * (ci[0] - com[0])
             + (ci[1] - com[1]) * (ci[1] - com[1])
             + (ci[2] - com[2]) * (ci[2] - com[2]));
        if (d2 > d2max) {
          d2max = d2;
        }
      }
      if constexpr (D == Device::CPU) {
        dmax = sqrt(d2max);
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        per_thread_dist_to_com);

    if constexpr (D == Device::CPU) {
      dmax = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
          dmax, tmol::portable::maximum_t<Real>());
    } else {
      d2max = DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
          d2max, tmol::portable::maximum_t<Real>());
    }

    auto thread0_write_out_result = ([=] TMOL_DEVICE_FUNC(int tid) {
      if (tid == 0) {
        block_spheres[pose_ind][block_ind][0] = com[0];
        block_spheres[pose_ind][block_ind][1] = com[1];
        block_spheres[pose_ind][block_ind][2] = com[2];
        if constexpr (D == Device::CPU) {
          block_spheres[pose_ind][block_ind][3] = dmax;
        } else {
          // Reduce squared distances first and take one square root per
          // block, rather than one per participating GPU thread.
          block_spheres[pose_ind][block_ind][3] = sqrt(d2max);
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        thread0_write_out_result);
  });

  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      mgr, block_ind_for_rot.size(0), compute_spheres);
}

template <typename Real, typename Int>
bool try_cpu_spatial_compact_block_neighbors(
    TView<Int, 2, tmol::Device::CPU> pose_stack_block_type,
    TView<Real, 3, tmol::Device::CPU> block_spheres,
    Real reach,
    std::vector<Int>& retained) {
  retained.clear();
  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  constexpr int min_spatial_sweep_blocks = 150;
  if (n_poses != 1 || max_n_blocks < min_spatial_sweep_blocks) {
    return false;
  }

  std::vector<Int> ordered_blocks;
  ordered_blocks.reserve(max_n_blocks);
  Real min_center[3] = {
      std::numeric_limits<Real>::infinity(),
      std::numeric_limits<Real>::infinity(),
      std::numeric_limits<Real>::infinity()};
  Real max_center[3] = {
      -std::numeric_limits<Real>::infinity(),
      -std::numeric_limits<Real>::infinity(),
      -std::numeric_limits<Real>::infinity()};
  bool finite_spheres = true;
  for (int block = 0; block < max_n_blocks; ++block) {
    if (pose_stack_block_type[0][block] < 0) continue;
    ordered_blocks.push_back(block);
    for (int axis = 0; axis < 3; ++axis) {
      Real const center = block_spheres[0][block][axis];
      finite_spheres = finite_spheres && std::isfinite(center);
      min_center[axis] = std::min(min_center[axis], center);
      max_center[axis] = std::max(max_center[axis], center);
    }
    finite_spheres =
        finite_spheres && std::isfinite(block_spheres[0][block][3]);
  }

  // NaN/Inf coordinates follow the established quadratic comparison behavior
  // rather than entering a comparator without a strict order.
  if (!finite_spheres || ordered_blocks.size() < min_spatial_sweep_blocks) {
    return false;
  }

  int sweep_axis = 0;
  for (int axis = 1; axis < 3; ++axis) {
    if (max_center[axis] - min_center[axis]
        > max_center[sweep_axis] - min_center[sweep_axis]) {
      sweep_axis = axis;
    }
  }
  std::sort(
      ordered_blocks.begin(), ordered_blocks.end(), [=](Int left, Int right) {
        Real const left_center = block_spheres[0][left][sweep_axis];
        Real const right_center = block_spheres[0][right][sweep_axis];
        return left_center < right_center
               || (left_center == right_center && left < right);
      });

  int const n_valid_blocks = static_cast<int>(ordered_blocks.size());
  std::vector<Real> suffix_max_radius(n_valid_blocks);
  Real suffix_radius = 0;
  for (int position = n_valid_blocks - 1; position >= 0; --position) {
    suffix_radius =
        std::max(suffix_radius, block_spheres[0][ordered_blocks[position]][3]);
    suffix_max_radius[position] = suffix_radius;
  }

  int64_t const n_pairs =
      (int64_t(max_n_blocks) * (int64_t(max_n_blocks) + 1)) / 2;
  retained.reserve(
      std::min<int64_t>(
          n_pairs, std::max<int64_t>(16, int64_t(n_valid_blocks) * 32)));
  int const triangle_dimension = max_n_blocks + 1;
  for (int left_position = 0; left_position < n_valid_blocks; ++left_position) {
    Int const left = ordered_blocks[left_position];
    Real const left_axis_center = block_spheres[0][left][sweep_axis];
    Real const left_radius = block_spheres[0][left][3];
    for (int right_position = left_position; right_position < n_valid_blocks;
         ++right_position) {
      Int const right = ordered_blocks[right_position];
      Real const axis_separation =
          block_spheres[0][right][sweep_axis] - left_axis_center;
      if (right_position != left_position
          && axis_separation
                 >= left_radius + suffix_max_radius[right_position] + reach) {
        break;
      }

      Real const dx = block_spheres[0][left][0] - block_spheres[0][right][0];
      Real const dy = block_spheres[0][left][1] - block_spheres[0][right][1];
      Real const dz = block_spheres[0][left][2] - block_spheres[0][right][2];
      Real const d2 = dx * dx + dy * dy + dz * dz;
      Real const threshold = left_radius + block_spheres[0][right][3] + reach;
      if (d2 >= threshold * threshold) continue;

      Int const block1 = std::min(left, right);
      Int const block2 = std::max(left, right);
      int64_t const candidate =
          int64_t(block1) * (2 * int64_t(triangle_dimension) - block1 - 1) / 2
          + block2 - block1;
      retained.push_back(static_cast<Int>(candidate));
    }
  }
  std::sort(retained.begin(), retained.end());
  return true;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct detect_compact_block_neighbors {
  static void f(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Real, 3, D> block_spheres,
      TView<Int, 1, D> neighbor_indices,
      Real reach) {
    LAUNCH_BOX_32;

    int const n_poses = pose_stack_block_type.size(0);
    int const max_n_blocks = pose_stack_block_type.size(1);
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
      // Sorting retained candidate IDs restores the exact canonical order used
      // by the quadratic path, so downstream accumulation is unchanged.
      std::vector<Int> retained;
      if (try_cpu_spatial_compact_block_neighbors(
              pose_stack_block_type, block_spheres, reach, retained)) {
        neighbor_indices[0] = static_cast<Int>(retained.size());
        if (!retained.empty()) {
          std::memcpy(
              neighbor_indices.data() + 1,
              retained.data(),
              retained.size() * sizeof(Int));
        }
        return;
      }

      auto detect_neighbors =
          ([=] TMOL_DEVICE_FUNC(
               int candidate, int pose_ind, int block_ind1, int block_ind2) {
            // CPU candidates run concurrently. Classify into disjoint slots;
            // compact them deterministically after the parallel loop instead of
            // racing on the intentionally non-atomic CPU accumulator.
            neighbor_indices[candidate + 1] = -1;
            int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
            if (block_type1 < 0) return;
            int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
            if (block_type2 < 0) return;

            Vec<Real, 4> sphere1{0, 0, 0, 0};
            Vec<Real, 4> sphere2{0, 0, 0, 0};
            for (int i = 0; i < 4; ++i) {
              sphere1[i] = block_spheres[pose_ind][block_ind1][i];
              sphere2[i] = block_spheres[pose_ind][block_ind2][i];
            }
            Real const d2 =
                ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
                 + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
                 + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));
            Real const threshold = sphere1[3] + sphere2[3] + reach;
            if (d2 >= threshold * threshold) return;

            neighbor_indices[candidate + 1] = candidate;
          });
      // Decode one triangular index per chunk, then advance through adjacent
      // pairs. Chunks own disjoint output slots and retain canonical ordering.
      constexpr int chunk_size = 128;
      int const n_chunks =
          int((int64_t(n_candidates) + chunk_size - 1) / chunk_size);
      auto detect_chunk = ([=] TMOL_DEVICE_FUNC(int chunk) {
        int const first = chunk * chunk_size;
        int const end = first + min(chunk_size, n_candidates - first);
        int pose = first / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            first % n_pairs, max_n_blocks + 1);
        int block1 = common::get<0>(pair);
        int block2 = common::get<1>(pair) - 1;
        for (int candidate = first; candidate < end; ++candidate) {
          detect_neighbors(candidate, pose, block1, block2);
          if (++block2 == max_n_blocks) {
            if (++block1 == max_n_blocks) {
              ++pose;
              block1 = 0;
            }
            block2 = block1;
          }
        }
      });
      DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
          mgr, n_chunks, detect_chunk);
      Int n_neighbors = 0;
      for (int candidate = 0; candidate < n_candidates; ++candidate) {
        Int const value = neighbor_indices[candidate + 1];
        if (value >= 0) neighbor_indices[++n_neighbors] = value;
      }
      neighbor_indices[0] = n_neighbors;
    } else {
      auto detect_neighbors = ([=] TMOL_DEVICE_FUNC(int candidate) {
        int const pose_ind = candidate / n_pairs;
        auto pair = common::upper_triangle_inds_from_linear_index(
            candidate % n_pairs, max_n_blocks + 1);
        int const block_ind1 = common::get<0>(pair);
        int const block_ind2 = common::get<1>(pair) - 1;

        int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
        if (block_type1 < 0) return;
        int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];
        if (block_type2 < 0) return;

        Vec<Real, 4> sphere1{0, 0, 0, 0};
        Vec<Real, 4> sphere2{0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
          sphere1[i] = block_spheres[pose_ind][block_ind1][i];
          sphere2[i] = block_spheres[pose_ind][block_ind2][i];
        }
        Real const d2 =
            ((sphere1[0] - sphere2[0]) * (sphere1[0] - sphere2[0])
             + (sphere1[1] - sphere2[1]) * (sphere1[1] - sphere2[1])
             + (sphere1[2] - sphere2[2]) * (sphere1[2] - sphere2[2]));
        Real const threshold = sphere1[3] + sphere2[3] + reach;
        if (d2 >= threshold * threshold) return;

        Int const output = accumulate<D, Int>::add(neighbor_indices[0], Int(1));
        neighbor_indices[output + 1] = candidate;
      });
      DeviceDispatch<D>::template forall_independent<launch_t>(
          mgr, n_candidates, detect_neighbors);
    }
  }
};

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
