#pragma once

// Retained-rotamer topology, independent of tensor allocation or registration.
// Keep masks are zero-initialized for bump checks (keep_rotamer is all ones
// without bump checking); fixed-block assignments start at -1. Prefix maps
// are exclusive scans performed by the adapter between stages. Counts and
// offsets start at zero, including padding and leading/trailing empty poses.
// Reject a clash only when its block has a strictly favorable alternative.
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <cstdint>

namespace tmol::pack::compiled {
template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2,
    typename View3,
    typename View4>
void launch_ig_keep_rotamers(
    Context& mgr,
    View0 pose_for_rot,
    View1 block_ind_for_rot,
    View2 best_energy_for_rot,
    View3 best_energy_per_block,
    View4 keep_rotamer) {
  LAUNCH_BOX_32;
  int const n_rotamers = score::common::checked_dispatch_size(
      keep_rotamer.size(0), "retained rotamers");
  auto decide_keep_rotamers = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rot = index;
    int const block = block_ind_for_rot[rot];
    int const pose = pose_for_rot[rot];
    auto energy_for_rot = best_energy_for_rot[rot];
    auto best_energy_for_block = best_energy_per_block[pose][block];
    if (energy_for_rot <= 5.0 || best_energy_for_block >= 5.0) {
      keep_rotamer[rot] = 1;
    }
  });
  Ops<D>::template forall<launch_t>(mgr, n_rotamers, decide_keep_rotamers);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2,
    typename View3,
    typename View4>
void launch_ig_keep_blocks(
    Context& mgr,
    View0 n_rots_for_block,
    View1 rot_offset_for_block,
    View2 keep_rotamer,
    View3 keep_block,
    View4 rotamer_for_nonmolten_block) {
  LAUNCH_BOX_32;
  constexpr int nt = launch_t::sm_ptx::nt;
  int const max_n_blocks = n_rots_for_block.size(1);
  int const n_pose_block_cells = score::common::checked_dispatch_product(
      n_rots_for_block.size(0), max_n_blocks, "retained blocks");
  auto decide_keep_blocks = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const pose = cta / max_n_blocks;
    int const block = cta % max_n_blocks;
    int const block_n_rots = n_rots_for_block[pose][block];
    if (block_n_rots == 0) {
      return;
    }
    int const block_rot_offset = rot_offset_for_block[pose][block];
    auto count_n_rots_for_block = ([&] TMOL_DEVICE_FUNC(int tid) {
      // Threads will distribute work of looking at all the rotamers
      // for this block; then we'll reduce on the number of kept rotamers
      // and the index of the biggest kept rotamer
      int tid_n_kept_rotamers_for_block = 0;
      int tid_biggest_kept_rotamer = -1;
      for (int rot_in_block = tid; rot_in_block < block_n_rots;
           rot_in_block += nt) {
        int const rot = block_rot_offset + rot_in_block;
        if (keep_rotamer[rot]) {
          tid_n_kept_rotamers_for_block += 1;
          if (rot > tid_biggest_kept_rotamer) {
            tid_biggest_kept_rotamer = rot;
          }
        }
      }
      int n_kept_rotamers_for_block =
          Ops<D>::template shuffle_reduce_in_workgroup<nt>(
              tid_n_kept_rotamers_for_block, tmol::portable::plus_t<int>());
      int biggest_kept_rotamer =
          Ops<D>::template shuffle_reduce_in_workgroup<nt>(
              tid_biggest_kept_rotamer, tmol::portable::maximum_t<int>());

      if (tid == 0) {
        if (n_kept_rotamers_for_block > 1) {
          keep_block[pose][block] = 1;
        } else {
          // we are not keeping this block,
          // so if it has exactly 1 rotamer, then we have to also
          // note that we are not keeping that rotamer
          if (n_kept_rotamers_for_block == 1) {
            keep_rotamer[biggest_kept_rotamer] = 0;
            rotamer_for_nonmolten_block[pose][block] = biggest_kept_rotamer;
          }
        }
      }
    });
    Ops<D>::template for_each_in_workgroup<nt>(count_n_rots_for_block);
  });
  Ops<D>::template foreach_workgroup<launch_t>(
      mgr, n_pose_block_cells, decide_keep_blocks);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2>
void launch_ig_invert_rotamers(
    Context& mgr,
    View0 keep_rotamer,
    View1 old_to_new_rotamer_index,
    View2 new_to_old_rotamer_index) {
  LAUNCH_BOX_32;
  int const n_rotamers = score::common::checked_dispatch_size(
      keep_rotamer.size(0), "retained rotamers");
  auto record_new_rotamer_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rot = index;
    if (keep_rotamer[rot]) {
      int64_t new_index = old_to_new_rotamer_index[rot];
      new_to_old_rotamer_index[new_index] = rot;
    }
  });
  Ops<D>::template forall<launch_t>(
      mgr, n_rotamers, record_new_rotamer_indices);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2>
void launch_ig_invert_blocks(
    Context& mgr,
    View0 keep_block,
    View1 block_to_molten_block_inds,
    View2 molten_block_to_block_inds) {
  LAUNCH_BOX_32;
  int const max_n_blocks = keep_block.size(1);
  int const n_pose_block_cells = score::common::checked_dispatch_product(
      keep_block.size(0), max_n_blocks, "retained blocks");
  auto record_molten_block_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / max_n_blocks;
    int const block = index % max_n_blocks;
    if (keep_block[pose][block]) {
      int64_t molten_block_index = block_to_molten_block_inds[index];
      molten_block_to_block_inds[molten_block_index] = index;
    }
  });
  Ops<D>::template forall<launch_t>(
      mgr, n_pose_block_cells, record_molten_block_indices);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2>
void launch_ig_pose_block_offsets(
    Context& mgr,
    View0 keep_block,
    View1 block_to_molten_block_inds,
    View2 molten_block_offset_for_pose) {
  LAUNCH_BOX_32;
  int const n_poses = keep_block.size(0);
  int const max_n_blocks = keep_block.size(1);
  auto record_molten_block_offset_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    if (max_n_blocks == 0) {
      return;
    }
    int const pose = index;
    int last_block_kept = keep_block[pose][max_n_blocks - 1];
    int last_offset_for_pose =
        block_to_molten_block_inds[pose * max_n_blocks + (max_n_blocks - 1)];
    int pose_n_molten_blocks = last_offset_for_pose + last_block_kept;
    molten_block_offset_for_pose[pose + 1] = pose_n_molten_blocks;
  });
  // Note: launch n-poses - 1 threads here as thread i writes to position i+1
  // and position 0's value does not get written to by any thread
  Ops<D>::template forall<launch_t>(
      mgr, n_poses - 1, record_molten_block_offset_per_pose);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1>
void launch_ig_pose_block_counts(
    Context& mgr,
    View0 molten_block_offset_for_pose,
    View1 n_molten_blocks_per_pose,
    int n_molten_blocks) {
  LAUNCH_BOX_32;
  int const n_poses = n_molten_blocks_per_pose.size(0);
  auto compute_n_molten_blocks_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    if (pose < n_poses - 1) {
      n_molten_blocks_per_pose[pose] = molten_block_offset_for_pose[pose + 1]
                                       - molten_block_offset_for_pose[pose];
    } else {
      n_molten_blocks_per_pose[pose] =
          n_molten_blocks - molten_block_offset_for_pose[pose];
    }
  });
  Ops<D>::template forall<launch_t>(
      mgr, n_poses, compute_n_molten_blocks_per_pose);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2>
void launch_ig_pose_rotamer_offsets(
    Context& mgr,
    View0 rot_offset_for_pose,
    View1 old_to_new_rotamer_index,
    View2 bc_rot_offset_for_pose,
    int64_t n_kept_rotamers) {
  LAUNCH_BOX_32;
  int const n_poses = rot_offset_for_pose.size(0);
  int const n_rotamers = score::common::checked_dispatch_size(
      old_to_new_rotamer_index.size(0), "retained rotamers");
  auto fill_bc_rot_offsets_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    int orig_first_rotamer_for_pose = rot_offset_for_pose[pose];
    // Trailing empty poses start one past the original rotamer array.
    // Their compacted offset is the total number of retained rotamers.
    int new_first_rotamer_for_pose =
        orig_first_rotamer_for_pose < n_rotamers
            ? old_to_new_rotamer_index[orig_first_rotamer_for_pose]
            : n_kept_rotamers;
    // This will be the right offset whether or not the first rotamer
    // in this pose has been bump-checked away
    bc_rot_offset_for_pose[pose] = new_first_rotamer_for_pose;
  });
  Ops<D>::template forall<launch_t>(mgr, n_poses, fill_bc_rot_offsets_per_pose);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1>
void launch_ig_pose_rotamer_counts(
    Context& mgr,
    View0 bc_rot_offset_for_pose,
    View1 n_bc_rots_per_pose,
    int64_t n_kept_rotamers) {
  LAUNCH_BOX_32;
  int const n_poses = n_bc_rots_per_pose.size(0);
  auto fill_n_bc_rots_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    int pose_bc_rot_offset = bc_rot_offset_for_pose[pose];
    if (pose < n_poses - 1) {
      n_bc_rots_per_pose[pose] =
          bc_rot_offset_for_pose[pose + 1] - pose_bc_rot_offset;
    } else {
      n_bc_rots_per_pose[pose] = n_kept_rotamers - pose_bc_rot_offset;
    }
  });
  Ops<D>::template forall<launch_t>(mgr, n_poses, fill_n_bc_rots_per_pose);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2,
    typename View3,
    typename View4>
void launch_ig_block_rotamer_offsets(
    Context& mgr,
    View0 molten_block_to_block_inds,
    View1 molten_block_offset_for_pose,
    View2 rot_offset_for_block,
    View3 old_to_new_rotamer_index,
    View4 bc_rot_offset_for_molten_block) {
  LAUNCH_BOX_32;
  int const max_n_blocks = rot_offset_for_block.size(1);
  int const n_molten_blocks = score::common::checked_dispatch_size(
      molten_block_to_block_inds.size(0), "retained blocks");
  auto fill_bc_rot_offset_for_molten_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const molten_block_index = index;
    int const block_and_pose = molten_block_to_block_inds[molten_block_index];
    int const pose = block_and_pose / max_n_blocks;
    int const block = block_and_pose % max_n_blocks;
    int const pose_molten_block_offset = molten_block_offset_for_pose[pose];
    int const block_rot_offset = rot_offset_for_block[pose][block];
    int const bc_rot_for_block = old_to_new_rotamer_index[block_rot_offset];
    int const local_molten_block_index =
        molten_block_index - pose_molten_block_offset;
    bc_rot_offset_for_molten_block[pose][local_molten_block_index] =
        bc_rot_for_block;
  });
  Ops<D>::template forall<launch_t>(
      mgr, n_molten_blocks, fill_bc_rot_offset_for_molten_block);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2,
    typename View3,
    typename View4,
    typename View5>
void launch_ig_block_rotamer_counts(
    Context& mgr,
    View0 molten_block_to_block_inds,
    View1 molten_block_offset_for_pose,
    View2 n_molten_blocks_per_pose,
    View3 bc_rot_offset_for_molten_block,
    View4 bc_rot_offset_for_pose,
    View5 n_bc_rots_for_molten_block,
    int max_n_blocks,
    int64_t n_kept_rotamers) {
  LAUNCH_BOX_32;
  int const n_poses = n_molten_blocks_per_pose.size(0);
  int const n_molten_blocks = score::common::checked_dispatch_size(
      molten_block_to_block_inds.size(0), "retained blocks");
  auto fill_n_bc_rots_for_molten_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const molten_block_index = index;
    int const block_and_pose = molten_block_to_block_inds[molten_block_index];
    int const pose = block_and_pose / max_n_blocks;
    int const block = block_and_pose % max_n_blocks;
    int const local_molten_block_index =
        molten_block_index - molten_block_offset_for_pose[pose];
    int const pose_n_molten_blocks = n_molten_blocks_per_pose[pose];
    int const offset_for_molten_block =
        bc_rot_offset_for_molten_block[pose][local_molten_block_index];
    if (local_molten_block_index < pose_n_molten_blocks - 1) {
      n_bc_rots_for_molten_block[pose][local_molten_block_index] =
          (bc_rot_offset_for_molten_block[pose][local_molten_block_index + 1]
           - offset_for_molten_block);
    } else {
      if (pose < n_poses - 1) {
        n_bc_rots_for_molten_block[pose][local_molten_block_index] =
            (bc_rot_offset_for_pose[pose + 1] - offset_for_molten_block);
      } else {
        n_bc_rots_for_molten_block[pose][local_molten_block_index] =
            (n_kept_rotamers - offset_for_molten_block);
      }
    }
  });
  Ops<D>::template forall<launch_t>(
      mgr, n_molten_blocks, fill_n_bc_rots_for_molten_block);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename View0,
    typename View1,
    typename View2,
    typename View3,
    typename View4,
    typename View5>
void launch_ig_rotamer_blocks(
    Context& mgr,
    View0 new_to_old_rotamer_index,
    View1 block_ind_for_rot,
    View2 pose_for_rot,
    View3 block_to_molten_block_inds,
    View4 molten_block_offset_for_pose,
    View5 molten_block_ind_for_bc_rot,
    int max_n_blocks) {
  LAUNCH_BOX_32;
  int const n_kept_rotamers = score::common::checked_dispatch_size(
      new_to_old_rotamer_index.size(0), "retained rotamers");
  auto fill_molten_block_ind_for_bc_rot = ([=] TMOL_DEVICE_FUNC(int index) {
    int const bc_rot = index;
    int const orig_rot = new_to_old_rotamer_index[bc_rot];
    int const block = block_ind_for_rot[orig_rot];
    int const pose = pose_for_rot[orig_rot];
    int const molten_block_index =
        block_to_molten_block_inds[pose * max_n_blocks + block];

    // This needs to be the *local* molten block index, counting from 0
    // within the pose that it is part of
    molten_block_ind_for_bc_rot[bc_rot] =
        molten_block_index - molten_block_offset_for_pose[pose];
  });
  Ops<D>::template forall<launch_t>(
      mgr, n_kept_rotamers, fill_molten_block_ind_for_bc_rot);
}

}  // namespace tmol::pack::compiled
