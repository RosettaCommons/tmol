#pragma once

// Mature preliminary graph and rotamer/block lower bounds for bump pruning.
// Pack owns allocation; Ops and Context own execution and scans. Inputs must
// have canonical upper-triangle inter-block entries and diagonal one-body
// entries. A unique prefix can execute independently; its duplicate tail is
// accumulated serially on CPU. Bounds group alternatives by BLOCK, not type.
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/accumulate.hh>
#include <tuple>

namespace tmol::pack::compiled {
template <template <Device> class DeviceDispatch, Device D, class Real, class Int,
          template <typename, size_t, Device> class Pack, class Context>
auto prepare_ig_bump_bounds(
    Context& mgr, int chunk_size,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    TView<Int, 1, D> pose_for_rot,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int32_t, 2, D> sparse_inds,
    TView<Real, 1, D> sparse_energies, int64_t n_unique_entries)
    -> std::tuple<Pack<Real, 1, D>, Pack<Real, 2, D>> {
  int const n_poses = n_rots_for_block.size(0);
  int const max_n_blocks = n_rots_for_block.size(1);
  int const n_rotamers = pose_for_rot.size(0);
  int const n_pose_block_cells = score::common::checked_dispatch_product(
      n_poses, max_n_blocks, "bump-bound block dispatch");
  int const n_sparse_entries_dispatch = score::common::checked_dispatch_size(
      sparse_inds.size(1), "bump-bound sparse entries");
  int64_t const n_unique = n_unique_entries < 0 ? n_sparse_entries_dispatch : n_unique_entries;
  LAUNCH_BOX_32;
  constexpr int nt = launch_t::sm_ptx::nt;
  // The bump check builds dense block-pair tables; only it needs their
  // square dispatch. The no-bump path below is sparse in block pairs.
  int const n_block_pair_cells = score::common::checked_dispatch_product(
      max_n_blocks,
      max_n_blocks,
      "interaction-graph block-pair dispatch per pose");
  int const n_pose_block_pair_cells = score::common::checked_dispatch_product(
      n_poses, n_block_pair_cells, "interaction-graph block-pair dispatch");
  auto energy1b_tp = Pack<Real, 1, D>::zeros({n_rotamers});
  auto energy1b = energy1b_tp.view;
  auto n_chunks_for_block_tp =
      Pack<int32_t, 2, D>::zeros({n_poses, max_n_blocks});
  auto n_chunks_for_block = n_chunks_for_block_tp.view;

  auto count_n_chunks_for_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / max_n_blocks;
    int const block = index % max_n_blocks;
    int const n_rots = n_rots_for_block[pose][block];
    if (n_rots != 0) {
      int const n_chunks = (n_rots - 1) / chunk_size + 1;
      n_chunks_for_block[pose][block] = n_chunks;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_pose_block_cells, count_n_chunks_for_block);

  auto respair_is_adjacent_tp =
      Pack<int32_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto respair_is_adjacent = respair_is_adjacent_tp.view;

  auto note_adjacent_respairs = ([=] TMOL_DEVICE_FUNC(int64_t index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    if (block1 == block2) {
      return;
    }
    // Assert: block1 < block2
    // All terms must report scores in the upper triangle of the
    // (hypothetical) n-rotamers x n-rotamers matrix of energies;
    // Code below relies on this.
    if (block1 > block2) {
      printf(
          "Assumption violated! block1 %d < block2 %d for index %lld (pose "
          "%d)\n",
          block1,
          block2,
          static_cast<long long>(index),
          pose);
    } else {
      DeviceDispatch<D>::store_idempotent(
          respair_is_adjacent[pose][block1][block2], int32_t(1));
    }
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_sparse_entries_dispatch, note_adjacent_respairs);

  auto n_chunks_for_block_pair_tp =
      Pack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto n_chunks_for_block_pair = n_chunks_for_block_pair_tp.view;

  auto note_n_chunks_for_block_pair = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / n_block_pair_cells;
    index = index - pose * n_block_pair_cells;
    int const block1 = index / max_n_blocks;
    int const block2 = index % max_n_blocks;

    // We don't have to worry about block1 > block2 as those will not
    // have entries in the sparse_inds input tensors, but we do
    // have to worry about block1 == block2 for one-body energies
    if (respair_is_adjacent[pose][block1][block2]) {
      int const n_chunks1 = n_chunks_for_block[pose][block1];
      int const n_chunks2 = n_chunks_for_block[pose][block2];
      int64_t const n_chunk_pairs = int64_t(n_chunks1) * n_chunks2;
      n_chunks_for_block_pair[pose][block1][block2] = n_chunk_pairs;
      n_chunks_for_block_pair[pose][block2][block1] = n_chunk_pairs;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_pose_block_pair_cells, note_n_chunks_for_block_pair);

  auto chunk_pair_offset_for_block_pair_tp =
      Pack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto chunk_pair_offset_for_block_pair =
      chunk_pair_offset_for_block_pair_tp.view;

  int64_t const n_adjacent_chunk_pairs_total_64 =
      DeviceDispatch<D>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          n_chunks_for_block_pair.data(),
          chunk_pair_offset_for_block_pair.data(),
          n_pose_block_pair_cells,
          tmol::portable::plus_t<int64_t>());
  int const n_adjacent_chunk_pairs_total =
      score::common::checked_dispatch_size(
          n_adjacent_chunk_pairs_total_64,
          "interaction-graph chunk-pair dispatch");

  auto chunk_pair_adjacency_tp =
      Pack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
  auto chunk_pair_adjacency = chunk_pair_adjacency_tp.view;

  auto note_adjacent_chunk_pairs = ([=] TMOL_DEVICE_FUNC(int64_t index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    if (block1 == block2) {
      return;
    }

    int const block1_rot_offset = rot_offset_for_block[pose][block1];
    int const block2_rot_offset = rot_offset_for_block[pose][block2];
    int const local_rot1 = rot1 - block1_rot_offset;
    int const local_rot2 = rot2 - block2_rot_offset;
    int const chunk1 = local_rot1 / chunk_size;
    int const chunk2 = local_rot2 / chunk_size;
    int const n_rots_block1 = n_rots_for_block[pose][block1];
    int const n_rots_block2 = n_rots_for_block[pose][block2];
    int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;

    int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
    int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
    int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);
    int const chunk2_size = (overhang2 > chunk_size ? chunk_size : overhang2);

    int64_t const block_pair_chunk_offset_ij =
        chunk_pair_offset_for_block_pair[pose][block1][block2];
    int64_t const block_pair_chunk_offset_ji =
        chunk_pair_offset_for_block_pair[pose][block2][block1];

    // multiple threads will write exactly these values to these entries in
    // the chunk_pair_adjacency table
    int64_t const n_pairs = int64_t(chunk1_size) * chunk2_size;
    DeviceDispatch<D>::store_idempotent(
        chunk_pair_adjacency
            [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2],
        n_pairs);
    DeviceDispatch<D>::store_idempotent(
        chunk_pair_adjacency
            [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1],
        n_pairs);
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_sparse_entries_dispatch, note_adjacent_chunk_pairs);

  auto chunk_pair_offsets_tp =
      Pack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
  auto chunk_pair_offsets = chunk_pair_offsets_tp.view;

  int64_t const n_two_body_energies =
      DeviceDispatch<D>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          chunk_pair_adjacency.data(),
          chunk_pair_offsets.data(),
          n_adjacent_chunk_pairs_total,
          tmol::portable::plus_t<int64_t>());

  auto energy2b_tp = Pack<Real, 1, D>::zeros({n_two_body_energies});
  auto energy2b = energy2b_tp.view;

  auto record_energies_in_energy1b_and_energy2b =
      ([=] TMOL_DEVICE_FUNC(int64_t index) {
        int const pose = sparse_inds[0][index];
        int const rot1 = sparse_inds[1][index];
        int const rot2 = sparse_inds[2][index];
        Real const energy = sparse_energies[index];
        int const block1 = block_ind_for_rot[rot1];
        int const block2 = block_ind_for_rot[rot2];
        if (block1 == block2) {
          score::common::accumulate<D, Real>::add(energy1b[rot1], energy);
        } else {
          int const block1_rot_offset = rot_offset_for_block[pose][block1];
          int const block2_rot_offset = rot_offset_for_block[pose][block2];
          int const local_rot1 = rot1 - block1_rot_offset;
          int const local_rot2 = rot2 - block2_rot_offset;
          int const chunk1 = local_rot1 / chunk_size;
          int const chunk2 = local_rot2 / chunk_size;
          int const n_rots_block1 = n_rots_for_block[pose][block1];
          int const n_rots_block2 = n_rots_for_block[pose][block2];
          int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
          int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;

          int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
          int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
          int const chunk1_size =
              (overhang1 > chunk_size ? chunk_size : overhang1);
          int const chunk2_size =
              (overhang2 > chunk_size ? chunk_size : overhang2);

          int const rot_ind_wi_chunk1 = local_rot1 - chunk1 * chunk_size;
          int const rot_ind_wi_chunk2 = local_rot2 - chunk2 * chunk_size;

          int64_t const block_pair_chunk_offset_ij =
              chunk_pair_offset_for_block_pair[pose][block1][block2];
          int64_t const block_pair_chunk_offset_ji =
              chunk_pair_offset_for_block_pair[pose][block2][block1];

          int64_t const chunk_offset_ij = chunk_pair_offsets
              [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2];
          int64_t const chunk_offset_ji = chunk_pair_offsets
              [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1];

          score::common::accumulate<D, Real>::add(
              energy2b
                  [chunk_offset_ij + rot_ind_wi_chunk1 * chunk2_size
                   + rot_ind_wi_chunk2],
              energy);
          score::common::accumulate<D, Real>::add(
              energy2b
                  [chunk_offset_ji + rot_ind_wi_chunk2 * chunk1_size
                   + rot_ind_wi_chunk1],
              energy);
        }
      });
  // Same split as the main accumulation below: this lambda's writes are
  // disjoint only while the keys are distinct, so the parallel pass stops
  // at the unique prefix and the duplicate tail runs serially.
  auto record_duplicate_energies = ([=] TMOL_DEVICE_FUNC(int64_t offset) {
    record_energies_in_energy1b_and_energy2b(n_unique + offset);
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_unique, record_energies_in_energy1b_and_energy2b);
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_sparse_entries_dispatch - n_unique, record_duplicate_energies);

  // Mark the chunk_pair_offset_for_block_pair that are not adjacent w/ -1s
  // Mark the chunk_pair_offsets that are not adjacent w/ -1s

  auto sentinel_out_non_adjacent_block_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const pose = index / n_block_pair_cells;
        index = index - pose * n_block_pair_cells;
        int const block1 = index / max_n_blocks;
        int const block2 = index % max_n_blocks;

        // We don't have to worry about block1 >= block2 as those will not
        // have entries in the sparse_inds input tensors
        if (block1 <= block2 && !respair_is_adjacent[pose][block1][block2]) {
          chunk_pair_offset_for_block_pair[pose][block1][block2] = -1;
          chunk_pair_offset_for_block_pair[pose][block2][block1] = -1;
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_pose_block_pair_cells, sentinel_out_non_adjacent_block_pairs);

  auto sentinel_out_non_adjacent_chunk_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const n_pairs_for_chunk = chunk_pair_adjacency.data()[index];
        if (n_pairs_for_chunk == 0) {
          chunk_pair_offsets[index] = -1;
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr,
      n_adjacent_chunk_pairs_total,
      sentinel_out_non_adjacent_chunk_pairs);

  // Okay, from here we want to compute the best energy
  // for each rotamer, akin to (original, weakest) DEE criterion:
  // What's the best energy a rotamer can get in any assignment
  // of rotamers.
  // Then we will eliminate any rotamer iff:
  // 1. a rotamer's best energy is worse than +5 kcal/mol, and
  // 2. there is at least one rotamer of that block with a best energy
  // better +5

  Pack<Real, 1, D> best_energy_for_rot_tp =
      Pack<Real, 1, D>::full({n_rotamers}, 0);
  auto best_energy_for_rot = best_energy_for_rot_tp.view;

  auto compute_best_energy_for_rotamers = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const rot1 = cta / max_n_blocks;
    int const block2 = cta % max_n_blocks;
    int const block1 = block_ind_for_rot[rot1];
    int const pose = pose_for_rot[rot1];
    int const n_rots_block1 = n_rots_for_block[pose][block1];
    int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
    int const rot_in_block1 = rot1 - rot_offset_for_block[pose][block1];
    int const chunk1 = rot_in_block1 / chunk_size;
    int const rot_ind_wi_chunk1 = rot_in_block1 - chunk1 * chunk_size;
    int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
    int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);

    if (block2 == block1) {
      auto accum_one_body_energy = ([&] TMOL_DEVICE_FUNC(int tid) {
        if (tid == 0) {
          Real one_body_energy = energy1b[rot1];
          score::common::accumulate<D, Real>::add(
              best_energy_for_rot[rot1], one_body_energy);
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          accum_one_body_energy);
      return;
    }

    int64_t const block_pair_chunk_offset =
        chunk_pair_offset_for_block_pair[pose][block1][block2];
    if (block_pair_chunk_offset == -1) {
      return;
    }
    auto find_min_energy_for_rotamer_and_block2 =
        ([&] TMOL_DEVICE_FUNC(int tid) {
          bool first_rotamer = true;
          Real best_energy_for_rot1_w_block2 = 1234;
          int const n_rots_block2 = n_rots_for_block[pose][block2];
          int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;
          for (int local_rot2 = tid; local_rot2 < n_rots_block2;
               local_rot2 += nt) {
            int const chunk2 = local_rot2 / chunk_size;
            int const rot_ind_wi_chunk2 = local_rot2 - chunk2 * chunk_size;

            int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
            int const chunk2_size =
                (overhang2 > chunk_size ? chunk_size : overhang2);

            int64_t const chunk_pair_offset = chunk_pair_offsets
                [block_pair_chunk_offset + chunk1 * n_chunks2 + chunk2];
            if (chunk_pair_offset == -1) {
              continue;
            }

            Real e2b = energy2b
                [chunk_pair_offset + rot_ind_wi_chunk1 * chunk2_size
                 + rot_ind_wi_chunk2];
            if (first_rotamer || e2b < best_energy_for_rot1_w_block2) {
              best_energy_for_rot1_w_block2 = e2b;
              first_rotamer = false;
            }
          }

          bool all_first_rotamer =
              DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
                  first_rotamer, tmol::portable::minimum_t<bool>());
          Real best_energy =
              DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
                  best_energy_for_rot1_w_block2,
                  tmol::portable::minimum_t<Real>());

          if (!all_first_rotamer && tid == 0) {
            if (best_energy != 0.0) {
              score::common::accumulate<D, Real>::add(
                  best_energy_for_rot[rot1], best_energy);
            }
          }
        });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        find_min_energy_for_rotamer_and_block2);
  });
  DeviceDispatch<D>::template foreach_grouped_workgroup<launch_t>(
      mgr, n_rotamers, max_n_blocks, compute_best_energy_for_rotamers);

  // Now let's figure out the best energy for each block
  auto best_energy_per_block_tp =
      Pack<Real, 2, D>::full({n_poses, max_n_blocks}, 1234);
  auto best_energy_per_block = best_energy_per_block_tp.view;

  // assign a CTA to each block and stride across the energy2b table to
  // find its best rotamer's energy
  auto compute_best_energy_per_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const pose = cta / max_n_blocks;
    int const block = cta % max_n_blocks;

    if (n_rots_for_block[pose][block] == 0) {
      return;
    }
    auto find_min_energy_for_block = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real best_energy = 1234;
      bool first_rotamer = true;
      int const block_n_rots = n_rots_for_block[pose][block];
      for (int rot_in_block = tid; rot_in_block < block_n_rots;
           rot_in_block += nt) {
        int const rot = rot_offset_for_block[pose][block] + rot_in_block;
        Real energy_for_rot = best_energy_for_rot[rot];
        if (first_rotamer || energy_for_rot < best_energy) {
          best_energy = energy_for_rot;
          first_rotamer = false;
        }
      }
      bool none_found =
          DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
              first_rotamer, tmol::portable::minimum_t<bool>());
      Real block_best_energy =
          DeviceDispatch<D>::template shuffle_reduce_in_workgroup<nt>(
              best_energy, tmol::portable::minimum_t<Real>());
      if (!none_found && tid == 0) {
        best_energy_per_block[pose][block] = block_best_energy;
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        find_min_energy_for_block);
  });
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      mgr, n_pose_block_cells, compute_best_energy_per_block);

  return std::make_tuple(best_energy_for_rot_tp, best_energy_per_block_tp);
}
}  // namespace tmol::pack::compiled
