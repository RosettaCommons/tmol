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
#include <stdexcept>

namespace tmol::pack::compiled {
struct BumpGraphSizes {
  int poses, blocks, rotamers, entries;
  int block_cells, pair_cells_per_pose, pair_cells;
  int64_t unique_entries;
};

inline BumpGraphSizes ig_bump_workspace_sizes(
    int64_t poses,
    int64_t blocks,
    int64_t rotamers,
    int64_t entries,
    int chunk_size,
    int64_t unique_entries) {
  if (chunk_size <= 0 || unique_entries > entries)
    throw std::invalid_argument(
        "Invalid bump graph chunk size or unique prefix");
  using score::common::checked_dispatch_product;
  using score::common::checked_dispatch_size;
  return {
      checked_dispatch_size(poses, "bump poses"),
      checked_dispatch_size(blocks, "bump blocks"),
      checked_dispatch_size(rotamers, "bump rotamers"),
      checked_dispatch_size(entries, "bump entries"),
      checked_dispatch_product(poses, blocks, "bump block cells"),
      checked_dispatch_product(blocks, blocks, "bump block pairs"),
      checked_dispatch_product(
          poses,
          checked_dispatch_product(blocks, blocks, "bump block pairs"),
          "bump pose block pairs"),
      unique_entries < 0 ? entries : unique_entries};
}

template <class Real, class Int, Device D>
struct BumpGraphContext {
  int chunk_size;
  int64_t unique_entries;
  TView<Int, 2, D> n_rots_for_block, rot_offset_for_block;
  TView<Int, 1, D> pose_for_rot;
  TView<int32_t, 1, D> block_ind_for_rot;
  TView<int32_t, 2, D> sparse_inds;
  TView<Real, 1, D> sparse_energies;
  TView<int32_t, 2, D> n_chunks_for_block;
  TView<int32_t, 3, D> respair_is_adjacent;
  TView<int64_t, 3, D> n_chunks_for_block_pair,
      chunk_pair_offset_for_block_pair;
  TView<int64_t, 1, D> chunk_pair_adjacency, chunk_pair_offsets;
  TView<Real, 1, D> energy1b, energy2b, best_energy_for_rot;
  TView<Real, 2, D> best_energy_per_block;

  BumpGraphSizes sizes() const {
    return ig_bump_workspace_sizes(
        n_rots_for_block.size(0),
        n_rots_for_block.size(1),
        pose_for_rot.size(0),
        sparse_inds.size(1),
        chunk_size,
        unique_entries);
  }
};

template <
    template <Device> class DeviceDispatch,
    Device D,
    class Real,
    class Int,
    class Context>
void launch_ig_bump_block_counts(
    Context& mgr, BumpGraphContext<Real, Int, D> data) {
  auto const sizes = data.sizes();
  LAUNCH_BOX_32;
  auto count_n_chunks_for_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / sizes.blocks;
    int const block = index % sizes.blocks;
    int const n_rots = data.n_rots_for_block[pose][block];
    if (n_rots != 0) {
      int const n_chunks = (n_rots - 1) / data.chunk_size + 1;
      data.n_chunks_for_block[pose][block] = n_chunks;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, sizes.block_cells, count_n_chunks_for_block);

  auto note_adjacent_respairs = ([=] TMOL_DEVICE_FUNC(int64_t index) {
    int const pose = data.sparse_inds[0][index];
    int const rot1 = data.sparse_inds[1][index];
    int const rot2 = data.sparse_inds[2][index];
    int const block1 = data.block_ind_for_rot[rot1];
    int const block2 = data.block_ind_for_rot[rot2];
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
          data.respair_is_adjacent[pose][block1][block2], int32_t(1));
    }
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, sizes.entries, note_adjacent_respairs);

  auto note_n_chunks_for_block_pair = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / sizes.pair_cells_per_pose;
    index = index - pose * sizes.pair_cells_per_pose;
    int const block1 = index / sizes.blocks;
    int const block2 = index % sizes.blocks;

    // We don't have to worry about block1 > block2 as those will not
    // have entries in the data.sparse_inds input tensors, but we do
    // have to worry about block1 == block2 for one-body energies
    if (data.respair_is_adjacent[pose][block1][block2]) {
      int const n_chunks1 = data.n_chunks_for_block[pose][block1];
      int const n_chunks2 = data.n_chunks_for_block[pose][block2];
      int64_t const n_chunk_pairs = int64_t(n_chunks1) * n_chunks2;
      data.n_chunks_for_block_pair[pose][block1][block2] = n_chunk_pairs;
      data.n_chunks_for_block_pair[pose][block2][block1] = n_chunk_pairs;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, sizes.pair_cells, note_n_chunks_for_block_pair);
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    class Real,
    class Int,
    class Context>
void launch_ig_bump_chunk_sizes(
    Context& mgr, BumpGraphContext<Real, Int, D> data) {
  auto const sizes = data.sizes();
  LAUNCH_BOX_32;
  auto note_adjacent_chunk_pairs = ([=] TMOL_DEVICE_FUNC(int64_t index) {
    int const pose = data.sparse_inds[0][index];
    int const rot1 = data.sparse_inds[1][index];
    int const rot2 = data.sparse_inds[2][index];
    int const block1 = data.block_ind_for_rot[rot1];
    int const block2 = data.block_ind_for_rot[rot2];
    if (block1 == block2) {
      return;
    }

    int const block1_rot_offset = data.rot_offset_for_block[pose][block1];
    int const block2_rot_offset = data.rot_offset_for_block[pose][block2];
    int const local_rot1 = rot1 - block1_rot_offset;
    int const local_rot2 = rot2 - block2_rot_offset;
    int const chunk1 = local_rot1 / data.chunk_size;
    int const chunk2 = local_rot2 / data.chunk_size;
    int const n_rots_block1 = data.n_rots_for_block[pose][block1];
    int const n_rots_block2 = data.n_rots_for_block[pose][block2];
    int const n_chunks1 = (n_rots_block1 - 1) / data.chunk_size + 1;
    int const n_chunks2 = (n_rots_block2 - 1) / data.chunk_size + 1;

    int const overhang1 = n_rots_block1 - chunk1 * data.chunk_size;
    int const overhang2 = n_rots_block2 - chunk2 * data.chunk_size;
    int const chunk1_size =
        (overhang1 > data.chunk_size ? data.chunk_size : overhang1);
    int const chunk2_size =
        (overhang2 > data.chunk_size ? data.chunk_size : overhang2);

    int64_t const block_pair_chunk_offset_ij =
        data.chunk_pair_offset_for_block_pair[pose][block1][block2];
    int64_t const block_pair_chunk_offset_ji =
        data.chunk_pair_offset_for_block_pair[pose][block2][block1];

    // multiple threads will write exactly these values to these entries in
    // the data.chunk_pair_adjacency table
    int64_t const n_pairs = int64_t(chunk1_size) * chunk2_size;
    DeviceDispatch<D>::store_idempotent(
        data.chunk_pair_adjacency
            [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2],
        n_pairs);
    DeviceDispatch<D>::store_idempotent(
        data.chunk_pair_adjacency
            [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1],
        n_pairs);
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, sizes.entries, note_adjacent_chunk_pairs);
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    class Real,
    class Int,
    class Context>
void launch_ig_bump_bounds(Context& mgr, BumpGraphContext<Real, Int, D> data) {
  auto const sizes = data.sizes();
  LAUNCH_BOX_32;
  constexpr int nt = launch_t::sm_ptx::nt;
  int const n_adjacent_chunk_pairs_total = score::common::checked_dispatch_size(
      data.chunk_pair_adjacency.size(0), "bump chunk pairs");
  auto record_energies_in_energy1b_and_energy2b =
      ([=] TMOL_DEVICE_FUNC(int64_t index) {
        int const pose = data.sparse_inds[0][index];
        int const rot1 = data.sparse_inds[1][index];
        int const rot2 = data.sparse_inds[2][index];
        Real const energy = data.sparse_energies[index];
        int const block1 = data.block_ind_for_rot[rot1];
        int const block2 = data.block_ind_for_rot[rot2];
        if (block1 == block2) {
          score::common::accumulate<D, Real>::add(data.energy1b[rot1], energy);
        } else {
          int const block1_rot_offset = data.rot_offset_for_block[pose][block1];
          int const block2_rot_offset = data.rot_offset_for_block[pose][block2];
          int const local_rot1 = rot1 - block1_rot_offset;
          int const local_rot2 = rot2 - block2_rot_offset;
          int const chunk1 = local_rot1 / data.chunk_size;
          int const chunk2 = local_rot2 / data.chunk_size;
          int const n_rots_block1 = data.n_rots_for_block[pose][block1];
          int const n_rots_block2 = data.n_rots_for_block[pose][block2];
          int const n_chunks1 = (n_rots_block1 - 1) / data.chunk_size + 1;
          int const n_chunks2 = (n_rots_block2 - 1) / data.chunk_size + 1;

          int const overhang1 = n_rots_block1 - chunk1 * data.chunk_size;
          int const overhang2 = n_rots_block2 - chunk2 * data.chunk_size;
          int const chunk1_size =
              (overhang1 > data.chunk_size ? data.chunk_size : overhang1);
          int const chunk2_size =
              (overhang2 > data.chunk_size ? data.chunk_size : overhang2);

          int const rot_ind_wi_chunk1 = local_rot1 - chunk1 * data.chunk_size;
          int const rot_ind_wi_chunk2 = local_rot2 - chunk2 * data.chunk_size;

          int64_t const block_pair_chunk_offset_ij =
              data.chunk_pair_offset_for_block_pair[pose][block1][block2];
          int64_t const block_pair_chunk_offset_ji =
              data.chunk_pair_offset_for_block_pair[pose][block2][block1];

          int64_t const chunk_offset_ij =
              data.chunk_pair_offsets
                  [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2];
          int64_t const chunk_offset_ji =
              data.chunk_pair_offsets
                  [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1];

          score::common::accumulate<D, Real>::add(
              data.energy2b
                  [chunk_offset_ij + rot_ind_wi_chunk1 * chunk2_size
                   + rot_ind_wi_chunk2],
              energy);
          score::common::accumulate<D, Real>::add(
              data.energy2b
                  [chunk_offset_ji + rot_ind_wi_chunk2 * chunk1_size
                   + rot_ind_wi_chunk1],
              energy);
        }
      });
  // Same split as the main accumulation below: this lambda's writes are
  // disjoint only while the keys are distinct, so the parallel pass stops
  // at the unique prefix and the duplicate tail runs serially.
  auto record_duplicate_energies = ([=] TMOL_DEVICE_FUNC(int64_t offset) {
    record_energies_in_energy1b_and_energy2b(sizes.unique_entries + offset);
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, sizes.unique_entries, record_energies_in_energy1b_and_energy2b);
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, sizes.entries - sizes.unique_entries, record_duplicate_energies);

  // Mark the data.chunk_pair_offset_for_block_pair that are not adjacent w/ -1s
  // Mark the data.chunk_pair_offsets that are not adjacent w/ -1s

  auto sentinel_out_non_adjacent_block_pairs = ([=] TMOL_DEVICE_FUNC(
                                                    int index) {
    int const pose = index / sizes.pair_cells_per_pose;
    index = index - pose * sizes.pair_cells_per_pose;
    int const block1 = index / sizes.blocks;
    int const block2 = index % sizes.blocks;

    // We don't have to worry about block1 >= block2 as those will not
    // have entries in the data.sparse_inds input tensors
    if (block1 <= block2 && !data.respair_is_adjacent[pose][block1][block2]) {
      data.chunk_pair_offset_for_block_pair[pose][block1][block2] = -1;
      data.chunk_pair_offset_for_block_pair[pose][block2][block1] = -1;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, sizes.pair_cells, sentinel_out_non_adjacent_block_pairs);

  auto sentinel_out_non_adjacent_chunk_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const n_pairs_for_chunk = data.chunk_pair_adjacency.data()[index];
        if (n_pairs_for_chunk == 0) {
          data.chunk_pair_offsets[index] = -1;
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_adjacent_chunk_pairs_total, sentinel_out_non_adjacent_chunk_pairs);

  // Okay, from here we want to compute the best energy
  // for each rotamer, akin to (original, weakest) DEE criterion:
  // What's the best energy a rotamer can get in any assignment
  // of rotamers.
  // Then we will eliminate any rotamer iff:
  // 1. a rotamer's best energy is worse than +5 kcal/mol, and
  // 2. there is at least one rotamer of that block with a best energy
  // better +5

  auto compute_best_energy_for_rotamers = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const rot1 = cta / sizes.blocks;
    int const block2 = cta % sizes.blocks;
    int const block1 = data.block_ind_for_rot[rot1];
    int const pose = data.pose_for_rot[rot1];
    int const n_rots_block1 = data.n_rots_for_block[pose][block1];
    int const n_chunks1 = (n_rots_block1 - 1) / data.chunk_size + 1;
    int const rot_in_block1 = rot1 - data.rot_offset_for_block[pose][block1];
    int const chunk1 = rot_in_block1 / data.chunk_size;
    int const rot_ind_wi_chunk1 = rot_in_block1 - chunk1 * data.chunk_size;
    int const overhang1 = n_rots_block1 - chunk1 * data.chunk_size;
    int const chunk1_size =
        (overhang1 > data.chunk_size ? data.chunk_size : overhang1);

    if (block2 == block1) {
      auto accum_one_body_energy = ([&] TMOL_DEVICE_FUNC(int tid) {
        if (tid == 0) {
          Real one_body_energy = data.energy1b[rot1];
          score::common::accumulate<D, Real>::add(
              data.best_energy_for_rot[rot1], one_body_energy);
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          accum_one_body_energy);
      return;
    }

    int64_t const block_pair_chunk_offset =
        data.chunk_pair_offset_for_block_pair[pose][block1][block2];
    if (block_pair_chunk_offset == -1) {
      return;
    }
    auto find_min_energy_for_rotamer_and_block2 = ([&] TMOL_DEVICE_FUNC(
                                                       int tid) {
      bool first_rotamer = true;
      Real best_energy_for_rot1_w_block2 = 1234;
      int const n_rots_block2 = data.n_rots_for_block[pose][block2];
      int const n_chunks2 = (n_rots_block2 - 1) / data.chunk_size + 1;
      for (int local_rot2 = tid; local_rot2 < n_rots_block2; local_rot2 += nt) {
        int const chunk2 = local_rot2 / data.chunk_size;
        int const rot_ind_wi_chunk2 = local_rot2 - chunk2 * data.chunk_size;

        int const overhang2 = n_rots_block2 - chunk2 * data.chunk_size;
        int const chunk2_size =
            (overhang2 > data.chunk_size ? data.chunk_size : overhang2);

        int64_t const chunk_pair_offset =
            data.chunk_pair_offsets
                [block_pair_chunk_offset + chunk1 * n_chunks2 + chunk2];
        if (chunk_pair_offset == -1) {
          continue;
        }

        Real e2b = data.energy2b
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
              best_energy_for_rot1_w_block2, tmol::portable::minimum_t<Real>());

      if (!all_first_rotamer && tid == 0) {
        if (best_energy != 0.0) {
          score::common::accumulate<D, Real>::add(
              data.best_energy_for_rot[rot1], best_energy);
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        find_min_energy_for_rotamer_and_block2);
  });
  DeviceDispatch<D>::template foreach_grouped_workgroup<launch_t>(
      mgr, sizes.rotamers, sizes.blocks, compute_best_energy_for_rotamers);

  // Now let's figure out the best energy for each block
  // assign a CTA to each block and stride across the data.energy2b table to
  // find its best rotamer's energy
  auto compute_best_energy_per_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const pose = cta / sizes.blocks;
    int const block = cta % sizes.blocks;

    if (data.n_rots_for_block[pose][block] == 0) {
      return;
    }
    auto find_min_energy_for_block = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real best_energy = 1234;
      bool first_rotamer = true;
      int const block_n_rots = data.n_rots_for_block[pose][block];
      for (int rot_in_block = tid; rot_in_block < block_n_rots;
           rot_in_block += nt) {
        int const rot = data.rot_offset_for_block[pose][block] + rot_in_block;
        Real energy_for_rot = data.best_energy_for_rot[rot];
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
        data.best_energy_per_block[pose][block] = block_best_energy;
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        find_min_energy_for_block);
  });
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      mgr, sizes.block_cells, compute_best_energy_per_block);
}

template <
    template <Device> class DeviceDispatch,
    Device D,
    class Real,
    class Int,
    template <typename, size_t, Device> class Pack,
    class Context>
auto prepare_ig_bump_bounds(
    Context& mgr,
    int chunk_size,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    TView<Int, 1, D> pose_for_rot,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int32_t, 2, D> sparse_inds,
    TView<Real, 1, D> sparse_energies,
    int64_t n_unique_entries)
    -> std::tuple<Pack<Real, 1, D>, Pack<Real, 2, D>> {
  BumpGraphContext<Real, Int, D> data{
      chunk_size,
      n_unique_entries,
      n_rots_for_block,
      rot_offset_for_block,
      pose_for_rot,
      block_ind_for_rot,
      sparse_inds,
      sparse_energies};
  auto const sizes = data.sizes();
  auto energy1b = Pack<Real, 1, D>::zeros({sizes.rotamers});
  auto n_chunks = Pack<int32_t, 2, D>::zeros({sizes.poses, sizes.blocks});
  auto adjacent =
      Pack<int32_t, 3, D>::zeros({sizes.poses, sizes.blocks, sizes.blocks});
  auto pair_counts =
      Pack<int64_t, 3, D>::zeros({sizes.poses, sizes.blocks, sizes.blocks});
  auto pair_offsets =
      Pack<int64_t, 3, D>::zeros({sizes.poses, sizes.blocks, sizes.blocks});
  data.energy1b = energy1b.view;
  data.n_chunks_for_block = n_chunks.view;
  data.respair_is_adjacent = adjacent.view;
  data.n_chunks_for_block_pair = pair_counts.view;
  data.chunk_pair_offset_for_block_pair = pair_offsets.view;
  launch_ig_bump_block_counts<DeviceDispatch, D>(mgr, data);
  int64_t const chunk_pairs = DeviceDispatch<D>::template scan_and_return_total<
      tmol::portable::scan_type::exclusive>(
      mgr,
      pair_counts.view.data(),
      pair_offsets.view.data(),
      sizes.pair_cells,
      tmol::portable::plus_t<int64_t>());
  int const chunk_pair_dispatch =
      score::common::checked_dispatch_size(chunk_pairs, "bump chunk pairs");
  auto chunk_sizes = Pack<int64_t, 1, D>::zeros({chunk_pairs});
  auto chunk_offsets = Pack<int64_t, 1, D>::zeros({chunk_pairs});
  data.chunk_pair_adjacency = chunk_sizes.view;
  data.chunk_pair_offsets = chunk_offsets.view;
  launch_ig_bump_chunk_sizes<DeviceDispatch, D>(mgr, data);
  int64_t const pair_energies =
      DeviceDispatch<D>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          chunk_sizes.view.data(),
          chunk_offsets.view.data(),
          chunk_pair_dispatch,
          tmol::portable::plus_t<int64_t>());
  auto energy2b = Pack<Real, 1, D>::zeros({pair_energies});
  auto best_rotamer = Pack<Real, 1, D>::zeros({sizes.rotamers});
  auto best_block = Pack<Real, 2, D>::full({sizes.poses, sizes.blocks}, 1234);
  data.energy2b = energy2b.view;
  data.best_energy_for_rot = best_rotamer.view;
  data.best_energy_per_block = best_block.view;
  launch_ig_bump_bounds<DeviceDispatch, D>(mgr, data);
  return std::make_tuple(best_rotamer, best_block);
}
}  // namespace tmol::pack::compiled
