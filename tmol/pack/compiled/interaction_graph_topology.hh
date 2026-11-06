#pragma once

// Caller-owned sparse topology stages. Prefix sums and dynamic storage are
// owned by the execution adapter. Count/bitset buffers start at zero; molten
// maps and absent chunks use -1. Preserve directed-edge and partial-chunk
// order.
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/utility/portable/device.hh>
#include <cstdint>

namespace tmol::pack::compiled {

template <typename View>
TMOL_HOST_DEVICE_FUNC bool ig_blocks_are_adjacent(
    View adjacency, int pose, int block, int neighbor) {
  int const word = neighbor / 32;
  int32_t const mask = int32_t(uint32_t(1) << (neighbor % 32));
  return (adjacency[pose][block][word] & mask) != 0;
}

template <typename View>
TMOL_HOST_DEVICE_FUNC int64_t ig_row_for_edge(View row_offsets, int edge) {
  int64_t lower = 0, upper = row_offsets.size(0) - 1;
  while (lower < upper) {
    int64_t const middle = lower + (upper - lower) / 2;
    if (row_offsets[middle + 1] <= edge)
      lower = middle + 1;
    else
      upper = middle;
  }
  return lower;
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Counts,
    typename Map>
void launch_ig_initialize_pose(
    Context& mgr, Counts n_rots_for_block, Map orig_block_to_molten) {
  int const n_poses = n_rots_for_block.size(0);
  int const max_n_blocks = n_rots_for_block.size(1);
  LAUNCH_BOX_32;
  auto initialize_pose = ([=] TMOL_DEVICE_FUNC(int pose) {
    int molten = 0;
    for (int block = 0; block < max_n_blocks; ++block) {
      if (n_rots_for_block[pose][block] <= 1) {
        continue;
      }
      orig_block_to_molten[pose][block] = molten;
      ++molten;
    }
  });
  Ops<D>::template forall<launch_t>(mgr, n_poses, initialize_pose);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Adjacency,
    typename Wide>
void launch_ig_count_neighbors(
    Context& mgr, Adjacency block_adjacency, Wide n_neighbors_for_block) {
  int const max_n_molten_blocks = block_adjacency.size(1);
  int const n_rows = score::common::checked_dispatch_product(
      block_adjacency.size(0), max_n_molten_blocks, "graph rows");
  LAUNCH_BOX_32;
  auto count_neighbors = ([=] TMOL_DEVICE_FUNC(int row) {
    int const pose = row / max_n_molten_blocks;
    int const block = row % max_n_molten_blocks;
    int64_t count = 0;
    for (int neighbor = 0; neighbor < max_n_molten_blocks; ++neighbor) {
      count += ig_blocks_are_adjacent(block_adjacency, pose, block, neighbor)
                   ? 1
                   : 0;
    }
    n_neighbors_for_block[row] = count;
  });
  Ops<D>::template forall<launch_t>(mgr, n_rows, count_neighbors);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Counts,
    typename Adjacency,
    typename Wide,
    typename Indices>
void launch_ig_fill_neighbors(
    Context& mgr,
    int chunk_size,
    Counts n_bc_rots_for_molten_block,
    Adjacency block_adjacency,
    Wide neighbor_row_offsets,
    Indices neighbor_blocks,
    Wide n_chunks_for_neighbor_pair) {
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  int const n_rows = score::common::checked_dispatch_product(
      n_bc_rots_for_molten_block.size(0), max_n_molten_blocks, "graph rows");
  LAUNCH_BOX_32;
  auto fill_neighbors = ([=] TMOL_DEVICE_FUNC(int row) {
    int const pose = row / max_n_molten_blocks;
    int const block = row % max_n_molten_blocks;
    int64_t output = neighbor_row_offsets[row];
    int const n_rots1 = n_bc_rots_for_molten_block[pose][block];
    int const n_chunks1 = n_rots1 == 0 ? 0 : (n_rots1 - 1) / chunk_size + 1;
    for (int neighbor = 0; neighbor < max_n_molten_blocks; ++neighbor) {
      if (!ig_blocks_are_adjacent(block_adjacency, pose, block, neighbor)) {
        continue;
      }
      int const n_rots2 = n_bc_rots_for_molten_block[pose][neighbor];
      int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
      neighbor_blocks[output] = neighbor;
      n_chunks_for_neighbor_pair[output] = int64_t(n_chunks1) * n_chunks2;
      ++output;
    }
  });
  Ops<D>::template forall<launch_t>(mgr, n_rows, fill_neighbors);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Wide>
void launch_ig_count_chunk_words(
    Context& mgr,
    Wide n_chunks_for_neighbor_pair,
    Wide n_chunk_words_for_neighbor_pair) {
  LAUNCH_BOX_32;
  auto count_chunk_words = ([=] TMOL_DEVICE_FUNC(int edge) {
    n_chunk_words_for_neighbor_pair[edge] =
        (n_chunks_for_neighbor_pair[edge] + 31) / 32;
  });
  Ops<D>::template forall<launch_t>(
      mgr,
      score::common::checked_dispatch_size(
          n_chunks_for_neighbor_pair.size(0), "graph edges"),
      count_chunk_words);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Counts,
    typename Wide,
    typename Indices>
void launch_ig_count_chunk_pairs(
    Context& mgr,
    int chunk_size,
    Counts n_bc_rots_for_molten_block,
    Wide neighbor_row_offsets,
    Indices neighbor_blocks,
    Wide n_chunks_for_neighbor_pair) {
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  LAUNCH_BOX_32;
  auto count_chunk_pairs = ([=] TMOL_DEVICE_FUNC(int edge) {
    int const row = ig_row_for_edge(neighbor_row_offsets, edge);
    int const pose = row / max_n_molten_blocks;
    int const block1 = row % max_n_molten_blocks;
    int const block2 = neighbor_blocks[edge];
    int const n_rots1 = n_bc_rots_for_molten_block[pose][block1];
    int const n_rots2 = n_bc_rots_for_molten_block[pose][block2];
    int64_t const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
    int64_t const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
    n_chunks_for_neighbor_pair[edge] = n_chunks1 * n_chunks2;
  });
  Ops<D>::template forall<launch_t>(
      mgr,
      score::common::checked_dispatch_size(
          neighbor_blocks.size(0), "graph edges"),
      count_chunk_pairs);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Counts,
    typename Wide,
    typename Indices>
void launch_ig_fill_chunk_pair_sizes(
    Context& mgr,
    int chunk_size,
    Counts n_bc_rots_for_molten_block,
    Wide neighbor_row_offsets,
    Indices neighbor_blocks,
    Wide neighbor_chunk_offset_offsets,
    Wide neighbor_chunk_bitset_offsets,
    Indices chunk_adjacency,
    Wide chunk_pair_sizes) {
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  LAUNCH_BOX_32;
  auto fill_chunk_pair_sizes = ([=] TMOL_DEVICE_FUNC(int edge) {
    int const row = ig_row_for_edge(neighbor_row_offsets, edge);
    int const pose = row / max_n_molten_blocks;
    int const block1 = row % max_n_molten_blocks;
    int const block2 = neighbor_blocks[edge];
    int const n_rots1 = n_bc_rots_for_molten_block[pose][block1];
    int const n_rots2 = n_bc_rots_for_molten_block[pose][block2];
    int const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
    int64_t const output_begin = neighbor_chunk_offset_offsets[edge];
    int64_t const bitset_begin = neighbor_chunk_bitset_offsets[edge];
    for (int chunk1 = 0; chunk1 < n_chunks1; ++chunk1) {
      int const chunk1_size = min(chunk_size, n_rots1 - chunk1 * chunk_size);
      for (int chunk2 = 0; chunk2 < n_chunks2; ++chunk2) {
        int64_t const bit = int64_t(chunk1) * n_chunks2 + chunk2;
        int32_t const mask = int32_t(uint32_t(1) << static_cast<int>(bit % 32));
        if ((chunk_adjacency[bitset_begin + bit / 32] & mask) != 0) {
          int const chunk2_size =
              min(chunk_size, n_rots2 - chunk2 * chunk_size);
          chunk_pair_sizes[output_begin + bit] =
              int64_t(chunk1_size) * chunk2_size;
        }
      }
    }
  });
  Ops<D>::template forall<launch_t>(
      mgr,
      score::common::checked_dispatch_size(
          neighbor_blocks.size(0), "graph edges"),
      fill_chunk_pair_sizes);
}

template <
    template <Device> class Ops,
    Device D,
    typename Context,
    typename Wide>
void launch_ig_sentinel_empty_chunks(
    Context& mgr, Wide chunk_pair_sizes, Wide chunk_offsets) {
  LAUNCH_BOX_32;
  auto sentinel_empty_chunks = ([=] TMOL_DEVICE_FUNC(int index) {
    if (chunk_pair_sizes[index] == 0) {
      chunk_offsets[index] = -1;
    }
  });
  Ops<D>::template forall<launch_t>(
      mgr,
      score::common::checked_dispatch_size(
          chunk_pair_sizes.size(0), "graph chunks"),
      sentinel_empty_chunks);
}

}  // namespace tmol::pack::compiled
