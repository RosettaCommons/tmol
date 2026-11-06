#include <tmol/pack/compiled/interaction_graph_topology.hh>
#include "native_test_ops.hh"
#include <cassert>

using namespace tmol::pack::compiled;
using tmol_test::view;

template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int counts[] = {0, 1, 0, 3, 1, 2};
  Int molten_counts[] = {0, 0, 3, 2};
  int64_t mapping[] = {-1, -1, -1, -1, -1, -1};
  launch_ig_initialize_pose<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr, view(counts, {2, 3}), view(mapping, {2, 3}));
  int64_t expected_mapping[] = {-1, -1, -1, 0, -1, 1};
  for (int i = 0; i < 6; ++i) assert(mapping[i] == expected_mapping[i]);
  // One frozen pose, then two interacting molten residues. The last chunk
  // of the three-rotamer residue has only one rotamer.
  int32_t adjacency[] = {0, 0, 2, 1};
  int64_t neighbor_counts[5] = {};
  launch_ig_count_neighbors<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr, view(adjacency, {2, 2, 1}), view(neighbor_counts, {5}));
  int64_t expected_counts[] = {0, 0, 1, 1, 0};
  for (int i = 0; i < 5; ++i) assert(neighbor_counts[i] == expected_counts[i]);
  int64_t rows[] = {0, 0, 0, 1, 2}, pairs[2] = {}, words[2] = {};
  int32_t neighbors[2] = {};
  launch_ig_fill_neighbors<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr,
      2,
      view(molten_counts, {2, 2}),
      view(adjacency, {2, 2, 1}),
      view(rows, {5}),
      view(neighbors, {2}),
      view(pairs, {2}));
  assert(neighbors[0] == 1 && neighbors[1] == 0);
  assert(pairs[0] == 2 && pairs[1] == 2);
  launch_ig_count_chunk_words<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr, view(pairs, {2}), view(words, {2}));
  assert(words[0] == 1 && words[1] == 1);
  pairs[0] = pairs[1] = 0;
  launch_ig_count_chunk_pairs<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr,
      2,
      view(molten_counts, {2, 2}),
      view(rows, {5}),
      view(neighbors, {2}),
      view(pairs, {2}));
  assert(pairs[0] == 2 && pairs[1] == 2);
  int64_t pair_offsets[] = {0, 2}, bitset_offsets[] = {0, 1}, sizes[4] = {};
  int32_t chunk_adjacency[] = {2, 2};
  launch_ig_fill_chunk_pair_sizes<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr,
      2,
      view(molten_counts, {2, 2}),
      view(rows, {5}),
      view(neighbors, {2}),
      view(pair_offsets, {2}),
      view(bitset_offsets, {2}),
      view(chunk_adjacency, {2}),
      view(sizes, {4}));
  assert(sizes[0] == 0 && sizes[1] == 2 && sizes[2] == 0 && sizes[3] == 2);
  int64_t offsets[] = {0, 0, 2, 2};
  launch_ig_sentinel_empty_chunks<tmol_test::SerialOps, tmol::Device::CPU>(
      mgr, view(sizes, {4}), view(offsets, {4}));
  assert(offsets[0] == -1 && offsets[1] == 0);
  assert(offsets[2] == -1 && offsets[3] == 2);
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
