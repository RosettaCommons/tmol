#include <tmol/score/constraint/potentials/constraint_topology.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::score::constraint::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int const start = sizeof(Int) == 8 ? Int(2500000000LL) : Int(10);
  Vec<Int, 3> atoms[12] = {
      {0, 1, 0},
      {0, 1, 1},
      {-1, -1, -1},
      {0, 1, 1},
      {0, 0, 1},
      {0, 1, 0},
      {-1, -1, -1},
      {0, 1, 0},
      {0, 2, 0},
      {0, 2, 1},
      {-1, -1, -1},
      {0, 2, 1}};
  Int bodies[3] = {1, 2, 1};
  Vec<Int, 3> blocks[3] = {{0, 1, 1}, {0, 0, 1}, {0, 2, 2}};
  Int coord[5] = {
      start,
      Int(start + 10),
      Int(start + 20),
      Int(start + 30),
      Int(start + 40)};
  Int sizes[3] = {2, 3, 0}, first[3] = {0, 2, 4};
  ConstraintCandidateTopology<Int, tmol::Device::CPU> ctx{
      view(atoms, {3, 4}),
      view(bodies, {3}),
      view(coord, {5}),
      view(blocks, {3}),
      view(sizes, {1, 3}),
      view(first, {1, 3})};
  int64_t counts[6];
  constraint_candidate_count_native<SerialOps, tmol::Device::CPU>(
      mgr, ctx, view(counts, {6}));
  int64_t expected_counts[] = {0, 6, 0, 3, 0, 0};
  for (int i = 0; i < 6; ++i) assert(counts[i] == expected_counts[i]);
  int64_t offsets[] = {0, 0, 6, 6, 9, 9};
  Int owners[] = {1, 1, 1, 1, 1, 1, 3, 3, 3}, sources[9];
  Vec<Int, 4> mapped[9];
  Vec<int64_t, 3> indices[9];
  constraint_candidate_fill_native<SerialOps, tmol::Device::CPU>(
      mgr,
      ctx,
      true,
      view(offsets, {6}),
      view(owners, {9}),
      view(sources, {9}),
      view(mapped, {9}),
      view(indices, {9}));
  int64_t expected_pairs[9][2] = {
      {0, 2}, {1, 2}, {0, 3}, {1, 3}, {0, 4}, {1, 4}, {2, 2}, {3, 3}, {4, 4}};
  for (int i = 0; i < 9; ++i) {
    assert(sources[i] == (i < 6 ? 1 : 0));
    assert(
        indices[i][0] == 0 && indices[i][1] == expected_pairs[i][0]
        && indices[i][2] == expected_pairs[i][1]);
  }
  assert(
      mapped[0][0] == start + 1 && mapped[0][1] == start + 20
      && mapped[0][2] == start + 19);
  assert(mapped[1][0] == start + 11 && mapped[2][1] == start + 30);
  assert(
      mapped[6][0] == start + 20 && mapped[6][1] == start + 21
      && mapped[6][2] == -1);
  ctx.atoms = view(atoms, {2, 4});
  ctx.n_unique = view(bodies, {2});
  ctx.unique_blocks = view(blocks, {2});
  constraint_candidate_fill_native<SerialOps, tmol::Device::CPU>(
      mgr,
      ctx,
      false,
      {},
      {},
      view(sources, {2}),
      view(mapped, {2}),
      view(indices, {2}));
  assert(
      sources[0] == 0 && sources[1] == 1 && indices[0][1] == 1
      && indices[0][2] == 1);
  assert(mapped[0][0] == start + 20 && mapped[0][2] == start + 39);
  assert(mapped[1][0] == start + 1 && mapped[1][1] == start + 20);
  ctx.atoms = view(atoms, {0, 4});
  ctx.n_unique = view(bodies, {0});
  ctx.unique_blocks = view(blocks, {0});
  constraint_candidate_count_native<SerialOps, tmol::Device::CPU>(
      mgr, ctx, view(counts, {0}));
  constraint_candidate_fill_native<SerialOps, tmol::Device::CPU>(
      mgr,
      ctx,
      true,
      {},
      {},
      view(sources, {0}),
      view(mapped, {0}),
      view(indices, {0}));
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
