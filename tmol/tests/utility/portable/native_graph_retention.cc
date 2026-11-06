#include <tmol/pack/compiled/interaction_graph_retention.hh>
#include "native_test_ops.hh"
#include <cassert>

using namespace tmol::pack::compiled;
using tmol_test::view;
template <tmol::Device D>
struct RetentionOps : tmol_test::SerialOps<D> {
  template <int nt, class T, class Op>
  static T shuffle_reduce_in_workgroup(T v, Op) {
    return v;
  }
};

template <typename Int>
void check() {
  tmol::ContextManager mgr;
  // A fixed first pose, mixed retained/fixed blocks in the second, then empty
  // trailing pose. Sparse selection is deliberately not a contiguous prefix.
  Int counts[] = {1, 0, 0, 3, 2, 1, 0, 0, 0};
  Int offsets[] = {0, 1, 1, 1, 4, 6, 7, 7, 7};
  int64_t keep[] = {1, 1, 0, 1, 0, 1, 1}, blocks[9] = {};
  int64_t fixed[9];
  for (auto& x : fixed) x = -1;
  launch_ig_keep_blocks<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(counts, {3, 3}),
      view(offsets, {3, 3}),
      view(keep, {7}),
      view(blocks, {3, 3}),
      view(fixed, {3, 3}));
  assert(fixed[0] == 0 && fixed[4] == 5 && fixed[5] == 6);
  int64_t expected_keep[] = {0, 1, 0, 1, 0, 0, 0};
  for (int i = 0; i < 7; ++i) assert(keep[i] == expected_keep[i]);
  for (int i = 0; i < 9; ++i) assert(blocks[i] == (i == 3));
  int64_t rot_prefix[7], block_prefix[9];
  int64_t nr = 0, nb = 0;
  for (int i = 0; i < 7; ++i) {
    rot_prefix[i] = nr;
    nr += keep[i];
  }
  for (int i = 0; i < 9; ++i) {
    block_prefix[i] = nb;
    nb += blocks[i];
  }
  assert(nr == 2 && nb == 1);
  int64_t rot_inverse[2] = {}, block_inverse[1] = {};
  launch_ig_invert_rotamers<RetentionOps, tmol::Device::CPU>(
      mgr, view(keep, {7}), view(rot_prefix, {7}), view(rot_inverse, {2}));
  launch_ig_invert_blocks<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(blocks, {3, 3}),
      view(block_prefix, {9}),
      view(block_inverse, {1}));
  assert(rot_inverse[0] == 1 && rot_inverse[1] == 3 && block_inverse[0] == 3);
  Int block_pose_offsets[3] = {}, block_pose_counts[3] = {};
  launch_ig_pose_block_offsets<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(blocks, {3, 3}),
      view(block_prefix, {9}),
      view(block_pose_offsets, {3}));
  launch_ig_pose_block_counts<RetentionOps, tmol::Device::CPU>(
      mgr, view(block_pose_offsets, {3}), view(block_pose_counts, {3}), nb);
  Int original_pose_offsets[] = {0, 1, 7}, rot_pose_offsets[3] = {},
      rot_pose_counts[3] = {};
  launch_ig_pose_rotamer_offsets<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(original_pose_offsets, {3}),
      view(rot_prefix, {7}),
      view(rot_pose_offsets, {3}),
      nr);
  launch_ig_pose_rotamer_counts<RetentionOps, tmol::Device::CPU>(
      mgr, view(rot_pose_offsets, {3}), view(rot_pose_counts, {3}), nr);
  assert(
      block_pose_offsets[0] == 0 && block_pose_offsets[1] == 0
      && block_pose_offsets[2] == 1);
  assert(
      rot_pose_offsets[0] == 0 && rot_pose_offsets[1] == 0
      && rot_pose_offsets[2] == 2);
  for (int p = 0; p < 3; ++p) {
    assert(block_pose_counts[p] == (p == 1));
    assert(rot_pose_counts[p] == (p == 1 ? 2 : 0));
  }
  Int compact_offsets[3] = {}, compact_counts[3] = {};
  launch_ig_block_rotamer_offsets<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(block_inverse, {1}),
      view(block_pose_offsets, {3}),
      view(offsets, {3, 3}),
      view(rot_prefix, {7}),
      view(compact_offsets, {3, 1}));
  launch_ig_block_rotamer_counts<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(block_inverse, {1}),
      view(block_pose_offsets, {3}),
      view(block_pose_counts, {3}),
      view(compact_offsets, {3, 1}),
      view(rot_pose_offsets, {3}),
      view(compact_counts, {3, 1}),
      3,
      nr);
  assert(
      compact_counts[0] == 0 && compact_counts[1] == 2
      && compact_counts[2] == 0);
  Int pose_for_rot[] = {0, 1, 1, 1, 1, 1, 1}, local_rot_blocks[2] = {-1, -1};
  int32_t block_for_rot[] = {0, 0, 0, 0, 1, 1, 2};
  launch_ig_rotamer_blocks<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(rot_inverse, {2}),
      view(block_for_rot, {7}),
      view(pose_for_rot, {7}),
      view(block_prefix, {9}),
      view(block_pose_offsets, {3}),
      view(local_rot_blocks, {2}),
      3);
  assert(local_rot_blocks[0] == 0 && local_rot_blocks[1] == 0);
  // Conservation: every originally selected rotamer is either fixed or mapped
  // bijectively into the compact graph; removed rotamers occur in neither.
  for (int i : {0, 1, 3, 5, 6}) {
    int appearances = 0;
    for (auto x : fixed) appearances += x == i;
    for (auto x : rot_inverse) appearances += x == i;
    assert(appearances == 1);
  }
  // Ordinary bump energies: a favorable alternative rejects a clash, while
  // an entirely clashing block remains available rather than losing support.
  double energy[] = {0, 4, 6, 2, 8, 9, 0}, best[] = {0, 0, 0, 2, 8, 0, 0, 0, 0};
  int64_t bump_keep[7] = {};
  launch_ig_keep_rotamers<RetentionOps, tmol::Device::CPU>(
      mgr,
      view(pose_for_rot, {7}),
      view(block_for_rot, {7}),
      view(energy, {7}),
      view(best, {3, 3}),
      view(bump_keep, {7}));
  assert(bump_keep[1] == 1 && bump_keep[2] == 0 && bump_keep[3] == 1);
  assert(bump_keep[4] == 1 && bump_keep[5] == 1);
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
