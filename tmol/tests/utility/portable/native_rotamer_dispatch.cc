#include <tmol/score/common/rotamer_dispatch.hh>
#include <tmol/score/common/tview_binder.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <vector>

using namespace tmol::score::common::sphere_overlap;
using tmol_test::view;
using Binder = tmol::score::common::TViewBinder<tmol::Device::CPU>;
template <tmol::Device D>
struct SphereOps : tmol_test::SerialOps<D> {
  template <class L, class F>
  static void forall_independent(tmol::ContextManager& c, int n, F f) {
    tmol_test::SerialOps<D>::template forall<L>(c, n, f);
  }
  template <int nt, class T, class Op, class... Lane>
  static T shuffle_reduce_in_workgroup(T v, Op, Lane...) {
    return v;
  }
  template <int nt, class T, class Op, class... Lane>
  static T shuffle_reduce_and_broadcast_in_workgroup(T v, Op, Lane...) {
    return v;
  }
};
template <class Real, class Int>
void check() {
  tmol::ContextManager mgr;
  using Vec = tmol::portable::Vec<Real, 3>;
  Vec xyz[10];
  Real centres[] = {0, 10, 11, 1, 50};
  for (int r = 0; r < 5; ++r) {
    xyz[2 * r] = Vec{centres[r] - Real(.5), 0, 0};
    xyz[2 * r + 1] = Vec{centres[r] + Real(.5), 0, 0};
  }
  Int offsets[] = {0, 2, 4, 6, 8}, types[] = {0, 0, 0, 0, 0}, natoms[] = {2};
  Real spheres[20] = {}, blocks[16] = {};
  RotSpheresCtxT<Binder::template View, Real, Int> s;
  s.rot_coords = view(xyz, {10});
  s.rot_coord_offset = view(offsets, {5});
  s.block_type_ind_for_rot = view(types, {5});
  s.block_type_n_atoms = view(natoms, {1});
  s.rot_spheres = view(spheres, {5, 4});
  launch_rotamer_spheres<SphereOps, tmol::Device::CPU, Real, Int>(mgr, s);
  for (int r = 0; r < 5; ++r) {
    assert(spheres[4 * r] == centres[r]);
    assert(spheres[4 * r + 3] == Real(.5));
  }
  Int counts[] = {2, 2, 1, 0}, rot_offsets[] = {0, 2, 4, 5},
      block_types[] = {0, 0, 0, -1}, groups[] = {0, 0, -1, -1};
  BlockSpheresFromRotCtxT<Binder::template View, Real, Int> b;
  b.rot_spheres = s.rot_spheres;
  b.n_rots_for_block = view(counts, {1, 4});
  b.rot_offset_for_block = view(rot_offsets, {1, 4});
  b.block_spheres = view(blocks, {1, 4, 4});
  b.max_n_blocks = 4;
  launch_blocks_from_rotamer_spheres<SphereOps, tmol::Device::CPU, Real, Int>(
      mgr, b);
  assert(blocks[0] == 5 && blocks[3] == Real(5.5));
  auto full = rotamer_pair_window(1, 4);
  assert(full.n_candidates == 10);
  auto enumerate = [&](int begin, int count) {
    auto w = rotamer_pair_window(1, 4, begin, count);
    std::vector<Int> pc(w.n_candidates, 0), po(w.n_candidates, 0);
    int64_t overflow = 0;
    RotPairListCtxT<Binder::template View, Real, Int> c;
    c.pose_stack_block_type = view(block_types, {1, 4});
    c.block_spheres = b.block_spheres;
    c.n_rots_for_block = b.n_rots_for_block;
    c.rot_offset_for_block = b.rot_offset_for_block;
    c.rot_spheres = s.rot_spheres;
    c.lockstep_group_for_block = view(groups, {1, 4});
    c.pair_counts = view(pc.data(), {w.n_candidates});
    c.pair_offsets = view(po.data(), {w.n_candidates});
    c.pair_count_overflow = view(&overflow, {1});
    c.max_n_blocks = 4;
    c.block_pairs_per_pose = w.block_pairs_per_pose;
    c.candidate_begin = w.candidate_begin;
    c.reach = 2;
    launch_rotamer_pair_counts<SphereOps, tmol::Device::CPU, Real, Int>(mgr, c);
    assert(!overflow);
    Int total = 0;
    for (int i = 0; i < w.n_candidates; ++i) {
      po[i] = total;
      total += pc[i];
    }
    std::vector<Int> output(3 * total, -1);
    c.indices = view(output.data(), {3, total});
    launch_rotamer_pair_fill<SphereOps, tmol::Device::CPU, Real, Int>(mgr, c);
    std::vector<Int> pairs;
    for (int i = 0; i < total; ++i) {
      assert(output[i] == 0);
      pairs.push_back(10 * output[total + i] + output[2 * total + i]);
    }
    return pairs;
  };
  auto coupled = enumerate(0, -1);
  assert((coupled == std::vector<Int>{0, 11, 2, 13, 22, 33, 44}));
  auto left = enumerate(0, 2), right = enumerate(2, 8);
  left.insert(left.end(), right.begin(), right.end());
  assert(left == coupled);
  groups[0] = groups[1] = -1;
  auto independent = enumerate(0, -1);
  assert((independent == std::vector<Int>{0, 11, 3, 12, 22, 33, 44}));
  for (auto& x : xyz) x[0] += 8;
  launch_rotamer_spheres<SphereOps, tmol::Device::CPU, Real, Int>(mgr, s);
  launch_blocks_from_rotamer_spheres<SphereOps, tmol::Device::CPU, Real, Int>(
      mgr, b);
  assert(enumerate(0, -1) == independent);
  assert(enumerate(10, 0).empty());
  bool invalid = false;
  try {
    rotamer_pair_window(1, 4, 1, INT64_MAX);
  } catch (std::out_of_range const&) {
    invalid = true;
  }
  assert(invalid);
}
int main() {
  check<float, int32_t>();
  check<float, int64_t>();
  check<double, int32_t>();
  check<double, int64_t>();
}
