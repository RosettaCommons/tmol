#include <tmol/score/disulfide/potentials/disulfide_rotamer_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <set>
#include <utility>
using namespace tmol::score::disulfide::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int blocks[] = {0, 0, 1, 2, 2, -1, 0}, poses[] = {0, 0, 0, 0, 0, -1, 0};
  Int types[] = {0, 0, 0, 0, 0, -1, -1}, first[] = {0, 2, 3},
      counts[] = {2, 1, 2};
  bool capable[] = {true, true};
  tmol::portable::Vec<Int, 2> conns[6];
  for (auto& x : conns) {
    x[0] = -1;
    x[1] = -1;
  }
  conns[0][0] = 1;
  conns[0][1] = 0;
  conns[1][0] = 2;
  conns[1][1] = 0;
  conns[2][0] = 0;
  conns[2][1] = 0;
  conns[4][0] = 0;
  conns[4][1] = 1;
  int64_t sizes[7] = {}, offsets[7] = {};
  disulfide_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(blocks, {7}),
      view(poses, {7}),
      view(types, {7}),
      view(counts, {1, 3}),
      view(conns, {1, 3, 2}),
      view(capable, {1, 2}),
      view(sizes, {7}));
  assert(sizes[0] == 3 && sizes[1] == 3);
  for (int i = 2; i < 7; ++i) assert(sizes[i] == 0);
  int total = 0;
  for (int i = 0; i < 7; ++i) {
    offsets[i] = total;
    total += sizes[i];
  }
  assert(total == 6);
  Int pairs[18] = {}, edge_ids[12] = {};
  disulfide_rotamer_fill_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(first, {1, 3}),
      view(blocks, {7}),
      view(poses, {7}),
      view(types, {7}),
      view(counts, {1, 3}),
      view(conns, {1, 3, 2}),
      view(capable, {1, 2}),
      view(sizes, {7}),
      view(offsets, {7}),
      view(pairs, {3, 6}),
      view(edge_ids, {2, 6}),
      3,
      21);
  std::set<std::pair<Int, Int>> actual;
  for (int i = 0; i < 6; ++i) {
    assert(pairs[i] == 0);
    Int a = pairs[6 + i], b = pairs[12 + i];
    assert(a < 2 && b >= 2 && b < 5);
    assert(edge_ids[i] == (b == 2 ? 0 : 1) && edge_ids[6 + i] == 0);
    actual.emplace(a, b);
  }
  assert(actual.size() == 6);
  disulfide_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(blocks, {0}),
      view(poses, {0}),
      view(types, {0}),
      view(counts, {0, 3}),
      view(conns, {0, 3, 2}),
      view(capable, {1, 2}),
      view(sizes, {0}));
  disulfide_rotamer_fill_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(first, {0, 3}),
      view(blocks, {0}),
      view(poses, {0}),
      view(types, {0}),
      view(counts, {0, 3}),
      view(conns, {0, 3, 2}),
      view(capable, {1, 2}),
      view(sizes, {0}),
      view(offsets, {0}),
      view(pairs, {3, 0}),
      view(edge_ids, {2, 0}),
      0,
      0);
}
template <typename Real, typename Int>
void instantiate() {
  auto volatile f = &disulfide_rotamer_score_native<
      false,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  auto volatile b = &disulfide_rotamer_score_native<
      true,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  (void)f;
  (void)b;
}
int main() {
  check<int32_t>();
  check<int64_t>();
  instantiate<float, int32_t>();
  instantiate<double, int32_t>();
  instantiate<float, int64_t>();
  instantiate<double, int64_t>();
}
