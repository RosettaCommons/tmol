#include <tmol/score/genbonded/potentials/genbonded_rotamer_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cstdint>
using namespace tmol::score::genbonded::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int>
void check(bool coupled) {
  tmol::ContextManager mgr;
  Int blocks[] = {0, 0, 1, 1}, poses[] = {0, 0, 0, 0}, types[] = {0, 0, 0, 0};
  Int first[] = {0, 2}, counts[] = {2, 2}, groups[] = {-1, -1};
  if (coupled) groups[0] = groups[1] = 9;
  tmol::portable::Vec<Int, 2> connections[2];
  connections[0][0] = 1;
  connections[0][1] = 0;
  connections[1][0] = 0;
  connections[1][1] = 0;
  int64_t sizes[8] = {};
  genbonded_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(blocks, {4}),
      view(poses, {4}),
      view(types, {4}),
      view(counts, {1, 2}),
      view(groups, {1, 2}),
      view(connections, {1, 2, 1}),
      view(sizes, {8}));
  int64_t expected[] = {coupled ? 1 : 2, 1, coupled ? 1 : 2, 1, 0, 1, 0, 1};
  int64_t offsets[8];
  int total = 0;
  for (int i = 0; i < 8; ++i) {
    assert(sizes[i] == expected[i]);
    offsets[i] = total;
    total += sizes[i];
  }
  assert(total == (coupled ? 6 : 8));
  Int uncoupled_owners[] = {0, 0, 1, 2, 2, 3, 5, 7};
  Int coupled_owners[] = {0, 1, 2, 3, 5, 7};
  Int pairs[24] = {};
  genbonded_rotamer_fill_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(first, {1, 2}),
      view(blocks, {4}),
      view(poses, {4}),
      view(groups, {1, 2}),
      view(connections, {1, 2, 1}),
      view(offsets, {8}),
      view(coupled ? coupled_owners : uncoupled_owners, {total}),
      view(pairs, {3, total}));
  int cross = 0, diagonal = 0;
  for (int i = 0; i < total; ++i) {
    assert(pairs[i] == 0);
    Int r1 = pairs[total + i], r2 = pairs[2 * total + i];
    assert(r1 <= r2);
    if (r1 == r2)
      ++diagonal;
    else {
      ++cross;
      assert(blocks[r1] == 0 && blocks[r2] == 1);
      if (coupled) assert(r2 - r1 == 2);
    }
  }
  assert(diagonal == 4 && cross == (coupled ? 2 : 4));
  genbonded_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(blocks, {0}),
      view(poses, {0}),
      view(types, {0}),
      view(counts, {0, 2}),
      view(groups, {0, 2}),
      view(connections, {0, 2, 1}),
      view(sizes, {0}));
}
template <typename Real>
void instantiate() {
  auto volatile f = &genbonded_rotamer_score_native<
      false,
      SerialOps,
      tmol::Device::CPU,
      Real,
      int32_t>;
  auto volatile b = &genbonded_rotamer_score_native<
      true,
      SerialOps,
      tmol::Device::CPU,
      Real,
      int32_t>;
  (void)f;
  (void)b;
}
int main() {
  check<int32_t>(false);
  check<int32_t>(true);
  check<int64_t>(false);
  check<int64_t>(true);
  instantiate<float>();
  instantiate<double>();
}
