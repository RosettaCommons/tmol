#include <tmol/score/metal/potentials/metal_rotamer_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <set>
#include <utility>
using namespace tmol::score::metal::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int, typename Real>
void check() {
  tmol::ContextManager mgr;
  Int first_types[] = {0, 1}, first[] = {0, 2}, blocks[] = {0, 0, 1, 1, -1},
      poses[] = {0, 0, 0, 0, -1}, types[] = {0, 0, 1, 1, -1}, counts[] = {2, 2};
  Int atom[] = {0, 0}, metal[] = {0, -1}, virt[] = {-1, NOT_A_SITE},
      key[] = {-1, 0}, internal[] = {-1, -1};
  Real d0[] = {0, 0};
  tmol::portable::Vec<Int, 2> conns[2], fans[2];
  conns[0][0] = 1;
  conns[0][1] = 0;
  conns[1][0] = 0;
  conns[1][1] = 0;
  fans[0][0] = 0;
  fans[0][1] = 1;
  fans[1][0] = -1;
  fans[1][1] = -1;
  MetalSiteParams<Real> params[2] = {};
  int64_t sizes[5] = {}, offsets[5] = {};
  metal_rotamer_count_native<SerialOps, tmol::Device::CPU, Real, Int>(
      mgr,
      view(first_types, {1, 2}),
      view(blocks, {5}),
      view(poses, {5}),
      view(types, {5}),
      view(counts, {1, 2}),
      view(conns, {1, 2, 1}),
      view(atom, {2, 1}),
      view(metal, {2, 1}),
      view(virt, {2, 1}),
      view(key, {2, 1}),
      view(params, {2, 1, 1}),
      view(fans, {2, 1}),
      view(internal, {2, 1, 1}),
      view(d0, {2, 1, 1}),
      view(sizes, {5}));
  assert(sizes[0] == 3 && sizes[1] == 3);
  for (int i = 2; i < 5; ++i) assert(sizes[i] == 0);
  int total = 0;
  for (int i = 0; i < 5; ++i) {
    offsets[i] = total;
    total += sizes[i];
  }
  assert(total == 6);
  Int pairs[18] = {}, terms[18] = {};
  metal_rotamer_fill_native<SerialOps, tmol::Device::CPU, Real, Int>(
      mgr,
      view(first, {1, 2}),
      view(first_types, {1, 2}),
      view(blocks, {5}),
      view(poses, {5}),
      view(types, {5}),
      view(counts, {1, 2}),
      view(conns, {1, 2, 1}),
      view(atom, {2, 1}),
      view(metal, {2, 1}),
      view(virt, {2, 1}),
      view(key, {2, 1}),
      view(params, {2, 1, 1}),
      view(fans, {2, 1}),
      view(internal, {2, 1, 1}),
      view(d0, {2, 1, 1}),
      view(sizes, {5}),
      view(offsets, {5}),
      view(pairs, {3, 6}),
      view(terms, {3, 6}),
      3,
      15);
  std::set<std::pair<Int, Int>> actual;
  int fan_count = 0;
  for (int i = 0; i < 6; ++i) {
    assert(pairs[i] == 0);
    Int a = pairs[6 + i], b = pairs[12 + i];
    assert(a < 2);
    assert(terms[i] == a && terms[6 + i] == b);
    if (a == b) {
      assert(terms[12 + i] == -1);
      ++fan_count;
    } else {
      assert(b >= 2 && b < 4 && terms[12 + i] == 0);
    }
    actual.emplace(a, b);
  }
  assert(actual.size() == 6 && fan_count == 2);
  auto volatile f = &metal_rotamer_score_native<
      false,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  auto volatile b = &metal_rotamer_score_native<
      true,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  (void)f;
  (void)b;
}
int main() {
  check<int32_t, float>();
  check<int32_t, double>();
  check<int64_t, float>();
  check<int64_t, double>();
}
