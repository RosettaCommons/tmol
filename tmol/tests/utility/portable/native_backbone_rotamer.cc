#include <tmol/score/backbone_torsion/potentials/backbone_rotamer_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cstdint>
using namespace tmol::score::backbone_torsion::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int counts[] = {2, 3, 0, 0}, types[] = {0, 0, 0, -1};
  Int upper[] = {1}, first[] = {0, 2, 5, 5};
  tmol::portable::Vec<Int, 2> connections[8];
  for (auto& c : connections) c = tmol::portable::Vec<Int, 2>::Constant(-1);
  connections[1][0] = 1;
  connections[3][0] = 2;
  int64_t sizes[4] = {}, offsets[] = {0, 6, 6, 6};
  auto count = [&] {
    backbone_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
        mgr,
        view(counts, {1, 4}),
        view(types, {1, 4}),
        view(connections, {1, 4, 2}),
        view(upper, {1}),
        view(sizes, {1, 4}));
  };
  count();
  assert(sizes[0] == 6 && sizes[1] == 0 && sizes[2] == 0 && sizes[3] == 0);
  Int owners[6] = {}, pairs[18] = {};
  backbone_rotamer_fill_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(first, {1, 4}),
      view(counts, {1, 4}),
      view(types, {1, 4}),
      view(connections, {1, 4, 2}),
      view(upper, {1}),
      view(offsets, {1, 4}),
      view(owners, {6}),
      view(pairs, {3, 6}));
  for (int i = 0; i < 6; ++i) {
    assert(pairs[i] == 0);
    assert(pairs[6 + i] == i / 3);
    assert(pairs[12 + i] == 2 + i % 3);
  }
  counts[0] = counts[1] = 50000;
  count();
  assert(sizes[0] == INT64_C(2500000000));
  backbone_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(counts, {0, 4}),
      view(types, {0, 4}),
      view(connections, {0, 4, 2}),
      view(upper, {1}),
      view(sizes, {0, 4}));
}
template <typename Real, typename Int>
void instantiate() {
  auto volatile f = &backbone_rotamer_score_native<
      false,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  auto volatile b = &backbone_rotamer_score_native<
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
