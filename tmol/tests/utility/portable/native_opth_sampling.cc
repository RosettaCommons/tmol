#include <tmol/pack/rotamer/opth_sampling.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cmath>
using namespace tmol::pack::rotamer;
using tmol_test::SerialOps;
using tmol_test::view;
constexpr auto D = tmol::Device::CPU;
int main() {
  tmol::ContextManager mgr;
  int64_t types[] = {0, 1, 2, 0}, poses[] = {0, 0, 0, 0},
          blocks[] = {0, 0, 0, 0}, original[] = {1};
  bool allowed[] = {true, true, true, false}, sampler[] = {true},
       his[] = {false, true, true}, proton[] = {true, false, false};
  int32_t type_counts[18] = {}, cols[] = {-1, 1, 1}, nhq_atoms[] = {-1, 7, 8},
          atoms[] = {4, 5, -1, -1, -1, -1}, counts[] = {2, 3, 0, 0, 0, 0};
  float samples[18] = {10, 11, 0, 20, 21, 22}, flips[] = {.25f};
  type_counts[12] = 6;
  type_counts[13] = 2;
  type_counts[14] = 2;
  OptHSamplingContext<D> c{
      view(types, {4}),
      view(poses, {4}),
      view(blocks, {4}),
      view(original, {1, 1}),
      view(allowed, {4}),
      view(sampler, {1, 1}),
      view(type_counts, {2, 3, 3}),
      view(cols, {3}),
      view(nhq_atoms, {3}),
      view(his, {3}),
      {view(proton, {3}),
       view(atoms, {3, 2}),
       view(counts, {3, 2}),
       view(samples, {3, 2, 3})},
      view(flips, {1, 1})};
  int64_t n[4], nf[4];
  bool measure[4];
  opth_count_native<SerialOps, D>(
      mgr, c, true, view(n, {4}), view(measure, {4}));
  assert(n[0] == 6 && n[1] == 2 && n[2] == 2 && n[3] == 0);
  assert(!measure[0] && measure[1] && measure[2] && !measure[3]);
  opth_flip_counts_native<SerialOps, D>(mgr, c, view(n, {4}), view(nf, {4}));
  assert(nf[0] == 0 && nf[1] == 2 && nf[2] == 2 && nf[3] == 0);
  int64_t offsets[] = {0, 6, 8, 10}, prefix[] = {0, 0, 2, 4}, sources[20];
  int32_t owners[] = {0, 0, 0, 0, 0, 0, 1, 1, 2, 2}, result_atoms[20];
  float values[20];
  opth_samples_native<SerialOps, D>(
      mgr,
      c,
      view(offsets, {4}),
      view(prefix, {4}),
      view(owners, {10}),
      view(result_atoms, {10, 2}),
      view(values, {10, 2}),
      view(sources, {10, 2}));
  const float expected[] = {10, 20, 11, 20, 10, 21, 11, 21, 10, 22, 11, 22};
  for (int i = 0; i < 12; ++i) {
    assert(values[i] == expected[i]);
    assert(samples[sources[i]] == values[i]);
    assert(result_atoms[i] == (i % 2 ? 5 : 4));
  }
  assert(result_atoms[13] == -1 && values[13] == 0 && sources[13] == -1);
  assert(
      result_atoms[15] == 7 && result_atoms[17] == 8 && result_atoms[19] == 8);
  assert(values[17] == .25f && sources[17] == 18);
  for (int i : {15, 19}) {
    assert(std::abs(values[i] - (.25f + float(3.14159265358979323846))) < 1e-6);
    assert(sources[i] == 18);
  }
  sampler[0] = false;
  opth_count_native<SerialOps, D>(
      mgr, c, true, view(n, {4}), view(measure, {4}));
  for (int i = 0; i < 4; ++i) assert(n[i] == 0 && !measure[i]);
  c.types = view(types, {0});
  opth_count_native<SerialOps, D>(
      mgr, c, true, view(n, {0}), view(measure, {0}));
  opth_samples_native<SerialOps, D>(
      mgr,
      c,
      view(offsets, {0}),
      view(prefix, {0}),
      view(owners, {0}),
      view(result_atoms, {0, 2}),
      view(values, {0, 2}),
      view(sources, {0, 2}));
}
