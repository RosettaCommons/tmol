#include <tmol/pack/rotamer/na_sampling.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cmath>
#include <limits>
using namespace tmol::pack::rotamer;
using tmol_test::SerialOps;
using tmol_test::view;
constexpr auto D = tmol::Device::CPU;
void check_samples() {
  tmol::ContextManager mgr;
  int64_t types[] = {0, 1, 1, 2, 3}, base[] = {0, 5, 7, -1},
          pucker[] = {0, 1, -1, 1, 0}, combos[] = {2, 2, 1, 0};
  bool active[] = {true, true, true, true, true};
  int32_t atoms[] = {3, 4, 7, 8, 11, 12, -1, -1};
  float proton[] = {1, 2, 3, 4, 5, 0, 0, 0}, wells[200] = {}, means[100] = {},
        sdev[] = {10, 20}, steps[] = {0, 1, -1};
  means[0] = 120;
  means[51] = 160;
  means[71] = 210;
  wells[117] = 6;
  NaChiSamplingContext<D> ctx{
      view(types, {5}),
      view(base, {4}),
      view(pucker, {5}),
      view(combos, {4}),
      view(active, {5}),
      view(atoms, {4, 2}),
      view(proton, {4, 2, 1}),
      view(wells, {2, 10, 10}),
      view(means, {10, 10}),
      view(sdev, {2}),
      view(steps, {3}),
      {true, 5, 5, 50}};
  int64_t counts[5];
  na_chi_count_native<SerialOps, D>(mgr, ctx, view(counts, {5}));
  int expected[] = {6, 12, 0, 3, 0};
  for (int i = 0; i < 5; ++i) assert(counts[i] == expected[i]);
  int64_t offsets[] = {0, 6, 18, 18, 21};
  int32_t owners[21], out_atoms[42];
  float chi[42];
  for (int i = 0; i < 21; ++i) owners[i] = i < 6 ? 0 : (i < 18 ? 1 : 3);
  na_chi_fill_native<SerialOps, D>(
      mgr,
      ctx,
      view(offsets, {5}),
      view(owners, {21}),
      view(out_atoms, {21, 2}),
      view(chi, {21, 2}));
  float angles[] = {120, 120, 130, 130, 110, 110, 160, 160, 180, 180, 140,
                    140, 50,  50,  70,  70,  30,  30,  210, 230, 190};
  for (int i = 0; i < 21; ++i) {
    assert(
        std::abs(chi[2 * i] - angles[i] * float(3.14159265358979323846 / 180))
        < 1e-6);
    int bt = i < 6 ? 0 : (i < 18 ? 1 : 2);
    assert(out_atoms[2 * i] == atoms[2 * bt]);
    assert(out_atoms[2 * i + 1] == atoms[2 * bt + 1]);
    float expected_proton =
        i < 6 ? float(1 + i % 2) : (i < 18 ? float(3 + i % 2) : 5.f);
    assert(chi[2 * i + 1] == expected_proton);
  }
  ctx.policy.sample_syn = false;
  na_chi_count_native<SerialOps, D>(mgr, ctx, view(counts, {5}));
  assert(counts[1] == 6);
  combos[0] = std::numeric_limits<int64_t>::max();
  na_chi_count_native<SerialOps, D>(mgr, ctx, view(counts, {5}));
  assert(counts[0] == 3LL * (int64_t(INT32_MAX) + 1));
  combos[0] = -100;
  na_chi_count_native<SerialOps, D>(mgr, ctx, view(counts, {5}));
  assert(counts[0] == -3);
  ctx.types = view(types, {0});
  na_chi_count_native<SerialOps, D>(mgr, ctx, view(counts, {0}));
  na_chi_fill_native<SerialOps, D>(
      mgr,
      ctx,
      view(offsets, {0}),
      view(owners, {0}),
      view(out_atoms, {0, 2}),
      view(chi, {0, 2}));
}
template <typename Real>
void check_pucker() {
  tmol::ContextManager mgr;
  Real xyz[15] = {
      0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, Real(.4), Real(-.3), Real(.5), 0};
  int64_t ring[10] = {0, 1, 2, 3, 4, 2500000000LL, 1, 2, 3, 4}, out[2];
  bool valid[10] = {
      true, true, true, true, true, false, true, true, true, true};
  na_sampling_pucker_native<SerialOps, D>(
      mgr,
      view(xyz, {5, 3}),
      view(ring, {2, 5}),
      view(valid, {2, 5}),
      Real(.1),
      view(out, {2}));
  assert(out[0] >= 0 && out[0] < 10 && out[1] == -1);
  auto first = out[0];
  for (int i = 0; i < 5; ++i) {
    Real x = xyz[3 * i], y = xyz[3 * i + 1];
    xyz[3 * i] = -y + 2;
    xyz[3 * i + 1] = x - 3;
    xyz[3 * i + 2] += 1;
  }
  na_sampling_pucker_native<SerialOps, D>(
      mgr,
      view(xyz, {5, 3}),
      view(ring, {2, 5}),
      view(valid, {2, 5}),
      Real(.1),
      view(out, {2}));
  assert(out[0] == first && out[1] == -1);
}
int main() {
  check_samples();
  check_pucker<float>();
  check_pucker<double>();
}
