#include <tmol/io/details/compiled/canonical_policies.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <limits>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;
template <class Int, class Real>
void check() {
  tmol::ContextManager mgr;
  Int types[] = {0, 0, 1, -1}, variants[] = {3, 3, 1, 0},
      terms[] = {0, 1, 0, 0}, effective[4];
  bool valid[2 * 4 * 4] = {};
  valid[3] = true;
  CanonicalVariantContext<Int, CPU> v{
      view(types, {1, 4}),
      view(variants, {1, 4}),
      view(terms, {1, 4}),
      view(valid, {2, 4, 4, 1}),
      3,
      view(effective, {1, 4})};
  launch_canonical_variants<tmol_test::SerialOps, CPU>(mgr, v);
  assert(
      effective[0] == 3 && effective[1] == 0 && effective[2] == 1
      && effective[3] == 0);
  Int up[] = {1, -1}, down[] = {0, -1}, partner[8],
      cyctypes[] = {0, 0, 0, 1, 0, 0, -1, -1};
  int32_t chains[] = {0, 0, 0, 0, 0, 0, -1, -1};
  Vec<Real, 3> coords[16];
  for (auto& c : coords) c = {std::numeric_limits<Real>::quiet_NaN(), 0, 0};
  coords[0] = {0, 0, 0};
  coords[5] = {1, 0, 0};
  coords[8] = {0, 0, 0};
  coords[11] = {Real(1.5), 0, 0};
  Int explicit_rows[] = {0, 2, 0};
  CanonicalCycleContext<Real, Int, CPU> cycle{
      view(cyctypes, {2, 4}),
      view(chains, {2, 4}),
      view(up, {2}),
      view(down, {2}),
      view(coords, {2, 4, 2}),
      view(explicit_rows, {0, 3}),
      Real(2),
      view(partner, {2, 4})};
  launch_canonical_cycles<tmol_test::SerialOps, CPU>(mgr, cycle);
  assert(
      partner[2] == 0 && partner[5] == 0 && partner[3] == -1
      && partner[7] == -1);
  cycle.explicit_pairs = view(explicit_rows, {1, 3});
  launch_canonical_cycles<tmol_test::SerialOps, CPU>(mgr, cycle);
  assert(partner[2] == -1 && partner[5] == 0);
  coords[11] = {2, 0, 0};
  launch_canonical_cycles<tmol_test::SerialOps, CPU>(mgr, cycle);
  assert(partner[5] == -1);  // Strict distance cutoff.
  // Nonuniform port identities and padding: polymer 0-1, disulfide 0-2.
  Int wtypes[] = {0, 1, 2, -1}, wup[] = {2, 1, -1}, wdown[] = {1, 0, -1},
      dslf[] = {0, 2, 1};
  bool polymer[] = {1, 1, 0, 0}, first[] = {1, 0, 1, 0}, last[] = {0, 1, 1, 0},
       broken[8] = {};
  Int pair[] = {0, 0, 2}, empty[5] = {}, status[1];
  Vec<Int, 2> connections[12];
  bool bad[12];
  CanonicalWiringContext<Int, CPU> w{
      view(wtypes, {1, 4}),
      view(polymer, {1, 4}),
      view(first, {1, 4}),
      view(last, {1, 4}),
      view(broken, {1, 4, 2}),
      view(wup, {3}),
      view(wdown, {3}),
      view(dslf, {3}),
      view(pair, {1, 3}),
      view(empty, {0, 3}),
      view(empty, {0, 5}),
      view(connections, {1, 4, 3}),
      view(status, {1})};
  launch_canonical_wiring<tmol_test::SerialOps, CPU>(mgr, w);
  assert(status[0] == 0 && connections[2][0] == 1 && connections[2][1] == 0);
  assert(
      connections[0][0] == 2 && connections[0][1] == 1
      && connections[7][0] == 0);
  for (int i = 9; i < 12; ++i)
    assert(connections[i][0] == -1 && connections[i][1] == -1);
  connections[3] = {9, 0};
  launch_canonical_connection_validation<tmol_test::SerialOps, CPU>(
      mgr, w.types, w.connections, view(bad, {1, 4, 3}));
  assert(bad[2] && bad[3]);
  broken[1] = true;
  launch_canonical_wiring<tmol_test::SerialOps, CPU>(mgr, w);
  assert(
      status[0]
      == 1);  // An asymmetric break is an error, never silently dropped.
  broken[1] = false;
  pair[0] = 3;
  launch_canonical_wiring<tmol_test::SerialOps, CPU>(mgr, w);
  assert(status[0] == 2);
  pair[0] = 0;
  wtypes[0] = 5;
  launch_canonical_wiring<tmol_test::SerialOps, CPU>(mgr, w);
  assert(status[0] == 2);
}
int main() {
  check<int32_t, float>();
  check<int64_t, float>();
  check<int32_t, double>();
  check<int64_t, double>();
}
