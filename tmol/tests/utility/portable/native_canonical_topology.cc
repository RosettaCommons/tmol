#include <tmol/io/details/compiled/canonical_topology.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;
template <typename Int>
void check() {
  int32_t chains[] = {0, 0, 0, 0, 0, 0, -1, -1};
  Int types[] = {0, 0, 1, 0, 0, 0, -1, -1};
  bool real[] = {1, 1, 1, 1, 1, 1, 0, 0}, disconnected[16] = {},
       explicit_ports[16] = {};
  bool polymer[] = {1, 0, 1};
  Int ports[] = {0, 1, -1, -1, -1, 1};
  Int cycles[] = {1, 1, 0}, termini[8];
  bool first[8], last[8], poly[8];
  CanonicalChainContext<Int, CPU> c{
      view(chains, {2, 4}),
      view(types, {2, 4}),
      view(disconnected, {2, 4, 2}),
      view(real, {2, 4}),
      view(polymer, {3}),
      view(ports, {3, 2}),
      view(cycles, {1, 3}),
      view(explicit_ports, {2, 4, 2}),
      view(termini, {2, 4}),
      view(first, {2, 4}),
      view(last, {2, 4}),
      view(poly, {2, 4})};
  tmol::ContextManager mgr;
  launch_canonical_chain_endings<tmol_test::SerialOps, CPU>(mgr, c);
  // A nonpolymer splits polymer connectivity despite an equal chain label.
  assert(
      termini[0] == 0 && termini[1] == 2 && termini[2] == 1 && termini[3] == 3);
  assert(!poly[2] && first[3] && last[1]);
  // Closing a cycle removes chemical termini but retains sequence boundaries.
  assert(termini[4] == 1 && termini[5] == 1 && first[4] && last[5]);
  assert(termini[6] == 1 && termini[7] == 1 && !poly[6] && !poly[7]);
  explicit_ports[1] = true;
  launch_canonical_chain_endings<tmol_test::SerialOps, CPU>(mgr, c);
  assert(last[0] && first[1] && termini[0] == 0 && termini[1] == 3);
  // An already closed chemical port is terminal even at a user-declared gap.
  c.cycles = view(cycles, {0, 3});
  types[4] = 2;
  disconnected[8] = true;
  launch_canonical_chain_endings<tmol_test::SerialOps, CPU>(mgr, c);
  assert(termini[4] == 0 && termini[5] == 2);

  bool present[] = {1, 1, 1}, one_real[] = {1}, broken[] = {1, 1};
  Int io[] = {0}, variants[] = {0}, original[] = {1}, output[1], counts[4];
  // Original retains a heavy atom; one tied alternative loses it while gaining
  // hydrogen names. Even a smaller extraneous count must not select that group.
  Int candidates[] = {-1, -1, 0, -1, -1, -1, 1, 2};
  bool valid[] = {0, 0, 1, 0, 0, 0, 1, 1};
  bool absent[] = {0, 1, 1, 0, 0, 1, 1, 0, 0};
  CanonicalTerminusContext<Int, CPU> t{
      view(present, {1, 1, 3}),
      view(io, {1, 1}),
      view(variants, {1, 1}),
      view(original, {1, 1}),
      view(broken, {1, 1, 2}),
      view(one_real, {1, 1}),
      view(candidates, {1, 4, 1, 2}),
      view(valid, {1, 4, 1, 2}),
      view(absent, {3, 3}),
      view(counts, {1, 1, 2, 2}),
      view(output, {1, 1})};
  launch_canonical_atom_termini<tmol_test::SerialOps, CPU>(mgr, t);
  assert(output[0] == 1);
  valid[7] = false;
  launch_canonical_atom_termini<tmol_test::SerialOps, CPU>(mgr, t);
  assert(
      output[0]
      == 3);  // All remaining best alternatives retain the heavy atom.
  one_real[0] = false;
  launch_canonical_atom_termini<tmol_test::SerialOps, CPU>(mgr, t);
  assert(output[0] == 1);
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
