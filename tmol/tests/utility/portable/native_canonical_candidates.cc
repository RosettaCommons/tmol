#include <tmol/io/details/compiled/canonical_candidates.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;
template <typename Int>
void check() {
  // Distinguish exact ties, non-default termini, missing nonterminal atoms,
  // extraneous supplied atoms, no candidate, padding and invalid class/variant.
  bool present[] = {1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  bool real[] = {1, 1, 1, 1, 0, 1};
  Int io[] = {0, 0, 0, 1, -1, 2}, term[6] = {}, var[6] = {};
  Int candidates[] = {0, 1, 2, 3, -1, -1, -1, -1};
  bool valid[] = {1, 1, 1, 1, 0, 0, 0, 0};
  bool absent[] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1};
  bool nonterminal[] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  bool nondefault[] = {1, 0, 0, 0};
  Int selected[6], indices[6];
  int64_t scores[6];
  CanonicalCandidateContext<Int, CPU> c{
      view(present, {2, 3, 3}),
      view(real, {2, 3}),
      view(io, {2, 3}),
      view(term, {2, 3}),
      view(var, {2, 3}),
      view(candidates, {2, 1, 1, 4}),
      view(valid, {2, 1, 1, 4}),
      view(absent, {4, 3}),
      view(nonterminal, {4, 3}),
      view(nondefault, {4}),
      view(selected, {2, 3}),
      view(scores, {2, 3}),
      view(indices, {2, 3})};
  tmol::ContextManager mgr;
  launch_canonical_candidates<tmol_test::SerialOps, CPU>(mgr, c);
  assert(selected[0] == 1 && indices[0] == 1 && scores[0] == 0);
  assert(selected[1] == 1 && scores[1] == 0);
  assert(selected[2] == 1 && scores[2] == 8);
  assert(selected[3] == -1 && scores[3] == (int64_t(1) << 30));
  assert(selected[4] == -1 && scores[4] == 0);
  assert(selected[5] == -1 && scores[5] == (int64_t(1) << 30));
  // Reordering genuinely tied candidates changes the selected palette ID.
  std::swap(candidates[1], candidates[2]);
  nonterminal[7] = true;  // type 2 now requires missing nonterminal atom 1
  launch_canonical_candidates<tmol_test::SerialOps, CPU>(mgr, c);
  assert(selected[0] == 1 && indices[0] == 2);
  assert(selected[1] == 2 && indices[1] == 1);
  var[0] = -1;
  launch_canonical_candidates<tmol_test::SerialOps, CPU>(mgr, c);
  assert(selected[0] == -1 && scores[0] == (int64_t(1) << 30));
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
