#include <tmol/io/details/compiled/histidine_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;
template <typename Real, typename Int>
void check() {
  using V = tmol::portable::Vec<Real, 3>;
  V coords[72] = {};
  bool present[72] = {};
  Int types[9] = {}, variants[9] = {};
  int64_t poses[9] = {0, 0, 0, 0, 0, 0, 0, 0, -1},
          residues[9] = {0, 1, 2, 3, 4, 5, 6, 7, -1};
  int64_t mapping[72], states[9] = {};
  for (int r = 0; r < 9; ++r)
    for (int a = 0; a < 8; ++a) mapping[r * 8 + a] = a;
  present[2] = true;
  present[8 + 3] = true;
  present[16 + 2] = present[16 + 3] = true;
  present[32] = present[33] = present[36] = true;
  coords[32] = V{0, 0, 0};
  coords[33] = V{4, 0, 0};
  coords[36] = V{Real(.7), 0, 0};
  for (int r : {5, 6}) {
    present[r * 8 + 4] = present[r * 8 + 5] = present[r * 8 + 6] =
        present[r * 8 + 7] = true;
    coords[r * 8 + 5] = V{0, 0, 0};
    coords[r * 8 + 6] = V{4, 0, 0};
  }
  coords[47] = V{-1, 0, 0};
  coords[55] = V{5, 0, 0};
  present[60] = true;
  variants[8] = 7;
  types[8] = -1;
  HisAtomIndsInCanonicalOrdering<Int> indices[] = {{0, 1, 2, 3, 4, 5, 6, 7}};
  HisTautContext<Real, Int, CPU> c{
      view(coords, {1, 9, 8}),
      view(types, {1, 9}),
      view(variants, {1, 9}),
      view(poses, {9}),
      view(residues, {9}),
      view(present, {1, 9, 8}),
      view(indices, {1}),
      view(mapping, {1, 9, 8}),
      view(states, {1, 9})};
  tmol::ContextManager mgr;
  launch_resolve_his_taut<tmol_test::SerialOps, CPU>(mgr, c);
  int expected_states[] = {1, 2, 5, 2, 1, 3, 4, 6, 0};
  int expected_variants[] = {1, 0, 2, 0, 1, 1, 0, 3, 7};
  for (int r = 0; r < 9; ++r) {
    assert(states[r] == expected_states[r]);
    assert(variants[r] == expected_variants[r]);
  }
  assert(mapping[34] == 4);
  assert(mapping[40] == 5 && mapping[42] == 4 && mapping[41] == 6);
  assert(mapping[48] == 6 && mapping[51] == 4 && mapping[49] == 5);
  for (int a = 0; a < 8; ++a) assert(mapping[64 + a] == a);
  // Relabel alternate classes' atom slots. Physical tautomer decisions and
  // alias destinations must be invariant to that canonical permutation.
  V reordered[72];
  bool reordered_present[72];
  int64_t expected_mapping[72];
  HisAtomIndsInCanonicalOrdering<Int> by_class[] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {7, 6, 5, 4, 3, 2, 1, 0}};
  for (int r = 0; r < 9; ++r) {
    if (r < 8) types[r] = r % 2;
    for (int a = 0; a < 8; ++a) {
      auto slot = r % 2 ? 7 - a : a;
      reordered[r * 8 + slot] = coords[r * 8 + a];
      reordered_present[r * 8 + slot] = present[r * 8 + a];
      expected_mapping[r * 8 + slot] =
          r % 2 ? 7 - mapping[r * 8 + a] : mapping[r * 8 + a];
    }
  }
  for (int r = 0; r < 9; ++r)
    for (int a = 0; a < 8; ++a) mapping[r * 8 + a] = a;
  c.coords = view(reordered, {1, 9, 8});
  c.atom_is_present = view(reordered_present, {1, 9, 8});
  c.his_atom_inds = view(by_class, {2});
  launch_resolve_his_taut<tmol_test::SerialOps, CPU>(mgr, c);
  for (int r = 0; r < 9; ++r) {
    assert(
        states[r] == expected_states[r] && variants[r] == expected_variants[r]);
    for (int a = 0; a < 8; ++a)
      assert(mapping[r * 8 + a] == expected_mapping[r * 8 + a]);
  }
}
int main() {
  check<float, int32_t>();
  check<double, int32_t>();
  check<double, int64_t>();
  check<float, int64_t>();
}
