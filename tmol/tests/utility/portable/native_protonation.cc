#include <tmol/io/details/compiled/protonation.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;
template <class Int>
void check() {
  tmol::ContextManager mgr;
  Int types[] = {0, 0, 0, 0, 0, 0, -1}, variants[] = {0, 0, 0, 0, 0, 0, 7},
      out[7];
  bool present[28] = {}, hydrogen[] = {true, true, false, false},
       titr[] = {false, true, false, false},
       parents[] = {false, false, true, false},
       shared[] = {true, false, false, false},
       expects[] = {true, false, false, false, false}, donates[20] = {}, bad[7];
  present[0] = true;  // complete shared H but no titratable H -> deprotonated
  present[5] = true;  // only titratable H: incomplete evidence -> unchanged
  present[8] = true;  // covalent attachment prevents deprotonation
  present[20] = present[21] =
      true;           // supplied H prevents forced metal donor form
  donates[6] = true;  // variant1 can donate atom2
  int64_t bonds[] = {0, 2, 2, 5, 3}, required[] = {0, 3, 2, 0, 4, 1, 0, 5, 2};
  ProtonationContext<Int, CPU> c{
      view(types, {1, 7}),
      view(variants, {1, 7}),
      view(present, {1, 7, 4}),
      view(hydrogen, {1, 4}),
      view(titr, {1, 4}),
      view(parents, {1, 4}),
      view(shared, {1, 4}),
      view(expects, {1, 5}),
      view(donates, {1, 5, 4}),
      view(bonds, {1, 5}),
      view(required, {3, 3}),
      Int(4),
      view(out, {1, 7}),
      view(bad, {1, 7})};
  launch_protonation<tmol_test::SerialOps, CPU>(mgr, c);
  Int expected[] = {4, 0, 0, 1, 0, 0, 7};
  for (int r = 0; r < 7; ++r) {
    assert(out[r] == expected[r]);
    assert(bad[r] == (r == 4));
    assert(variants[r] == (r == 6 ? 7 : 0));
  }
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
