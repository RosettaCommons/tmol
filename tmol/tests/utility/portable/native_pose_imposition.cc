#include <tmol/pack/rotamer/pose_imposition.hh>
#include "native_test_ops.hh"
#include <cassert>
using namespace tmol::pack::rotamer;
using tmol_test::SerialOps;
using tmol_test::view;
constexpr auto D = tmol::Device::CPU;
int main() {
  tmol::ContextManager mgr;
  int64_t original[] = {0, 1}, fixed[] = {-1, 1}, retained_offsets[] = {0, 0};
  int64_t molten_counts[] = {1, 0}, mapping[] = {0}, flags[2],
          prefix[] = {0, 1};
  int32_t choices[] = {0, INT32_MAX}, errors[2];
  int64_t selected[2];
  PoseImpositionSelection<D> c{
      view(original, {2, 1}),
      view(fixed, {2, 1}),
      view(retained_offsets, {2, 1}),
      view(molten_counts, {2}),
      view(mapping, {1}),
      view(choices, {2, 1})};
  imposition_molten_flags_native<SerialOps, D>(mgr, c, view(flags, {2}));
  assert(flags[0] == 1 && flags[1] == 0);
  imposition_select_native<SerialOps, D>(
      mgr,
      c,
      view(flags, {2}),
      view(prefix, {2}),
      view(selected, {2, 1}),
      view(errors, {2, 1}));
  assert(selected[0] == 0 && selected[1] == 1 && !errors[0] && !errors[1]);
  // An invalid 64-bit retained offset must be rejected before dereference.
  retained_offsets[0] = 2500000000LL;
  imposition_select_native<SerialOps, D>(
      mgr,
      c,
      view(flags, {2}),
      view(prefix, {2}),
      view(selected, {2, 1}),
      view(errors, {2, 1}));
  assert(errors[0] && selected[0] == -1);
  selected[0] = 0;
  errors[0] = 0;
  int64_t rot_types[] = {0, 1}, types[2], counts[2], atom_prefix[] = {0, 1},
          totals[2];
  int32_t n_atoms[] = {1, 2}, rot_offsets[] = {0, 1}, changed[2],
          old_offsets[] = {0, 5}, offsets[2];
  imposition_types_native<SerialOps, D>(
      mgr,
      view(original, {2, 1}),
      view(selected, {2, 1}),
      view(rot_types, {2}),
      view(n_atoms, {2}),
      view(rot_offsets, {2}),
      3,
      view(types, {2, 1}),
      view(counts, {2}),
      view(changed, {2}),
      view(errors, {2, 1}));
  assert(counts[0] == 1 && counts[1] == 2 && !changed[0] && !changed[1]);
  imposition_layout_native<SerialOps, D>(
      mgr,
      view(counts, {2}),
      view(atom_prefix, {2}),
      view(old_offsets, {2, 1}),
      false,
      2147483647LL,
      view(offsets, {2, 1}),
      view(totals, {2}),
      view(errors, {2, 1}));
  assert(offsets[1] == 5 && totals[1] == 2);
  int32_t owners[] = {0, 1, 1};
  int64_t source[3], target[3];
  imposition_atom_indices_native<SerialOps, D>(
      mgr,
      view(selected, {2, 1}),
      view(rot_offsets, {2}),
      view(atom_prefix, {2}),
      view(owners, {3}),
      view(offsets, {2, 1}),
      2147483647LL,
      view(source, {3}),
      view(target, {3}));
  assert(source[0] == 0 && source[1] == 1 && source[2] == 2);
  assert(
      target[0] == 0 && target[1] == 2147483652LL && target[2] == 2147483653LL);
  // Compact layout discards gaps while retaining per-pose atom totals.
  imposition_layout_native<SerialOps, D>(
      mgr,
      view(counts, {2}),
      view(atom_prefix, {2}),
      view(old_offsets, {2, 1}),
      true,
      2147483647LL,
      view(offsets, {2, 1}),
      view(totals, {2}),
      view(errors, {2, 1}));
  assert(
      offsets[0] == 0 && offsets[1] == 0 && totals[0] == 1 && totals[1] == 2);
}
