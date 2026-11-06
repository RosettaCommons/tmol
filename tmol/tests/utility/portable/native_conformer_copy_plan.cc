#include <tmol/pack/rotamer/conformer_copy_plan.hh>
#include <tmol/score/common/tview_binder.hh>
#include "native_test_ops.hh"
#include <cassert>
using tmol_test::SerialOps;
using tmol_test::view;
using namespace tmol::pack::rotamer;
constexpr auto D = tmol::Device::CPU;
using Binder = tmol::score::common::TViewBinder<D>;
template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int original[] = {0, -1, 1, 2, 0, -1}, considered[] = {2, 0, 3},
      global[] = {4, 0, 3, 2};
  Int types[] = {2, 0, 1}, offsets[] = {7, 15, 20}, selected[] = {2, 0};
  int32_t natoms[] = {2, 3, 5};
  int64_t counts[6], prefix[] = {137, 2, 2, 5, 10, 12}, segments[6];
  ConformerCopyPlan<Binder::template View, Int> plan{
      view(original, {2, 3}),
      view(natoms, {3}),
      view(considered, {3}),
      view(global, {4}),
      view(types, {3}),
      view(offsets, {3}),
      view(selected, {2}),
      view(counts, {6}),
      view(prefix, {6}),
      view(segments, {3, 2})};
  launch_conformer_source_counts<SerialOps, D>(mgr, plan);
  assert(prefix[0] == 0);
  int64_t expected_counts[] = {2, 0, 3, 5, 2, 0};
  for (int i = 0; i < 6; ++i) assert(counts[i] == expected_counts[i]);
  launch_conformer_copy_segments<SerialOps, D>(mgr, plan);
  int64_t expected_segments[] = {3, 5, 2, 5, 20, 7};
  for (int i = 0; i < 6; ++i) assert(segments[i] == expected_segments[i]);
  int32_t chi[] = {-1, -1, 3, -1}, down[] = {-1, -1, -1, 1, 2, -1, 2, 4, -1},
          ndown[] = {0, 2, 2};
  int64_t indices[12];
  OptHResetPlan<Binder::template View, Int> reset{
      view(types, {3}),
      view(offsets, {3}),
      view(selected, {2}),
      view(chi, {2, 2}),
      view(down, {3, 3}),
      view(ndown, {3}),
      view(indices, {2, 6}),
      5,
      true};
  launch_opth_reset_indices<SerialOps, D>(mgr, reset);
  int64_t expected[] = {-1, -1, -1, 12, 14, -1, -1, -1, -1, 10, 12, -1};
  for (int i = 0; i < 12; ++i) assert(indices[i] == expected[i]);
  if constexpr (sizeof(Int) == 8) {
    offsets[0] = Int(2500000000LL);
    launch_opth_reset_indices<SerialOps, D>(mgr, reset);
    assert(indices[9] == 2500000003LL && indices[10] == 2500000005LL);
  }
  reset.flip = false;
  launch_opth_reset_indices<SerialOps, D>(mgr, reset);
  for (auto i : indices) assert(i == -1);
  plan.original_types = view(original, {0, 0});
  plan.selected = view(selected, {0});
  launch_conformer_source_counts<SerialOps, D>(mgr, plan);
  launch_conformer_copy_segments<SerialOps, D>(mgr, plan);
  reset.selected = view(selected, {0});
  launch_opth_reset_indices<SerialOps, D>(mgr, reset);
}
int main() {
  check<int32_t>();
  check<int64_t>();
}
