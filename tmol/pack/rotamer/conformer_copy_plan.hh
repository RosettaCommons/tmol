#pragma once
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

namespace tmol::pack::rotamer {
// Caller owns source counts/prefix workspace and the [size,source,target]
// output rows. Padded pose blocks contribute zero to the compact DOF forest.
template <template <typename, int> class View, typename Int>
struct ConformerCopyPlan {
  View<Int, 2> original_types;
  View<int32_t, 1> type_natoms;
  View<Int, 1> considered, global_blocks, types, offsets, selected;
  View<int64_t, 1> source_counts, source_offsets;
  View<int64_t, 2> segments;
};
template <template <Device> class Ops, Device D, typename Context, typename Ctx>
void launch_conformer_source_counts(Context& context, Ctx c) {
  LAUNCH_BOX_128;
  int n = score::common::checked_dispatch_product(
      c.original_types.size(0),
      c.original_types.size(1),
      "conformer original blocks");
  Ops<D>::template forall<launch_t>(
      context, n, [=] TMOL_DEVICE_FUNC(int index) {
        auto type = c.original_types[index / c.original_types.size(1)]
                                    [index % c.original_types.size(1)];
        c.source_counts[index] = type < 0 ? 0 : c.type_natoms[type];
        // The common CPU exclusive scan preserves its first destination
        // element. Initialize that identity in this shared stage, not in
        // frontend allocators.
        if (index == 0) c.source_offsets[0] = 0;
      });
}
template <template <Device> class Ops, Device D, typename Context, typename Ctx>
void launch_conformer_copy_segments(Context& context, Ctx c) {
  LAUNCH_BOX_128;
  int n = score::common::checked_dispatch_size(
      c.selected.size(0), "conformer copy segments");
  Ops<D>::template forall<launch_t>(context, n, [=] TMOL_DEVICE_FUNC(int row) {
    auto conf = c.selected[row], type = c.types[conf];
    auto block = c.global_blocks[c.considered[conf]];
    c.segments[0][row] = c.type_natoms[type];
    c.segments[1][row] = c.source_offsets[block];
    c.segments[2][row] = c.offsets[conf];
  });
}
// Reset rows address flattened ideal_dofs[type,atom,9] and the packed target
// DOF forest (including its virtual-root row). Invalid slots remain -1.
template <template <typename, int> class View, typename Int>
struct OptHResetPlan {
  View<Int, 1> types, offsets, selected;
  View<int32_t, 2> chi_atoms, downstream;
  View<int32_t, 1> downstream_count;
  View<int64_t, 2> indices;
  int64_t ideal_width;
  bool flip;
};
template <template <Device> class Ops, Device D, typename Context, typename Ctx>
void launch_opth_reset_indices(Context& context, Ctx c) {
  LAUNCH_BOX_128;
  int n = score::common::checked_dispatch_size(
      c.selected.size(0), "OptH reset conformers");
  Ops<D>::template forall<launch_t>(context, n, [=] TMOL_DEVICE_FUNC(int row) {
    auto conf = c.selected[row], type = c.types[conf];
    bool active = false;
    for (int chi = 0; chi < c.chi_atoms.size(1); ++chi)
      active |= c.chi_atoms[row][chi] >= 0;
    for (int slot = 0; slot < c.downstream.size(1); ++slot) {
      auto index = int64_t(row) * c.downstream.size(1) + slot;
      bool reset = c.flip && active && slot < c.downstream_count[type];
      auto atom = reset ? c.downstream[type][slot] : -1;
      c.indices[0][index] = atom < 0 ? -1 : type * c.ideal_width + atom;
      c.indices[1][index] = atom < 0 ? -1 : c.offsets[conf] + atom + 1;
    }
  });
}
}  // namespace tmol::pack::rotamer
