#pragma once
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/rotamer_correlation.hh>
namespace tmol::score::constraint::potentials {
using tmol::portable::Vec;
// Structural inputs only. Parameters and custom callbacks stay with the
// frontend, which gathers their rows through the returned source indices.
// Expanded ordering preserves the mature two-body rows followed by one-body
// rows; the first block's local conformer index advances fastest. This entry
// does not add support for three-body constraints. Nonempty group metadata
// requires validated equal active conformer counts within each coupled group.
template <typename Int, tmol::Device D>
struct ConstraintCandidateTopology {
  TView<Vec<Int, 3>, 2, D> atoms;
  TView<Int, 1, D> n_unique, coordinate_offsets;
  TView<Vec<Int, 3>, 1, D> unique_blocks;
  TView<Int, 2, D> n_conformers, first_conformer;
  TView<Int, 2, D> lockstep_group_for_block;
};
template <template <tmol::Device> class Ops, tmol::Device D, typename Int>
void constraint_candidate_count_native(
    ContextManager& mgr,
    ConstraintCandidateTopology<Int, D> ctx,
    TView<int64_t, 1, D> counts) {
  LAUNCH_BOX_128;
  int const n = ctx.atoms.size(0);
  auto count = [ctx, n, counts] TMOL_DEVICE_FUNC(int row) {
    int const source = row % n, phase = row / n;
    int const body = ctx.n_unique[source];
    auto const blocks = ctx.unique_blocks[source];
    int64_t total = 0;
    if ((phase == 0 && body == 2) || (phase == 1 && body == 1)) {
      total = ctx.n_conformers[blocks[0]][blocks[1]];
      if (body == 2) {
        Int const other = ctx.n_conformers[blocks[0]][blocks[2]];
        bool const coupled = common::same_rotamer_group(
            ctx.lockstep_group_for_block, blocks[0], blocks[1], blocks[2]);
        total *= coupled && other > 0 ? 1 : other;
      }
    }
    counts[row] = total;
  };
  Ops<D>::template forall<launch_t>(mgr, 2 * n, count);
}
template <
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Int,
    typename Index>
void constraint_candidate_fill_native(
    ContextManager& mgr,
    ConstraintCandidateTopology<Int, D> ctx,
    bool expanded,
    TView<int64_t, 1, D> offsets,
    TView<Int, 1, D> owners,
    TView<Int, 1, D> source_rows,
    TView<Vec<Int, 4>, 1, D> atom_indices,
    TView<Vec<Index, 3>, 1, D> sparse_indices) {
  LAUNCH_BOX_128;
  int const n = ctx.atoms.size(0);
  auto fill = [ctx,
               expanded,
               offsets,
               owners,
               source_rows,
               atom_indices,
               sparse_indices,
               n] TMOL_DEVICE_FUNC(int output) {
    Int const owner = expanded ? owners[output] : Int(output);
    Int const source = expanded ? owner % n : owner;
    source_rows[output] = source;
    auto const blocks = ctx.unique_blocks[source];
    if (!expanded) {
      sparse_indices[output] = Vec<Index, 3>{blocks[0], blocks[1], blocks[2]};
      for (int a = 0; a < 4; ++a) {
        auto const atom = ctx.atoms[source][a];
        // Preserve Python indexing for unused padded atom slots. Callbacks
        // define how many of the four slots are physically meaningful.
        Int const pose =
            atom[0] < 0 ? ctx.first_conformer.size(0) + atom[0] : atom[0];
        Int const block =
            atom[1] < 0 ? ctx.first_conformer.size(1) + atom[1] : atom[1];
        Int rot = ctx.first_conformer[pose][block];
        if (rot < 0) rot += ctx.coordinate_offsets.size(0);
        atom_indices[output][a] = ctx.coordinate_offsets[rot] + atom[2];
      }
      return;
    }
    int64_t const local = int64_t(output) - offsets[owner];
    bool const two = ctx.n_unique[source] == 2;
    Int const n_first = ctx.n_conformers[blocks[0]][blocks[1]];
    bool const coupled =
        two
        && common::same_rotamer_group(
            ctx.lockstep_group_for_block, blocks[0], blocks[1], blocks[2]);
    Int const first = ctx.first_conformer[blocks[0]][blocks[1]]
                      + (two && !coupled ? local % n_first : local);
    Int const second = two ? ctx.first_conformer[blocks[0]][blocks[2]]
                                 + (coupled ? local : local / n_first)
                           : first;
    sparse_indices[output] = Vec<Index, 3>{blocks[0], first, second};
    for (int a = 0; a < 4; ++a) {
      auto const atom = ctx.atoms[source][a];
      bool const on_first = atom[1] == ctx.atoms[source][0][1];
      Int const offset = on_first
                             ? ctx.coordinate_offsets[first]
                             : (two ? ctx.coordinate_offsets[second] : Int(0));
      atom_indices[output][a] = offset + atom[2];
    }
  };
  Ops<D>::template forall<launch_t>(mgr, source_rows.size(0), fill);
}
}  // namespace tmol::score::constraint::potentials
