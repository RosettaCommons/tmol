#pragma once
#include <tmol/pack/rotamer/conformer_ops.hh>
namespace tmol::pack::rotamer {
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::copy(ContextManager& mgr, Copy c) {
  launch_copy_mainchain_dofs<Ops, D, Real>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::assign(
    ContextManager& mgr, Chi c) {
  launch_assign_chi_dofs<Ops, D, Real>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::rows(
    ContextManager& mgr, Rows c, int64_t n_rows, int root_shift) {
  launch_copy_conformer_rows<Ops, D>(mgr, c, n_rows, root_shift);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::correct(
    ContextManager& mgr, Correction c) {
  launch_correct_chi_dofs<Ops, D, Real>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::row_indices(
    ContextManager& mgr, RowIndices c, int shift) {
  launch_conformer_row_indices<Ops, D>(mgr, c, shift);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::chi_indices(
    ContextManager& mgr, ChiIndices c) {
  launch_chi_dof_indices<Ops, D>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::copy_plan(
    ContextManager& mgr, CopyPlan c) {
  launch_conformer_source_counts<Ops, D>(mgr, c);
  int n = score::common::checked_dispatch_size(
      c.source_counts.size(0), "conformer original blocks");
  if (n)
    Ops<D>::template scan<portable::scan_type::exclusive>(
        mgr,
        c.source_counts.data(),
        c.source_offsets.data(),
        n,
        portable::plus_t<int64_t>());
  launch_conformer_copy_segments<Ops, D>(mgr, c);
}
template <template <Device> class Ops, Device D, typename Real, typename Int>
void ConformerDofKernels<Ops, D, Real, Int>::opth_reset(
    ContextManager& mgr, ResetPlan c) {
  launch_opth_reset_indices<Ops, D>(mgr, c);
}
}  // namespace tmol::pack::rotamer
