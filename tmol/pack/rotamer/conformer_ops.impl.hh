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
}  // namespace tmol::pack::rotamer
