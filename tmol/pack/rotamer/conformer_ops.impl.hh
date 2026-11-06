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
}  // namespace tmol::pack::rotamer
