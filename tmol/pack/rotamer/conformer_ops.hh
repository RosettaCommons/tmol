#pragma once
#include <tmol/pack/rotamer/conformer_dofs.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/utility/tensor/context_manager.hh>
namespace tmol::pack::rotamer {
template <template <Device> class Ops, Device D, typename Real, typename Int>
struct ConformerDofKernels {
  using Binder = score::common::TViewBinder<D>;
  using Copy = MainchainDofCopy<Binder::template View, Real, Int>;
  using Chi = ChiDofAssignment<Binder::template View, Real, Int>;
  static void copy(ContextManager&, Copy);
  static void assign(ContextManager&, Chi);
};
}  // namespace tmol::pack::rotamer
