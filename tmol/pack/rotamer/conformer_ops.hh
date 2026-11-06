#pragma once
#include <tmol/pack/rotamer/conformer_dofs.hh>
#include <tmol/pack/rotamer/conformer_copy_plan.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/utility/tensor/context_manager.hh>
namespace tmol::pack::rotamer {
template <template <Device> class Ops, Device D, typename Real, typename Int>
struct ConformerDofKernels {
  using Binder = score::common::TViewBinder<D>;
  using Copy = MainchainDofCopy<Binder::template View, Real, Int>;
  using Chi = ChiDofAssignment<Binder::template View, Real, Int>;
  using Rows = ConformerRowCopy<Binder::template View, Real, Int>;
  static void rows(ContextManager&, Rows, int64_t, int);
  using RowIndices = ConformerRowIndexPlan<Binder::template View, Int>;
  static void row_indices(ContextManager&, RowIndices, int);
  using ChiIndices = ChiDofIndexPlan<Binder::template View, Int>;
  static void chi_indices(ContextManager&, ChiIndices);
  using CopyPlan = ConformerCopyPlan<Binder::template View, Int>;
  static void copy_plan(ContextManager&, CopyPlan);
  using ResetPlan = OptHResetPlan<Binder::template View, Int>;
  static void opth_reset(ContextManager&, ResetPlan);
  using Correction = ChiDofCorrection<Binder::template View, Real, Int>;
  static void correct(ContextManager&, Correction);
  static void copy(ContextManager&, Copy);
  using ValueIndices = typename Binder::template View<int64_t, 2>;
  using MainchainIndices = MainchainDofIndexPlan<Binder::template View, Int>;
  static void mainchain_rows(ContextManager&, MainchainIndices, ValueIndices);
  static void mainchain_indices(ContextManager&, Copy, ValueIndices);
  static void assign(ContextManager&, Chi);
};
}  // namespace tmol::pack::rotamer
