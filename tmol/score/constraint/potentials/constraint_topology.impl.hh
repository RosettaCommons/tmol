#pragma once
#include <tmol/score/constraint/potentials/constraint_score.hh>
#include <tmol/score/constraint/potentials/constraint_topology.hh>
#include <tmol/score/common/counting.hh>
namespace tmol::score::constraint::potentials {
template <
    typename Index,
    template <tmol::Device> class Ops,
    tmol::Device D,
    typename Int>
auto allocate_constraint_candidates(
    ContextManager& mgr,
    ConstraintCandidateTopology<Int, D> ctx,
    bool expanded,
    TView<int64_t, 1, D> offsets,
    TView<Int, 1, D> owners,
    int total) -> std::
    tuple<tmol::native::Tensor, tmol::native::Tensor, tmol::native::Tensor> {
  auto atoms = TPack<Vec<Int, 4>, 1, D>::empty({total});
  auto sources = TPack<Int, 1, D>::empty({total});
  auto indices = TPack<Vec<Index, 3>, 1, D>::empty({total});
  constraint_candidate_fill_native<Ops, D, Int, Index>(
      mgr,
      ctx,
      expanded,
      offsets,
      owners,
      sources.view,
      atoms.view,
      indices.view);
  return {atoms.tensor, sources.tensor, indices.tensor};
}
template <template <tmol::Device> class Ops, tmol::Device D, typename Int>
std::tuple<tmol::native::Tensor, tmol::native::Tensor, tmol::native::Tensor>
ConstraintCandidateDispatch<Ops, D, Int>::forward(
    ContextManager& mgr,
    TView<Vec<Int, 3>, 2, D> atoms,
    TView<Int, 1, D> n_unique,
    TView<Vec<Int, 3>, 1, D> unique_blocks,
    TView<Int, 1, D> coordinate_offsets,
    TView<Int, 2, D> n_conformers,
    TView<Int, 2, D> first_conformer,
    bool expanded) {
  TORCH_CHECK(atoms.size(1) == 4, "constraint atom slots must have width four");
  TORCH_CHECK(
      atoms.size(0) == n_unique.size(0)
          && atoms.size(0) == unique_blocks.size(0),
      "constraint topology row counts differ");
  TORCH_CHECK(
      n_conformers.size(0) == first_conformer.size(0)
          && n_conformers.size(1) == first_conformer.size(1),
      "constraint conformer table shapes differ");
  ConstraintCandidateTopology<Int, D> ctx{
      atoms,
      n_unique,
      coordinate_offsets,
      unique_blocks,
      n_conformers,
      first_conformer};
  int const n = tmol::score::common::checked_dispatch_size(
      atoms.size(0), "constraint source rows");
  if (!expanded)
    return allocate_constraint_candidates<Int, Ops, D>(
        mgr, ctx, false, {}, {}, n);
  int const generators = tmol::score::common::checked_dispatch_product(
      n, 2, "constraint count rows");
  auto counts = TPack<int64_t, 1, D>::empty({generators});
  auto offsets = TPack<int64_t, 1, D>::zeros({generators});
  int total = 0;
  if (generators) {
    constraint_candidate_count_native<Ops, D>(mgr, ctx, counts.view);
    int64_t const size = Ops<D>::template scan_and_return_total<
        tmol::portable::scan_type::exclusive>(
        mgr,
        counts.view.data(),
        offsets.view.data(),
        generators,
        tmol::portable::plus_t<int64_t>());
    total = tmol::score::common::checked_dispatch_size(
        size, "expanded constraint rows");
  }
  LAUNCH_BOX_128;
  auto owners = Ops<D>::template load_balancing_search<launch_t, int64_t, Int>(
      mgr, total, offsets.view.data(), generators);
  // Legacy sparse Cartesian products use int64 cumsum indices; whole-pose
  // attribution retains the structural input index dtype.
  return allocate_constraint_candidates<int64_t, Ops, D>(
      mgr, ctx, true, offsets.view, owners.view, total);
}
}  // namespace tmol::score::constraint::potentials
