#pragma once
#include <tmol/pack/rotamer/pose_imposition.hh>
#include <tmol/pack/rotamer/pose_imposition_torch.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/score/common/counting.hh>
namespace tmol::pack::rotamer {
template <Device D>
PoseImpositionSelection<D> imposition_selection(
    std::vector<tmol::native::Tensor> const& a) {
  TORCH_CHECK(a.size() >= 6, "Pose imposition requires six selection inputs");
  PoseImpositionSelection<D> c{
      TCAST(a[0]),
      TCAST(a[1]),
      TCAST(a[3]),
      TCAST(a[2]),
      TCAST(a[4]),
      TCAST(a[5])};
  auto p = c.original_types.size(0), b = c.original_types.size(1);
  TORCH_CHECK(
      c.fixed_rotamers.size(0) == p && c.fixed_rotamers.size(1) == b
          && c.molten_counts.size(0) == p && c.choices.size(0) == p
          && c.retained_offsets.size(0) == p
          && c.retained_offsets.size(1) == c.choices.size(1),
      "Pose imposition axes differ");
  return c;
}
template <template <Device> class Ops, Device D>
auto imposition_selection_buffers(
    ContextManager& mgr, PoseImpositionSelection<D> c) {
  int n = score::common::checked_dispatch_size(
      c.original_types.size(0) * c.original_types.size(1),
      "Pose imposition block count");
  auto flags = TPack<int64_t, 1, D>::empty({n});
  auto prefix = TPack<int64_t, 1, D>::zeros({n});
  auto selected = TPack<int64_t, 2, D>::empty(
      {c.original_types.size(0), c.original_types.size(1)});
  auto errors = TPack<int32_t, 2, D>::zeros(
      {c.original_types.size(0), c.original_types.size(1)});
  imposition_molten_flags_native<Ops, D>(mgr, c, flags.view);
  if (n)
    Ops<D>::template scan<portable::scan_type::exclusive>(
        mgr,
        flags.view.data(),
        prefix.view.data(),
        n,
        portable::plus_t<int64_t>());
  imposition_select_native<Ops, D>(
      mgr, c, flags.view, prefix.view, selected.view, errors.view);
  return std::make_pair(selected, errors);
}
template <template <Device> class Ops, Device D>
auto PoseImpositionTorch<Ops, D>::select(
    ContextManager& mgr, std::vector<Tensor> a) -> Tensor {
  auto c = imposition_selection<D>(a);
  auto result = imposition_selection_buffers<Ops, D>(mgr, c);
  int n = result.first.tensor.numel();
  TORCH_CHECK(
      !n
          || Ops<D>::reduce(
                 mgr,
                 result.second.view.data(),
                 n,
                 portable::maximum_t<int32_t>())
                 == 0,
      "Invalid pose imposition assignment or molten count");
  return result.first.tensor;
}
template <template <Device> class Ops, Device D>
auto PoseImpositionTorch<Ops, D>::plan(
    ContextManager& mgr,
    std::vector<Tensor> a,
    int64_t old_capacity,
    int64_t n_coordinates) -> std::vector<Tensor> {
  TORCH_CHECK(a.size() == 10, "Pose imposition requires ten plan inputs");
  auto c = imposition_selection<D>(a);
  auto selection = imposition_selection_buffers<Ops, D>(mgr, c);
  auto selected = selection.first;
  auto errors = selection.second;
  auto p = c.original_types.size(0), b = c.original_types.size(1);
  int n = selected.tensor.numel();
  TView<int64_t, 1, D> rot_types = TCAST(a[6]);
  TView<int32_t, 1, D> type_natoms = TCAST(a[7]), rot_offsets = TCAST(a[8]);
  TView<int32_t, 2, D> old_offsets = TCAST(a[9]);
  TORCH_CHECK(
      rot_types.size(0) == rot_offsets.size(0) && old_offsets.size(0) == p
          && old_offsets.size(1) == b && old_capacity >= 0
          && n_coordinates >= 0,
      "Pose imposition topology axes differ");
  auto types = TPack<int64_t, 2, D>::empty({p, b});
  auto counts = TPack<int64_t, 1, D>::empty({n});
  auto prefix = TPack<int64_t, 1, D>::zeros({n});
  auto changed = TPack<int32_t, 1, D>::empty({n});
  auto offsets = TPack<int32_t, 2, D>::empty({p, b});
  auto totals = TPack<int64_t, 1, D>::zeros({p});
  imposition_types_native<Ops, D>(
      mgr,
      c.original_types,
      selected.view,
      rot_types,
      type_natoms,
      rot_offsets,
      n_coordinates,
      types.view,
      counts.view,
      changed.view,
      errors.view);
  bool compact =
      n
      && Ops<D>::reduce(
             mgr, changed.view.data(), n, portable::maximum_t<int32_t>())
             != 0;
  int64_t total = n ? Ops<D>::template scan_and_return_total<
                          portable::scan_type::exclusive>(
                          mgr,
                          counts.view.data(),
                          prefix.view.data(),
                          n,
                          portable::plus_t<int64_t>())
                    : 0;
  int natoms =
      score::common::checked_dispatch_size(total, "Pose imposition atom count");
  imposition_layout_native<Ops, D>(
      mgr,
      counts.view,
      prefix.view,
      old_offsets,
      compact,
      old_capacity,
      offsets.view,
      totals.view,
      errors.view);
  TORCH_CHECK(
      !n
          || Ops<D>::reduce(
                 mgr, errors.view.data(), n, portable::maximum_t<int32_t>())
                 == 0,
      "Invalid pose imposition assignment, topology, or coordinate offset");
  int64_t capacity =
      compact && p
          ? Ops<D>::reduce(
                mgr, totals.view.data(), p, portable::maximum_t<int64_t>())
          : old_capacity;
  score::common::checked_dispatch_size(
      capacity, "Pose imposition pose capacity");
  LAUNCH_BOX_128;
  auto owners =
      Ops<D>::template load_balancing_search<launch_t, int64_t, int32_t>(
          mgr, natoms, prefix.view.data(), n);
  auto source = TPack<int64_t, 1, D>::empty({natoms}),
       target = TPack<int64_t, 1, D>::empty({natoms});
  imposition_atom_indices_native<Ops, D>(
      mgr,
      selected.view,
      rot_offsets,
      prefix.view,
      owners.view,
      offsets.view,
      capacity,
      source.view,
      target.view);
  // Allocation metadata is host-owned; only these two scalar values cross back.
  auto metadata = TPack<int64_t, 1, Device::CPU>::empty({2});
  metadata.view[0] = capacity;
  metadata.view[1] = compact;
  return {
      selected.tensor,
      types.tensor,
      offsets.tensor,
      source.tensor,
      target.tensor,
      metadata.tensor};
}
}  // namespace tmol::pack::rotamer
