#pragma once
#include <tmol/pack/rotamer/opth_sampling.hh>
#include <tmol/pack/rotamer/opth_sampling_torch.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/score/common/counting.hh>
namespace tmol::pack::rotamer {
template <Device D>
OptHSamplingContext<D> opth_context(
    std::vector<tmol::native::Tensor> const& a) {
  TORCH_CHECK(a.size() == 15, "OptH sampling requires fifteen input buffers");
  OptHSamplingContext<D> c{
      TCAST(a[0]),
      TCAST(a[1]),
      TCAST(a[2]),
      TCAST(a[3]),
      TCAST(a[4]),
      TCAST(a[5]),
      TCAST(a[6]),
      TCAST(a[7]),
      TCAST(a[8]),
      TCAST(a[9]),
      {TCAST(a[10]), TCAST(a[11]), TCAST(a[12]), TCAST(a[13])},
      TCAST(a[14])};
  int64_t n = c.types.size(0), t = c.nhq_col.size(0),
          nc = c.proton.atoms.size(1);
  TORCH_CHECK(
      c.poses.size(0) == n && c.blocks.size(0) == n && c.allowed.size(0) == n,
      "OptH candidate axes differ");
  TORCH_CHECK(
      c.type_counts.size(0) == 2 && c.type_counts.size(1) == t
          && c.type_counts.size(2) == t && c.nhq_atom.size(0) == t
          && c.is_his.size(0) == t && c.proton.has_proton.size(0) == t
          && c.proton.atoms.size(0) == t && c.proton.sample_counts.size(0) == t
          && c.proton.sample_counts.size(1) == nc
          && c.proton.samples.size(0) == t && c.proton.samples.size(1) == nc,
      "OptH chemical axes differ");
  TORCH_CHECK(
      c.original_types.size(0) == c.sampler_allowed.size(0)
          && c.original_types.size(1) == c.sampler_allowed.size(1)
          && c.flip_values.size(0) == c.original_types.size(0)
          && c.flip_values.size(1) == c.original_types.size(1),
      "OptH pose axes differ");
  return c;
}
template <template <Device> class Ops, Device D>
auto OptHSamplingTorch<Ops, D>::counts(
    ContextManager& mgr, std::vector<Tensor> inputs, bool flip)
    -> std::tuple<Tensor, Tensor> {
  auto c = opth_context<D>(inputs);
  int n = score::common::checked_dispatch_size(
      c.types.size(0), "OptH candidate count");
  auto counts = TPack<int64_t, 1, D>::empty({n});
  auto measure = TPack<bool, 1, D>::empty({n});
  opth_count_native<Ops, D>(mgr, c, flip, counts.view, measure.view);
  return {counts.tensor, measure.tensor};
}
template <template <Device> class Ops, Device D>
auto OptHSamplingTorch<Ops, D>::fill(
    ContextManager& mgr,
    std::vector<Tensor> inputs,
    Tensor count_tensor,
    int64_t n_rotamers) -> std::tuple<Tensor, Tensor, Tensor, Tensor> {
  auto c = opth_context<D>(inputs);
  int n = score::common::checked_dispatch_size(
      c.types.size(0), "OptH candidate count");
  int nr =
      score::common::checked_dispatch_size(n_rotamers, "OptH conformer count");
  TView<int64_t, 1, D> counts = TCAST(count_tensor);
  TORCH_CHECK(counts.size(0) == n, "OptH count axis differs");
  auto offsets = TPack<int64_t, 1, D>::zeros({n}),
       flip_counts = TPack<int64_t, 1, D>::empty({n}),
       flip_prefix = TPack<int64_t, 1, D>::zeros({n});
  if (n) {
    auto actual =
        Ops<D>::template scan_and_return_total<portable::scan_type::exclusive>(
            mgr,
            counts.data(),
            offsets.view.data(),
            n,
            portable::plus_t<int64_t>());
    TORCH_CHECK(actual == nr, "OptH allocation differs from sample counts");
    opth_flip_counts_native<Ops, D>(mgr, c, counts, flip_counts.view);
    Ops<D>::template scan<portable::scan_type::exclusive>(
        mgr,
        flip_counts.view.data(),
        flip_prefix.view.data(),
        n,
        portable::plus_t<int64_t>());
  } else
    TORCH_CHECK(nr == 0, "Empty OptH candidates require zero conformers");
  LAUNCH_BOX_128;
  auto owners =
      Ops<D>::template load_balancing_search<launch_t, int64_t, int32_t>(
          mgr, nr, offsets.view.data(), n);
  int64_t nc = c.proton.samples.size(1);
  auto atoms = TPack<int32_t, 2, D>::empty({nr, nc});
  auto chi = TPack<float, 2, D>::empty({nr, nc});
  auto sources = TPack<int64_t, 2, D>::empty({nr, nc});
  opth_samples_native<Ops, D>(
      mgr,
      c,
      offsets.view,
      flip_prefix.view,
      owners.view,
      atoms.view,
      chi.view,
      sources.view);
  return {owners.tensor, atoms.tensor, chi.tensor, sources.tensor};
}
template <template <Device> class Ops, Device D>
auto OptHSamplingTorch<Ops, D>::proton(
    ContextManager& mgr, std::vector<Tensor> a, int64_t width)
    -> std::tuple<Tensor, Tensor, Tensor> {
  TORCH_CHECK(
      a.size() == 7 && width >= 0,
      "OptH proton sampling requires seven buffers and nonnegative width");
  TView<int64_t, 1, D> types = TCAST(a[0]), offsets = TCAST(a[1]);
  TView<int32_t, 1, D> owners = TCAST(a[2]);
  OptHProtonContext<D> c{TCAST(a[3]), TCAST(a[4]), TCAST(a[5]), TCAST(a[6])};
  int nr = score::common::checked_dispatch_size(
      owners.size(0), "OptH proton sample count");
  auto atoms = TPack<int32_t, 2, D>::empty({nr, width});
  auto values = TPack<float, 2, D>::empty({nr, width});
  auto sources = TPack<int64_t, 2, D>::empty({nr, width});
  opth_proton_samples_native<Ops, D>(
      mgr, c, types, offsets, owners, atoms.view, values.view, sources.view);
  return {atoms.tensor, values.tensor, sources.tensor};
}
}  // namespace tmol::pack::rotamer
