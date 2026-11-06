#pragma once
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/pack/rotamer/na_sampling.hh>
#include <tmol/pack/rotamer/na_sampling_torch.hh>
namespace tmol::pack::rotamer {
template <tmol::Device D>
NaChiSamplingContext<D> na_sampling_context(
    std::vector<tmol::native::Tensor> const& a,
    bool sample_syn,
    int64_t bases_per_polymer,
    double max_syn_well,
    double syn_mean) {
  TORCH_CHECK(a.size() == 11, "NA sampler expects eleven input buffers");
  TORCH_CHECK(
      bases_per_polymer > 0, "NA sampler polymer base count must be positive");
  NaChiSamplingContext<D> ctx{
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
      TCAST(a[10]),
      {sample_syn,
       int(bases_per_polymer),
       float(max_syn_well),
       float(syn_mean)}};
  auto const n = ctx.types.size(0), nt = ctx.base.size(0),
             nc = ctx.chi_atom.size(1);
  TORCH_CHECK(
      ctx.pucker.size(0) == n && ctx.active.size(0) == n,
      "NA candidate axes differ");
  TORCH_CHECK(
      ctx.n_combos.size(0) == nt && ctx.chi_atom.size(0) == nt
          && ctx.proton.size(0) == nt,
      "NA type axes differ");
  TORCH_CHECK(nc >= 1 && ctx.proton.size(2) == nc - 1, "NA chi axes differ");
  TORCH_CHECK(
      ctx.steps.size(0) > 0 && ctx.steps.size(0) <= 5,
      "NA chi level needs one to five steps");
  TORCH_CHECK(
      ctx.means.size(0) == 2 * bases_per_polymer && ctx.means.size(1) == 10
          && ctx.well_syn.size(0) == 2 && ctx.well_syn.size(1) == 10
          && ctx.well_syn.size(2) == ctx.means.size(0) && ctx.sdev.size(0) == 2,
      "NA sampler parameter shapes differ");
  return ctx;
}
template <template <tmol::Device> class Ops, tmol::Device D>
auto NaSamplingTorchDispatch<Ops, D>::pucker(
    ContextManager& mgr,
    Tensor coords,
    Tensor rings,
    Tensor valid,
    double temperature) -> Tensor {
  TORCH_CHECK(
      coords.dim() == 2 && coords.size(1) == 3,
      "NA pucker coordinates must have width three");
  TORCH_CHECK(
      rings.dim() == 2 && rings.size(1) == 5 && valid.sizes() == rings.sizes(),
      "NA pucker rings must have width five");
  TORCH_CHECK(
      coords.is_contiguous() && rings.is_contiguous() && valid.is_contiguous(),
      "NA pucker input buffers must be contiguous");
  TORCH_CHECK(temperature > 0, "NA pucker temperature must be positive");
  auto output = TPack<int64_t, 1, D>::empty({rings.size(0)});
  AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "na_sampling_pucker", ([&] {
                               na_sampling_pucker_native<Ops, D, scalar_t>(
                                   mgr,
                                   TCAST(coords),
                                   TCAST(rings),
                                   TCAST(valid),
                                   scalar_t(temperature),
                                   output.view);
                             }));
  return output.tensor;
}
template <template <tmol::Device> class Ops, tmol::Device D>
auto NaSamplingTorchDispatch<Ops, D>::counts(
    ContextManager& mgr,
    std::vector<Tensor> inputs,
    bool syn,
    int64_t nbase,
    double maxwell,
    double mean) -> Tensor {
  auto ctx = na_sampling_context<D>(inputs, syn, nbase, maxwell, mean);
  int n = tmol::score::common::checked_dispatch_size(
      ctx.types.size(0), "NA candidate count");
  auto out = TPack<int64_t, 1, D>::empty({n});
  na_chi_count_native<Ops, D>(mgr, ctx, out.view);
  return out.tensor;
}
template <template <tmol::Device> class Ops, tmol::Device D>
auto NaSamplingTorchDispatch<Ops, D>::fill(
    ContextManager& mgr,
    std::vector<Tensor> inputs,
    Tensor counts_tensor,
    int64_t n_rotamers,
    bool syn,
    int64_t nbase,
    double maxwell,
    double mean) -> std::tuple<Tensor, Tensor, Tensor, Tensor> {
  auto ctx = na_sampling_context<D>(inputs, syn, nbase, maxwell, mean);
  int total = tmol::score::common::checked_dispatch_size(
      n_rotamers, "NA conformer count");
  int n = tmol::score::common::checked_dispatch_size(
      ctx.types.size(0), "NA candidate count");
  TView<int64_t, 1, D> counts = TCAST(counts_tensor);
  TORCH_CHECK(counts.size(0) == n, "NA count axis differs");
  auto offsets = TPack<int64_t, 1, D>::zeros({n});
  if (n) {
    auto actual = Ops<D>::template scan_and_return_total<
        tmol::portable::scan_type::exclusive>(
        mgr,
        counts.data(),
        offsets.view.data(),
        n,
        tmol::portable::plus_t<int64_t>());
    TORCH_CHECK(
        actual == total, "NA allocation count differs from sample counts");
  } else
    TORCH_CHECK(total == 0, "Empty NA candidates require zero conformers");
  LAUNCH_BOX_128;
  auto owners =
      Ops<D>::template load_balancing_search<launch_t, int64_t, int32_t>(
          mgr, total, offsets.view.data(), n);
  auto atoms = TPack<int32_t, 2, D>::empty({total, ctx.chi_atom.size(1)});
  auto chi = TPack<float, 2, D>::empty({total, ctx.chi_atom.size(1)});
  na_chi_fill_native<Ops, D>(
      mgr, ctx, offsets.view, owners.view, atoms.view, chi.view);
  return {owners.tensor, atoms.tensor, chi.tensor, offsets.tensor};
}
}  // namespace tmol::pack::rotamer
