#pragma once
#include <limits>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_native.hh>
namespace tmol::score::na_torsion::potentials {
template <template <tmol::Device> class Ops, tmol::Device D>
std::tuple<tmol::native::Tensor, tmol::native::Tensor>
na_torsion_score_dispatch(
    ContextManager& mgr,
    tmol::native::Tensor coords,
    tmol::native::Tensor base,
    tmol::native::Tensor is_na,
    tmol::native::Tensor torsion_indices,
    tmol::native::Tensor torsion_ok,
    tmol::native::Tensor ring_indices,
    tmol::native::Tensor ring_ok,
    tmol::native::Tensor prev,
    tmol::native::Tensor backbone_means,
    tmol::native::Tensor backbone_sdev,
    tmol::native::Tensor sugar_means,
    tmol::native::Tensor chi_means,
    tmol::native::Tensor sdev_sugar,
    tmol::native::Tensor sdev_chi,
    tmol::native::Tensor well_pucker,
    tmol::native::Tensor well_alpha_gamma,
    tmol::native::Tensor well_bibii_pucker,
    tmol::native::Tensor well_alphanext_bibii,
    tmol::native::Tensor well_chi_syn,
    tmol::native::Tensor is_north,
    tmol::native::Tensor weight_bb,
    tmol::native::Tensor weight_chi,
    tmol::native::Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev,
    bool compute_derivs,
    tmol::native::Tensor entry_weights) {
  TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
  int n_bases = int(chi_means.size(0));
  TORCH_CHECK(
      n_bases > 0 && n_bases % 2 == 0,
      "chi tables must have equal DNA and RNA base counts");
  TORCH_CHECK(
      chi_means.size(1) == N_PUCKER && well_chi_syn.size(2) == n_bases,
      "inconsistent nucleic-acid base table shapes");
  TORCH_CHECK(base.dim() == 2, "NA base indices must be rank two");
  TORCH_CHECK(
      base.numel() <= std::numeric_limits<int>::max()
          && coords.size(0) <= std::numeric_limits<int>::max() / 3,
      "NA dispatch exceeds native integer bounds");
  int n_poses = int(base.size(0));
  int max_n_blocks = int(base.size(1));
  auto output = tmol::native::zeros({2, n_poses}, coords.options());
  bool const weighted = entry_weights.defined();
  TORCH_CHECK(!weighted || compute_derivs, "NA cotangents require derivatives");
  if (weighted) {
    TORCH_CHECK(
        entry_weights.device() == coords.device()
            && entry_weights.scalar_type() == coords.scalar_type(),
        "NA cotangent device and dtype must match coordinates");
    TORCH_CHECK(
        entry_weights.dim() == 2 && entry_weights.size(0) == 2
            && entry_weights.size(1) == base.numel(),
        "NA cotangents must have shape [2,n_entries]");
  }
  auto derivatives =
      weighted ? tmol::native::zeros({coords.size(0), 3}, coords.options())
      : compute_derivs
          ? tmol::native::zeros({2, coords.size(0), 3}, coords.options())
          : tmol::native::empty({0}, coords.options());

  AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "na_torsion_score", [&] {
    NATorsionContext<scalar_t> ctx{};
    ctx.coords = coords.const_data_ptr<scalar_t>();
    ctx.base = base.const_data_ptr<int64_t>();
    ctx.is_na = is_na.const_data_ptr<bool>();
    ctx.torsion_indices = torsion_indices.const_data_ptr<int64_t>();
    ctx.torsion_ok = torsion_ok.const_data_ptr<bool>();
    ctx.ring_indices = ring_indices.const_data_ptr<int64_t>();
    ctx.ring_ok = ring_ok.const_data_ptr<bool>();
    ctx.prev = prev.const_data_ptr<int64_t>();
    ctx.backbone_means = backbone_means.const_data_ptr<scalar_t>();
    ctx.backbone_sdev = backbone_sdev.const_data_ptr<scalar_t>();
    ctx.sugar_means = sugar_means.const_data_ptr<scalar_t>();
    ctx.chi_means = chi_means.const_data_ptr<scalar_t>();
    ctx.sdev_sugar = sdev_sugar.const_data_ptr<scalar_t>();
    ctx.sdev_chi = sdev_chi.const_data_ptr<scalar_t>();
    ctx.well_pucker = well_pucker.const_data_ptr<scalar_t>();
    ctx.well_alpha_gamma = well_alpha_gamma.const_data_ptr<scalar_t>();
    ctx.well_bibii_pucker = well_bibii_pucker.const_data_ptr<scalar_t>();
    ctx.well_alphanext_bibii = well_alphanext_bibii.const_data_ptr<scalar_t>();
    ctx.well_chi_syn = well_chi_syn.const_data_ptr<scalar_t>();
    ctx.is_north = is_north.const_data_ptr<bool>();
    ctx.weight_bb = weight_bb.const_data_ptr<scalar_t>();
    ctx.weight_chi = weight_chi.const_data_ptr<scalar_t>();
    ctx.weight_sugar = weight_sugar.const_data_ptr<scalar_t>();
    ctx.pucker_temperature = scalar_t(pucker_temperature);
    ctx.bin_blend_sdev = scalar_t(bin_blend_sdev);
    ctx.n_poses = n_poses;
    ctx.max_n_blocks = max_n_blocks;
    ctx.n_bases = n_bases;
    ctx.n_atoms = int(coords.size(0));
    ctx.derivatives =
        compute_derivs ? derivatives.data_ptr<scalar_t>() : nullptr;
    ctx.output = output.data_ptr<scalar_t>();
    ctx.entry_weights =
        weighted ? entry_weights.const_data_ptr<scalar_t>() : nullptr;
    ctx.weight_stride_score = weighted ? entry_weights.stride(0) : 0;
    ctx.weight_stride_entry = weighted ? entry_weights.stride(1) : 0;
    if (weighted)
      launch_na_torsion_native<true, true, Ops, D>(mgr, ctx);
    else if (compute_derivs)
      launch_na_torsion_native<true, false, Ops, D>(mgr, ctx);
    else
      launch_na_torsion_native<false, false, Ops, D>(mgr, ctx);
  });
  return {output, derivatives};
}
}  // namespace tmol::score::na_torsion::potentials
