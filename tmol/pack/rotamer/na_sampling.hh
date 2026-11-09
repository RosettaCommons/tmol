#pragma once
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol::pack::rotamer {
// Each selected physical sugar is measured once. Selection and output storage
// belong to the frontend; the pucker physics is also used by NA scoring.
template <template <tmol::Device> class Ops, tmol::Device D, typename Real>
void na_sampling_pucker_native(
    ContextManager& mgr,
    TView<Real, 2, D> coords,
    TView<int64_t, 2, D> rings,
    TView<bool, 2, D> valid,
    Real temperature,
    TView<int64_t, 1, D> result) {
  LAUNCH_BOX_128;
  auto measure = [=] TMOL_DEVICE_FUNC(int row) {
    bool all = true;
    for (int a = 0; a < 5; ++a) all = all && valid[row][a];
    if (!all) {
      result[row] = -1;
      return;
    }
    Real weights[10];
    tmol::score::na_torsion::potentials::pucker_weights(
        coords.data(), rings.data(), valid.data(), row, temperature, weights);
    int best = 0;
    for (int p = 1; p < 10; ++p)
      if (weights[p] > weights[best]) best = p;
    result[row] = best;
  };
  Ops<D>::template forall<launch_t>(mgr, result.size(0), measure);
}

struct NaChiSamplingPolicy {
  bool sample_syn;
  int bases_per_polymer;
  float max_syn_well, syn_mean;
};
template <tmol::Device D>
struct NaChiSamplingContext {
  TView<int64_t, 1, D> types, base, pucker, n_combos;
  TView<bool, 1, D> active;
  TView<int32_t, 2, D> chi_atom;
  TView<float, 3, D> proton, well_syn;
  TView<float, 2, D> means;
  TView<float, 1, D> sdev, steps;
  NaChiSamplingPolicy policy;
};

template <template <tmol::Device> class Ops, tmol::Device D>
void na_chi_count_native(
    ContextManager& mgr,
    NaChiSamplingContext<D> ctx,
    TView<int64_t, 1, D> counts) {
  LAUNCH_BOX_128;
  auto count = [ctx, counts] TMOL_DEVICE_FUNC(int row) {
    auto const type = ctx.types[row];
    auto const base = ctx.base[type], pucker = ctx.pucker[row];
    if (!ctx.active[row] || pucker < 0 || base < 0) {
      counts[row] = 0;
      return;
    }
    int const polymer = base / ctx.policy.bases_per_polymer;
    bool const syn =
        ctx.policy.sample_syn && polymer == 1
        && ctx.well_syn[1][pucker][base] <= ctx.policy.max_syn_well;
    // Preserve invalid-count sentinels while preventing int64 multiplication
    // overflow. The allocation barrier validates both counts and total before
    // narrowing to the public int32 conformer index space.
    int64_t combo = ctx.n_combos[type];
    if (combo < -1) combo = -1;
    if (combo > int64_t(INT32_MAX) + 1) combo = int64_t(INT32_MAX) + 1;
    counts[row] = (syn ? 2 : 1) * ctx.steps.size(0) * combo;
  };
  Ops<D>::template forall<launch_t>(mgr, counts.size(0), count);
}

template <template <tmol::Device> class Ops, tmol::Device D>
void na_chi_fill_native(
    ContextManager& mgr,
    NaChiSamplingContext<D> ctx,
    TView<int64_t, 1, D> offsets,
    TView<int32_t, 1, D> owners,
    TView<int32_t, 2, D> chi_atom,
    TView<float, 2, D> chi) {
  LAUNCH_BOX_128;
  auto fill = [ctx, offsets, owners, chi_atom, chi] TMOL_DEVICE_FUNC(int row) {
    auto const owner = owners[row];
    auto const type = ctx.types[owner];
    int64_t const local = int64_t(row) - offsets[owner];
    int64_t const combo = local % ctx.n_combos[type];
    int64_t const slot = local / ctx.n_combos[type];
    auto const base = ctx.base[type], pucker = ctx.pucker[owner];
    int const polymer = base / ctx.policy.bases_per_polymer;
    float const center = slot / ctx.steps.size(0) == 0 ? ctx.means[base][pucker]
                                                       : ctx.policy.syn_mean;
    float const degrees =
        center + ctx.steps[slot % ctx.steps.size(0)] * ctx.sdev[polymer];
    // Torch's public sampler reports float32 radians even for double coords.
    chi[row][0] = degrees * float(0.01745329251994329576923690768489);
    for (int c = 0; c < chi.size(1); ++c) {
      chi_atom[row][c] = ctx.chi_atom[type][c];
      if (c) chi[row][c] = ctx.proton[type][combo][c - 1];
    }
  };
  Ops<D>::template forall<launch_t>(mgr, owners.size(0), fill);
}
}  // namespace tmol::pack::rotamer
