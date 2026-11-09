#pragma once
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
namespace tmol::pack::rotamer {
template <Device D>
struct OptHProtonContext {
  TView<bool, 1, D> has_proton;
  TView<int32_t, 2, D> atoms, sample_counts;
  TView<float, 3, D> samples;
};
template <Device D>
struct OptHSamplingContext {
  TView<int64_t, 1, D> types, poses, blocks;
  TView<int64_t, 2, D> original_types;
  TView<bool, 1, D> allowed;
  TView<bool, 2, D> sampler_allowed;
  TView<int32_t, 3, D> type_counts;
  TView<int32_t, 1, D> nhq_col, nhq_atom;
  TView<bool, 1, D> is_his;
  OptHProtonContext<D> proton;
  TView<float, 2, D> flip_values;
};
struct OptHChiValue {
  int32_t atom;
  float value;
  int64_t source;
};
// The first active proton chi varies fastest. Source indices expose the linear
// lookup for frontend VJP/JVPs, including learnable tables and measured flips.
template <Device D>
TMOL_DEVICE_FUNC OptHChiValue opth_proton_chi(
    OptHProtonContext<D> c, int64_t type, int col, int64_t& remaining) {
  if (!c.has_proton[type] || col >= c.sample_counts.size(1)) return {-1, 0, -1};
  int n = c.sample_counts[type][col];
  if (n <= 0) return {-1, 0, -1};
  int64_t slot = remaining % n;
  remaining /= n;
  return {
      c.atoms[type][col],
      c.samples[type][col][slot],
      (type * c.samples.size(1) + col) * c.samples.size(2) + slot};
}
template <template <Device> class Ops, Device D>
void opth_count_native(
    ContextManager& mgr,
    OptHSamplingContext<D> c,
    bool flip,
    TView<int64_t, 1, D> counts,
    TView<bool, 1, D> measure) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    int64_t p = c.poses[i], b = c.blocks[i], t = c.types[i];
    bool active = c.allowed[i] && c.sampler_allowed[p][b];
    counts[i] =
        active ? c.type_counts[flip ? 1 : 0][c.original_types[p][b]][t] : 0;
    measure[i] = flip && active && counts[i] > 0 && c.nhq_col[t] >= 0;
  };
  Ops<D>::template forall<launch_t>(mgr, counts.size(0), f);
}
template <template <Device> class Ops, Device D>
void opth_flip_counts_native(
    ContextManager& mgr,
    OptHSamplingContext<D> c,
    TView<int64_t, 1, D> counts,
    TView<int64_t, 1, D> flip_counts) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    flip_counts[i] = c.nhq_col[c.types[i]] >= 0 ? counts[i] : 0;
  };
  Ops<D>::template forall<launch_t>(mgr, counts.size(0), f);
}
template <template <Device> class Ops, Device D>
void opth_proton_samples_native(
    ContextManager& mgr,
    OptHProtonContext<D> c,
    TView<int64_t, 1, D> types,
    TView<int64_t, 1, D> offsets,
    TView<int32_t, 1, D> owners,
    TView<int32_t, 2, D> atoms,
    TView<float, 2, D> values,
    TView<int64_t, 2, D> sources) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int row) {
    int64_t choice = owners[row], type = types[choice],
            sample = row - offsets[choice];
    for (int col = 0; col < values.size(1); ++col) {
      auto v = opth_proton_chi(c, type, col, sample);
      atoms[row][col] = v.atom;
      values[row][col] = v.value;
      sources[row][col] = v.source;
    }
  };
  Ops<D>::template forall<launch_t>(mgr, owners.size(0), f);
}
template <template <Device> class Ops, Device D>
void opth_samples_native(
    ContextManager& mgr,
    OptHSamplingContext<D> c,
    TView<int64_t, 1, D> offsets,
    TView<int64_t, 1, D> flip_prefix,
    TView<int32_t, 1, D> owners,
    TView<int32_t, 2, D> atoms,
    TView<float, 2, D> values,
    TView<int64_t, 2, D> sources) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int row) {
    int64_t choice = owners[row], type = c.types[choice],
            local = row - offsets[choice];
    int64_t sample = local, p = c.poses[choice], b = c.blocks[choice];
    // Preserve the established global ordering of the native/flipped pairs,
    // also for custom tables mixing proton and NHQ annotations.
    bool flipped = (flip_prefix[choice] + local) % 2;
    bool tautomer = c.is_his[type] && type != c.original_types[p][b];
    int64_t proton_size = c.proton.samples.size(0) * c.proton.samples.size(1)
                          * c.proton.samples.size(2);
    for (int col = 0; col < values.size(1); ++col) {
      auto v = opth_proton_chi(c.proton, type, col, sample);
      if (col == c.nhq_col[type] && (flipped || tautomer)) {
        v.atom = c.nhq_atom[type];
        v.value = c.flip_values[p][b];
        if (flipped) v.value += float(3.14159265358979323846);
        v.source = proton_size + p * c.flip_values.size(1) + b;
      }
      atoms[row][col] = v.atom;
      values[row][col] = v.value;
      sources[row][col] = v.source;
    }
  };
  Ops<D>::template forall<launch_t>(mgr, owners.size(0), f);
}
}  // namespace tmol::pack::rotamer
