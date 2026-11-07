#pragma once
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <cstdint>

namespace tmol::pack::rotamer {
// All buffers and prefix scans belong to the caller. Indices are pose-major;
// retained choices refer to the per-molten-block retained conformer lists.
template <Device D>
struct PoseImpositionSelection {
  TView<int64_t, 2, D> original_types, fixed_rotamers, retained_offsets;
  TView<int64_t, 1, D> molten_counts, retained_to_original;
  TView<int32_t, 2, D> choices;
};
template <template <Device> class Ops, Device D>
void imposition_molten_flags_native(
    ContextManager& mgr,
    PoseImpositionSelection<D> c,
    TView<int64_t, 1, D> flags) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    int b = i % c.original_types.size(1), p = i / c.original_types.size(1);
    flags[i] = c.original_types[p][b] >= 0 && c.fixed_rotamers[p][b] == -1;
  };
  Ops<D>::template forall<launch_t>(mgr, flags.size(0), f);
}
template <template <Device> class Ops, Device D>
void imposition_select_native(
    ContextManager& mgr,
    PoseImpositionSelection<D> c,
    TView<int64_t, 1, D> flags,
    TView<int64_t, 1, D> prefix,
    TView<int64_t, 2, D> selected,
    TView<int32_t, 2, D> errors) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    int const nb = c.original_types.size(1), p = i / nb, b = i % nb;
    int64_t const begin = p * nb, last = begin + nb - 1;
    int64_t const n_molten = prefix[last] + flags[last] - prefix[begin];
    int32_t error = n_molten != c.molten_counts[p];
    int64_t rot = c.fixed_rotamers[p][b];
    if (flags[i]) {
      int64_t m = prefix[i] - prefix[begin];
      if (m >= c.choices.size(1)) {
        error = 1;
        rot = -1;
      } else {
        int64_t choice = c.choices[p][m], offset = c.retained_offsets[p][m];
        // Check before addition or lookup, including negative sentinels.
        if (choice < 0 || offset < 0 || offset >= c.retained_to_original.size(0)
            || choice >= c.retained_to_original.size(0) - offset) {
          error = 1;
          rot = -1;
        } else
          rot = c.retained_to_original[offset + choice];
      }
    }
    if (c.original_types[p][b] < -1 || (c.original_types[p][b] >= 0 && rot < 0))
      error = 1;
    selected[p][b] = rot;
    errors[p][b] = error;
  };
  Ops<D>::template forall<launch_t>(mgr, flags.size(0), f);
}
template <template <Device> class Ops, Device D>
void imposition_types_native(
    ContextManager& mgr,
    TView<int64_t, 2, D> original,
    TView<int64_t, 2, D> selected,
    TView<int64_t, 1, D> rotamer_types,
    TView<int32_t, 1, D> type_natoms,
    TView<int32_t, 1, D> rotamer_offsets,
    int64_t n_coordinates,
    TView<int64_t, 2, D> types,
    TView<int64_t, 1, D> counts,
    TView<int32_t, 1, D> changed,
    TView<int32_t, 2, D> errors) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    int const b = i % original.size(1), p = i / original.size(1);
    int64_t type = -1, count = 0;
    if (original[p][b] >= 0) {
      int64_t rot = selected[p][b];
      if (rot < 0 || rot >= rotamer_types.size(0))
        errors[p][b] = 1;
      else {
        type = rotamer_types[rot];
        if (type < 0 || type >= type_natoms.size(0))
          errors[p][b] = 1;
        else {
          count = type_natoms[type];
          int64_t offset = rotamer_offsets[rot];
          if (count < 0 || offset < 0 || offset > n_coordinates
              || count > n_coordinates - offset)
            errors[p][b] = 1;
        }
      }
    }
    types[p][b] = type;
    counts[i] = errors[p][b] ? 0 : count;
    changed[i] = type != original[p][b];
  };
  Ops<D>::template forall<launch_t>(mgr, counts.size(0), f);
}
// A type change compacts every pose. Otherwise preserve the original offsets,
// including gaps and padding, and retain the original coordinate capacity.
template <template <Device> class Ops, Device D>
void imposition_layout_native(
    ContextManager& mgr,
    TView<int64_t, 1, D> counts,
    TView<int64_t, 1, D> prefix,
    TView<int32_t, 2, D> original_offsets,
    bool compact,
    int64_t original_capacity,
    TView<int32_t, 2, D> offsets,
    TView<int64_t, 1, D> totals,
    TView<int32_t, 2, D> errors) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    int const nb = offsets.size(1), p = i / nb, b = i % nb;
    int64_t const begin = p * nb, last = begin + nb - 1;
    int64_t offset =
        compact ? prefix[i] - prefix[begin] : original_offsets[p][b];
    if (b == 0) totals[p] = prefix[last] + counts[last] - prefix[begin];
    if (offset > INT32_MAX || (counts[i] && offset < 0)
        || (!compact && counts[i]
            && (offset > original_capacity
                || counts[i] > original_capacity - offset)))
      errors[p][b] = 1;
    offsets[p][b] = int32_t(offset);
  };
  Ops<D>::template forall<launch_t>(mgr, counts.size(0), f);
}
template <template <Device> class Ops, Device D>
void imposition_atom_indices_native(
    ContextManager& mgr,
    TView<int64_t, 2, D> selected,
    TView<int32_t, 1, D> rotamer_offsets,
    TView<int64_t, 1, D> prefix,
    TView<int32_t, 1, D> owners,
    TView<int32_t, 2, D> offsets,
    int64_t pose_capacity,
    TView<int64_t, 1, D> source,
    TView<int64_t, 1, D> target) {
  LAUNCH_BOX_128;
  auto f = [=] TMOL_DEVICE_FUNC(int i) {
    int64_t const owner = owners[i], atom = i - prefix[owner];
    int const p = owner / selected.size(1), b = owner % selected.size(1);
    source[i] = int64_t(rotamer_offsets[selected[p][b]]) + atom;
    target[i] = int64_t(p) * pose_capacity + offsets[p][b] + atom;
  };
  Ops<D>::template forall<launch_t>(mgr, owners.size(0), f);
}
}  // namespace tmol::pack::rotamer
