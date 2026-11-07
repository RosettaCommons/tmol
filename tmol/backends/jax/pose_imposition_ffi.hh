// Caller-owned selection, layout and gather indices from the shared packer.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/rotamer/pose_imposition.hh>

namespace tmol::jax {
struct PoseImpositionStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    constexpr int inputs[] = {6, 8, 5, 3, 5};
    constexpr int outputs[] = {1, 2, 4, 3, 2};
    if (op < 0 || op > 4 || a.size() != inputs[op] || out.size() != outputs[op]
        || extent < 0 || extent > INT32_MAX || width < 0 || width > 1)
      throw std::invalid_argument("invalid pose imposition stage");
    auto i = ffi::DataType::S32, u = ffi::DataType::U8;
    auto wide = [&](ffi::AnyBuffer x, std::initializer_list<int64_t> shape) {
      require_shape(x, u, shape);
      if (reinterpret_cast<uintptr_t>(x.untyped_data()) % alignof(int64_t))
        throw std::invalid_argument("unaligned pose imposition metadata");
    };
    if (op < 3 || op == 4) {
      wide(a[0], {-1, -1, 8});
      auto p = a[0].dimensions()[0], b = a[0].dimensions()[1];
      auto n = score::common::checked_dispatch_product(
          p, b, "pose imposition blocks");
      if (op < 2) {
        wide(a[1], {p, b, 8});
        graph_wide(a[2], p);
        wide(a[3], {p, -1, 8});
        graph_wide(a[4]);
        require_shape(a[5], i, {p, a[3].dimensions()[1]});
        if (op == 0)
          graph_wide(out[0], n);
        else {
          graph_wide(a[6], n);
          graph_wide(a[7], n);
          wide(out[0], {p, b, 8});
          require_shape(out[1], i, {p, b});
        }
      } else if (op == 2) {
        wide(a[1], {p, b, 8});
        graph_wide(a[2]);
        require_shape(a[3], i, {-1});
        require_shape(a[4], i, {a[2].dimensions()[0]});
        wide(out[0], {p, b, 8});
        graph_wide(out[1], n);
        require_shape(out[2], i, {n});
        require_shape(out[3], i, {p, b});
      } else {
        require_shape(a[1], i, {-1});
        graph_wide(a[2], n);
        require_shape(a[3], i, {-1});
        require_shape(a[4], i, {p, b});
        auto atoms = a[3].dimensions()[0];
        score::common::checked_dispatch_size(atoms, "pose imposition atoms");
        graph_wide(out[0], atoms);
        graph_wide(out[1], atoms);
      }
    } else {
      require_shape(a[2], i, {-1, -1});
      auto p = a[2].dimensions()[0], b = a[2].dimensions()[1];
      auto n = score::common::checked_dispatch_product(
          p, b, "pose imposition blocks");
      graph_wide(a[0], n);
      graph_wide(a[1], n);
      require_shape(out[0], i, {p, b});
      graph_wide(out[1], p);
      require_shape(out[2], i, {p, b});
    }
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    using namespace pack::rotamer;
    ContextManager manager;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    if (op < 2) {
      PoseImpositionSelection<D> ctx{
          IN(int64_t, 2, 0),
          IN(int64_t, 2, 1),
          IN(int64_t, 2, 3),
          IN(int64_t, 1, 2),
          IN(int64_t, 1, 4),
          IN(int32_t, 2, 5)};
      if (op == 0)
        imposition_molten_flags_native<Ops, D>(
            manager, ctx, OUT(int64_t, 1, 0));
      else
        imposition_select_native<Ops, D>(
            manager,
            ctx,
            IN(int64_t, 1, 6),
            IN(int64_t, 1, 7),
            OUT(int64_t, 2, 0),
            OUT(int32_t, 2, 1));
    } else if (op == 2) {
      imposition_types_native<Ops, D>(
          manager,
          IN(int64_t, 2, 0),
          IN(int64_t, 2, 1),
          IN(int64_t, 1, 2),
          IN(int32_t, 1, 3),
          IN(int32_t, 1, 4),
          extent,
          OUT(int64_t, 2, 0),
          OUT(int64_t, 1, 1),
          OUT(int32_t, 1, 2),
          OUT(int32_t, 2, 3));
    } else if (op == 3) {
      imposition_layout_native<Ops, D>(
          manager,
          IN(int64_t, 1, 0),
          IN(int64_t, 1, 1),
          IN(int32_t, 2, 2),
          bool(width),
          extent,
          OUT(int32_t, 2, 0),
          OUT(int64_t, 1, 1),
          OUT(int32_t, 2, 2));
    } else {
      imposition_atom_indices_native<Ops, D>(
          manager,
          IN(int64_t, 2, 0),
          IN(int32_t, 1, 1),
          IN(int64_t, 1, 2),
          IN(int32_t, 1, 3),
          IN(int32_t, 2, 4),
          extent,
          OUT(int64_t, 1, 0),
          OUT(int64_t, 1, 1));
    }
#undef IN
#undef OUT
  }
};
}  // namespace tmol::jax
