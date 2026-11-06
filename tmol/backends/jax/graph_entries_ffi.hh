// Typed entry stages for the canonical streaming interaction graph.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/compiled/streaming_interaction_graph_kernels.hh>

namespace tmol::jax {
struct GraphEntryStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    if (op < 0 || op > 2
        || a.size()
               != (op == 0   ? 4
                   : op == 1 ? 8
                             : 12)
        || out.size() != (op == 2 ? 3 : 1) || extent < 0 || extent > INT32_MAX
        || width < 1 || width > INT32_MAX)
      throw std::invalid_argument("invalid interaction graph entry stage");
    auto i = ffi::DataType::S32, u = ffi::DataType::U8;
    require_shape(a[0], i, {3, -1});
    score::common::checked_dispatch_size(a[0].dimensions()[1], "graph entries");
    int shift = op == 2;
    require_shape(a[1 + shift], i, {-1});
    require_shape(a[2 + shift], u, {-1, -1, 8});
    auto p = a[2 + shift].dimensions()[0], b = a[2 + shift].dimensions()[1];
    if (!op) {
      require_shape(a[3], i, {p, -1});
      auto m = a[3].dimensions()[1];
      require_shape(out[0], i, {p, m, (m + 31) / 32});
      return;
    }
    require_shape(a[3 + shift], i, {p, b});
    int counts = op == 1 ? 4 : 6;
    require_shape(a[counts], i, {p, -1});
    auto m = a[counts].dimensions()[1];
    int csr = op == 1 ? 5 : 8;
    graph_wide(a[csr], p * m + 1);
    require_shape(a[csr + 1], i, {-1});
    graph_wide(a[csr + 2], a[csr + 1].dimensions()[0]);
    if (op == 1) {
      require_shape(out[0], i, {-1});
    } else {
      auto real = a[1].element_type();
      if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
        throw std::invalid_argument("graph energies require float32/64");
      require_shape(a[1], real, {a[0].dimensions()[1]});
      require_shape(a[5], u, {p, b, 8});
      require_shape(a[7], i, {p, m});
      graph_wide(a[11]);
      require_shape(out[0], real, {p});
      require_shape(out[1], real, {extent});
      require_shape(out[2], real, {-1});
    }
  }

  template <template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t width) {
    using namespace pack::compiled;
    InteractionGraphEntryCtxT<
        score::common::TViewBinder<D>::template View,
        Real,
        int32_t>
        c{};
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    int shift = op == 2;
    c.sparse_inds = IN(int32_t, 2, 0);
    c.block_ind_for_rot = IN(int32_t, 1, 1 + shift);
    c.orig_block_to_molten = IN(int64_t, 2, 2 + shift);
    c.n_bc_rots_for_molten_block =
        IN(int32_t,
           2,
           op == 0   ? 3
           : op == 1 ? 4
                     : 6);
    c.max_n_molten_blocks = c.n_bc_rots_for_molten_block.size(1);
    c.chunk_size = width;
    if (!op)
      c.block_adjacency = OUT(int32_t, 3, 0);
    else {
      c.rot_offset_for_block = IN(int32_t, 2, 3 + shift);
      int csr = op == 1 ? 5 : 8;
      c.neighbor_row_offsets = IN(int64_t, 1, csr);
      c.neighbor_blocks = IN(int32_t, 1, csr + 1);
      if (op == 1) {
        c.neighbor_chunk_bitset_offsets = IN(int64_t, 1, 7);
        c.chunk_adjacency = OUT(int32_t, 1, 0);
      } else {
        c.sparse_energies = IN(Real, 1, 1);
        c.rotamer_for_nonmolten_block = IN(int64_t, 2, 5);
        c.bc_rot_offset_for_molten_block = IN(int32_t, 2, 7);
        c.neighbor_chunk_offset_offsets = IN(int64_t, 1, 10);
        c.chunk_offsets = IN(int64_t, 1, 11);
        c.bg_bg_energies = OUT(Real, 1, 0);
        c.energy1b = OUT(Real, 1, 1);
        c.energy2b = OUT(Real, 1, 2);
      }
    }
    auto entry = [=] TMOL_DEVICE_FUNC(int index) {
      if (!op)
        ig_note_block_pair<Ops, D>(c, index);
      else if (op == 1)
        ig_note_chunk_pair<Ops, D>(c, index);
      else
        ig_accumulate_entry<D, Real>(c, index);
    };
    int n = c.sparse_inds.size(1);
    // CPU accumulation is deliberately serial: duplicate entries can target
    // the same output. CUDA's canonical accumulator supplies atomic addition.
    if constexpr (D == Device::CPU) {
      for (int row = 0; row < n; ++row) entry(row);
    } else {
      ContextManager manager;
      LAUNCH_BOX_32;
      Ops<D>::template forall<launch_t>(manager, n, entry);
    }
#undef IN
#undef OUT
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t width) {
    if (op == 2 && a[1].element_type() == ffi::DataType::F64)
      launch<Ops, D, double>(a, out, op, width);
    else
      launch<Ops, D, float>(a, out, op, width);
  }
};
}  // namespace tmol::jax
