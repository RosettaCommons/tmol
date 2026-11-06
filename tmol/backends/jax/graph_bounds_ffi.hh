// Caller-owned storage for the shared preliminary graph and bump bounds.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/compiled/interaction_graph_bounds.hh>

namespace tmol::jax {
// The shared bound producer serializes duplicate accumulation on CPU.
// Other FFI stages use forall for disjoint elementwise work, so retain this
// operation's execution contract without changing their scheduling policy.
template <template <Device> class Parent>
struct BumpOperations {
  template <Device D>
  struct Type : Parent<D> {
    template <typename Launch, typename F>
    static void forall(ContextManager& mgr, int count, F fn) {
      if constexpr (D == Device::CPU) {
        for (int i = 0; i < count; ++i) fn(i);
      } else
        Parent<D>::template forall<Launch>(mgr, count, fn);
    }
    template <typename Launch, typename F>
    static void forall_independent(ContextManager& mgr, int count, F fn) {
      forall<Launch>(mgr, count, fn);
    }
    template <typename Launch, typename F>
    static void foreach_grouped_workgroup(
        ContextManager& mgr, int groups, int size, F fn) {
      Parent<D>::template foreach_pose_workgroup<Launch>(mgr, groups, size, fn);
    }
    static TMOL_DEVICE_FUNC void store_idempotent(
        int32_t& target, int32_t value) {
#ifdef __CUDACC__
      atomicExch(&target, value);
#else
      target = value;
#endif
    }
    static TMOL_DEVICE_FUNC void store_idempotent(
        int64_t& target, int64_t value) {
#ifdef __CUDACC__
      atomicExch(
          reinterpret_cast<unsigned long long*>(&target),
          static_cast<unsigned long long>(value));
#else
      target = value;
#endif
    }
  };
};

struct GraphBoundsStage {
  static int fill(int64_t op, size_t out) {
    return op == 2 && (out == 1 || out == 4 || out == 5) ? -1 : 0;
  }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    if (op < 0 || op > 2
        || a.size()
               != (op == 0   ? 6
                   : op == 1 ? 7
                             : 11)
        || out.size()
               != (op == 0   ? 3
                   : op == 1 ? 1
                             : 6)
        || extent < 0)
      throw std::invalid_argument("invalid bump-bound stage operands");
    auto i = ffi::DataType::S32, u = ffi::DataType::U8;
    auto real = a[5].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("bump energies require float32/64");
    require_shape(a[0], i, {-1, -1});
    auto p = a[0].dimensions()[0], b = a[0].dimensions()[1];
    require_shape(a[1], i, {p, b});
    require_shape(a[2], i, {-1});
    auto r = a[2].dimensions()[0];
    require_shape(a[3], i, {r});
    require_shape(a[4], i, {3, -1});
    require_shape(a[5], real, {a[4].dimensions()[1]});
    if (width < 1 || width > INT32_MAX)
      throw std::invalid_argument("invalid bump chunk size");
    pack::compiled::ig_bump_workspace_sizes(
        p, b, r, a[4].dimensions()[1], width, 0);
    if (op == 0) {
      require_shape(out[0], i, {p, b});
      require_shape(out[1], i, {p, b, b});
      require_shape(out[2], u, {p, b, b, 8});
    } else if (op == 1) {
      require_shape(a[6], u, {p, b, b, 8});
      graph_wide(out[0], extent);
      score::common::checked_dispatch_size(extent, "bump chunk pairs");
    } else {
      require_shape(a[6], i, {p, b, b});
      require_shape(a[7], u, {p, b, b, 8});
      graph_wide(a[8]);
      auto chunks = a[8].dimensions()[0];
      score::common::checked_dispatch_size(chunks, "bump chunk pairs");
      graph_wide(a[9], chunks);
      require_shape(a[10], real, {p, b});
      require_shape(out[0], real, {r});
      require_shape(out[1], real, {p, b});
      require_shape(out[2], real, {r});
      require_shape(out[3], real, {extent});
      require_shape(out[4], u, {p, b, b, 8});
      graph_wide(out[5], chunks);
      if (out[1].untyped_data() != a[10].untyped_data()
          || out[4].untyped_data() != a[7].untyped_data()
          || out[5].untyped_data() != a[9].untyped_data())
        throw std::invalid_argument(
            "bump bounds require aliased initialized buffers");
    }
  }
  template <typename Real, template <Device> class Ops, Device D>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t width) {
    using namespace pack::compiled;
#define IN(T, R, K) buffer_view<T, R, D>(a[K])
#define OUT(T, R, K) buffer_view<T, R, D>(out[K])
    BumpGraphContext<Real, int32_t, D> data{
        int(width),
        0,
        IN(int32_t, 2, 0),
        IN(int32_t, 2, 1),
        IN(int32_t, 1, 2),
        IN(int32_t, 1, 3),
        IN(int32_t, 2, 4),
        IN(Real, 1, 5)};
    ContextManager manager;
    if (op == 0) {
      data.n_chunks_for_block = OUT(int32_t, 2, 0);
      data.respair_is_adjacent = OUT(int32_t, 3, 1);
      data.n_chunks_for_block_pair = OUT(int64_t, 3, 2);
      launch_ig_bump_block_counts<BumpOperations<Ops>::template Type, D>(
          manager, data);
    } else if (op == 1) {
      data.chunk_pair_offset_for_block_pair = IN(int64_t, 3, 6);
      data.chunk_pair_adjacency = OUT(int64_t, 1, 0);
      launch_ig_bump_chunk_sizes<BumpOperations<Ops>::template Type, D>(
          manager, data);
    } else {
      data.respair_is_adjacent = IN(int32_t, 3, 6);
      data.chunk_pair_offset_for_block_pair = OUT(int64_t, 3, 4);
      data.chunk_pair_adjacency = IN(int64_t, 1, 8);
      data.chunk_pair_offsets = OUT(int64_t, 1, 5);
      data.energy1b = OUT(Real, 1, 2);
      data.energy2b = OUT(Real, 1, 3);
      data.best_energy_for_rot = OUT(Real, 1, 0);
      data.best_energy_per_block = OUT(Real, 2, 1);
      launch_ig_bump_bounds<BumpOperations<Ops>::template Type, D>(
          manager, data);
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
    if (a[5].element_type() == ffi::DataType::F32)
      launch<float, Ops, D>(a, out, op, width);
    else
      launch<double, Ops, D>(a, out, op, width);
  }
};
}  // namespace tmol::jax
