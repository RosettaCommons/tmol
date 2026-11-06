// Typed execution adapters for common canonical input and reconstruction.
#pragma once
#include "graph_topology_ffi.hh"
#include "io_policy_ffi.hh"
#include "io_hydrogen_ffi.hh"
#include <tmol/io/details/compiled/leaf_atom_kernels.hh>
#include <tmol/io/details/compiled/canonical_candidates.hh>
#include <tmol/io/details/compiled/canonical_topology.hh>
#include <tmol/io/details/compiled/histidine_kernels.hh>

namespace tmol::jax {
struct LeafAtomStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if ((op != 0 && op != 1) || a.size() != (op ? 13 : 12) || out.size() != 1)
      throw std::invalid_argument("invalid leaf atom execution stage");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument(
          "leaf atom coordinates require float32 or float64");
    auto i = ffi::DataType::S32, b = ffi::DataType::PRED;
    require_shape(a[0], real, {-1, -1, 3});
    auto poses = a[0].dimensions()[0], atoms = a[0].dimensions()[1];
    require_shape(a[1], b, {poses, -1, -1});
    auto blocks = a[1].dimensions()[1], width = a[1].dimensions()[2];
    require_shape(a[2], b, {poses, atoms});
    require_shape(a[3], i, {poses, blocks});
    require_shape(a[4], i, {poses, blocks});
    require_shape(a[5], i, {poses, blocks, -1, 2});
    auto ports = a[5].dimensions()[2];
    require_shape(a[6], i, {-1});
    auto types = a[6].dimensions()[0];
    require_shape(a[7], i, {types, ports, width});
    require_shape(a[8], i, {types, width, 3, 3});
    require_shape(a[9], real, {types, width, 3});
    require_shape(a[10], i, {types, width, 3, 3});
    require_shape(a[11], real, {types, width, 3});
    require_shape(out[0], real, {poses, atoms, 3});
    if (op) require_shape(a[12], real, {poses, atoms, 3});
  }
  template <template <Device> class Ops, Device D, typename Real, bool Backward>
  static void launch(const std::vector<ffi::AnyBuffer>& a, ffi::AnyBuffer out) {
    using V = portable::Vec<Real, 3>;
    using C = portable::Vec<int32_t, 2>;
    using U = UnresolvedAtomID<int32_t>;
    io::details::compiled::LeafAtomContext<Real, int32_t, D> c{
        buffer_view<V, 2, D>(a[0]),
        buffer_view<bool, 3, D>(a[1]),
        buffer_view<bool, 2, D>(a[2]),
        buffer_view<int32_t, 2, D>(a[3]),
        buffer_view<int32_t, 2, D>(a[4]),
        buffer_view<C, 3, D>(a[5]),
        buffer_view<int32_t, 1, D>(a[6]),
        buffer_view<int32_t, 3, D>(a[7]),
        buffer_view<U, 3, D>(a[8]),
        buffer_view<Real, 3, D>(a[9]),
        buffer_view<U, 3, D>(a[10]),
        buffer_view<Real, 3, D>(a[11])};
    ContextManager manager;
    io::details::compiled::
        launch_pose_leaf_atoms<Backward, Ops, D, Real, int32_t>(
            manager,
            c,
            buffer_view<V, 2, D>(Backward ? a[12] : a[0]),
            buffer_view<V, 2, D>(out));
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if (a[0].element_type() == ffi::DataType::F32) {
      if (op)
        launch<Ops, D, float, true>(a, out[0]);
      else
        launch<Ops, D, float, false>(a, out[0]);
    } else {
      if (op)
        launch<Ops, D, double, true>(a, out[0]);
      else
        launch<Ops, D, double, false>(a, out[0]);
    }
  }
};
struct CanonicalCandidateStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if (op != 0 || a.size() != 10 || out.size() != 3)
      throw std::invalid_argument("invalid canonical candidate arguments");
    auto b = ffi::DataType::PRED, i = ffi::DataType::S32;
    require_shape(a[0], b, {-1, -1, -1});
    auto p = a[0].dimensions()[0], r = a[0].dimensions()[1],
         n = a[0].dimensions()[2];
    require_shape(a[1], b, {p, r});
    for (int k = 2; k < 5; ++k) require_shape(a[k], i, {p, r});
    require_shape(a[5], i, {-1, 4, -1, -1});
    require_shape(
        a[6],
        b,
        {a[5].dimensions()[0], 4, a[5].dimensions()[2], a[5].dimensions()[3]});
    require_shape(a[7], b, {-1, n});
    auto types = a[7].dimensions()[0];
    require_shape(a[8], b, {types, n});
    require_shape(a[9], b, {types});
    require_shape(out[0], i, {p, r});
    require_shape(out[1], ffi::DataType::U8, {p, r, 8});
    require_shape(out[2], i, {p, r});
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t,
      int64_t,
      int64_t) {
    io::details::compiled::CanonicalCandidateContext<int32_t, D> c{
        buffer_view<bool, 3, D>(a[0]),
        buffer_view<bool, 2, D>(a[1]),
        buffer_view<int32_t, 2, D>(a[2]),
        buffer_view<int32_t, 2, D>(a[3]),
        buffer_view<int32_t, 2, D>(a[4]),
        buffer_view<int32_t, 4, D>(a[5]),
        buffer_view<bool, 4, D>(a[6]),
        buffer_view<bool, 2, D>(a[7]),
        buffer_view<bool, 2, D>(a[8]),
        buffer_view<bool, 1, D>(a[9]),
        buffer_view<int32_t, 2, D>(out[0]),
        buffer_view<int64_t, 2, D>(out[1]),
        buffer_view<int32_t, 2, D>(out[2])};
    ContextManager manager;
    io::details::compiled::launch_canonical_candidates<Ops, D, int32_t>(
        manager, c);
  }
};
struct CanonicalTopologyStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if ((op != 0 && op != 1) || a.size() != (op ? 9 : 8)
        || out.size() != (op ? 2 : 4))
      throw std::invalid_argument("invalid canonical topology arguments");
    auto i = ffi::DataType::S32, b = ffi::DataType::PRED;
    require_shape(a[1], i, {-1, -1});
    auto p = a[1].dimensions()[0], r = a[1].dimensions()[1];
    require_shape(out[0], i, {p, r});
    if (!op) {
      require_shape(a[0], i, {p, r});
      require_shape(a[2], b, {p, r, 2});
      require_shape(a[3], b, {p, r});
      require_shape(a[4], b, {-1});
      require_shape(a[5], i, {a[4].dimensions()[0], 2});
      require_shape(a[6], i, {-1, 3});
      require_shape(a[7], b, {p, r, 2});
      for (int k = 1; k < 4; ++k) require_shape(out[k], b, {p, r});
    } else {
      require_shape(a[0], b, {p, r, -1});
      require_shape(a[2], i, {p, r});
      require_shape(a[3], i, {p, r});
      require_shape(a[4], b, {p, r, 2});
      require_shape(a[5], b, {p, r});
      require_shape(a[6], i, {-1, 4, -1, -1});
      require_shape(
          a[7],
          b,
          {a[6].dimensions()[0],
           4,
           a[6].dimensions()[2],
           a[6].dimensions()[3]});
      require_shape(a[8], b, {-1, a[0].dimensions()[2]});
      require_shape(out[1], i, {p, r, 2, a[6].dimensions()[3]});
    }
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    ContextManager manager;
    if (!op) {
      io::details::compiled::CanonicalChainContext<int32_t, D> c{
          buffer_view<int32_t, 2, D>(a[0]),
          buffer_view<int32_t, 2, D>(a[1]),
          buffer_view<bool, 3, D>(a[2]),
          buffer_view<bool, 2, D>(a[3]),
          buffer_view<bool, 1, D>(a[4]),
          buffer_view<int32_t, 2, D>(a[5]),
          buffer_view<int32_t, 2, D>(a[6]),
          buffer_view<bool, 3, D>(a[7]),
          buffer_view<int32_t, 2, D>(out[0]),
          buffer_view<bool, 2, D>(out[1]),
          buffer_view<bool, 2, D>(out[2]),
          buffer_view<bool, 2, D>(out[3])};
      io::details::compiled::launch_canonical_chain_endings<Ops, D, int32_t>(
          manager, c);
    } else {
      io::details::compiled::CanonicalTerminusContext<int32_t, D> c{
          buffer_view<bool, 3, D>(a[0]),
          buffer_view<int32_t, 2, D>(a[1]),
          buffer_view<int32_t, 2, D>(a[2]),
          buffer_view<int32_t, 2, D>(a[3]),
          buffer_view<bool, 3, D>(a[4]),
          buffer_view<bool, 2, D>(a[5]),
          buffer_view<int32_t, 4, D>(a[6]),
          buffer_view<bool, 4, D>(a[7]),
          buffer_view<bool, 2, D>(a[8]),
          buffer_view<int32_t, 4, D>(out[1]),
          buffer_view<int32_t, 2, D>(out[0])};
      io::details::compiled::launch_canonical_atom_termini<Ops, D, int32_t>(
          manager, c);
    }
  }
};
struct HistidineStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t,
      int64_t) {
    if (op || a.size() != 7 || out.size() != 3)
      throw std::invalid_argument("invalid histidine resolution arguments");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument(
          "histidine coordinates require float32 or float64");
    require_shape(a[0], real, {-1, -1, -1, 3});
    auto p = a[0].dimensions()[0], r = a[0].dimensions()[1],
         n = a[0].dimensions()[2];
    auto i = ffi::DataType::S32, b = ffi::DataType::PRED, w = ffi::DataType::U8;
    require_shape(a[1], i, {p, r});
    require_shape(a[2], i, {p, r});
    graph_wide(a[3]);
    graph_wide(a[4], a[3].dimensions()[0]);
    require_shape(a[5], b, {p, r, n});
    require_shape(a[6], i, {1, 8});
    require_shape(out[0], i, {p, r});
    require_shape(out[1], w, {p, r, n, 8});
    require_shape(out[2], w, {p, r, 8});
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out) {
    using V = portable::Vec<Real, 3>;
    using H = io::details::compiled::HisAtomIndsInCanonicalOrdering<int32_t>;
    io::details::compiled::HisTautContext<Real, int32_t, D> c{
        buffer_view<V, 3, D>(a[0]),
        buffer_view<int32_t, 2, D>(a[1]),
        buffer_view<int32_t, 2, D>(out[0]),
        buffer_view<int64_t, 1, D>(a[3]),
        buffer_view<int64_t, 1, D>(a[4]),
        buffer_view<bool, 3, D>(a[5]),
        buffer_view<H, 1, D>(a[6]),
        buffer_view<int64_t, 3, D>(out[1]),
        buffer_view<int64_t, 2, D>(out[2])};
    auto source = buffer_view<int32_t, 2, D>(a[2]);
    int rows = score::common::checked_dispatch_product(
        c.coords.size(0), c.coords.size(1), "histidine rows");
    int atoms = score::common::checked_dispatch_product(
        rows, c.coords.size(2), "histidine remapping");
    auto width = c.coords.size(1), na = c.coords.size(2);
    ContextManager manager;
    LAUNCH_BOX_128;
    Ops<D>::template forall<launch_t>(
        manager, std::max(rows, atoms), [=] TMOL_DEVICE_FUNC(int index) {
          if (index < rows)
            c.res_type_variants[index / width][index % width] =
                source[index / width][index % width];
          if (index < atoms)
            c.his_remapping_dst_index[index / (width * na)]
                                     [(index / na) % width][index % na] =
                index % na;
        });
    io::details::compiled::launch_resolve_his_taut<Ops, D, Real, int32_t>(
        manager, c);
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t,
      int64_t,
      int64_t) {
    if (a[0].element_type() == ffi::DataType::F32)
      launch<Ops, D, float>(a, out);
    else
      launch<Ops, D, double>(a, out);
  }
};
}  // namespace tmol::jax
