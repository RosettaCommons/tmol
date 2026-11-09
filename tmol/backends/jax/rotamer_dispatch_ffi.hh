// Sparse conformer spheres and candidate enumeration with caller-owned arrays.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/score/common/rotamer_dispatch.hh>
#include <tmol/score/common/tview_binder.hh>

namespace tmol::jax {
struct RotamerDispatchStage {
  static int fill(int64_t, size_t) { return 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t begin,
      int64_t count,
      double reach) {
    constexpr size_t inputs[] = {4, 3, 6, 8};
    if (op < 0 || op >= 4 || a.size() != inputs[op]
        || out.size() != (op == 2 ? 2 : 1) || !std::isfinite(reach)
        || reach < 0)
      throw std::invalid_argument("invalid rotamer dispatch stage");
    auto real = a[0].element_type();
    if (real != ffi::DataType::F32 && real != ffi::DataType::F64)
      throw std::invalid_argument("rotamer spheres require float32/64");
    auto integer = ffi::DataType::S32;
    if (op == 0) {
      require_shape(a[0], real, {-1, 3});
      require_shape(a[1], integer, {-1});
      auto n = a[1].dimensions()[0];
      require_shape(a[2], integer, {n});
      require_shape(a[3], integer, {-1});
      require_shape(out[0], real, {n, 4});
    } else if (op == 1) {
      require_shape(a[0], real, {-1, 4});
      require_shape(a[1], integer, {-1, -1});
      auto p = a[1].dimensions()[0], m = a[1].dimensions()[1];
      require_shape(a[2], integer, {p, m});
      require_shape(out[0], real, {p, m, 4});
    } else {
      require_shape(a[0], real, {-1, -1, 4});
      auto p = a[0].dimensions()[0], m = a[0].dimensions()[1];
      for (int i : {1, 2, 3, 5}) require_shape(a[i], integer, {p, m});
      require_shape(a[4], real, {-1, 4});
      auto window = score::common::sphere_overlap::rotamer_pair_window(
          p, m, begin, count);
      if (op == 2) {
        require_shape(out[0], integer, {window.n_candidates});
        graph_wide(out[1], 1);
      } else {
        require_shape(a[6], integer, {window.n_candidates});
        require_shape(a[7], integer, {window.n_candidates});
        require_shape(out[0], integer, {3, -1});
      }
    }
  }
  template <template <Device> class Ops, Device D, typename Real>
  static void launch(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t begin,
      int64_t count,
      double reach) {
    using namespace score::common::sphere_overlap;
    using Binder = score::common::TViewBinder<D>;
    ContextManager context;
#define IN(T, R, I) buffer_view<T, R, D>(a[I])
#define OUT(T, R, I) buffer_view<T, R, D>(out[I])
    if (op == 0) {
      RotSpheresCtxT<Binder::template View, Real, int32_t> c;
      c.rot_coords = buffer_view<Vec<Real, 3>, 1, D>(a[0]);
      c.rot_coord_offset = IN(int32_t, 1, 1);
      c.block_type_ind_for_rot = IN(int32_t, 1, 2);
      c.block_type_n_atoms = IN(int32_t, 1, 3);
      c.rot_spheres = OUT(Real, 2, 0);
      launch_rotamer_spheres<Ops, D, Real, int32_t>(context, c);
    } else if (op == 1) {
      BlockSpheresFromRotCtxT<Binder::template View, Real, int32_t> c;
      c.rot_spheres = IN(Real, 2, 0);
      c.n_rots_for_block = IN(int32_t, 2, 1);
      c.rot_offset_for_block = IN(int32_t, 2, 2);
      c.block_spheres = OUT(Real, 3, 0);
      c.max_n_blocks = a[1].dimensions()[1];
      launch_blocks_from_rotamer_spheres<Ops, D, Real, int32_t>(context, c);
    } else {
      RotPairListCtxT<Binder::template View, Real, int32_t> c;
      c.block_spheres = IN(Real, 3, 0);
      c.pose_stack_block_type = IN(int32_t, 2, 1);
      c.n_rots_for_block = IN(int32_t, 2, 2);
      c.rot_offset_for_block = IN(int32_t, 2, 3);
      c.rot_spheres = IN(Real, 2, 4);
      c.lockstep_group_for_block = IN(int32_t, 2, 5);
      auto w = rotamer_pair_window(
          a[0].dimensions()[0], a[0].dimensions()[1], begin, count);
      c.max_n_blocks = w.max_n_blocks;
      c.block_pairs_per_pose = w.block_pairs_per_pose;
      c.candidate_begin = w.candidate_begin;
      c.reach = Real(reach);
      if (op == 2) {
        c.pair_counts = OUT(int32_t, 1, 0);
        c.pair_count_overflow = OUT(int64_t, 1, 1);
        launch_rotamer_pair_counts<Ops, D, Real, int32_t>(context, c);
      } else {
        c.pair_counts = IN(int32_t, 1, 6);
        c.pair_offsets = IN(int32_t, 1, 7);
        c.indices = OUT(int32_t, 2, 0);
        launch_rotamer_pair_fill<Ops, D, Real, int32_t>(context, c);
      }
    }
#undef IN
#undef OUT
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t begin,
      int64_t count,
      double reach) {
    if (a[0].element_type() == ffi::DataType::F32)
      launch<Ops, D, float>(a, out, op, begin, count, reach);
    else
      launch<Ops, D, double>(a, out, op, begin, count, reach);
  }
};
}  // namespace tmol::jax
