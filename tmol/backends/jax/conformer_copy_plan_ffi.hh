// Caller-owned buffers for shared current-conformer and OptH DOF plans.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/rotamer/conformer_copy_plan.hh>
#include <tmol/score/common/tview_binder.hh>

namespace tmol::jax {
struct ConformerCopyPlanStage {
  static int fill(int64_t op, size_t) { return op == 2 ? 255 : 0; }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t flip,
      int64_t width) {
    if (op < 0 || op > 2
        || a.size()
               != (op == 0   ? 2
                   : op == 1 ? 8
                             : 6)
        || out.size() != (op == 0 ? 2 : 1) || width < 0
        || (flip != 0 && flip != 1))
      throw std::invalid_argument("invalid conformer copy plan stage");
    auto i = ffi::DataType::S32;
    if (op < 2) {
      require_shape(a[0], i, {-1, -1});
      require_shape(a[1], i, {-1});
      auto n = score::common::checked_dispatch_product(
          a[0].dimensions()[0], a[0].dimensions()[1], "source conformers");
      if (op == 0) {
        graph_wide(out[0], n);
        graph_wide(out[1], n);
      } else {
        for (int index = 2; index <= 6; ++index)
          require_shape(a[index], i, {-1});
        require_shape(a[4], i, {a[2].dimensions()[0]});
        require_shape(a[5], i, {a[2].dimensions()[0]});
        graph_wide(a[7], n);
        score::common::checked_dispatch_size(
            a[6].dimensions()[0], "copy segments");
        require_shape(out[0], ffi::DataType::U8, {3, a[6].dimensions()[0], 8});
      }
    } else {
      require_shape(a[0], i, {-1});
      require_shape(a[1], i, {a[0].dimensions()[0]});
      require_shape(a[2], i, {-1});
      require_shape(a[3], i, {a[2].dimensions()[0], -1});
      require_shape(a[4], i, {-1, -1});
      require_shape(a[5], i, {a[4].dimensions()[0]});
      auto n = score::common::checked_dispatch_product(
          a[2].dimensions()[0], a[4].dimensions()[1], "OptH reset rows");
      require_shape(out[0], ffi::DataType::U8, {2, n, 8});
    }
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t flip,
      int64_t width) {
    using namespace pack::rotamer;
    ContextManager context;
    if (op < 2) {
      ConformerCopyPlan<score::common::TViewBinder<D>::template View, int32_t>
          c{};
      c.original_types = buffer_view<int32_t, 2, D>(a[0]);
      c.type_natoms = buffer_view<int32_t, 1, D>(a[1]);
      if (op == 0) {
        c.source_counts = buffer_view<int64_t, 1, D>(out[0]);
        c.source_offsets = buffer_view<int64_t, 1, D>(out[1]);
        launch_conformer_source_counts<Ops, D>(context, c);
      } else {
        c.considered = buffer_view<int32_t, 1, D>(a[2]);
        c.global_blocks = buffer_view<int32_t, 1, D>(a[3]);
        c.types = buffer_view<int32_t, 1, D>(a[4]);
        c.offsets = buffer_view<int32_t, 1, D>(a[5]);
        c.selected = buffer_view<int32_t, 1, D>(a[6]);
        c.source_offsets = buffer_view<int64_t, 1, D>(a[7]);
        c.segments = buffer_view<int64_t, 2, D>(out[0]);
        launch_conformer_copy_segments<Ops, D>(context, c);
      }
    } else {
      OptHResetPlan<score::common::TViewBinder<D>::template View, int32_t> c{
          buffer_view<int32_t, 1, D>(a[0]),
          buffer_view<int32_t, 1, D>(a[1]),
          buffer_view<int32_t, 1, D>(a[2]),
          buffer_view<int32_t, 2, D>(a[3]),
          buffer_view<int32_t, 2, D>(a[4]),
          buffer_view<int32_t, 1, D>(a[5]),
          buffer_view<int64_t, 2, D>(out[0]),
          width,
          bool(flip)};
      launch_opth_reset_indices<Ops, D>(context, c);
    }
  }
};
}  // namespace tmol::jax
