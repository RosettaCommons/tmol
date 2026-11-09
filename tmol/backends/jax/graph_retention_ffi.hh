// Typed transport for the common retained-conformer topology algorithms.
#pragma once
#include "graph_topology_ffi.hh"
#include <tmol/pack/compiled/interaction_graph_retention.hh>

namespace tmol::jax {
struct GraphRetentionStage {
  static int fill(int64_t operation, size_t output) {
    return operation == 1 ? (output == 0 ? -1 : output == 2 ? 255 : 0) : 0;
  }
  static void validate(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    constexpr size_t inputs[] = {4, 3, 2, 2, 2, 1, 2, 1, 4, 5, 5};
    if (op < 0 || op >= 11 || a.size() != inputs[op]
        || out.size() != (op == 1 ? 3 : 1) || extent < 0 || extent > INT32_MAX
        || width < 0 || width > INT32_MAX)
      throw std::invalid_argument("invalid graph retention stage");
    auto integer = [](ffi::AnyBuffer b, std::initializer_list<int64_t> s) {
      require_shape(b, ffi::DataType::S32, s);
    };
    auto matrix = [](ffi::AnyBuffer b, int64_t p, int64_t m) {
      require_shape(b, ffi::DataType::U8, {p, m, 8});
      if (reinterpret_cast<uintptr_t>(b.untyped_data()) % alignof(int64_t))
        throw std::invalid_argument("unaligned graph retention metadata");
    };
    if (op == 0) {
      integer(a[0], {-1});
      integer(a[1], {a[0].dimensions()[0]});
      auto type = a[2].element_type();
      if (type != ffi::DataType::F32 && type != ffi::DataType::F64)
        throw std::invalid_argument("bump energies must be floating point");
      require_shape(a[2], type, {a[0].dimensions()[0]});
      require_shape(a[3], type, {-1, -1});
      graph_wide(out[0], a[0].dimensions()[0]);
    } else if (op == 1) {
      integer(a[0], {-1, -1});
      auto p = a[0].dimensions()[0], m = a[0].dimensions()[1];
      integer(a[1], {p, m});
      graph_wide(a[2]);
      graph_wide(out[0], a[2].dimensions()[0]);
      matrix(out[1], p, m);
      matrix(out[2], p, m);
      if (out[0].untyped_data() != a[2].untyped_data())
        throw std::invalid_argument("retention requires aliased keep mask");
    } else if (op == 2) {
      graph_wide(a[0]);
      graph_wide(a[1], a[0].dimensions()[0]);
      graph_wide(out[0]);
    } else if (op == 3 || op == 4) {
      matrix(a[0], -1, -1);
      auto p = a[0].dimensions()[0], m = a[0].dimensions()[1];
      graph_wide(a[1], p * m);
      if (op == 3)
        graph_wide(out[0]);
      else
        integer(out[0], {p});
    } else if (op == 5 || op == 6 || op == 7) {
      integer(a[0], {-1});
      integer(out[0], {a[0].dimensions()[0]});
      if (op == 6) graph_wide(a[1]);
    } else if (op == 8) {
      graph_wide(a[0]);
      integer(a[2], {-1, -1});
      auto p = a[2].dimensions()[0];
      integer(a[1], {p});
      graph_wide(a[3]);
      integer(out[0], {p, width});
    } else if (op == 9) {
      graph_wide(a[0]);
      integer(a[1], {-1});
      auto p = a[1].dimensions()[0];
      integer(a[2], {p});
      integer(a[3], {p, -1});
      integer(a[4], {p});
      integer(out[0], {p, a[3].dimensions()[1]});
    } else {
      graph_wide(a[0]);
      integer(a[1], {-1});
      integer(a[2], {a[1].dimensions()[0]});
      integer(a[4], {-1});
      graph_wide(a[3], a[4].dimensions()[0] * width);
      integer(out[0], {a[0].dimensions()[0]});
    }
  }
  template <template <Device> class Ops, Device D>
  static void execute(
      const std::vector<ffi::AnyBuffer>& a,
      const std::vector<ffi::AnyBuffer>& out,
      int64_t op,
      int64_t extent,
      int64_t width) {
    using namespace pack::compiled;
    ContextManager context;
#define IN(T, R, I) buffer_view<T, R, D>(a[I])
#define OUT(T, R, I) buffer_view<T, R, D>(out[I])
    switch (op) {
      case 0:
        if (a[2].element_type() == ffi::DataType::F32)
          launch_ig_keep_rotamers<Ops, D>(
              context,
              IN(int32_t, 1, 0),
              IN(int32_t, 1, 1),
              IN(float, 1, 2),
              IN(float, 2, 3),
              OUT(int64_t, 1, 0));
        else
          launch_ig_keep_rotamers<Ops, D>(
              context,
              IN(int32_t, 1, 0),
              IN(int32_t, 1, 1),
              IN(double, 1, 2),
              IN(double, 2, 3),
              OUT(int64_t, 1, 0));
        break;
      case 1:
        launch_ig_keep_blocks<Ops, D>(
            context,
            IN(int32_t, 2, 0),
            IN(int32_t, 2, 1),
            OUT(int64_t, 1, 0),
            OUT(int64_t, 2, 1),
            OUT(int64_t, 2, 2));
        break;
      case 2:
        launch_ig_invert_rotamers<Ops, D>(
            context, IN(int64_t, 1, 0), IN(int64_t, 1, 1), OUT(int64_t, 1, 0));
        break;
      case 3:
        launch_ig_invert_blocks<Ops, D>(
            context, IN(int64_t, 2, 0), IN(int64_t, 1, 1), OUT(int64_t, 1, 0));
        break;
      case 4:
        launch_ig_pose_block_offsets<Ops, D>(
            context, IN(int64_t, 2, 0), IN(int64_t, 1, 1), OUT(int32_t, 1, 0));
        break;
      case 5:
        launch_ig_pose_block_counts<Ops, D>(
            context, IN(int32_t, 1, 0), OUT(int32_t, 1, 0), extent);
        break;
      case 6:
        launch_ig_pose_rotamer_offsets<Ops, D>(
            context,
            IN(int32_t, 1, 0),
            IN(int64_t, 1, 1),
            OUT(int32_t, 1, 0),
            extent);
        break;
      case 7:
        launch_ig_pose_rotamer_counts<Ops, D>(
            context, IN(int32_t, 1, 0), OUT(int32_t, 1, 0), extent);
        break;
      case 8:
        launch_ig_block_rotamer_offsets<Ops, D>(
            context,
            IN(int64_t, 1, 0),
            IN(int32_t, 1, 1),
            IN(int32_t, 2, 2),
            IN(int64_t, 1, 3),
            OUT(int32_t, 2, 0));
        break;
      case 9:
        launch_ig_block_rotamer_counts<Ops, D>(
            context,
            IN(int64_t, 1, 0),
            IN(int32_t, 1, 1),
            IN(int32_t, 1, 2),
            IN(int32_t, 2, 3),
            IN(int32_t, 1, 4),
            OUT(int32_t, 2, 0),
            width,
            extent);
        break;
      case 10:
        launch_ig_rotamer_blocks<Ops, D>(
            context,
            IN(int64_t, 1, 0),
            IN(int32_t, 1, 1),
            IN(int32_t, 1, 2),
            IN(int64_t, 1, 3),
            IN(int32_t, 1, 4),
            OUT(int32_t, 1, 0),
            width);
        break;
    }
#undef IN
#undef OUT
  }
};
}  // namespace tmol::jax
