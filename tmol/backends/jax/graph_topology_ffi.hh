// Typed array contracts for shared sparse interaction-graph topology stages.
#pragma once
#include "genbonded_ffi.hh"
#include <tmol/pack/compiled/interaction_graph_topology.hh>

namespace tmol::jax {
inline std::vector<ffi::AnyBuffer> native_results(ffi::RemainingRets results) {
  std::vector<ffi::AnyBuffer> buffers;
  for (size_t i = 0; i < results.size(); ++i) {
    auto result = results.get<ffi::AnyBuffer>(i);
    if (!result)
      throw std::invalid_argument("native FFI expected result buffers");
    buffers.push_back(**result);
  }
  return buffers;
}

inline void graph_wide(ffi::AnyBuffer a, int64_t size = -1) {
  require_shape(a, ffi::DataType::U8, {size, 8});
  if (reinterpret_cast<uintptr_t>(a.untyped_data()) % alignof(int64_t))
    throw std::invalid_argument("unaligned graph int64 metadata");
}

inline void validate_graph_topology(
    const std::vector<ffi::AnyBuffer>& a,
    const std::vector<ffi::AnyBuffer>& out,
    int64_t operation,
    int64_t chunk_size) {
  constexpr size_t inputs[] = {1, 1, 3, 1, 3, 6, 2};
  constexpr size_t outputs[] = {1, 1, 2, 1, 1, 1, 1};
  if (operation < 0 || operation >= 7 || a.size() != inputs[operation]
      || out.size() != outputs[operation] || chunk_size < 1
      || chunk_size > INT32_MAX)
    throw std::invalid_argument("invalid graph topology stage");
  if (operation == 0) {
    require_shape(a[0], ffi::DataType::S32, {-1, -1});
    require_shape(
        out[0],
        ffi::DataType::U8,
        {a[0].dimensions()[0], a[0].dimensions()[1], 8});
    return;
  }
  if (operation == 1) {
    require_shape(a[0], ffi::DataType::S32, {-1, -1, -1});
    auto p = a[0].dimensions()[0], m = a[0].dimensions()[1];
    require_shape(a[0], ffi::DataType::S32, {p, m, (m + 31) / 32});
    graph_wide(out[0], p * m + 1);
    return;
  }
  if (operation == 3 || operation == 6) {
    graph_wide(a[0]);
    graph_wide(out[0], a[0].dimensions()[0]);
    if (operation == 6) {
      graph_wide(a[1], a[0].dimensions()[0]);
      if (out[0].untyped_data() != a[1].untyped_data())
        throw std::invalid_argument("graph sentinel requires aliased offsets");
    }
    return;
  }
  require_shape(a[0], ffi::DataType::S32, {-1, -1});
  auto p = a[0].dimensions()[0], m = a[0].dimensions()[1];
  if (operation == 2) {
    require_shape(a[1], ffi::DataType::S32, {p, m, (m + 31) / 32});
    graph_wide(a[2], p * m + 1);
    require_shape(out[0], ffi::DataType::S32, {-1});
    graph_wide(out[1], out[0].dimensions()[0]);
  } else {
    graph_wide(a[1], p * m + 1);
    require_shape(a[2], ffi::DataType::S32, {-1});
    auto edges = a[2].dimensions()[0];
    if (operation == 4) {
      graph_wide(out[0], edges);
    } else {
      graph_wide(a[3], edges);
      graph_wide(a[4], edges);
      require_shape(a[5], ffi::DataType::S32, {-1});
      graph_wide(out[0]);
    }
  }
}

template <template <Device> class Ops, Device D>
void execute_graph_topology(
    const std::vector<ffi::AnyBuffer>& a,
    const std::vector<ffi::AnyBuffer>& out,
    int64_t operation,
    int64_t chunk_size) {
  using namespace pack::compiled;
  ContextManager context;
#define IN(T, R, I) buffer_view<T, R, D>(a[I])
#define OUT(T, R, I) buffer_view<T, R, D>(out[I])
  switch (operation) {
    case 0:
      launch_ig_initialize_pose<Ops, D>(
          context, IN(int32_t, 2, 0), OUT(int64_t, 2, 0));
      break;
    case 1:
      launch_ig_count_neighbors<Ops, D>(
          context, IN(int32_t, 3, 0), OUT(int64_t, 1, 0));
      break;
    case 2:
      launch_ig_fill_neighbors<Ops, D>(
          context,
          chunk_size,
          IN(int32_t, 2, 0),
          IN(int32_t, 3, 1),
          IN(int64_t, 1, 2),
          OUT(int32_t, 1, 0),
          OUT(int64_t, 1, 1));
      break;
    case 3:
      launch_ig_count_chunk_words<Ops, D>(
          context, IN(int64_t, 1, 0), OUT(int64_t, 1, 0));
      break;
    case 4:
      launch_ig_count_chunk_pairs<Ops, D>(
          context,
          chunk_size,
          IN(int32_t, 2, 0),
          IN(int64_t, 1, 1),
          IN(int32_t, 1, 2),
          OUT(int64_t, 1, 0));
      break;
    case 5:
      launch_ig_fill_chunk_pair_sizes<Ops, D>(
          context,
          chunk_size,
          IN(int32_t, 2, 0),
          IN(int64_t, 1, 1),
          IN(int32_t, 1, 2),
          IN(int64_t, 1, 3),
          IN(int64_t, 1, 4),
          IN(int32_t, 1, 5),
          OUT(int64_t, 1, 0));
      break;
    case 6:
      launch_ig_sentinel_empty_chunks<Ops, D>(
          context, IN(int64_t, 1, 0), OUT(int64_t, 1, 0));
      break;
  }
#undef IN
#undef OUT
}
}  // namespace tmol::jax
