#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tmol::pack::compiled {
// These are the established CUDA annealing schedule, not tunable frontend
// defaults.
struct CudaAnnealerPolicy {
  static constexpr int high_trajectories = 500, low_expansions = 10;
  static constexpr int low_trajectories = 1250, final_trajectories = 312;
  static constexpr int high_iterations = 10, low_iterations = 10;
  static constexpr float high_initial = 30.f, low_initial = .3f;
  static constexpr float high_later = .2f, low_later = .1f;
};

// Conservative storage for the existing ModernGPU float/int segmented sort.
// Its smallest default tile is 128*11. All allocations are made once before
// merge passes; the bound covers both arrays, merge/copy metadata, partitions,
// and alignment. The native arena rejects any future contract mismatch.
inline int64_t cuda_annealer_sort_bytes(int64_t count) {
  if (count < 0 || count > INT32_MAX)
    throw std::invalid_argument("CUDA annealer sort count exceeds int32");
  if (!count) return 0;
  int64_t ctas = (count + 1407) / 1408, passes = 0;
  while ((int64_t(1) << passes) < ctas) ++passes;
  return (passes ? 8 * count : 0) + 64 * ctas + 32 * passes + 256;
}

struct CudaAnnealerWorkspace {
  static constexpr Device D = Device::CUDA;
  TView<float, 2, D> scores_hitemp, scores_lotemp;
  TView<int32_t, 3, D> current_rotamer_assignments_hitemp,
      best_rotamer_assignments_hitemp;
  TView<int32_t, 3, D> current_rotamer_assignments_lotemp,
      best_rotamer_assignments_lotemp;
  TView<int32_t, 3, D> current_rotamer_assignments_fullquench;
  TView<int32_t, 2, D> sorted_hitemp_traj, sorted_lotemp_traj,
      sorted_fullquench_traj;
  TView<int32_t, 1, D> segment_heads_hitemp, segment_heads_lotemp,
      segment_heads_fullquench;
  TView<int32_t, 3, D> quench_lite_order, fullquench_order;
  TView<int64_t, 1, D> reverse_offsets;
  unsigned char* data;
  void* sort_data = nullptr;
  int64_t bytes = 0, zero_bytes = 0, sort_bytes = 0;

  template <class T, size_t N>
  TView<T, N, D> take(std::array<int64_t, N> shape, int alignment = 16) {
    bytes = (bytes + alignment - 1) / alignment * alignment;
    int64_t stride[N], count = 1;
    for (int i = N - 1; i >= 0; --i) {
      if (shape[i] < 0 || (shape[i] && count > INT64_MAX / shape[i]))
        throw std::overflow_error("CUDA annealer workspace shape overflow");
      stride[i] = count;
      count *= shape[i];
    }
    if (count > (INT64_MAX - bytes) / int64_t(sizeof(T)))
      throw std::overflow_error("CUDA annealer workspace byte overflow");
    T* ptr = data ? reinterpret_cast<T*>(data + bytes) : nullptr;
    bytes += count * sizeof(T);
    return TView<T, N, D>(ptr, shape.data(), stride);
  }

  CudaAnnealerWorkspace(
      int64_t poses,
      int64_t residues,
      int64_t rotamers,
      int64_t edges,
      void* storage = nullptr,
      int64_t storage_bytes = -1)
      : data(static_cast<unsigned char*>(storage)) {
    if (poses < 0 || residues < 0 || rotamers < 0 || edges < 0
        || poses > INT32_MAX / (32 * CudaAnnealerPolicy::low_trajectories)
        || residues > INT32_MAX || rotamers > INT32_MAX / 2
        || (residues && poses > INT32_MAX / residues))
      throw std::invalid_argument(
          "CUDA annealer dimensions exceed dispatch capacity");
    constexpr int h = CudaAnnealerPolicy::high_trajectories;
    constexpr int l = CudaAnnealerPolicy::low_trajectories;
    constexpr int q = CudaAnnealerPolicy::final_trajectories;
    scores_hitemp = take<float, 2>({poses, h});
    current_rotamer_assignments_hitemp = take<int32_t, 3>({poses, h, residues});
    best_rotamer_assignments_hitemp = take<int32_t, 3>({poses, h, residues});
    sorted_hitemp_traj = take<int32_t, 2>({poses, h});
    segment_heads_hitemp = take<int32_t, 1>({poses});
    segment_heads_lotemp = take<int32_t, 1>({poses});
    segment_heads_fullquench = take<int32_t, 1>({poses});
    scores_lotemp = take<float, 2>({poses, l});
    current_rotamer_assignments_lotemp = take<int32_t, 3>({poses, l, residues});
    best_rotamer_assignments_lotemp = take<int32_t, 3>({poses, l, residues});
    sorted_lotemp_traj = take<int32_t, 2>({poses, l});
    current_rotamer_assignments_fullquench =
        take<int32_t, 3>({poses, q, residues});
    sorted_fullquench_traj = take<int32_t, 2>({poses, q});
    zero_bytes = bytes;
    int64_t lite = (rotamers - 1) / 31 + residues;
    int64_t lite_size = poses * l * lite, full_size = poses * q * rotamers;
    auto order = take<int32_t, 1>({std::max(lite_size, full_size)});
    int64_t ls[3] = {poses, l, lite}, lst[3] = {l * lite, lite, 1};
    int64_t fs[3] = {poses, q, rotamers}, fst[3] = {q * rotamers, rotamers, 1};
    quench_lite_order = TView<int32_t, 3, D>(order.data(), ls, lst);
    fullquench_order = TView<int32_t, 3, D>(order.data(), fs, fst);
    reverse_offsets = take<int64_t, 1>({edges});
    sort_bytes = cuda_annealer_sort_bytes(poses * l);
    sort_data = take<unsigned char, 1>({sort_bytes}, 256).data();
    if (storage_bytes >= 0 && storage_bytes < bytes)
      throw std::invalid_argument("CUDA annealer workspace is too small");
    if (data && reinterpret_cast<uintptr_t>(data) % 16)
      throw std::invalid_argument(
          "CUDA annealer workspace must be 16-byte aligned");
  }
};
}  // namespace tmol::pack::compiled
