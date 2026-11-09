#pragma once
#include <tmol/pack/compiled/annealer_workspace.hh>
#include <tmol/pack/compiled/annealer_policy.hh>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tmol::pack::compiled {
// PR626 CUDA schedule; trajectory counts are selected by the caller.
struct CudaAnnealerPolicy {
  static constexpr int high_trajectories =
                           AnnealerTrajectoryCounts::default_high,
                       low_expansions = 10;
  static constexpr int low_trajectories = 320, final_trajectories = 80;
  static constexpr int high_iterations = 5, low_iterations = 5;
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

struct CudaAnnealerWorkspace : AnnealerWorkspace<Device::CUDA> {
  AnnealerTrajectoryCounts counts;
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
  void* sort_data = nullptr;
  int64_t zero_bytes = 0, sort_bytes = 0;

  CudaAnnealerWorkspace(
      int64_t poses,
      int64_t residues,
      int64_t rotamers,
      int64_t edges,
      void* storage = nullptr,
      int64_t storage_bytes = -1,
      int high = AnnealerTrajectoryCounts::default_high)
      : AnnealerWorkspace(storage), counts(high) {
    if (poses < 0 || residues < 0 || rotamers < 0 || edges < 0
        || poses > INT32_MAX / (int64_t(32) * counts.low)
        || residues > INT32_MAX || rotamers > INT32_MAX / 2
        || (residues && poses > INT32_MAX / residues))
      throw std::invalid_argument(
          "CUDA annealer dimensions exceed dispatch capacity");
    int const h = counts.high;
    int const l = counts.low;
    int const q = counts.final;
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
    validate(storage_bytes);
  }
};
}  // namespace tmol::pack::compiled
