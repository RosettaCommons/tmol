// Caller-owned XLA buffers for the shared native CPU annealing schedule.
#pragma once
#include "annealer_common_ffi.hh"
#include <tmol/pack/compiled/annealer_cpu_funnel.hh>
#include <tmol/score/common/tview_binder.hh>
#include <cstring>

namespace tmol::jax {
template <typename ParallelFor>
inline ffi::Error anneal_cpu(
    ffi::RemainingArgs inputs,
    ffi::RemainingRets outputs,
    int64_t chunk_size,
    int64_t seed,
    int64_t n_trajectories,
    int64_t threads,
    ParallelFor parallel_for) {
  try {
    auto a = native_arguments(inputs);
    if (a.size() != 12 || outputs.size() != 5 || chunk_size < 1
        || chunk_size > INT32_MAX)
      throw std::invalid_argument("invalid CPU annealer arguments");
    if (n_trajectories < 4 || n_trajectories > INT32_MAX || threads < 1)
      throw std::invalid_argument(
          "invalid CPU annealer trajectory/thread count");
    pack::compiled::AnnealerTrajectoryCounts counts(n_trajectories, true);
    bool funnel =
        threads >= pack::compiled::AnnealerTrajectoryCounts::min_cpu_threads;
    int final = funnel ? counts.final : 1;
    validate_annealer_graph(a);
    auto poses = a[0].dimensions()[0], residues = a[3].dimensions()[1],
         rotamers = a[5].dimensions()[0];
    std::vector<ffi::AnyBuffer> out;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto result = outputs.get<ffi::AnyBuffer>(i);
      if (!result)
        throw std::invalid_argument("annealer expected result buffers");
      out.push_back(**result);
    }
    require_shape(out[0], ffi::DataType::F32, {poses, final});
    require_shape(out[1], ffi::DataType::S32, {poses, final, residues});
    require_shape(out[2], ffi::DataType::S32, {poses, 1, residues});
    require_shape(out[3], ffi::DataType::S32, {rotamers});
    require_shape(out[4], ffi::DataType::U8, {-1});
    for (auto result : out)
      if (result.size_bytes())
        std::memset(result.untyped_data(), 0, result.size_bytes());
    constexpr Device D = Device::CPU;
    pack::compiled::CpuAnnealerContext<
        score::common::TViewBinder<D>::template View>
        c;
    c.pose_n_res = buffer_view<int32_t, 1, D>(a[0]);
    c.n_rotamers_for_pose = buffer_view<int32_t, 1, D>(a[1]);
    c.rotamer_offset_for_pose = buffer_view<int32_t, 1, D>(a[2]);
    c.n_rotamers_for_res = buffer_view<int32_t, 2, D>(a[3]);
    c.oneb_offsets = buffer_view<int32_t, 2, D>(a[4]);
    c.res_for_rot = buffer_view<int32_t, 1, D>(a[5]);
    c.neighbor_row_offsets = buffer_view<int64_t, 1, D>(a[6]);
    c.neighbor_blocks = buffer_view<int32_t, 1, D>(a[7]);
    c.neighbor_chunk_offset_offsets = buffer_view<int64_t, 1, D>(a[8]);
    c.chunk_offsets = buffer_view<int64_t, 1, D>(a[9]);
    c.energy1b = buffer_view<float, 1, D>(a[10]);
    c.energy2b = buffer_view<float, 1, D>(a[11]);
    c.scores = buffer_view<float, 2, D>(out[0]);
    c.best_rotamer_assignments = buffer_view<int32_t, 3, D>(out[1]);
    c.current_rotamer_assignments = buffer_view<int32_t, 3, D>(out[2]);
    c.quench_order = buffer_view<int32_t, 1, D>(out[3]);
    c.chunk_size = static_cast<int>(chunk_size);
    if (funnel) {
      pack::compiled::CpuFunnelWorkspace workspace(
          poses,
          residues,
          counts.high,
          out[4].untyped_data(),
          out[4].size_bytes());
      pack::compiled::run_cpu_funnel(
          c, workspace, static_cast<uint64_t>(seed), parallel_for);
    } else {
      pack::compiled::run_cpu_annealer(c, static_cast<uint64_t>(seed));
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::InvalidArgument(error.what());
  }
}
}  // namespace tmol::jax
