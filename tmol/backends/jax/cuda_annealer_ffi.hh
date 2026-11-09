// XLA stream and caller-owned workspace for the common CUDA search schedule.
#pragma once
#include "annealer_common_ffi.hh"
#include "forward_kinematics_ffi.hh"
#include <tmol/pack/compiled/cuda_annealer.cuh>

namespace tmol::jax {
inline ffi::Error anneal_cuda(
    cudaStream_t stream,
    ffi::RemainingArgs inputs,
    ffi::RemainingRets outputs,
    int64_t chunk_size,
    int64_t max_rotamers,
    int64_t n_trajectories) {
  try {
    auto a = native_arguments(inputs);
    if (a.size() != 13 || outputs.size() != 3 || chunk_size < 1
        || chunk_size > INT32_MAX)
      throw std::invalid_argument("invalid CUDA annealer arguments");
    validate_annealer_graph(a);
    require_shape(a[12], ffi::DataType::U8, {3, 8});
    if (reinterpret_cast<uintptr_t>(a[12].untyped_data()) % alignof(uint64_t))
      throw std::invalid_argument("unaligned CUDA phase seeds");
    std::vector<ffi::AnyBuffer> out;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto result = outputs.get<ffi::AnyBuffer>(i);
      if (!result) throw std::invalid_argument("missing CUDA annealer output");
      out.push_back(**result);
    }
    auto poses = a[0].dimensions()[0], residues = a[3].dimensions()[1];
    using namespace pack::compiled;
    if (n_trajectories < 4 || n_trajectories > INT32_MAX)
      throw std::invalid_argument("annealer trajectory count exceeds int32");
    AnnealerTrajectoryCounts counts(static_cast<int>(n_trajectories));
    int trajectories = counts.final;
    require_shape(out[0], ffi::DataType::F32, {poses, trajectories});
    require_shape(out[1], ffi::DataType::S32, {poses, trajectories, residues});
    require_shape(out[2], ffi::DataType::U8, {-1});
    CudaAnnealerWorkspace workspace(
        poses,
        residues,
        max_rotamers,
        a[7].dimensions()[0],
        out[2].untyped_data(),
        out[2].size_bytes(),
        counts.high);
    constexpr Device D = Device::CUDA;
    CudaInteractionGraph<D, int, float> graph{
        static_cast<int>(max_rotamers),
        buffer_view<int32_t, 1, D>(a[0]),
        buffer_view<int32_t, 1, D>(a[1]),
        buffer_view<int32_t, 1, D>(a[2]),
        buffer_view<int32_t, 2, D>(a[3]),
        buffer_view<int32_t, 2, D>(a[4]),
        buffer_view<int32_t, 1, D>(a[5]),
        static_cast<int32_t>(chunk_size),
        buffer_view<int64_t, 1, D>(a[6]),
        buffer_view<int32_t, 1, D>(a[7]),
        buffer_view<int64_t, 1, D>(a[8]),
        workspace.reverse_offsets,
        buffer_view<int64_t, 1, D>(a[9]),
        buffer_view<float, 1, D>(a[10]),
        buffer_view<float, 1, D>(a[11])};
    auto seeds = reinterpret_cast<const uint64_t*>(a[12].untyped_data());
    auto scores = buffer_view<float, 2, D>(out[0]);
    auto assignments = buffer_view<int32_t, 3, D>(out[1]);
    KinematicStream context(stream);
    switch (chunk_size) {
      case 16:
        run_cuda_annealer<16>(
            context, graph, seeds, workspace, scores, assignments);
        break;
      case 32:
        run_cuda_annealer<32>(
            context, graph, seeds, workspace, scores, assignments);
        break;
      default:
        run_cuda_annealer<0>(
            context, graph, seeds, workspace, scores, assignments);
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::InvalidArgument(error.what());
  }
}
}  // namespace tmol::jax
