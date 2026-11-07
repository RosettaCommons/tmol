#include <tmol/backends/cuda/score/common/device_operations.cuda.impl.cuh>
#include <tmol/utility/tensor/torch_context.hh>
#include <tmol/pack/compiled/compiled.impl.hh>
#include <tmol/pack/compiled/streaming_interaction_graph.impl.hh>
#include <tmol/pack/compiled/cuda_annealer.cuh>

namespace tmol {
namespace pack {
namespace compiled {
template <tmol::Device D>
auto AnnealerDispatch<D>::forward(
    ContextManager& mgr,
    int max_n_rotamers_per_pose,
    TView<int, 1, D> pose_n_res,
    TView<int, 1, D> pose_n_rotamers,
    TView<int, 1, D> pose_rotamer_offset,
    TView<int, 2, D> n_rotamers_for_res,
    TView<int, 2, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    int32_t chunk_size,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_offset_offsets,
    TView<int64_t, 1, D> chunk_offsets,
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b)
    -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D>> {
  int const n_poses_cpu = pose_n_res.size(0);
  int const max_n_res_cpu = n_rotamers_for_res.size(1);

  int const n_final = CudaAnnealerPolicy::final_trajectories;
  CudaAnnealerWorkspace shape(
      n_poses_cpu,
      max_n_res_cpu,
      max_n_rotamers_per_pose,
      neighbor_blocks.size(0));
  auto scratch = TPack<uint8_t, 1, D>::empty({shape.bytes});
  CudaAnnealerWorkspace workspace(
      n_poses_cpu,
      max_n_res_cpu,
      max_n_rotamers_per_pose,
      neighbor_blocks.size(0),
      scratch.view.data());
  auto scores = TPack<float, 2, D>::empty({n_poses_cpu, n_final});
  auto assignments =
      TPack<int32_t, 3, D>::empty({n_poses_cpu, n_final, max_n_res_cpu});
  auto phase_seeds =
      native::random_seed_tensor({3}, native::Device(native::kCUDA));
  auto seeds = reinterpret_cast<uint64_t const*>(phase_seeds.data_ptr());
  auto context = current_context(mgr);
  CudaInteractionGraph<D, int, float> ig(
      {max_n_rotamers_per_pose,
       pose_n_res,
       pose_n_rotamers,
       pose_rotamer_offset,
       n_rotamers_for_res,
       oneb_offsets,
       res_for_rot,
       chunk_size,
       neighbor_row_offsets,
       neighbor_blocks,
       neighbor_chunk_offset_offsets,
       workspace.reverse_offsets,
       chunk_offsets,
       energy1b,
       energy2b});

  switch (chunk_size) {
    case 16:
      run_cuda_annealer<16>(
          *context, ig, seeds, workspace, scores.view, assignments.view);
      break;
    case 32:
      run_cuda_annealer<32>(
          *context, ig, seeds, workspace, scores.view, assignments.view);
      break;
    default:
      run_cuda_annealer<0>(
          *context, ig, seeds, workspace, scores.view, assignments.view);
  }
  return {scores, assignments};
}

template struct AnnealerDispatch<tmol::Device::CUDA>;

template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int64_t>;
template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
