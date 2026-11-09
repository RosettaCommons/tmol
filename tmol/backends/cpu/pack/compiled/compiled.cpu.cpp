#include <tmol/utility/portable/math.hh>
#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>

#include <tmol/pack/compiled/annealer_cpu_funnel.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/pack/compiled/compiled.impl.hh>
#include <tmol/pack/compiled/streaming_interaction_graph.impl.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <tmol::Device D>
auto AnnealerDispatch<D>::forward(
    ContextManager&,
    int max_n_rotamers_per_pose,
    TView<int, 1, D> pose_n_res,               // n-poses
    TView<int, 1, D> n_rotamers_for_pose,      // n-poses
    TView<int, 1, D> rotamer_offset_for_pose,  // n-poses
    TView<int, 2, D> n_rotamers_for_res,       // n-poses x max-n-res
    TView<int, 2, D> oneb_offsets,             // n-poses x max-n-res
    TView<int, 1, D> res_for_rot,              // n-rots
    int32_t chunk_size,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_offset_offsets,
    TView<int64_t, 1, D> chunk_offsets,  // n-chunks-on-interacting-res
    TView<float, 1, D> energy1b,
    TView<float, 1, D> energy2b,
    int n_hitemp_traj) -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D>> {
  int const n_poses = pose_n_res.size(0);
  int const max_n_res = n_rotamers_for_res.size(1);
  int const n_rotamers = res_for_rot.size(0);

  AnnealerTrajectoryCounts counts(n_hitemp_traj, true);
  bool const funnel =
      native::get_num_threads() >= AnnealerTrajectoryCounts::min_cpu_threads;
  int const n_traj = funnel ? counts.final : 1;

  auto scores_t = TPack<float, 2, D>::zeros({n_poses, n_traj});
  auto current_rotamer_assignments_t =
      TPack<int, 3, D>::zeros({funnel ? 0 : n_poses, 1, max_n_res});
  auto best_rotamer_assignments_t =
      TPack<int, 3, D>::zeros({n_poses, n_traj, max_n_res});
  auto quench_order_t = TPack<int, 1, D>::zeros({funnel ? 0 : n_rotamers});

  auto scores = scores_t.view;
  auto current_rotamer_assignments = current_rotamer_assignments_t.view;
  auto best_rotamer_assignments = best_rotamer_assignments_t.view;
  auto quench_order = quench_order_t.view;

  using Binder = score::common::TViewBinder<D>;
  CpuAnnealerContext<Binder::template View> c;
  c.pose_n_res = pose_n_res;
  c.n_rotamers_for_pose = n_rotamers_for_pose;
  c.rotamer_offset_for_pose = rotamer_offset_for_pose;
  c.res_for_rot = res_for_rot;
  c.quench_order = quench_order;
  c.n_rotamers_for_res = n_rotamers_for_res;
  c.oneb_offsets = oneb_offsets;
  c.neighbor_row_offsets = neighbor_row_offsets;
  c.neighbor_chunk_offset_offsets = neighbor_chunk_offset_offsets;
  c.chunk_offsets = chunk_offsets;
  c.energy1b = energy1b;
  c.energy2b = energy2b;
  c.scores = scores;
  c.current_rotamer_assignments = current_rotamer_assignments;
  c.best_rotamer_assignments = best_rotamer_assignments;
  c.neighbor_blocks = neighbor_blocks;
  c.chunk_size = chunk_size;
  auto seed = native::random_seed_tensor({}, native::Device(native::kCPU));
  if (funnel) {
    CpuFunnelWorkspace shape(n_poses, max_n_res, n_hitemp_traj);
    auto scratch = TPack<uint8_t, 1, D>::empty({shape.bytes});
    CpuFunnelWorkspace workspace(
        n_poses, max_n_res, n_hitemp_traj, scratch.view.data(), shape.bytes);
    run_cpu_funnel(
        c,
        workspace,
        static_cast<uint64_t>(seed.item<int64_t>()),
        [](int64_t begin, int64_t end, int64_t grain, auto fn) {
          native::parallel_for(begin, end, grain, fn);
        });
  } else {
    run_cpu_annealer(c, static_cast<uint64_t>(seed.item<int64_t>()));
  }

  return {scores_t, best_rotamer_assignments_t};
}

template struct AnnealerDispatch<tmol::Device::CPU>;

template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct InteractionGraphBuilder<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int64_t>;
template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int64_t>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
