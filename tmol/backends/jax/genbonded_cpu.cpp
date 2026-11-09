// CPU execution policy and XLA registration; science lives in the common
// kernel.
#include <initializer_list>
#include "armijo_ffi.hh"
#include "cartbonded_ffi.hh"
#include "backbone_torsion_ffi.hh"
#include "rotamer_backbone_ffi.hh"
#include "metal_ffi.hh"
#include "na_torsion_ffi.hh"
#include "na_sampling_ffi.hh"
#include "pose_imposition_ffi.hh"
#include "opth_sampling_ffi.hh"
#include "workspace_ffi.hh"
#include "lk_ball_ffi.hh"
#include "hbond_ffi.hh"
#include "disulfide_ffi.hh"
#include "rotamer_connection_ffi.hh"
#include "dunbrack_ffi.hh"
#include "nonbonded_ffi.hh"
#include "independent_nonbonded_ffi.hh"
#include "constraint_torsion_ffi.hh"
#include "kinematics_ffi.hh"
#include "conformer_ffi.hh"
#include "io_ffi.hh"
#include "conformer_copy_plan_ffi.hh"
#include "graph_topology_ffi.hh"
#include "graph_entries_ffi.hh"
#include "graph_bounds_ffi.hh"
#include "constraint_topology_ffi.hh"
#include "graph_retention_ffi.hh"
#include "rotamer_dispatch_ffi.hh"
#include "rotamer_nonbonded_ffi.hh"
#include "graph_prefix_ffi.hh"
#include "load_balance_ffi.hh"
#include "rotamer_bonded_ffi.hh"
#include "annealer_ffi.hh"
#include "forward_kinematics_ffi.hh"
#include <tmol/score/common/serial_cpu_operations.hh>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <tmol/numeric/bspline_compiled/bspline_coeffs.hh>

namespace tmol::jax {
namespace {
thread_local ffi::ThreadPool* active_pool = nullptr;
std::mutex parallel_workgroups;

template <Device D>
struct CpuOps;
template <>
struct CpuOps<Device::CPU> : score::common::SerialCPUOperations {
  static void bitwise_or(int32_t& target, int32_t value) { target |= value; }
  static void load_balance(
      int total,
      const int64_t* offsets,
      int generators,
      int32_t* output,
      int64_t* scratch,
      size_t capacity) {
    score::common::load_balancing_search_into_cpu(
        total, offsets, generators, output);
  }

  template <typename Launch, typename F>
  static void foreach_pose_workgroup(
      ContextManager&, int poses, int groups, F f) {
    auto run = [=](int pose) {
      for (int group = 0; group < groups; ++group) f(pose * groups + group);
    };
    // Concurrent XLA calls may occupy the same pool. Only one call may wait
    // for child work; competing calls execute their own work without blocking
    // a pool worker needed by the first call. Keep the uncontended parallel
    // path, including its caller's share of the work.
    std::unique_lock<std::mutex> dispatch(
        parallel_workgroups, std::try_to_lock);
    if (poses <= 1 || !active_pool || active_pool->num_threads() <= 1
        || !dispatch.owns_lock()) {
      for (int pose = 0; pose < poses; ++pose) run(pose);
      return;
    }
    std::mutex mutex;
    std::condition_variable completed;
    int remaining = poses - 1;
    for (int pose = 0; pose < poses - 1; ++pose) {
      active_pool->Schedule([&, pose] {
        run(pose);
        std::lock_guard<std::mutex> lock(mutex);
        if (--remaining == 0) completed.notify_one();
      });
    }
    run(poses - 1);
    std::unique_lock<std::mutex> lock(mutex);
    completed.wait(lock, [&] { return remaining == 0; });
  }
  template <typename Launch, typename F>
  static void forall_grouped(
      ContextManager& context, int poses, int groups, F f) {
    foreach_pose_workgroup<Launch>(context, poses, groups, f);
  }
  template <typename Launch, typename F>
  static void foreach_workgroup(ContextManager&, int count, F f) {
    for (int index = 0; index < count; ++index) f(index);
  }
  template <typename Launch, typename F>
  static void foreach_independent_workgroup(
      ContextManager& context, int count, F f) {
    foreach_pose_workgroup<Launch>(context, count, 1, f);
  }
  template <typename Launch, typename F>
  static void forall(ContextManager& context, int count, F f) {
    // Elementwise initialization is too small for one pool task per element.
    // Bound scheduling overhead and keep contiguous writes in each task.
    constexpr int grain = 4096;
    int chunks = count / grain + (count % grain != 0);
    if (active_pool)
      chunks = std::min<int64_t>(chunks, active_pool->num_threads());
    if (chunks <= 1) {
      for (int i = 0; i < count; ++i) f(i);
      return;
    }
    int width = count / chunks + (count % chunks != 0);
    foreach_pose_workgroup<Launch>(context, chunks, 1, [=](int chunk) {
      int begin = chunk * width;
      int end = begin + std::min(width, count - begin);
      for (int i = begin; i < end; ++i) f(i);
    });
  }
  template <typename Launch, typename F>
  static void forall_independent(ContextManager& context, int count, F f) {
    forall<Launch>(context, count, f);
  }
  template <int N, int Width, typename T>
  static void copy_contiguous_data(
      T* destination,
      T* source,
      int count,
      score::common::WorkgroupLane<Device::CPU> = {}) {
    std::copy_n(source, count, destination);
  }
  template <int N, int Width, typename T, typename U>
  static void copy_contiguous_data_and_cast(
      T* destination,
      const U* source,
      int count,
      score::common::WorkgroupLane<Device::CPU> = {}) {
    std::copy_n(source, count, destination);
  }
  static void synchronize_workgroup(
      score::common::WorkgroupLane<Device::CPU> = {}) {}
  template <int N, typename T, typename Op>
  static T shuffle_reduce_in_workgroup(
      T value, Op, score::common::WorkgroupLane<Device::CPU> = {}) {
    static_assert(N == 1);
    return value;
  }
  template <int N, typename T, typename Op>
  static T shuffle_reduce_and_broadcast_in_workgroup(
      T value, Op op, score::common::WorkgroupLane<Device::CPU> lane = {}) {
    return shuffle_reduce_in_workgroup<N>(value, op, lane);
  }
  template <int N, typename F>
  static void for_each_in_workgroup(
      F f, score::common::WorkgroupLane<Device::CPU> = {}) {
    static_assert(N == 1, "CPU workgroup must have one lane");
    f(0);
  }
  template <int N, typename T, typename S, typename Op>
  static T reduce_in_workgroup(
      T value, S&, Op, score::common::WorkgroupLane<Device::CPU> = {}) {
    return value;
  }
};

template <typename Kernel, typename... Attributes>
ffi::Error run(
    ffi::ThreadPool pool,
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> energy,
    ffi::RemainingRets rets,
    Attributes... attributes) {
  auto* previous = active_pool;
  try {
    auto buffers = native_arguments(args);
    auto outputs = native_outputs<Kernel>(buffers, *energy, rets);
    const auto& gradient = outputs.gradient;
    if constexpr (ZeroEnergy<Kernel>::value)
      std::memset(energy->untyped_data(), 0, energy->size_bytes());
    if (gradient)
      std::memset(gradient->untyped_data(), 0, gradient->size_bytes());
    if (outputs.workspace && ZeroWorkspace<Kernel>::value)
      std::memset(
          outputs.workspace->untyped_data(),
          0,
          outputs.workspace->size_bytes());
    active_pool = &pool;
    if (buffers[0].element_type() == ffi::DataType::F32) {
      execute_native<Kernel, CpuOps, Device::CPU, float>(
          buffers, *energy, outputs, attributes...);
    } else {
      execute_native<Kernel, CpuOps, Device::CPU, double>(
          buffers, *energy, outputs, attributes...);
    }
    active_pool = previous;
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    active_pool = previous;
    return ffi::Error::InvalidArgument(error.what());
  }
}
template <typename Stage, typename... Attributes>
ffi::Error run_graph_stage(
    ffi::ThreadPool pool,
    ffi::RemainingArgs args,
    ffi::RemainingRets rets,
    int64_t operation,
    int64_t extent,
    int64_t width,
    Attributes... attributes) {
  auto previous = active_pool;
  try {
    auto inputs = native_arguments(args);
    auto outputs = native_results(rets);
    Stage::validate(inputs, outputs, operation, extent, width, attributes...);
    for (size_t i = 0; i < outputs.size(); ++i) {
      int fill = Stage::fill(operation, i);
      auto result = outputs[i];
      if (fill >= 0 && result.size_bytes())
        std::memset(result.untyped_data(), fill, result.size_bytes());
    }
    active_pool = &pool;
    Stage::template execute<CpuOps, Device::CPU>(
        inputs, outputs, operation, extent, width, attributes...);
    active_pool = previous;

    return ffi::Error::Success();
  } catch (const std::exception& error) {
    active_pool = previous;
    return ffi::Error::InvalidArgument(error.what());
  }
}
// Runtime pool size is queried once before compiling shape-dependent search.
ffi::Error cpu_thread_count(
    ffi::ThreadPool pool, ffi::Result<ffi::AnyBuffer> out) {
  require_shape(*out, ffi::DataType::S32, {});
  *static_cast<int32_t*>(out->untyped_data()) =
      std::max<int>(1, pool.num_threads());
  return ffi::Error::Success();
}

ffi::Error run_annealer(
    ffi::ThreadPool pool,
    ffi::RemainingArgs args,
    ffi::RemainingRets rets,
    int64_t chunk_size,
    int64_t seed,
    int64_t n_trajectories,
    int64_t threads) {
  if (threads < 1 || threads > std::max<int64_t>(1, pool.num_threads()))
    return ffi::Error::InvalidArgument(
        "annealing thread budget exceeds XLA pool");
  auto* previous = active_pool;
  active_pool = &pool;
  auto parallel_for = [threads](
                          int64_t begin, int64_t end, int64_t grain, auto fn) {
    if (end <= begin) return;
    int chunks = static_cast<int>(
        std::min<int64_t>(threads, (end - begin + grain - 1) / grain));
    int64_t size = (end - begin + chunks - 1) / chunks;
    ContextManager context;
    CpuOps<Device::CPU>::foreach_pose_workgroup<void>(
        context, chunks, 1, [&](int part) {
          int64_t first = begin + part * size;
          if (first < end) fn(first, std::min(end, first + size));
        });
  };
  auto result = anneal_cpu(
      args, rets, chunk_size, seed, n_trajectories, threads, parallel_for);
  active_pool = previous;
  return result;
}
}  // namespace
}  // namespace tmol::jax

// Host setup reuses the execution policy above for topology-only kernels.
#include "host_pose_ffi.hh"

extern "C" int tmol_jax_native_abi_version() { return 6; }

namespace tmol::jax {
template <typename Real, size_t Rank>
void fit_spline(void* data, const int64_t* shape) {
  std::array<int64_t, Rank> sizes;
  std::copy_n(shape, Rank, sizes.begin());
  numeric::bspline::ndspline_coeffs<Rank, 3, Device::CPU, Real, int32_t>::
      computeCoeffs(contiguous_view<Real, Rank, Device::CPU>(data, sizes));
}

template <typename Real>
void fit_spline(void* data, int rank, const int64_t* shape) {
  switch (rank) {
    case 2:
      return fit_spline<Real, 2>(data, shape);
    case 3:
      return fit_spline<Real, 3>(data, shape);
    case 4:
      return fit_spline<Real, 4>(data, shape);
    default:
      throw std::invalid_argument("spline table rank must be 2, 3 or 4");
  }
}
}  // namespace tmol::jax

// Host-only parameter preparation; coordinate scoring never passes through
// here.
extern "C" int tmol_bspline_coefficients(
    void* data,
    int bits,
    int rank,
    const int64_t* shape,
    char* error,
    size_t error_size) {
  try {
    if (!data || !shape || rank < 2 || rank > 4)
      throw std::invalid_argument("invalid spline buffer or rank");
    int64_t size = 1;
    for (int i = 0; i < rank; ++i) {
      if (shape[i] <= 0
          || shape[i] > std::numeric_limits<int32_t>::max() / size)
        throw std::invalid_argument(
            "spline extents must be positive and fit int32");
      size *= shape[i];
    }
    if (bits == 32)
      tmol::jax::fit_spline<float>(data, rank, shape);
    else if (bits == 64)
      tmol::jax::fit_spline<double>(data, rank, shape);
    else
      throw std::invalid_argument("spline values must be float32 or float64");
    return 0;
  } catch (const std::exception& exception) {
    if (error && error_size) {
      std::strncpy(error, exception.what(), error_size - 1);
      error[error_size - 1] = '\0';
    }
    return 1;
  }
}

namespace {
// C ABI queries run during shape preparation and never allocate device storage.
// Keep precision dispatch and exception translation identical for every layout.
template <typename Query>
uint64_t workspace_bytes(
    int bits, std::initializer_list<int64_t> dimensions, Query query) noexcept {
  for (auto dimension : dimensions)
    if (dimension < 0 || dimension > INT32_MAX) return UINT64_MAX;
  try {
    if (bits == 32) return query(float{});
    if (bits == 64) return query(double{});
  } catch (...) {
  }
  return UINT64_MAX;
}
}  // namespace

extern "C" uint64_t tmol_dunbrack_workspace_bytes(
    int64_t rots, int64_t dihedrals, int bits, int derivative) {
  return workspace_bytes(bits, {rots, dihedrals}, [&](auto real) -> uint64_t {
    return tmol::jax::DunbrackWorkspace<decltype(real)>(
               rots, dihedrals, derivative)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_dunbrack_energy_workspace_bytes(
    int64_t rots, int64_t dihedrals, int64_t poses, int bits, int derivative) {
  if (poses > INT32_MAX / 3) return UINT64_MAX;
  return workspace_bytes(
      bits, {rots, dihedrals, poses}, [&](auto real) -> uint64_t {
        return tmol::jax::DunbrackWorkspace<decltype(real)>(
                   rots, dihedrals, derivative, nullptr, poses)
            .storage.size_bytes();
      });
}

extern "C" uint64_t tmol_dunbrack_precise_workspace_bytes(
    int64_t rots,
    int64_t dihedrals,
    int64_t poses,
    int64_t atoms,
    int bits,
    int derivative) {
  if (poses > INT32_MAX / 3) return UINT64_MAX;
  return workspace_bytes(
      bits, {rots, dihedrals, poses, atoms}, [&](auto real) -> uint64_t {
        auto gradients = derivative && sizeof(real) == sizeof(float)
                             ? tmol::jax::DunbrackKernel::gradient_values(atoms)
                             : 0;
        return tmol::jax::DunbrackWorkspace<decltype(real)>(
                   rots, dihedrals, derivative, nullptr, poses, gradients)
            .storage.size_bytes();
      });
}

extern "C" uint64_t tmol_backbone_torsion_workspace_bytes(
    int64_t poses, int bits, int) {
  if (poses > INT32_MAX / 2) return UINT64_MAX;
  return workspace_bytes(bits, {poses}, [&](auto real) -> uint64_t {
    return sizeof(real) == sizeof(float) ? 2 * poses * sizeof(double) : 0;
  });
}

extern "C" uint64_t tmol_backbone_torsion_precise_workspace_bytes(
    int64_t poses, int64_t atoms, int bits, int derivative) {
  if (poses > INT32_MAX / 2) return UINT64_MAX;
  return workspace_bytes(bits, {poses, atoms}, [&](auto real) -> uint64_t {
    auto values = derivative && sizeof(real) == sizeof(float)
                      ? tmol::jax::backbone_gradient_values(atoms)
                      : 0;
    return tmol::jax::BackboneWorkspace<decltype(real)>(poses, values)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_rotamer_backbone_precise_workspace_bytes(
    int64_t atoms, int bits, int derivative) {
  return workspace_bytes(bits, {atoms}, [&](auto real) -> uint64_t {
    auto values = derivative && sizeof(real) == sizeof(float)
                      ? tmol::jax::backbone_gradient_values(atoms, true)
                      : 0;
    return tmol::jax::BackboneWorkspace<decltype(real)>(0, values)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_rotamer_elec_precise_workspace_bytes(
    int64_t atoms, int bits, int derivative) {
  return workspace_bytes(bits, {atoms}, [&](auto real) -> uint64_t {
    return derivative && sizeof(real) == sizeof(float)
               ? tmol::jax::rotamer_elec_gradient_bytes(atoms)
               : 0;
  });
}

extern "C" uint64_t tmol_nonbonded_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  return workspace_bytes(bits, {poses, blocks}, [&](auto real) -> uint64_t {
    return tmol::jax::NonbondedWorkspace<decltype(real)>(poses, blocks)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_nonbonded_weighted_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  return workspace_bytes(bits, {poses, blocks}, [&](auto real) -> uint64_t {
    return tmol::jax::NonbondedWorkspace<decltype(real), true>(poses, blocks)
        .storage.size_bytes();
  });
}

namespace {
template <bool Weighted>
uint64_t precise_nonbonded_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        size_t values = 0;
        if (bits == 32 && derivative) {
          auto rows = tmol::score::common::checked_dispatch_product(
              poses, atoms, "JAX nonbonded gradient atoms");
          values = tmol::score::common::checked_dispatch_product(
              rows, (Weighted ? 1 : 4) * 3, "JAX nonbonded gradient values");
        }
        return tmol::jax::NonbondedWorkspace<decltype(real), Weighted>(
                   poses, blocks, nullptr, values)
            .storage.size_bytes();
      });
}
}  // namespace

extern "C" uint64_t tmol_nonbonded_precise_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return precise_nonbonded_workspace_bytes<false>(
      poses, blocks, atoms, bits, derivative);
}

extern "C" uint64_t tmol_nonbonded_weighted_precise_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return precise_nonbonded_workspace_bytes<true>(
      poses, blocks, atoms, bits, derivative);
}

extern "C" uint64_t tmol_cartbonded_precise_workspace_bytes(
    int64_t poses, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(bits, {poses, atoms}, [&](auto) -> uint64_t {
    return bits == 32 ? tmol::jax::CartBondedPreciseKernel::values(
                            poses, atoms, derivative)
                            * sizeof(double)
                      : 0;
  });
}

extern "C" uint64_t tmol_ljlk_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  return workspace_bytes(bits, {poses, blocks}, [&](auto real) -> uint64_t {
    return tmol::jax::IndependentNonbondedWorkspace<decltype(real), false>(
               poses, blocks)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_elec_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  return workspace_bytes(bits, {poses, blocks}, [&](auto real) -> uint64_t {
    return tmol::jax::IndependentNonbondedWorkspace<decltype(real), true>(
               poses, blocks)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_ljlk_precise_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        auto values =
            tmol::jax::IndependentNonbondedKernel<false, true>::gradient_values(
                poses, atoms, bits, derivative);
        return tmol::jax::IndependentNonbondedWorkspace<decltype(real), false>(
                   poses, blocks, nullptr, values)
            .storage.size_bytes();
      });
}

extern "C" uint64_t tmol_inverse_kinematics_workspace_bytes(
    int64_t nodes, int bits, int) {
  return workspace_bytes(bits, {nodes}, [&](auto real) -> uint64_t {
    return tmol::jax::InverseKinematicsKernel::workspace_bytes<decltype(real)>(
        nodes);
  });
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_forward_kinematics,
    tmol::jax::run_kinematic_scan<false>,
    xla::ffi::Ffi::Bind()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<tmol::jax::Generations>("generations"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_kinematic_vjp,
    tmol::jax::run_kinematic_scan<true>,
    xla::ffi::Ffi::Bind()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<tmol::jax::Generations>("generations"));

extern "C" uint64_t tmol_kinematic_scan_workspace_bytes(
    int64_t count,
    const int32_t* generations,
    int64_t generation_count,
    int bits,
    int derivative) {
  return tmol::jax::kinematic_workspace_bytes(
      count, generations, generation_count, bits, derivative);
}

extern "C" uint64_t tmol_lk_ball_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        return tmol::jax::LKBallWorkspace<decltype(real)>(
                   poses, blocks, atoms, derivative)
            .storage.size_bytes();
      });
}

namespace {
template <bool Weighted>
uint64_t precise_lk_ball_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        if constexpr (std::is_same_v<decltype(real), float>)
          return tmol::jax::LKBallWorkspace<float, Weighted, false, true>(
                     poses, blocks, atoms, derivative)
              .storage.size_bytes();
        else
          return tmol::jax::LKBallWorkspace<double, Weighted>(
                     poses, blocks, atoms, derivative)
              .storage.size_bytes();
      });
}
}  // namespace
extern "C" uint64_t tmol_lk_ball_precise_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return precise_lk_ball_workspace_bytes<false>(
      poses, blocks, atoms, bits, derivative);
}
extern "C" uint64_t tmol_lk_ball_weighted_precise_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return precise_lk_ball_workspace_bytes<true>(
      poses, blocks, atoms, bits, derivative);
}

extern "C" uint64_t tmol_genbonded_precise_workspace_bytes(
    int64_t poses, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(bits, {poses, atoms}, [&](auto real) -> uint64_t {
    if constexpr (std::is_same_v<decltype(real), float>)
      return tmol::jax::GenBondedPreciseKernel::values(poses, atoms, derivative)
             * sizeof(double);
    return 0;
  });
}

extern "C" uint64_t tmol_hbond_precise_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        return tmol::jax::HBondWorkspace<decltype(real), false, true>(
                   poses, blocks, atoms, nullptr, derivative)
            .storage.size_bytes();
      });
}

extern "C" uint64_t tmol_hbond_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        return tmol::jax::HBondWorkspace<decltype(real)>(poses, blocks, atoms)
            .storage.size_bytes();
      });
}

extern "C" uint64_t tmol_lk_ball_weighted_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        return tmol::jax::LKBallWorkspace<decltype(real), true>(
                   poses, blocks, atoms, derivative)
            .storage.size_bytes();
      });
}

extern "C" uint64_t tmol_cpu_annealer_workspace(
    int64_t poses, int64_t residues, int64_t high) {
  try {
    if (high < 4 || high > INT32_MAX) return UINT64_MAX;
    return tmol::pack::compiled::CpuFunnelWorkspace(poses, residues, high)
        .bytes;
  } catch (...) {
    return UINT64_MAX;
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_cpu_thread_count,
    tmol::jax::cpu_thread_count,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .Ret<xla::ffi::AnyBuffer>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_anneal_cpu,
    tmol::jax::run_annealer,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("chunk_size")
        .Attr<int64_t>("seed")
        .Attr<int64_t>("n_trajectories")
        .Attr<int64_t>("threads"));

extern "C" uint64_t tmol_graph_prefix_workspace_bytes(
    int64_t count, int bits, int) {
  return workspace_bytes(bits, {count}, [&](auto) -> uint64_t {
    if (count >= INT32_MAX || bits != 64) return UINT64_MAX;
    return tmol::jax::GraphPrefixWorkspace<tmol::Device::CPU>(count)
        .storage.size_bytes();
  });
}

extern "C" uint64_t tmol_rotamer_lk_ball_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int derivative) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        return tmol::jax::LKBallWorkspace<decltype(real), false, true>(
                   poses, blocks, atoms, derivative)
            .storage.size_bytes();
      });
}

extern "C" double tmol_hbond_neighbor_reach() {
  return tmol::score::hbond::potentials::hbond_neighbor_reach();
}
extern "C" uint64_t tmol_rotamer_hbond_workspace_bytes(
    int64_t poses, int64_t blocks, int64_t atoms, int bits, int) {
  return workspace_bytes(
      bits, {poses, blocks, atoms}, [&](auto real) -> uint64_t {
        if (atoms > INT32_MAX / 9) return UINT64_MAX;
        return tmol::jax::HBondWorkspace<decltype(real), true>(
                   poses, blocks, atoms)
            .storage.size_bytes();
      });
}

#include "handlers_ffi.hh"
