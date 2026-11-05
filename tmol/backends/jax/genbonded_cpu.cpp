// CPU execution policy and XLA registration; science lives in the common
// kernel.
#include "cartbonded_ffi.hh"
#include "backbone_torsion_ffi.hh"
#include "metal_ffi.hh"
#include "na_torsion_ffi.hh"
#include "workspace_ffi.hh"
#include "dunbrack_ffi.hh"
#include "nonbonded_ffi.hh"
#include "independent_nonbonded_ffi.hh"
#include "constraint_torsion_ffi.hh"
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <tmol/numeric/bspline_compiled/bspline_coeffs.hh>

namespace tmol::jax {
namespace {
thread_local ffi::ThreadPool* active_pool = nullptr;

template <Device D>
struct CpuOps;
template <>
struct CpuOps<Device::CPU> {
  template <typename Launch, typename F>
  static void foreach_pose_workgroup(
      ContextManager&, int poses, int groups, F f) {
    auto run = [=](int pose) {
      for (int group = 0; group < groups; ++group) f(pose * groups + group);
    };
    if (poses <= 1 || !active_pool || active_pool->num_threads() <= 1) {
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
    foreach_independent_workgroup<Launch>(context, count, f);
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
}  // namespace
}  // namespace tmol::jax

extern "C" int tmol_jax_native_abi_version() { return 1; }
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_genbonded,
    tmol::jax::run<tmol::jax::GenBondedKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_cartbonded,
    tmol::jax::run<tmol::jax::CartBondedKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_backbone_torsion,
    tmol::jax::run<tmol::jax::BackboneTorsionKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_metal_coordination,
    tmol::jax::run<tmol::jax::MetalCoordinationKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_na_torsion,
    (tmol::jax::run<tmol::jax::NaTorsionKernel, double, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("pucker_temperature")
        .Attr<double>("bin_blend_sdev"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_dunbrack,
    tmol::jax::run<tmol::jax::DunbrackKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

extern "C" uint64_t tmol_dunbrack_workspace_bytes(
    int64_t rots, int64_t dihedrals, int bits, int derivative) {
  try {
    if (rots < 0 || dihedrals < 0 || rots > INT32_MAX || dihedrals > INT32_MAX)
      throw std::invalid_argument("invalid Dunbrack workspace extents");
    if (bits == 32)
      return tmol::jax::DunbrackWorkspace<float>(rots, dihedrals, derivative)
          .storage.size_bytes();
    if (bits == 64)
      return tmol::jax::DunbrackWorkspace<double>(rots, dihedrals, derivative)
          .storage.size_bytes();
  } catch (const std::exception&) {
  }
  return std::numeric_limits<uint64_t>::max();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_nonbonded,
    (tmol::jax::run<tmol::jax::NonbondedKernel<false>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"));

extern "C" uint64_t tmol_nonbonded_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  try {
    if (poses < 0 || blocks < 0 || poses > INT32_MAX || blocks > INT32_MAX)
      throw std::invalid_argument("invalid nonbonded workspace extents");
    if (bits == 32)
      return tmol::jax::NonbondedWorkspace<float>(poses, blocks)
          .storage.size_bytes();
    if (bits == 64)
      return tmol::jax::NonbondedWorkspace<double>(poses, blocks)
          .storage.size_bytes();
  } catch (const std::exception&) {
  }
  return std::numeric_limits<uint64_t>::max();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_nonbonded_weighted,
    (tmol::jax::run<tmol::jax::NonbondedKernel<true>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"));

extern "C" uint64_t tmol_nonbonded_weighted_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  try {
    if (poses < 0 || blocks < 0 || poses > INT32_MAX || blocks > INT32_MAX)
      throw std::invalid_argument("invalid nonbonded workspace extents");
    if (bits == 32)
      return tmol::jax::NonbondedWorkspace<float, true>(poses, blocks)
          .storage.size_bytes();
    if (bits == 64)
      return tmol::jax::NonbondedWorkspace<double, true>(poses, blocks)
          .storage.size_bytes();
  } catch (const std::exception&) {
  }
  return std::numeric_limits<uint64_t>::max();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_constraint_torsion,
    tmol::jax::run<tmol::jax::ConstraintTorsionKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_ljlk,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<false>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"));

extern "C" uint64_t tmol_ljlk_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  try {
    if (poses < 0 || blocks < 0 || (bits != 32 && bits != 64))
      throw std::invalid_argument("invalid independent nonbonded workspace");
    return bits == 32 ? tmol::jax::IndependentNonbondedWorkspace<float, false>(
                            poses, blocks)
                            .storage.size_bytes()
                      : tmol::jax::IndependentNonbondedWorkspace<double, false>(
                            poses, blocks)
                            .storage.size_bytes();
  } catch (...) {
    return std::numeric_limits<uint64_t>::max();
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_elec,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<true>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::ThreadPool>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"));

extern "C" uint64_t tmol_elec_workspace_bytes(
    int64_t poses, int64_t blocks, int bits, int) {
  try {
    if (poses < 0 || blocks < 0 || (bits != 32 && bits != 64))
      throw std::invalid_argument("invalid independent nonbonded workspace");
    return bits == 32 ? tmol::jax::IndependentNonbondedWorkspace<float, true>(
                            poses, blocks)
                            .storage.size_bytes()
                      : tmol::jax::IndependentNonbondedWorkspace<double, true>(
                            poses, blocks)
                            .storage.size_bytes();
  } catch (...) {
    return std::numeric_limits<uint64_t>::max();
  }
}
