// CPU execution policy and XLA registration; science lives in the common
// kernel.
#include "cartbonded_ffi.hh"
#include "backbone_torsion_ffi.hh"
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
  template <int N, typename F>
  static void for_each_in_workgroup(F f) {
    static_assert(N == 1, "CPU workgroup must have one lane");
    f(0);
  }
  template <int N, typename T, typename S, typename Op>
  static T reduce_in_workgroup(T value, S&, Op) {
    return value;
  }
};

template <typename Kernel>
ffi::Error run(
    ffi::ThreadPool pool,
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> energy,
    ffi::RemainingRets rets) {
  auto* previous = active_pool;
  try {
    auto buffers = native_arguments(args);
    if (rets.size() > 1)
      throw std::invalid_argument("too many generic bonded results");
    std::optional<ffi::AnyBuffer> gradient;
    if (rets.size()) {
      auto value = rets.get<ffi::AnyBuffer>(0);
      if (!value)
        throw std::invalid_argument("invalid generic bonded gradient buffer");
      gradient = **value;
    }
    Kernel::validate(buffers, *energy, gradient ? &*gradient : nullptr);
    std::memset(energy->untyped_data(), 0, energy->size_bytes());
    if (gradient)
      std::memset(gradient->untyped_data(), 0, gradient->size_bytes());
    active_pool = &pool;
    if (buffers[0].element_type() == ffi::DataType::F32) {
      Kernel::template execute<CpuOps, Device::CPU, float>(
          buffers, *energy, gradient ? &*gradient : nullptr);
    } else {
      Kernel::template execute<CpuOps, Device::CPU, double>(
          buffers, *energy, gradient ? &*gradient : nullptr);
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
