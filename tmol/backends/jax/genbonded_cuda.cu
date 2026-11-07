// Exact CUDA policy for bonded physics, topology and kinematics.
#include "cuda_execution_ffi.hh"

extern "C" uint64_t tmol_cuda_annealer_workspace(
    int64_t poses, int64_t residues, int64_t rotamers, int64_t edges) {
  try {
    return tmol::pack::compiled::CudaAnnealerWorkspace(
               poses, residues, rotamers, edges)
        .bytes;
  } catch (...) {
    return UINT64_MAX;
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_anneal_cuda,
    tmol::jax::anneal_cuda,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("chunk_size")
        .Attr<int64_t>("max_rotamers"),
    {xla::ffi::Traits::kCmdBufferCompatible});

extern "C" int tmol_jax_native_abi_version() { return 4; }

extern "C" int tmol_cuda_memory_info(
    int device, uint64_t* free_bytes, uint64_t* total_bytes) {
  if (!free_bytes || !total_bytes) return cudaErrorInvalidValue;
  // Reject invalid ordinals before calling cudaSetDevice: its failure leaves
  // a runtime error that can poison subsequent unrelated XLA launches.
  int count = 0;
  auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) return status;
  if (device < 0 || device >= count) return cudaErrorInvalidDevice;
  int previous = -1;
  status = cudaGetDevice(&previous);
  if (status != cudaSuccess) return status;
  status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    cudaSetDevice(previous);
    return status;
  }
  size_t available = 0, capacity = 0;
  status = cudaMemGetInfo(&available, &capacity);
  auto restored = cudaSetDevice(previous);
  if (status != cudaSuccess) return status;
  if (restored != cudaSuccess) return restored;
  *free_bytes = available;
  *total_bytes = capacity;
  return cudaSuccess;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_forward_kinematics,
    tmol::jax::run_kinematic_scan<false>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<tmol::jax::Generations>("generations"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_kinematic_vjp,
    tmol::jax::run_kinematic_scan<true>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_sampling_plan,
    tmol::jax::run_sampling<false>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<int64_t>("count"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_sampling_emit,
    tmol::jax::run_sampling<true>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<int64_t>("count"));
extern "C" uint64_t tmol_sampling_workspace_bytes(
    int64_t count, int64_t segments, int bits, int emit) {
  try {
    if (bits == 32)
      return tmol::jax::SamplingWorkspace<float>(count, segments, emit)
          .storage.size_bytes();
    if (bits == 64)
      return tmol::jax::SamplingWorkspace<double>(count, segments, emit)
          .storage.size_bytes();
  } catch (...) {
  }
  return UINT64_MAX;
}

extern "C" uint64_t tmol_load_balance_workspace(
    int total, int generators, int device) {
  int previous = -1;
  try {
    if (cudaGetDevice(&previous) != cudaSuccess
        || cudaSetDevice(device) != cudaSuccess)
      return UINT64_MAX;
    tmol::jax::KinematicStream context(nullptr);
    auto size =
        tmol::score::common::load_balancing_search_workspace_size<int64_t>(
            total, generators, context.ptx_version());
    if (cudaSetDevice(previous) != cudaSuccess) return UINT64_MAX;
    return size;
  } catch (...) {
    if (previous >= 0) cudaSetDevice(previous);
    return UINT64_MAX;
  }
}

#define TMOL_JAX_EXCLUDE_NONBONDED
#include "handlers_ffi.hh"
