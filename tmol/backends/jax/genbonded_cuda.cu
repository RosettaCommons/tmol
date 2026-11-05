// CUDA stream/workgroup policy for the same whole-pose kernels used by Torch.
#include <cuda_runtime_api.h>
#include <moderngpu/cta_reduce.hxx>
#include "genbonded_ffi.hh"

namespace tmol::jax {
namespace {
thread_local cudaStream_t active_stream = nullptr;

template <typename F>
__global__ void pose_workgroups(F f) {
  f(blockIdx.x);
}
template <Device D>
struct CudaOps;
template <>
struct CudaOps<Device::CUDA> {
  template <typename Launch, typename F>
  static void foreach_pose_workgroup(
      ContextManager&, int poses, int groups, F f) {
    int count = score::common::checked_dispatch_product(
        poses, groups, "JAX CUDA generic bonded");
    if (count)
      pose_workgroups<<<count, Launch::sm_ptx::nt, 0, active_stream>>>(f);
  }
  template <int N, typename F>
  __device__ static void for_each_in_workgroup(F f) {
    f(threadIdx.x);
  }
  template <int N, typename T, typename S, typename Op>
  __device__ static T reduce_in_workgroup(T value, S& shared, Op op) {
    return mgpu::cta_reduce_t<N, T>().reduce(
        threadIdx.x, value, shared.reduce, N, op);
  }
};

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

ffi::Error run(
    cudaStream_t stream,
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> energy,
    ffi::RemainingRets rets) {
  auto previous = active_stream;
  try {
    auto buffers = genbonded_arguments(args);
    if (rets.size() > 1)
      throw std::invalid_argument("too many generic bonded results");
    std::optional<ffi::AnyBuffer> gradient;
    if (rets.size()) {
      auto value = rets.get<ffi::AnyBuffer>(0);
      if (!value)
        throw std::invalid_argument("invalid generic bonded gradient buffer");
      gradient = **value;
    }
    validate_genbonded(buffers, *energy, gradient ? &*gradient : nullptr);
    check_cuda(cudaMemsetAsync(
        energy->untyped_data(), 0, energy->size_bytes(), stream));
    if (gradient)
      check_cuda(cudaMemsetAsync(
          gradient->untyped_data(), 0, gradient->size_bytes(), stream));
    active_stream = stream;
    if (buffers[0].element_type() == ffi::DataType::F32) {
      execute_genbonded<CudaOps, Device::CUDA, float>(
          buffers, *energy, gradient ? &*gradient : nullptr);
    } else {
      execute_genbonded<CudaOps, Device::CUDA, double>(
          buffers, *energy, gradient ? &*gradient : nullptr);
    }
    active_stream = previous;
    check_cuda(cudaPeekAtLastError());
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    active_stream = previous;
    return ffi::Error::Internal(error.what());
  }
}
}  // namespace
}  // namespace tmol::jax

extern "C" int tmol_jax_native_abi_version() { return 1; }
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_genbonded,
    tmol::jax::run,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());
