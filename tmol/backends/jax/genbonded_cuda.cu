// CUDA stream/workgroup policy for the same whole-pose kernels used by Torch.
#include <cuda_runtime_api.h>
#include <moderngpu/cta_reduce.hxx>
#include "cartbonded_ffi.hh"
#include "backbone_torsion_ffi.hh"
#include "metal_ffi.hh"

namespace tmol::jax {
namespace {
thread_local cudaStream_t active_stream = nullptr;

template <typename F>
__global__ void pose_workgroups(F f) {
  f(blockIdx.x);
}
template <typename F>
__global__ void independent_work(int count, F f) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) f(index);
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
  template <typename Launch, typename F>
  static void forall_grouped(
      ContextManager& context, int poses, int groups, F f) {
    int count = score::common::checked_dispatch_product(
        poses, groups, "JAX CUDA backbone");
    if (count)
      independent_work<<<(count + 127) / 128, 128, 0, active_stream>>>(
          count, f);
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

template <typename Kernel>
ffi::Error run(
    cudaStream_t stream,
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> energy,
    ffi::RemainingRets rets) {
  auto previous = active_stream;
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
    check_cuda(cudaMemsetAsync(
        energy->untyped_data(), 0, energy->size_bytes(), stream));
    if (gradient)
      check_cuda(cudaMemsetAsync(
          gradient->untyped_data(), 0, gradient->size_bytes(), stream));
    active_stream = stream;
    if (buffers[0].element_type() == ffi::DataType::F32) {
      Kernel::template execute<CudaOps, Device::CUDA, float>(
          buffers, *energy, gradient ? &*gradient : nullptr);
    } else {
      Kernel::template execute<CudaOps, Device::CUDA, double>(
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
    tmol::jax::run<tmol::jax::GenBondedKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_cartbonded,
    tmol::jax::run<tmol::jax::CartBondedKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_backbone_torsion,
    tmol::jax::run<tmol::jax::BackboneTorsionKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_metal_coordination,
    tmol::jax::run<tmol::jax::MetalCoordinationKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());
