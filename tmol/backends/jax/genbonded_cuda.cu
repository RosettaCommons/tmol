// CUDA stream/workgroup policy for the same whole-pose kernels used by Torch.
#include <cuda_runtime_api.h>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/loadstore.hxx>
#include "cartbonded_ffi.hh"
#include "backbone_torsion_ffi.hh"
#include "metal_ffi.hh"
#include "na_torsion_ffi.hh"
#include "workspace_ffi.hh"
#include "dunbrack_ffi.hh"
#include "nonbonded_ffi.hh"

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
  template <typename Launch, typename F>
  static void foreach_workgroup(ContextManager&, int count, F f) {
    if (count)
      pose_workgroups<<<count, Launch::sm_ptx::nt, 0, active_stream>>>(f);
  }
  template <typename Launch, typename F>
  static void foreach_independent_workgroup(
      ContextManager& context, int count, F f) {
    foreach_workgroup<Launch>(context, count, f);
  }
  template <typename Launch, typename F>
  static void forall(ContextManager& context, int count, F f) {
    forall_grouped<Launch>(context, count, 1, f);
  }
  template <typename Launch, typename F>
  static void forall_independent(ContextManager& context, int count, F f) {
    forall<Launch>(context, count, f);
  }
  template <int N, int Width, typename T>
  __device__ static void copy_contiguous_data(
      T* destination,
      T* source,
      int count,
      score::common::WorkgroupLane<Device::CUDA> = {}) {
    mgpu::mem_to_shared<N, Width>(
        source, threadIdx.x, count, destination, false);
  }
  __device__ static void synchronize_workgroup(
      score::common::WorkgroupLane<Device::CUDA> = {}) {
    __syncthreads();
  }
  template <int N, typename T, typename Op>
  __device__ static T shuffle_reduce_in_workgroup(
      T value, Op op, score::common::WorkgroupLane<Device::CUDA> = {}) {
    static_assert(N <= 32);
    auto group = cooperative_groups::coalesced_threads();
    return score::common::reduce<Device::CUDA, T>::reduce_to_head(
        group, value, op);
  }
  template <int N, typename T, typename Op>
  __device__ static T shuffle_reduce_and_broadcast_in_workgroup(
      T value, Op op, score::common::WorkgroupLane<Device::CUDA> = {}) {
    static_assert(N <= 32);
    auto group = cooperative_groups::coalesced_threads();
    return score::common::reduce<Device::CUDA, T>::reduce_to_all(
        group, value, op);
  }
  template <int N, typename F>
  __device__ static void for_each_in_workgroup(
      F f, score::common::WorkgroupLane<Device::CUDA> = {}) {
    f(threadIdx.x);
  }
  template <int N, typename T, typename S, typename Op>
  __device__ static T reduce_in_workgroup(
      T value,
      S& shared,
      Op op,
      score::common::WorkgroupLane<Device::CUDA> = {}) {
    return mgpu::cta_reduce_t<N, T>().reduce(
        threadIdx.x, value, shared.reduce, N, op);
  }
};

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename Kernel, typename... Attributes>
ffi::Error run(
    cudaStream_t stream,
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> energy,
    ffi::RemainingRets rets,
    Attributes... attributes) {
  auto previous = active_stream;
  try {
    auto buffers = native_arguments(args);
    auto outputs = native_outputs<Kernel>(buffers, *energy, rets);
    const auto& gradient = outputs.gradient;
    check_cuda(cudaMemsetAsync(
        energy->untyped_data(), 0, energy->size_bytes(), stream));
    if (gradient)
      check_cuda(cudaMemsetAsync(
          gradient->untyped_data(), 0, gradient->size_bytes(), stream));
    if (outputs.workspace)
      check_cuda(cudaMemsetAsync(
          outputs.workspace->untyped_data(),
          0,
          outputs.workspace->size_bytes(),
          stream));
    active_stream = stream;
    if (buffers[0].element_type() == ffi::DataType::F32) {
      execute_native<Kernel, CudaOps, Device::CUDA, float>(
          buffers, *energy, outputs, attributes...);
    } else {
      execute_native<Kernel, CudaOps, Device::CUDA, double>(
          buffers, *energy, outputs, attributes...);
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_na_torsion,
    (tmol::jax::run<tmol::jax::NaTorsionKernel, double, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("pucker_temperature")
        .Attr<double>("bin_blend_sdev"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_dunbrack,
    tmol::jax::run<tmol::jax::DunbrackKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_nonbonded,
    (tmol::jax::run<tmol::jax::NonbondedKernel<false>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_nonbonded_weighted,
    (tmol::jax::run<tmol::jax::NonbondedKernel<true>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"));
