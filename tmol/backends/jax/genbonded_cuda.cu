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
#include "independent_nonbonded_ffi.hh"
#include "constraint_torsion_ffi.hh"
#include "kinematics_ffi.hh"
#include "forward_kinematics_ffi.hh"

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
  template <int N, int Width, typename T, typename U>
  __device__ static void copy_contiguous_data_and_cast(
      T* destination,
      const U* source,
      int count,
      score::common::WorkgroupLane<Device::CUDA> = {}) {
    for (int i = threadIdx.x; i < count; i += N)
      destination[i] = static_cast<T>(source[i]);
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

// Sparse whole-pose Dunbrack work otherwise occupies a single CUDA block.
// Keep its independent residue evaluations on separate SM workgroups; the
// common kernel owns all geometry, table lookup and analytic derivatives.
template <Device D>
struct DunbrackCudaOps : CudaOps<D> {
  template <typename Launch, typename F>
  static void forall_grouped(ContextManager&, int poses, int groups, F f) {
    int count = score::common::checked_dispatch_product(
        poses, groups, "JAX CUDA Dunbrack");
    // Small batches need many workgroups; large batches fill active warp lanes.
    int threads = count >= 2048 ? 16 : 1;
    if (count)
      independent_work<<<
          (count + threads - 1) / threads,
          threads,
          0,
          active_stream>>>(count, f);
  }
};

template <typename Kernel>
struct KernelDispatch {
  template <Device D>
  using Ops = CudaOps<D>;
};
template <>
struct KernelDispatch<DunbrackKernel> {
  template <Device D>
  using Ops = DunbrackCudaOps<D>;
};

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Output buffers are separate XLA allocations. Clear them in one stream
// operation while retaining cudaMemsetAsync for a single buffer.
__global__ void zero_native_buffers(
    uint32_t* a,
    size_t na,
    uint32_t* b,
    size_t nb,
    uint32_t* c,
    size_t nc,
    size_t count) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    if (i < na) a[i] = 0;
    if (i < nb) b[i] = 0;
    if (i < nc) c[i] = 0;
  }
}

void zero_outputs(
    cudaStream_t stream,
    ffi::AnyBuffer energy,
    const NativeOutputs& outputs,
    bool clear_energy,
    bool clear_workspace) {
  void* buffers[] = {
      energy.untyped_data(),
      outputs.gradient ? outputs.gradient->untyped_data() : nullptr,
      outputs.workspace ? outputs.workspace->untyped_data() : nullptr};
  size_t bytes[] = {
      clear_energy ? energy.size_bytes() : 0,
      outputs.gradient ? outputs.gradient->size_bytes() : 0,
      clear_workspace && outputs.workspace ? outputs.workspace->size_bytes()
                                           : 0};
  int present = 0;
  size_t count = 0;
  for (int i = 0; i != 3; ++i) {
    present += bytes[i] != 0;
    count = std::max(count, bytes[i] / sizeof(uint32_t));
    if (bytes[i] % sizeof(uint32_t))
      throw std::invalid_argument(
          "native zero buffer must contain whole words");
  }
  if (present < 2) {
    for (int i = 0; i != 3; ++i)
      if (bytes[i])
        check_cuda(cudaMemsetAsync(buffers[i], 0, bytes[i], stream));
  } else {
    int blocks = int(std::min(size_t(1024), (count + 255) / 256));
    zero_native_buffers<<<blocks, 256, 0, stream>>>(
        static_cast<uint32_t*>(buffers[0]),
        bytes[0] / sizeof(uint32_t),
        static_cast<uint32_t*>(buffers[1]),
        bytes[1] / sizeof(uint32_t),
        static_cast<uint32_t*>(buffers[2]),
        bytes[2] / sizeof(uint32_t),
        count);
  }
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
    zero_outputs(
        stream,
        *energy,
        outputs,
        ZeroEnergy<Kernel>::value,
        ZeroWorkspace<Kernel>::value);
    active_stream = stream;
    if (buffers[0].element_type() == ffi::DataType::F32) {
      execute_native<
          Kernel,
          KernelDispatch<Kernel>::template Ops,
          Device::CUDA,
          float>(buffers, *energy, outputs, attributes...);
    } else {
      execute_native<
          Kernel,
          KernelDispatch<Kernel>::template Ops,
          Device::CUDA,
          double>(buffers, *energy, outputs, attributes...);
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
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_cartbonded,
    tmol::jax::run<tmol::jax::CartBondedKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_backbone_torsion,
    tmol::jax::run<tmol::jax::BackboneTorsionKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_metal_coordination,
    tmol::jax::run<tmol::jax::MetalCoordinationKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_na_torsion,
    (tmol::jax::run<tmol::jax::NaTorsionKernel, double, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("pucker_temperature")
        .Attr<double>("bin_blend_sdev"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_dunbrack,
    tmol::jax::run<tmol::jax::DunbrackKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_nonbonded,
    (tmol::jax::run<tmol::jax::NonbondedKernel<false>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_nonbonded_weighted,
    (tmol::jax::run<tmol::jax::NonbondedKernel<true>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_constraint_torsion,
    tmol::jax::run<tmol::jax::ConstraintTorsionKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_ljlk,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<false>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_elec,
    (tmol::jax::run<tmol::jax::IndependentNonbondedKernel<true>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_inverse_kinematics,
    tmol::jax::run<tmol::jax::InverseKinematicsKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

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
