// CUDA stream/workgroup policy for the same whole-pose kernels used by Torch.
#include <cuda_runtime_api.h>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/loadstore.hxx>
#include "cartbonded_ffi.hh"
#include "backbone_torsion_ffi.hh"
#include "metal_ffi.hh"
#include "na_torsion_ffi.hh"
#include "workspace_ffi.hh"
#include "lk_ball_ffi.hh"
#include "hbond_ffi.hh"
#include "disulfide_ffi.hh"
#include "dunbrack_ffi.hh"
#include "nonbonded_ffi.hh"
#include "independent_nonbonded_ffi.hh"
#include "constraint_torsion_ffi.hh"
#include "kinematics_ffi.hh"
#include "conformer_ffi.hh"
#include "graph_topology_ffi.hh"
#include "graph_retention_ffi.hh"
#include "rotamer_dispatch_ffi.hh"
#include "rotamer_nonbonded_ffi.hh"
#include "graph_prefix_ffi.hh"
#include "forward_kinematics_ffi.hh"
#include "sampling_ffi.hh"

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
  template <portable::scan_type Scan, typename T, typename I, typename Op>
  static void segmented_scan_into(
      ContextManager&,
      T* source,
      I* starts,
      T* destination,
      int count,
      int segments,
      Op op,
      T identity,
      score::common::SegmentedScanWorkspace<T, Device::CUDA> workspace) {
    KinematicStream context(active_stream);
    score::common::segmented_scan_into_cuda<Scan>(
        context,
        source,
        starts,
        destination,
        count,
        segments,
        op,
        identity,
        workspace);
  }

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

// Sampling reuses the canonical segmented scan for both segmented sums and
// the single-segment maximum. XLA owns scratch, and every launch uses its
// stream.
template <Device D>
struct SamplingCudaOps : CudaOps<D> {
  template <portable::scan_type Scan, typename T, typename I, typename Op>
  static void segmented_scan_into(
      SamplingContext& context,
      T* source,
      I* starts,
      T* dest,
      int count,
      int segments,
      Op op,
      T identity,
      score::common::SegmentedScanWorkspace<T, D> workspace) {
    score::common::segmented_scan_into_cuda<Scan>(
        context.execution,
        source,
        starts,
        dest,
        count,
        segments,
        op,
        identity,
        workspace);
  }
  template <portable::scan_type Scan, typename Op>
  static void scan(
      SamplingContext& context,
      int32_t* source,
      int32_t* dest,
      int count,
      Op op) {
    static_assert(Scan == portable::scan_type::inclusive);
    segmented_scan_into<Scan>(
        context,
        source,
        context.starts,
        dest,
        count,
        1,
        op,
        int32_t(0),
        context.scan_workspace);
  }
};

template <bool Emit>
ffi::Error run_sampling(
    cudaStream_t stream,
    ffi::RemainingArgs arguments,
    ffi::Result<ffi::AnyBuffer> first,
    ffi::RemainingRets remaining,
    int64_t count) {
  try {
    active_stream = stream;
    auto inputs = native_arguments(arguments);
    std::vector<ffi::AnyBuffer> outputs{*first};
    for (size_t i = 0; i < remaining.size(); ++i) {
      auto result = remaining.get<ffi::AnyBuffer>(i);
      if (!result)
        throw std::invalid_argument("sampling output is not a buffer");
      outputs.push_back(**result);
    }
    if (inputs.empty()) throw std::invalid_argument("missing sampling inputs");
    if (inputs[0].element_type() == ffi::DataType::F32)
      execute_sampling<SamplingCudaOps, float, Emit>(
          stream, inputs, outputs, count);
    else if (inputs[0].element_type() == ffi::DataType::F64)
      execute_sampling<SamplingCudaOps, double, Emit>(
          stream, inputs, outputs, count);
    else
      throw std::invalid_argument("sampling coordinates require float32/64");
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::InvalidArgument(error.what());
  }
}

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
template <typename Stage, typename... Attributes>
ffi::Error run_graph_stage(
    cudaStream_t stream,
    ffi::RemainingArgs args,
    ffi::RemainingRets rets,
    int64_t operation,
    int64_t extent,
    int64_t width,
    Attributes... attributes) {
  auto previous = active_stream;
  try {
    auto inputs = native_arguments(args);
    auto outputs = native_results(rets);
    Stage::validate(inputs, outputs, operation, extent, width, attributes...);
    for (size_t i = 0; i < outputs.size(); ++i) {
      int fill = Stage::fill(operation, i);
      auto result = outputs[i];
      if (fill >= 0 && result.size_bytes())
        check_cuda(cudaMemsetAsync(
            result.untyped_data(), fill, result.size_bytes(), stream));
    }
    active_stream = stream;
    Stage::template execute<CudaOps, Device::CUDA>(
        inputs, outputs, operation, extent, width, attributes...);
    active_stream = previous;
    check_cuda(cudaPeekAtLastError());
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    active_stream = previous;
    return ffi::Error::InvalidArgument(error.what());
  }
}
}  // namespace
}  // namespace tmol::jax

extern "C" int tmol_jax_native_abi_version() { return 2; }
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_lk_ball,
    (tmol::jax::run<tmol::jax::LKBallKernel, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_disulfide,
    tmol::jax::run<tmol::jax::DisulfideKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_hbond,
    tmol::jax::run<tmol::jax::HBondKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_lk_ball_weighted,
    (tmol::jax::run<tmol::jax::LKBallWeightedKernel, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_copy_conformer_dofs,
    (tmol::jax::run<tmol::jax::CopyConformerDofsKernel, int64_t>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<int64_t>("sampler"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_assign_conformer_chi,
    tmol::jax::run<tmol::jax::AssignConformerChiKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_copy_conformer_rows,
    (tmol::jax::run<tmol::jax::CopyConformerRowsKernel, int64_t, int64_t>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets()
        .Attr<int64_t>("n_rows")
        .Attr<int64_t>("root_shift"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_correct_conformer_chi,
    tmol::jax::run<tmol::jax::CorrectConformerChiKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_graph_topology,
    tmol::jax::run_graph_stage<tmol::jax::GraphTopologyStage>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("operation")
        .Attr<int64_t>("extent")
        .Attr<int64_t>("width"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_graph_retention,
    tmol::jax::run_graph_stage<tmol::jax::GraphRetentionStage>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("operation")
        .Attr<int64_t>("extent")
        .Attr<int64_t>("width"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_rotamer_dispatch,
    (tmol::jax::run_graph_stage<tmol::jax::RotamerDispatchStage, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("operation")
        .Attr<int64_t>("extent")
        .Attr<int64_t>("width")
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_rotamer_elec,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerNonbondedStage<true>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("operation")
        .Attr<int64_t>("extent")
        .Attr<int64_t>("width")
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_rotamer_ljlk,
    (tmol::jax::
         run_graph_stage<tmol::jax::RotamerNonbondedStage<false>, double>),
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("operation")
        .Attr<int64_t>("extent")
        .Attr<int64_t>("width")
        .Attr<double>("reach"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tmol_graph_prefix,
    tmol::jax::run<tmol::jax::GraphPrefixKernel>,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .RemainingArgs()
        .Ret<xla::ffi::AnyBuffer>()
        .RemainingRets(),
    {xla::ffi::Traits::kCmdBufferCompatible});
