// Typed JAX execution for the shared forward scan and its analytic VJP.
#pragma once
#include "workspace_ffi.hh"
#ifdef __CUDACC__
#include <tmol/kinematics/compiled/native_kinematics_cuda.cuh>
#include <map>
#else
#include <tmol/kinematics/compiled/native_kinematics_cpu.hh>
#endif

namespace tmol::jax {
using Generations = ffi::Span<const int32_t>;
using HostGenerations =
    TView<kinematics::KinForestGenData<int32_t>, 1, Device::CPU>;

inline HostGenerations generation_view(Generations generations) {
  if (generations.size() < 2 || generations.size() % 2)
    throw std::invalid_argument(
        "kinematic generations require pairs and a sentinel");
  if (generations[0] || generations[1])
    throw std::invalid_argument("kinematic generations must start at zero");
  for (size_t i = 2; i < generations.size(); ++i)
    if (generations[i] < generations[i - 2])
      throw std::invalid_argument("kinematic generations must be monotonic");
  return contiguous_view<kinematics::KinForestGenData<int32_t>, 1, Device::CPU>(
      const_cast<int32_t*>(generations.begin()),
      {int64_t(generations.size() / 2)});
}

#ifdef __CUDACC__
// The shared scans use caller-owned scratch. This context supplies the XLA
// stream and device launch metadata, without allocation, events or blocking.
class KinematicStream final : public mgpu::context_t {
  struct Metadata {
    cudaDeviceProp props;
    int ptx;
  };
  cudaStream_t stream_;
  const Metadata* metadata_;
  static void check(cudaError_t status) {
    if (status != cudaSuccess) throw mgpu::cuda_exception_t(status);
  }
  [[noreturn]] static void unsupported() {
    throw std::logic_error("kinematic FFI cannot allocate or synchronize");
  }

 public:
  explicit KinematicStream(cudaStream_t stream) : stream_(stream) {
    int device;
    check(cudaGetDevice(&device));
    static thread_local std::map<int, Metadata> metadata;
    auto found = metadata.find(device);
    if (found == metadata.end()) {
      Metadata value;
      cudaFuncAttributes attributes;
      check(cudaGetDeviceProperties(&value.props, device));
      check(cudaFuncGetAttributes(&attributes, mgpu::dummy_k<0>));
      value.ptx = attributes.ptxVersion;
      found = metadata.emplace(device, value).first;
    }
    metadata_ = &found->second;
  }
  const cudaDeviceProp& props() const override { return metadata_->props; }
  int ptx_version() const override { return metadata_->ptx; }
  cudaStream_t stream() override { return stream_; }
  void* alloc(size_t, mgpu::memory_space_t) override { unsupported(); }
  void free(void*, mgpu::memory_space_t) override { unsupported(); }
  void synchronize() override { unsupported(); }
  cudaEvent_t event() override { unsupported(); }
  void timer_begin() override { unsupported(); }
  double timer_end() override { unsupported(); }
};
using KinematicContext = cudaStream_t;
constexpr Device KinematicDevice = Device::CUDA;
#else
struct KinematicContext {};
constexpr Device KinematicDevice = Device::CPU;
#endif

template <typename Real, bool Derivative>
struct KinematicWorkspace {
  using Sum = Eigen::Matrix<Real, 6, 1>;
  using ScanValue =
      std::conditional_t<Derivative, Sum, kinematics::KinTransform<Real>>;
  ByteWorkspace storage;
  Sum* forces = nullptr;
  ScanValue *carryout = nullptr, *scan = nullptr;
  int32_t *codes = nullptr, *lbs = nullptr;
  int32_t scan_count = 0, carry_count = 0, lbs_count = 0;
  KinematicWorkspace(int64_t nodes, HostGenerations gens, void* data = nullptr)
      : storage(data) {
    if (nodes < 0 || nodes > INT32_MAX)
      throw std::invalid_argument("kinematic node count exceeds int32");
    if constexpr (Derivative) forces = storage.take<Sum>(nodes);
#ifdef __CUDACC__
    auto sizes = kinematics::getScanBufferSize<int32_t>(
        gens, Derivative ? 256 : 128, Derivative ? 3 : 2);
    scan_count = mgpu::get<0>(sizes);
    carry_count = mgpu::get<1>(sizes);
    lbs_count = mgpu::get<2>(sizes);
    carryout = storage.take<ScanValue>(carry_count);
    codes = storage.take<int32_t>(carry_count);
    lbs = storage.take<int32_t>(lbs_count);
    scan = storage.take<ScanValue>(scan_count);
#endif
  }
};

template <typename Real, bool Derivative>
void execute_kinematic_scan(
    KinematicContext context,
    const std::vector<ffi::AnyBuffer>& args,
    ffi::AnyBuffer result,
    ffi::RemainingRets rets,
    Generations generations) {
  constexpr Device D = KinematicDevice;
  using I = int32_t;
  using Transform = kinematics::KinTransform<Real>;
  using Dofs = kinematics::KinDofs<Real>;
  using Coord = kinematics::KinCoord<Real>;
  using Forest = kinematics::KinForestParams<I>;
  using Work = KinematicWorkspace<Real, Derivative>;
  constexpr int offset = Derivative ? 2 : 0;
  if (args.size() != 4 + offset || rets.size() != (Derivative ? 1 : 2))
    throw std::invalid_argument(
        "incorrect kinematic FFI argument or result count");
  auto real = result.element_type();
  require_shape(args[offset], real, {-1, 9});
  auto count = args[offset].dimensions()[0];
  require_shape(args[offset + 1], ffi::DataType::S32, {count, 6});
  require_shape(args[offset + 2], ffi::DataType::S32, {-1});
  require_shape(args[offset + 3], ffi::DataType::S32, {-1});
  auto gens = generation_view(generations);
  auto last = gens[gens.size(0) - 1];
  if (last.node_start != args[offset + 2].dimensions()[0]
      || last.scan_start != args[offset + 3].dimensions()[0])
    throw std::invalid_argument(
        "kinematic generation endpoints disagree with plan");
  auto output = [&](int i) {
    auto value = rets.get<ffi::AnyBuffer>(i);
    if (!value)
      throw std::invalid_argument("kinematic FFI expected output buffer");
    return **value;
  };
  auto workspace = output(Derivative ? 0 : 1);
  Work work(count, gens, workspace.untyped_data());
  require_shape(
      workspace, ffi::DataType::U8, {int64_t(work.storage.size_bytes())});
  if (workspace.size_bytes()
      && reinterpret_cast<uintptr_t>(workspace.untyped_data())
             % alignof(typename Work::ScanValue))
    throw std::invalid_argument("kinematic workspace is misaligned");
  auto dofs = buffer_view<Dofs, 1, D>(args[offset]);
  auto tree = buffer_view<Forest, 1, D>(args[offset + 1]);
  auto nodes = buffer_view<I, 1, D>(args[offset + 2]);
  auto scans = buffer_view<I, 1, D>(args[offset + 3]);
  auto transforms = Derivative ? args[1] : output(0);
  require_shape(transforms, real, {count, 16});
  auto hts = buffer_view<Transform, 1, D>(transforms);
  if constexpr (Derivative) {
    require_shape(args[0], real, {count, 3});
    require_shape(result, real, {count, 9});
  } else {
    require_shape(result, real, {count, 3});
  }
  if (!count) return;
#ifdef __CUDACC__
  KinematicStream stream(context);
  auto carryout = contiguous_view<typename Work::ScanValue, 1, D>(
      work.carryout, {work.carry_count});
  auto codes = contiguous_view<I, 1, D>(work.codes, {work.carry_count});
  auto lbs = contiguous_view<I, 1, D>(work.lbs, {work.lbs_count});
  auto scan = contiguous_view<typename Work::ScanValue, 1, D>(
      work.scan, {work.scan_count});
#endif
  if constexpr (Derivative) {
    auto forces =
        contiguous_view<typename Work::Sum, 1, D>(work.forces, {count});
    auto dx = buffer_view<Coord, 1, D>(args[0]);
    auto ddofs = buffer_view<Dofs, 1, D>(result);
#ifdef __CUDACC__
    kinematics::kin_derivatives_cuda_native<Real, I>(
        stream,
        dx,
        hts,
        dofs,
        nodes,
        scans,
        gens,
        tree,
        forces,
        ddofs,
        carryout,
        codes,
        lbs,
        scan);
#else
    kinematics::kin_derivatives_cpu_native<Real, I>(
        dx, hts, dofs, nodes, scans, gens, tree, forces, ddofs);
#endif
  } else {
    auto xyz = buffer_view<Coord, 1, D>(result);
#ifdef __CUDACC__
    kinematics::forward_kinematics_cuda_native<Real, I>(
        stream,
        dofs,
        nodes,
        scans,
        gens,
        tree,
        hts,
        xyz,
        carryout,
        codes,
        lbs,
        scan);
#else
    kinematics::forward_kinematics_cpu_native<Real, I>(
        dofs, nodes, scans, gens, tree, hts, xyz);
#endif
  }
#ifdef __CUDACC__
  auto status = cudaGetLastError();
  if (status != cudaSuccess) throw mgpu::cuda_exception_t(status);
#endif
}

template <bool Derivative>
ffi::Error run_kinematic_scan(
#ifdef __CUDACC__
    cudaStream_t stream,
#endif
    ffi::RemainingArgs args,
    ffi::Result<ffi::AnyBuffer> result,
    ffi::RemainingRets rets,
    Generations generations) {
  try {
    auto buffers = native_arguments(args);
#ifdef __CUDACC__
    KinematicContext context = stream;
#else
    KinematicContext context;
#endif
    if (result->element_type() == ffi::DataType::F32)
      execute_kinematic_scan<float, Derivative>(
          context, buffers, *result, rets, generations);
    else if (result->element_type() == ffi::DataType::F64)
      execute_kinematic_scan<double, Derivative>(
          context, buffers, *result, rets, generations);
    else
      throw std::invalid_argument("kinematics requires float32/64 arrays");
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::InvalidArgument(error.what());
  }
}

inline uint64_t kinematic_workspace_bytes(
    int64_t count,
    const int32_t* generations,
    int64_t generation_count,
    int bits,
    int derivative) {
  try {
    if (generation_count < 1 || generation_count > INT32_MAX / 2
        || (bits != 32 && bits != 64) || (derivative != 0 && derivative != 1))
      return UINT64_MAX;
    auto gens = generation_view(Generations(generations, generation_count * 2));
    if (bits == 32)
      return derivative ? KinematicWorkspace<float, true>(count, gens)
                              .storage.size_bytes()
                        : KinematicWorkspace<float, false>(count, gens)
                              .storage.size_bytes();
    return derivative ? KinematicWorkspace<double, true>(count, gens)
                            .storage.size_bytes()
                      : KinematicWorkspace<double, false>(count, gens)
                            .storage.size_bytes();
  } catch (...) {
    return UINT64_MAX;
  }
}
}  // namespace tmol::jax
