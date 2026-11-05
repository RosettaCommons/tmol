// Buffer ownership and stream execution for the common dynamic chi sampler.
#pragma once
#include "forward_kinematics_ffi.hh"
#include <tmol/pack/rotamer/dunbrack/native_sampling.impl.hh>
#include <tmol/score/common/segmented_scan_cuda.cuh>

namespace tmol::jax {
namespace sampling = pack::rotamer::dunbrack;
constexpr Device SamplingDevice = Device::CUDA;
using SamplingInt = int32_t;
template <typename T, size_t N, Device D>
struct SamplingExternalPack {};

template <typename T>
struct SamplingElement {
  using Scalar = T;
  static constexpr int lanes = 1;
};
template <typename T, int N>
struct SamplingElement<Vec<T, N>> {
  using Scalar = T;
  static constexpr int lanes = N;
};

template <typename T, size_t Rank>
auto sampling_view(ffi::AnyBuffer buffer) {
  using E = SamplingElement<T>;
  using S = typename E::Scalar;
  constexpr bool wide = std::is_same_v<S, int64_t>;
  constexpr int extra = (E::lanes > 1) + wide;
  auto dims = buffer.dimensions();
  auto dtype = wide                        ? ffi::DataType::U8
               : std::is_same_v<S, float>  ? ffi::DataType::F32
               : std::is_same_v<S, double> ? ffi::DataType::F64
               : std::is_same_v<S, bool>   ? ffi::DataType::PRED
                                           : ffi::DataType::S32;
  if (dims.size() != Rank + extra || buffer.element_type() != dtype)
    throw std::invalid_argument("sampling operand dtype/rank mismatch");
  if constexpr (E::lanes > 1)
    if (dims[Rank] != E::lanes)
      throw std::invalid_argument("sampling vector width mismatch");
  if constexpr (wide)
    if (dims[Rank + (E::lanes > 1)] != 8)
      throw std::invalid_argument(
          "sampling int64 storage requires eight bytes");
  if (reinterpret_cast<uintptr_t>(buffer.untyped_data()) % alignof(T))
    throw std::invalid_argument("sampling operand alignment mismatch");
  for (auto size : dims)
    if (size < 0 || size > INT32_MAX)
      throw std::invalid_argument("sampling operand exceeds int32 indexing");
  return buffer_view<T, Rank, SamplingDevice>(buffer);
}

template <typename Real>
auto sampling_inputs(const std::vector<ffi::AnyBuffer>& a) {
  using I = SamplingInt;
#define SV(T, N, K) sampling_view<T, N>(a[K])
#define VV(T, N, K) sampling_view<Vec<T, N>, 1>(a[K])
  sampling::DunbrackSamplingInputs<Real, I, SamplingDevice> in{
      VV(Real, 3, 0),    SV(Real, 3, 1),     VV(int64_t, 2, 2),
      VV(int64_t, 2, 3), SV(Real, 3, 4),     SV(Real, 3, 5),
      VV(int64_t, 2, 6), VV(int64_t, 2, 7),  SV(I, 1, 8),
      VV(Real, 2, 9),    VV(Real, 2, 10),    VV(Real, 2, 11),
      VV(Real, 2, 12),   SV(bool, 1, 13),    SV(int64_t, 1, 14),
      SV(I, 1, 15),      SV(int64_t, 3, 16), SV(I, 1, 17),
      SV(I, 1, 18),      SV(I, 1, 19),       VV(I, 4, 20),
      SV(I, 2, 21),      SV(I, 2, 22),       SV(Real, 3, 23),
      SV(I, 2, 24),      SV(Real, 1, 25),    SV(I, 1, 26)};
#undef SV
#undef VV
  auto count = in.nchi_for_buildable_restype.size(0);
  if (in.bubl_and_rottable_set_for_buildable_restype.size(0) != count
      || in.bubl_and_rottable_set_for_buildable_restype.size(1) != 2
      || in.prob_cumsum_limit_for_buildable_restype.size(0) != count
      || in.chi_expansion_for_buildable_restype.size(0) != count
      || in.non_dunbrack_expansion_for_buildable_restype.size(0) != count
      || in.non_dunbrack_expansion_counts_for_buildable_restype.size(0) != count
      || in.ndihe_for_res.size(0) != in.dihedral_offset_for_res.size(0))
    throw std::invalid_argument("sampling request extents differ");
  return in;
}

template <typename Real>
struct SamplingWorkspace {
  ByteWorkspace storage;
  using I = SamplingInt;
  using Scan = score::common::SegmentedScanWorkspace<I, SamplingDevice>;
  Scan integer_scan;
  score::common::SegmentedScanWorkspace<Real, SamplingDevice> real_scan;
  I *starts, *boundaries, *identities, *keep, *prefix;
  Real *probability, *probability_prefix;

  SamplingWorkspace(
      int64_t count, int64_t segments, bool emit, void* data = nullptr)
      : storage(data) {
    if (count < 0 || segments < 0 || count > INT32_MAX || segments > INT32_MAX)
      throw std::invalid_argument("sampling workspace extent exceeds int32");
    auto sizes = portable::scan_workspace_sizes<int>(
        count + std::max<int64_t>(segments, 1), 128, 2);
    auto one = [&](auto* pointer, int n) {
      using T = std::remove_pointer_t<decltype(pointer)>;
      return contiguous_view<T, 1, SamplingDevice>(pointer, {n});
    };
    starts = storage.take<I>(1);
    boundaries = storage.take<I>(count);
    integer_scan = {
        one(storage.take<I>(sizes[1]), sizes[1]),
        one(storage.take<I>(sizes[1]), sizes[1]),
        one(storage.take<I>(sizes[2]), sizes[2])};
    if (!emit) {
      identities = storage.take<I>(count);
      keep = storage.take<I>(count);
      prefix = storage.take<I>(count);
      probability = storage.take<Real>(count);
      probability_prefix = storage.take<Real>(count);
      real_scan = {
          one(storage.take<Real>(sizes[1]), sizes[1]),
          one(storage.take<I>(sizes[1]), sizes[1]),
          one(storage.take<I>(sizes[2]), sizes[2])};
    }
  }
};

struct SamplingContext : ContextManager {
  KinematicStream execution;
  int32_t* starts;
  score::common::SegmentedScanWorkspace<int32_t, SamplingDevice> scan_workspace;
  template <typename Real>
  SamplingContext(cudaStream_t stream, SamplingWorkspace<Real>& work)
      : execution(stream),
        starts(work.starts),
        scan_workspace(work.integer_scan) {
    auto status = cudaMemsetAsync(starts, 0, sizeof(int32_t), stream);
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
};

template <template <Device> class Ops, typename Real, bool Emit>
void execute_sampling(
    cudaStream_t stream,
    const std::vector<ffi::AnyBuffer>& a,
    const std::vector<ffi::AnyBuffer>& out,
    int64_t count) {
  using I = SamplingInt;
  constexpr auto D = SamplingDevice;
  using Sampler = sampling::NativeDunbrackChiSampler<
      Ops,
      D,
      Real,
      I,
      SamplingExternalPack,
      SamplingContext>;
  if (a.size() != (Emit ? 32 : 29) || out.size() != (Emit ? 3 : 5))
    throw std::invalid_argument("sampling operand/result count mismatch");
  auto in = sampling_inputs<Real>(a);
  auto nr = in.nchi_for_buildable_restype.size(0);
  auto plan_at = [&](int i) { return Emit ? a[27 + i] : out[i]; };
  typename Sampler::Plan plan{
      sampling_view<Real, 1>(plan_at(0)),
      sampling_view<I, 1>(plan_at(1)),
      sampling_view<I, 2>(plan_at(2)),
      sampling_view<I, 1>(plan_at(3))};
  if (plan.backbone_dihedrals.size(0) != 2 * in.ndihe_for_res.size(0)
      || plan.n_expansions_for_brt.size(0) != nr
      || plan.expansion_dim_prods_for_brt.size(0) != nr
      || plan.n_rotamers_to_build_per_brt.size(0) != nr)
    throw std::invalid_argument("sampling plan extents differ");
  if constexpr (Emit)
    if (out[0].dimensions().size() != 1 || out[0].dimensions()[0] != count)
      throw std::invalid_argument("sampling output count differs");
  auto workspace = out.back();
  SamplingWorkspace<Real> work(count, nr, Emit, workspace.untyped_data());
  require_shape(
      workspace, ffi::DataType::U8, {int64_t(work.storage.size_bytes())});
  if (reinterpret_cast<uintptr_t>(workspace.untyped_data()) % alignof(double))
    throw std::invalid_argument("sampling workspace alignment mismatch");
  SamplingContext context(stream, work);
  auto one = [&](auto* p) {
    using T = std::remove_pointer_t<decltype(p)>;
    return contiguous_view<T, 1, D>(p, {count});
  };
  if constexpr (Emit) {
    auto offsets = sampling_view<I, 1>(a[31]);
    auto brt = sampling_view<I, 1>(out[0]);
    auto chi = sampling_view<Real, 2>(out[1]);
    if (offsets.size(0) != nr || chi.size(0) != count
        || chi.size(1) != plan.expansion_dim_prods_for_brt.size(1))
      throw std::invalid_argument("sampling output extents differ");
    Sampler::emit_samples(
        context, in, plan, offsets, brt, one(work.boundaries), chi);
  } else {
    typename Sampler::Scratch scratch{
        sampling_view<I, 1>(a[27]),
        sampling_view<I, 1>(a[28]),
        one(work.identities),
        one(work.boundaries),
        one(work.probability),
        one(work.probability_prefix),
        one(work.keep),
        one(work.prefix),
        work.real_scan,
        work.integer_scan};
    if (scratch.n_possible_rotamers_per_brt.size(0) != nr
        || scratch.possible_rotamer_offset_for_brt.size(0) != nr)
      throw std::invalid_argument("sampling layout extents differ");
    Sampler::prepare_sampling_plan(context, in, plan, scratch);
  }
}
}  // namespace tmol::jax
