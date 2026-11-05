#pragma once
#include "native_sampling.impl.hh"
#include <tmol/score/common/serial_cpu_operations.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/tensor/owned_host_tensor.hh>

namespace tmol {
namespace pack {
namespace rotamer {
namespace dunbrack {

template <Device D>
struct HostSamplingOperations;
template <>
struct HostSamplingOperations<Device::CPU>
    : score::common::SerialCPUOperations {
  template <
      tmol::portable::scan_type scan_type,
      typename T,
      typename Int,
      typename OP,
      typename Context>
  static auto segmented_scan(
      Context& context,
      T* src,
      Int* starts,
      int n,
      int n_segments,
      OP op,
      T identity) {
    auto result = OwnedHostTensor<T, 1, Device::CPU>::empty({n});
    segmented_scan_into<scan_type>(
        context, src, starts, result.view.data(), n, n_segments, op, identity);
    return result;
  }
};

template <typename Real, typename Int = int32_t>
using HostDunbrackSampling = NativeDunbrackChiSampler<
    HostSamplingOperations,
    Device::CPU,
    Real,
    Int,
    OwnedHostTensor,
    ContextManager>;
// Small host allocation plan shared by eager and staged device adapters.
// Device execution receives these exact counts/offsets, never a frontend
// reinterpretation of rotamer-library or libraryless-residue semantics.
template <typename Int = int32_t>
struct DunbrackSamplingHostLayout {
  OwnedHostTensor<Int, 1, Device::CPU> possible_counts;
  OwnedHostTensor<Int, 1, Device::CPU> possible_offsets;
  Int n_possible_rotamers;
  Int max_n_chi;
};

template <typename Int>
DunbrackSamplingHostLayout<Int> prepare_dunbrack_sampling_layout(
    TView<Int, 2, Device::CPU> buildable_types,
    TView<int64_t, 1, Device::CPU> library_counts,
    TView<Int, 1, Device::CPU> chi_counts) {
  sampling_require(
      buildable_types.size(1) == 2
          && buildable_types.size(0) == chi_counts.size(0),
      "Dunbrack sampling layout shape mismatch");
  int64_t const count = chi_counts.size(0);
  sampling_require(
      count <= std::numeric_limits<Int>::max(),
      "Dunbrack sampling layout exceeds index capacity");
  for (int64_t i = 0; i < count; ++i) {
    sampling_require(
        buildable_types[i][1] < library_counts.size(0),
        "Dunbrack sampling table index out of range");
    sampling_require(chi_counts[i] >= 0, "Negative Dunbrack chi count");
  }
  ContextManager context;
  auto counts = OwnedHostTensor<Int, 1, Device::CPU>::empty(count);
  auto offsets = OwnedHostTensor<Int, 1, Device::CPU>::empty(count);
  using Sampler = HostDunbrackSampling<float, Int>;
  Sampler::determine_n_possible_rots(
      context, buildable_types, library_counts, counts.view);
  Int total =
      Sampler::checked_count_offsets(context, counts.view, offsets.view);
  Int max_chi =
      count == 0
          ? 0
          : HostSamplingOperations<Device::CPU>::reduce(
                context, chi_counts.data(), count, portable::maximum_t<Int>());
  return {counts, offsets, total, max_chi};
}
}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
