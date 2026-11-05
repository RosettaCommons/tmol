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
}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
