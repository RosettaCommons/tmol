#include <tmol/backends/cuda/score/common/device_operations.cuda.impl.cuh>
#include <tmol/io/details/compiled/gen_pose_leaf_atoms.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol

#include <tmol/io/details/compiled/canonical_candidates.impl.hh>
namespace tmol::io::details::compiled {
template struct CanonicalCandidates<
    score::common::DeviceOperations,
    Device::CUDA,
    int32_t>;
template struct CanonicalCandidates<
    score::common::DeviceOperations,
    Device::CUDA,
    int64_t>;
}  // namespace tmol::io::details::compiled

#include <tmol/io/details/compiled/canonical_topology.impl.hh>
namespace tmol::io::details::compiled {
template struct CanonicalTopology<
    score::common::DeviceOperations,
    Device::CUDA,
    int32_t>;
template struct CanonicalTopology<
    score::common::DeviceOperations,
    Device::CUDA,
    int64_t>;
}  // namespace tmol::io::details::compiled

#include <tmol/io/details/compiled/canonical_policies.impl.hh>
namespace tmol::io::details::compiled {
template struct CanonicalPolicies<
    score::common::DeviceOperations,
    Device::CUDA,
    int32_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CUDA,
    float,
    int32_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CUDA,
    double,
    int32_t>;
template struct CanonicalPolicies<
    score::common::DeviceOperations,
    Device::CUDA,
    int64_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CUDA,
    float,
    int64_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CUDA,
    double,
    int64_t>;
}  // namespace tmol::io::details::compiled
