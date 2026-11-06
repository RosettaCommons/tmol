#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>
#include <tmol/io/details/compiled/gen_pose_leaf_atoms.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CPU,
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
    Device::CPU,
    int32_t>;
template struct CanonicalCandidates<
    score::common::DeviceOperations,
    Device::CPU,
    int64_t>;
}  // namespace tmol::io::details::compiled

#include <tmol/io/details/compiled/canonical_topology.impl.hh>
namespace tmol::io::details::compiled {
template struct CanonicalTopology<
    score::common::DeviceOperations,
    Device::CPU,
    int32_t>;
template struct CanonicalTopology<
    score::common::DeviceOperations,
    Device::CPU,
    int64_t>;
}  // namespace tmol::io::details::compiled

#include <tmol/io/details/compiled/canonical_policies.impl.hh>
namespace tmol::io::details::compiled {
template struct CanonicalPolicies<
    score::common::DeviceOperations,
    Device::CPU,
    int32_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int32_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int32_t>;
template struct CanonicalPolicies<
    score::common::DeviceOperations,
    Device::CPU,
    int64_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int64_t>;
template struct CanonicalCycles<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int64_t>;
}  // namespace tmol::io::details::compiled

#include <tmol/io/details/compiled/hydrogen_completion.impl.hh>
namespace tmol::io::details::compiled {
template struct HydrogenCompletion<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int32_t>;
template struct HydrogenInput<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int32_t>;
template struct HydrogenCompletion<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int32_t>;
template struct HydrogenInput<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int32_t>;
template struct HydrogenCompletion<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int64_t>;
template struct HydrogenInput<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int64_t>;
template struct HydrogenCompletion<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int64_t>;
template struct HydrogenInput<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int64_t>;
}  // namespace tmol::io::details::compiled
