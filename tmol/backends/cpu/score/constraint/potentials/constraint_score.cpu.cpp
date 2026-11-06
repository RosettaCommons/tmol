#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/constraint/potentials/constraint_topology.impl.hh>
#include <tmol/score/constraint/potentials/constraint_score.impl.hh>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    float>;
template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    double>;

template struct ConstraintCandidateDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    int32_t>;
template struct ConstraintCandidateDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    int64_t>;

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
