#include <tmol/backends/cuda/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/constraint/potentials/constraint_topology.impl.hh>
#include <tmol/score/constraint/potentials/constraint_score.impl.hh>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float>;
template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double>;

template struct ConstraintCandidateDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    int32_t>;
template struct ConstraintCandidateDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    int64_t>;

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
