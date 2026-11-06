#include <tmol/backends/cpu/score/common/device_operations.cpu.impl.hh>
#include <tmol/pack/rotamer/conformer_ops.impl.hh>
namespace tmol::pack::rotamer {
template struct ConformerDofKernels<
    score::common::DeviceOperations,
    Device::CPU,
    float,
    int64_t>;
template struct ConformerDofKernels<
    score::common::DeviceOperations,
    Device::CPU,
    double,
    int64_t>;
}  // namespace tmol::pack::rotamer

#include <tmol/pack/rotamer/pose_imposition_torch.impl.hh>
template struct tmol::pack::rotamer::PoseImpositionTorch<
    tmol::score::common::DeviceOperations,
    tmol::Device::CPU>;
