#include <tmol/backends/cuda/score/common/device_operations.cuda.impl.cuh>
#include <tmol/pack/rotamer/conformer_ops.impl.hh>
namespace tmol::pack::rotamer {
template struct ConformerDofKernels<
    score::common::DeviceOperations,
    Device::CUDA,
    float,
    int64_t>;
template struct ConformerDofKernels<
    score::common::DeviceOperations,
    Device::CUDA,
    double,
    int64_t>;
}  // namespace tmol::pack::rotamer

#include <tmol/pack/rotamer/pose_imposition_torch.impl.hh>
template struct tmol::pack::rotamer::PoseImpositionTorch<
    tmol::score::common::DeviceOperations,
    tmol::Device::CUDA>;

#include <tmol/pack/rotamer/opth_sampling_torch.impl.hh>
template struct tmol::pack::rotamer::OptHSamplingTorch<tmol::score::common::DeviceOperations,tmol::Device::CUDA>;
