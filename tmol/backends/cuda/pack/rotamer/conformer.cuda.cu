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
