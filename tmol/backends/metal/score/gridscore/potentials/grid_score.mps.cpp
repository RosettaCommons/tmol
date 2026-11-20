// Instantiation of the grid-scoring dispatch for the Metal backend.
//
// Only float is instantiated: Metal has no double. Only ``atoms`` has a Metal
// entry; ``waters`` is instantiated with it but never reached on MPS, because
// its operator does not dispatch to this backend.

#include <tmol/backends/metal/score/common/device_operations.mps.host.impl.hh>
#include <tmol/score/gridscore/potentials/grid_score.impl.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template struct GridScoreDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    float,
    int64_t>;

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
