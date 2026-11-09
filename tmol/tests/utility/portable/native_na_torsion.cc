#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_native.hh>
#include <tmol/score/common/serial_cpu_operations.hh>
int main() { return 0; }
using namespace tmol::score::na_torsion::potentials;
auto forward32 = &na_torsion_block<float, tmol::Device::CPU, false>;
auto derivatives32 = &na_torsion_block<float, tmol::Device::CPU, true>;
auto forward64 = &na_torsion_block<double, tmol::Device::CPU, false>;
auto derivatives64 = &na_torsion_block<double, tmol::Device::CPU, true>;
auto backward32 = &na_torsion_weighted_derivative<float>;
auto backward64 = &na_torsion_weighted_derivative<double>;
auto indices32 = &na_torsion_block<double, tmol::Device::CPU, true, int32_t>;

auto weighted32 =
    &na_torsion_block<float, tmol::Device::CPU, true, int32_t, true>;
auto weighted64 =
    &na_torsion_block<double, tmol::Device::CPU, true, int64_t, true>;
