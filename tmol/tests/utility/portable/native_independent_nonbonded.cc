#include <tmol/score/ljlk/potentials/ljlk_pose_kernels.hh>
#include <tmol/score/elec/potentials/elec_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <algorithm>
int main() { return 0; }
#include "native_test_ops.hh"
using tmol_test::SerialOps;

using namespace tmol::score::elec::potentials;
auto f32 = &elec_pose_forward_native<SerialOps, tmol::Device::CPU, float, int>;
auto f64 = &elec_pose_forward_native<SerialOps, tmol::Device::CPU, double, int>;
auto b32 = &elec_pose_backward_native<SerialOps, tmol::Device::CPU, float, int>;
auto b64 =
    &elec_pose_backward_native<SerialOps, tmol::Device::CPU, double, int>;

auto lj32 =
    &tmol::score::ljlk::potentials::
        ljlk_pose_forward_native<SerialOps, tmol::Device::CPU, float, int>;
auto lj64 =
    &tmol::score::ljlk::potentials::
        ljlk_pose_forward_native<SerialOps, tmol::Device::CPU, double, int>;

using namespace tmol::score::ljlk::potentials;
auto lb32 =
    &ljlk_backward_native<false, SerialOps, tmol::Device::CPU, float, int>;
auto lb64 =
    &ljlk_backward_native<false, SerialOps, tmol::Device::CPU, double, int>;
auto lr32 =
    &ljlk_backward_native<true, SerialOps, tmol::Device::CPU, float, int>;
auto lr64 =
    &ljlk_backward_native<true, SerialOps, tmol::Device::CPU, double, int>;