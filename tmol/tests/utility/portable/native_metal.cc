#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <tmol/score/metal/potentials/metal_coordination_pose_kernels.hh>
int main() { return 0; }
template <tmol::Device D>
struct SerialOps {
  template <int nt, class F>
  static void for_each_in_workgroup(F f) {
    f(0);
  }
  template <int nt, class T, class S, class O>
  static T reduce_in_workgroup(T value, S&, O) {
    return value;
  }
  template <class L, class F>
  static void forall_grouped(
      tmol::ContextManager&, int poses, int groups, F f) {
    for (int i = 0; i < poses * groups; ++i) f(i);
  }
};
using namespace tmol::score::metal::potentials;
auto forward32 =
    &metal_coordination_pose_forward<SerialOps, tmol::Device::CPU, float, int>;
auto backward32 =
    &metal_coordination_pose_backward<SerialOps, tmol::Device::CPU, float, int>;
auto forward64 =
    &metal_coordination_pose_forward<SerialOps, tmol::Device::CPU, double, int>;
auto backward64 =
    &metal_coordination_pose_backward<SerialOps, tmol::Device::CPU, double, int>;
