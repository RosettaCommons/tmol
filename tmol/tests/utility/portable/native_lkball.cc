#include <tmol/score/lk_ball/potentials/lk_ball_pose_kernels.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_native.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <algorithm>
int main() { return 0; }
template <tmol::Device D>
struct SerialOps {
  template <int nt, class F, class... Lane>
  static void for_each_in_workgroup(F f, Lane...) {
    f(0);
  }
  template <int nt, class T, class S, class O, class... Lane>
  static T reduce_in_workgroup(T v, S&, O, Lane...) {
    return v;
  }
  template <int nt, int v, class T, class... Lane>
  static void copy_contiguous_data(T* dst, T const* src, int n, Lane...) {
    std::copy(src, src + n, dst);
  }
  template <int nt, int v, class T, class U, class... Lane>
  static void copy_contiguous_data_and_cast(
      T* dst, U const* src, int n, Lane...) {
    std::copy(src, src + n, dst);
  }
  template <class Lane>
  static void synchronize_workgroup(Lane const&) {}
  template <class L, class F>
  static void forall(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }
  template <class L, class F>
  static void foreach_workgroup(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }
  template <class L, class F>
  static void foreach_independent_workgroup(tmol::ContextManager&, int n, F f) {
    for (int i = 0; i < n; ++i) f(i);
  }
  template <class L, class F>
  static void foreach_pose_workgroup(tmol::ContextManager&, int p, int n, F f) {
    for (int i = 0; i < p * n; ++i) f(i);
  }
};
using namespace tmol::score::lk_ball::potentials;
auto f32 =
    &lk_ball_pose_native<false, SerialOps, tmol::Device::CPU, float, int>;
auto f64 =
    &lk_ball_pose_native<false, SerialOps, tmol::Device::CPU, double, int>;
auto b32 = &lk_ball_pose_native<true, SerialOps, tmol::Device::CPU, float, int>;
auto b64 =
    &lk_ball_pose_native<true, SerialOps, tmol::Device::CPU, double, int>;

auto wf32 = &water_gen_forward<SerialOps, tmol::Device::CPU, float, int>;
auto wf64 = &water_gen_forward<SerialOps, tmol::Device::CPU, double, int>;
auto wb32 = &water_gen_backward<SerialOps, tmol::Device::CPU, float, int>;
auto wb64 = &water_gen_backward<SerialOps, tmol::Device::CPU, double, int>;
