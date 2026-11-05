#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <algorithm>
int main() { return 0; }
template <tmol::Device D>
struct SerialOps {
  template <int nt, class F>
  static void for_each_in_workgroup(F f) {
    f(0);
  }
  template <int nt, class T, class S, class O>
  static T reduce_in_workgroup(T v, S&, O) {
    return v;
  }
  template <int nt, int v, class T>
  static void copy_contiguous_data(T* dst, T const* src, int n) {
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
using namespace tmol::score::ljlk::potentials;
auto f32 = &ljlk_elec_native_forward<
    true,
    false,
    false,
    SerialOps,
    tmol::Device::CPU,
    float,
    int>;
auto f64 = &ljlk_elec_native_forward<
    true,
    false,
    false,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
auto w64 = &ljlk_elec_native_forward<
    true,
    true,
    false,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
auto r64 = &ljlk_elec_native_forward<
    false,
    true,
    true,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
auto b64 = &ljlk_elec_native_forward<
    true,
    true,
    true,
    SerialOps,
    tmol::Device::CPU,
    double,
    int>;
