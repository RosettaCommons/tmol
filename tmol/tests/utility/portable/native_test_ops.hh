#pragma once
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <algorithm>
namespace tmol_test {
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

template <class T, size_t N>
tmol::TView<T, N, tmol::Device::CPU> view(T* ptr, int64_t const (&shape)[N]) {
  int64_t stride[N];
  int64_t st = 1;
  for (int i = N - 1; i >= 0; --i) {
    stride[i] = st;
    st *= shape[i];
  }
  return tmol::TView<T, N, tmol::Device::CPU>(ptr, shape, stride);
}
}  // namespace tmol_test
