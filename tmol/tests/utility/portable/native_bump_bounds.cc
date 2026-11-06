#include <tmol/pack/compiled/interaction_graph_bounds.hh>
#include <tmol/score/common/serial_cpu_operations.hh>
#include <tmol/utility/tensor/owned_host_tensor.hh>
#include "native_test_ops.hh"
#include <cassert>

using tmol::Device;
template <Device D>
struct HostOps : tmol::score::common::SerialCPUOperations {
  template <class L, class F>
  static void forall_independent(tmol::ContextManager& mgr, int n, F f) {
    forall<L>(mgr, n, f);
  }
  template <class L, class F>
  static void foreach_workgroup(tmol::ContextManager& mgr, int n, F f) {
    forall<L>(mgr, n, f);
  }
  template <class L, class F>
  static void foreach_grouped_workgroup(tmol::ContextManager& mgr, int p, int n, F f) {
    forall<L>(mgr, p*n, f);
  }
  template <int N, class F>
  static void for_each_in_workgroup(F f) { f(0); }
  template <int N, class T, class Op>
  static T shuffle_reduce_in_workgroup(T v, Op) { return v; }
  template <class T>
  static void store_idempotent(T& dst, T value) { dst=value; }
};

template <class Real, class Int>
void check() {
  using tmol_test::view;
  Int counts[] = {3,2,0, 0,0,0}, offsets[] = {0,3,5, 5,5,5};
  Int poses[] = {0,0,0,0,0};
  int32_t blocks[] = {0,0,0,1,1};
  // Five one-body entries, six pair entries, and a duplicate tail.
  int32_t indices[] = {0,0,0,0,0,0,0,0,0,0,0,0,
                      0,1,2,3,4,0,0,1,1,2,2,0,
                      0,1,2,3,4,3,4,3,4,3,4,3};
  Real energies[] = {1,8,2,3,4, 1,5,4,1,0,3, 1};
  tmol::ContextManager mgr;
  auto [rot, block] = tmol::pack::compiled::prepare_ig_bump_bounds<
      HostOps, Device::CPU, Real, Int, tmol::OwnedHostTensor>(
      mgr,2,view(counts,{2,3}),view(offsets,{2,3}),view(poses,{5}),
      view(blocks,{5}),view(indices,{3,12}),view(energies,{12}),11);
  Real expected[] = {3,9,2,3,5};
  for(int i=0;i<5;++i) assert(rot.view[i]==expected[i]);
  assert(block.view[0][0]==2 && block.view[0][1]==3);
  assert(block.view[0][2]==1234 && block.view[1][0]==1234);
}
int main() {
  check<float,int32_t>(); check<double,int32_t>();
  check<float,int64_t>(); check<double,int64_t>();
}
