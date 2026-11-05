#include <tmol/kinematics/compiled/host_pose_kinematics.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <limits>

using namespace tmol;
using namespace tmol::kinematics;
using tmol_test::view;

// Instantiate the complete assembly and minimizer-map ABI without any
// framework headers; the binding must not hide an uninstantiable method.
template struct tmol::kinematics::NativeKinForestFromStencil<
    HostPoseOperations,
    Device::CPU,
    int32_t,
    OwnedHostTensor,
    ContextManager>;

int main() {
  int32_t offsets[] = {0, 3, 0, 0};
  int32_t types[] = {0, 1, 1, -1};
  int32_t counts[] = {3, 1};
  bool real[] = {true, true, true, true, false, false};
  ContextManager context;
  auto [block_offset, mapping, atom_index] =
      HostPoseKinematics<>::get_kfo_indices_for_atoms(
          context,
          view(offsets, {2, 2}),
          view(types, {2, 2}),
          view(counts, {2}),
          view(real, {2, 3}));
  assert(
      mapping.view.size(0) == 6);  // one global root plus five physical atoms
  assert(block_offset.view[0][0] == 1 && block_offset.view[0][1] == 4);
  assert(block_offset.view[1][0] == 5 && block_offset.view[1][1] == 6);
  assert(mapping.view[0][0] == -1);
  int visited = 0;
  for (int p = 0; p < 2; ++p)
    for (int b = 0; b < 2; ++b) {
      int bt = types[p * 2 + b];
      for (int a = 0; a < 3; ++a) {
        int k = atom_index.view[p][b][a];
        if (bt == -1 || !real[bt * 3 + a]) {
          assert(k == -1);
          continue;
        }
        ++visited;
        assert(k > 0 && k < 6);
        assert(
            mapping.view[k][0] == p && mapping.view[k][1] == b
            && mapping.view[k][2] == a);
      }
    }
  assert(visited == 5);

  auto retained = [] {
    auto original = OwnedHostTensor<int32_t, 2, Device::CPU>::full({2, 3}, 7);
    return original;  // ownership survives the allocating scope
  }();
  assert(retained.view[1][2] == 7);
  auto empty = OwnedHostTensor<bool, 3, Device::CPU>::zeros({2, 0, 3});
  assert(
      empty.view.size(1) == 0 && empty.view.stride(0) == 3
      && empty.view.stride(1) == 3 && empty.view.stride(2) == 1);
  bool rejected = false;
  try {
    OwnedHostTensor<int32_t, 2, Device::CPU>::zeros(
        {std::numeric_limits<int64_t>::max(), 2});
  } catch (std::overflow_error const&) {
    rejected = true;
  }
  assert(rejected);
}
