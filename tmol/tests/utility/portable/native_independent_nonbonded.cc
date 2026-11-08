#include <tmol/score/ljlk/potentials/ljlk_pose_kernels.hh>
#include <tmol/score/elec/potentials/elec_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <algorithm>
#include <cassert>

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
auto sparse_lj32 =
    &ljlk_rotamer_forward_native<SerialOps, tmol::Device::CPU, float, int>;
auto sparse_lj64 =
    &ljlk_rotamer_forward_native<SerialOps, tmol::Device::CPU, double, int>;
auto sparse_e32 =
    &elec_rotamer_forward_native<SerialOps, tmol::Device::CPU, float, int>;
auto sparse_e64 =
    &elec_rotamer_forward_native<SerialOps, tmol::Device::CPU, double, int>;
auto sparse_eb32 =
    &elec_rotamer_backward_native<SerialOps, tmol::Device::CPU, float, int>;
auto sparse_eb64 =
    &elec_rotamer_backward_native<SerialOps, tmol::Device::CPU, double, int>;

// The wider accumulator must sum the original float pair forces, without
// changing the pair physics or requiring any electrostatic parameters.
int main() {
  using tmol::portable::Vec;
  using tmol_test::view;
  LJLKScoringData<float> data{};
  float coord1[3] = {}, coord2[3] = {};
  short atom_type = 0;
  data.r1.coords = coord1;
  data.r2.coords = coord2;
  data.r1.atom_types = data.r2.atom_types = &atom_type;
  data.r2.rot_coord_offset = 1;
  data.global_params = {6, 0.6f, 3, 2.6f, 1.75f};
  LJLKTypeParams<float> params{
      1.8f, 0.4f, -0.5f, 3.5f, 12, 0, 0, 0, 0, 1, 0, 0.2f, 1.f / 12.25f};
  Vec<double, 3> accumulated[6]{}, weighted[2]{};
  long double reference[6][3]{}, weighted_reference[2][3]{};
  for (int i = 0; i < 129; ++i) {
    coord1[0] = (i % 2 ? -1.f : 1.f) * (2.8f + i * 0.015f);
    coord1[1] = 0.3f;
    coord1[2] = -0.2f;
    Vec<float, 3> pair[6]{}, weighted_pair[2]{};
    auto energy = ljlk_atom_energy_and_derivs_full<float, tmol::Device::CPU>(
        0, 0, 0, 0, data, 5, view(pair, {3, 2}), &params);
    auto wide_energy =
        ljlk_atom_energy_and_derivs_full<float, tmol::Device::CPU, double>(
            0, 0, 0, 0, data, 5, view(accumulated, {3, 2}), &params);
    ljlk_atom_derivs<float, tmol::Device::CPU>(
        0,
        0,
        0,
        0,
        data,
        5,
        0.7f,
        -0.3f,
        1.2f,
        view(weighted_pair, {2}),
        &params);
    ljlk_atom_derivs<float, tmol::Device::CPU, double>(
        0, 0, 0, 0, data, 5, 0.7f, -0.3f, 1.2f, view(weighted, {2}), &params);
    for (int term = 0; term < 3; ++term) {
      assert(energy[term] == wide_energy[term]);
      for (int axis = 0; axis < 3; ++axis)
        assert(pair[2 * term][axis] == -pair[2 * term + 1][axis]);
    }
    for (int index = 0; index < 6; ++index)
      for (int axis = 0; axis < 3; ++axis)
        reference[index][axis] += pair[index][axis];
    for (int index = 0; index < 2; ++index)
      for (int axis = 0; axis < 3; ++axis)
        weighted_reference[index][axis] += weighted_pair[index][axis];
  }
  for (int index = 0; index < 6; ++index) {
    assert(reference[index][1] != 0);
    for (int axis = 0; axis < 3; ++axis)
      assert(accumulated[index][axis] == double(reference[index][axis]));
  }
  for (int index = 0; index < 2; ++index)
    for (int axis = 0; axis < 3; ++axis)
      assert(weighted[index][axis] == double(weighted_reference[index][axis]));
}
auto wide_lj =
    &ljlk_pose_forward_native<SerialOps, tmol::Device::CPU, float, int, double>;
auto wide_lb = &ljlk_backward_native<
    false,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double>;
auto wide_lr = &ljlk_backward_native<
    true,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double>;
auto wide_sparse = &ljlk_rotamer_forward_native<
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double>;
