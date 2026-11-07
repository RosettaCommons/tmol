#include <cmath>
#include <cstdlib>
#include <tmol/score/lk_ball/potentials/lk_ball_pose_kernels.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_native.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_pose_kernels.hh>
#include <tmol/score/na_torsion/potentials/na_torsion_pose_kernels.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include "native_test_ops.hh"
using tmol_test::SerialOps;
using tmol_test::view;
using namespace tmol::score::lk_ball::potentials;
auto f32 =
    &lk_ball_pose_native<false, SerialOps, tmol::Device::CPU, float, int>;
auto f64 =
    &lk_ball_pose_native<false, SerialOps, tmol::Device::CPU, double, int>;
auto b32 = &lk_ball_pose_native<true, SerialOps, tmol::Device::CPU, float, int>;
auto b64 =
    &lk_ball_pose_native<true, SerialOps, tmol::Device::CPU, double, int>;

auto rf32 =
    &lk_ball_rotamer_native<false, SerialOps, tmol::Device::CPU, float, int>;
auto rf64 =
    &lk_ball_rotamer_native<false, SerialOps, tmol::Device::CPU, double, int>;
auto rb32 =
    &lk_ball_rotamer_native<true, SerialOps, tmol::Device::CPU, float, int>;
auto rb64 =
    &lk_ball_rotamer_native<true, SerialOps, tmol::Device::CPU, double, int>;

auto wf32 = &water_gen_forward<SerialOps, tmol::Device::CPU, float, int>;
auto wf64 = &water_gen_forward<SerialOps, tmol::Device::CPU, double, int>;
auto wb32 = &water_gen_backward<SerialOps, tmol::Device::CPU, float, int>;
auto wb64 = &water_gen_backward<SerialOps, tmol::Device::CPU, double, int>;

// Explicitly instantiate wide outputs without introducing a framework ABI.
auto fw = &lk_ball_pose_native<
    false,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double,
    double>;
auto bw = &lk_ball_pose_native<
    true,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double,
    double>;
auto rw = &lk_ball_rotamer_native<
    true,
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double,
    double>;
auto ww = &water_gen_backward<
    SerialOps,
    tmol::Device::CPU,
    float,
    int,
    double,
    double>;

void check_water_chain() {
  using tmol::portable::Vec;
  Vec<float, 3> donor{0.2f, -0.7f, 1.1f}, hydrogen{1.0f, 0.5f, -0.3f};
  auto jacobian = build_don_water<float>::dV(donor, hydrogen, 2.65f);
  Vec<double, 3> force{0.25, -0.4, 0.7}, d_donor{}, d_hydrogen{};
  accumulate_water_derivative<tmol::Device::CPU>(d_donor, jacobian.dD, force);
  accumulate_water_derivative<tmol::Device::CPU>(
      d_hydrogen, jacobian.dH, force);
  // Translating both parent atoms translates the water by the same amount.
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(d_donor[axis] + d_hydrogen[axis] - force[axis]) > 1e-7)
      std::abort();
  }
}

template <class Real>
void check_padding() {
  // Padded blocks can retain a valid rotamer slot but a negative type.
  // No type-property arrays exist: both execution modes must reject padding
  // before consulting chemistry or writing an energy/gradient.
  int offsets[] = {0, 1}, types[] = {-1, -1}, candidates[] = {2, 0, 1};
  Real weights[] = {1, 1, 1, 1, 1, 1, 1, 1};
  LKBallCtx<Real, int, tmol::Device::CPU> ctx{};
  ctx.max_n_blocks = ctx.n_block_pairs = 1;
  ctx.use_shared_compact_block_neighbors = true;
  ctx.rot_offset_for_block = view(offsets, {2, 1});
  ctx.block_type_ind_for_rot = view(types, {2});
  ctx.dTdV = view(weights, {4, 2, 1, 1});
  tmol::ContextManager mgr;
  lk_ball_pose_native<false, SerialOps, tmol::Device::CPU, Real, int>(
      mgr, ctx, view(candidates, {3}), 2);
  lk_ball_pose_native<true, SerialOps, tmol::Device::CPU, Real, int>(
      mgr, ctx, view(candidates, {3}), 2);
}
int main() {
  check_water_chain();
  check_padding<float>();
  check_padding<double>();
}
