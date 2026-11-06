#include <tmol/io/details/compiled/leaf_atom_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cmath>
#include <limits>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;

template <typename Real, typename Int>
void check() {
  using V = tmol::portable::Vec<Real, 3>;
  using U = tmol::UnresolvedAtomID<Int>;
  using Connection = tmol::portable::Vec<Int, 2>;
  V coords[16] = {};
  coords[0] = V{0, 0, 0};
  coords[1] = V{1, 0, 0};
  coords[2] = V{1, 1, 0};
  coords[3] = V::Constant(std::numeric_limits<Real>::quiet_NaN());
  coords[4] = V{1, 1, 1};
  coords[5] = V{2, 1, 1};
  coords[6] = V{2, 2, 1};
  bool missing[16] = {}, pose_missing[16] = {};
  missing[3] = true;
  pose_missing[3] = true;
  Int offsets[] = {0, 4, 0, 0}, types[] = {0, 1, -1, -1}, counts[] = {4, 3};
  Connection connections[] = {{1, 0}, {0, 0}, {-1, -1}, {-1, -1}};
  Int downstream[8];
  std::fill(downstream, downstream + 8, Int(-1));
  downstream[4] = 0;
  U ancestors[24], backup[24];
  for (int i = 0; i < 24; ++i) ancestors[i] = backup[i] = U{-1, -1, -1};
  ancestors[9] = backup[9] = U{0, -1, -1};
  ancestors[10] = backup[10] = U{1, -1, -1};
  ancestors[11] = U{-1, 0, 0};
  backup[11] = U{2, -1, -1};
  Real geometry[24] = {};
  geometry[9] = Real(.712345678901);
  geometry[10] = Real(1.123456789012);
  geometry[11] = Real(1.234567890123);
  LeafAtomContext<Real, Int, CPU> c{
      view(coords, {2, 8}),
      view(missing, {2, 2, 4}),
      view(pose_missing, {2, 8}),
      view(offsets, {2, 2}),
      view(types, {2, 2}),
      view(connections, {2, 2, 1}),
      view(counts, {2}),
      view(downstream, {2, 1, 4}),
      view(ancestors, {2, 4, 3}),
      view(geometry, {2, 4, 3}),
      view(backup, {2, 4, 3}),
      view(geometry, {2, 4, 3})};
  tmol::ContextManager context;
  V output[16] = {}, weights[16] = {}, gradient[16] = {};
  weights[3] = V{Real(.7), Real(-.2), Real(.4)};
  auto forward = [&]() {
    std::fill(output, output + 16, V::Zero());
    launch_pose_leaf_atoms<false, tmol_test::SerialOps, CPU, Real, Int>(
        context, c, view(weights, {2, 8}), view(output, {2, 8}));
    return output[3].dot(weights[3]);
  };
  for (bool use_backup : {false, true}) {
    pose_missing[4] = use_backup;
    auto selected = select_leaf_ancestors<Int, Real, CPU>::find_ancestors(
        0,
        0,
        3,
        0,
        c.pose_stack_block_coord_offset,
        c.pose_stack_block_type,
        c.pose_stack_inter_block_connections,
        c.block_type_atom_downstream_of_conn,
        c.block_type_atom_ancestors,
        c.block_type_atom_icoors,
        c.block_type_atom_ancestors_backup,
        c.block_type_atom_icoors_backup,
        c.pose_stack_atom_missing);
    assert(
        selected.anc0 == 0 && selected.anc1 == 1
        && selected.anc2 == (use_backup ? 2 : 4));
    // Float64 geometry must retain its actual parameters, not round via float.
    assert(
        selected.D == geometry[11] && selected.theta == geometry[10]
        && selected.phi == geometry[9]);
    forward();
    assert(std::abs(output[3].norm() - geometry[11]) < Real(2e-6));
    for (int i = 0; i < 16; ++i)
      if (i != 3)
        for (int k = 0; k < 3; ++k)
          assert(output[i][k] == (i < 7 ? coords[i][k] : Real(0)));
    std::fill(gradient, gradient + 16, V::Zero());
    launch_pose_leaf_atoms<true, tmol_test::SerialOps, CPU, Real, Int>(
        context, c, view(weights, {2, 8}), view(gradient, {2, 8}));
    Real epsilon = sizeof(Real) == 4 ? Real(.002) : Real(1e-6);
    for (int atom = 0; atom < 7; ++atom)
      if (atom != 3)
        for (int k = 0; k < 3; ++k) {
          Real original = coords[atom][k];
          coords[atom][k] = original + epsilon;
          Real hi = forward();
          coords[atom][k] = original - epsilon;
          Real lo = forward();
          coords[atom][k] = original;
          Real finite_difference = (hi - lo) / (2 * epsilon);
          assert(
              std::abs(gradient[atom][k] - finite_difference)
              < (sizeof(Real) == 4 ? Real(2e-4) : Real(2e-8)));
        }
  }
  // Available metadata with an unresolved coordinate must not build a plausible
  // position or return a spurious derivative through that reference.
  pose_missing[4] = false;
  coords[4][0] = std::numeric_limits<Real>::quiet_NaN();
  forward();
  assert(std::isnan(output[3][0]));
  std::fill(gradient, gradient + 16, V::Zero());
  launch_pose_leaf_atoms<true, tmol_test::SerialOps, CPU, Real, Int>(
      context, c, view(weights, {2, 8}), view(gradient, {2, 8}));
  for (auto g : gradient) assert(g.isZero());
}
int main() {
  check<float, int32_t>();
  check<double, int32_t>();
  check<double, int64_t>();
}
