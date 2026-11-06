#include <tmol/io/details/compiled/hydrogen_completion.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cmath>
#include <limits>
using namespace tmol::io::details::compiled;
using tmol_test::view;
constexpr auto CPU = tmol::Device::CPU;
template <class Real, class Int>
void check() {
  tmol::ContextManager mgr;
  Vec<Real, 3> xyz[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {4, 5, 6}, {0, 0, 1}},
               out[5], weights[5], velocity[5], jvp[5];
  Int types[] = {0, 1, -1}, offsets[] = {0, 4, 5}, connection_atoms[] = {0, 0};
  Vec<Int, 2> connections[] = {{1, 0}, {0, 0}, {-1, -1}};
  bool rebuilt[12] = {}, eligible[8] = {};
  rebuilt[3] = eligible[3] = true;
  int64_t parent[8] = {}, neighbors[24] = {}, counts[8] = {}, ports[24] = {},
          port_counts[8] = {};
  neighbors[9] = 1;
  neighbors[10] = 2;
  counts[3] = 2;
  port_counts[3] = 1;
  Real distance[8] = {}, dg[8], dv[8] = {};
  distance[3] = Real(1.25);
  HydrogenCompletionContext<Real, Int, CPU> c{
      view(xyz, {1, 5}),
      view(types, {1, 3}),
      view(offsets, {1, 3}),
      view(connections, {1, 3, 1}),
      view(connection_atoms, {2, 1}),
      view(rebuilt, {1, 3, 4}),
      view(eligible, {2, 4}),
      view(parent, {2, 4}),
      view(neighbors, {2, 4, 3}),
      view(counts, {2, 4}),
      view(distance, {2, 4}),
      view(ports, {2, 4, 3}),
      view(port_counts, {2, 4})};
  auto run = [&]<int Mode>() {
    launch_hydrogen_completion<Mode, tmol_test::SerialOps, CPU>(
        mgr,
        c,
        view(weights, {1, 5}),
        view(velocity, {1, 5}),
        view(dv, {2, 4}),
        view(out, {1, 5}),
        view(dg, {2, 4}),
        view(jvp, {1, 5}));
  };
  Real tol = 20 * std::numeric_limits<Real>::epsilon();
  run.template operator()<0>();
  for (int d = 0; d < 3; ++d)
    assert(std::abs(out[3][d] + Real(1.25) / std::sqrt(Real(3))) < tol);
  for (int a : {0, 1, 2, 4}) assert((out[a] - xyz[a]).norm() == 0);
  Vec<Real, 3> total{0, 0, 0};
  for (int a = 0; a < 5; ++a) {
    weights[a] = {Real(a), Real(2 * a), Real(1 - a)};
    velocity[a] = {Real(.25), Real(-.5), Real(.75)};
    total += weights[a];
  }
  run.template operator()<1>();
  Vec<Real, 3> sum{0, 0, 0};
  for (auto g : out) sum += g;
  assert((sum - total).norm() < 10 * tol);
  assert(out[3].norm() == 0);
  assert(
      std::abs(
          dg[3]
          + (weights[3][0] + weights[3][1] + weights[3][2])
                / std::sqrt(Real(3)))
      < 10 * tol);
  run.template operator()<2>();
  for (int a = 0; a < 5; ++a) {
    assert(out[a].norm() < tol);
    assert((jvp[a] - velocity[a]).norm() < tol);
  }
  // Unused rebuilt hydrogens may have undefined intermediate geometry before
  // packing replaces them. Their zero cotangent must not poison retained atoms.
  xyz[1] = xyz[0];
  weights[3] = Vec<Real, 3>{0, 0, 0};
  run.template operator()<0>();
  assert(std::isnan(out[3][0]));
  run.template operator()<1>();
  for (int a = 0; a < 5; ++a) assert((out[a] - weights[a]).isZero());
  for (auto g : dg) assert(g == Real(0));
  // Do not suppress active singular geometry, even for a very small cotangent.
  weights[3] = Vec<Real, 3>::Constant(Real(1e-30));
  run.template operator()<1>();
  assert(!std::isfinite(out[0][0]) && !std::isfinite(dg[3]));
  // Strict retention tolerance, absent parent, NaNs and padded residues.
  Int input_types[] = {0, 0, 0, 0, -1};
  Vec<Real, 3> input_xyz[10];
  bool missing[10] = {}, retained[10];
  int64_t input_parent[] = {-1, 0};
  Real ideal[] = {0, 1};
  for (int b = 0; b < 5; ++b) {
    input_xyz[2 * b] = {0, 0, 0};
    input_xyz[2 * b + 1] = {1, 0, 0};
  }
  input_xyz[3] = {Real(1.5), 0, 0};
  missing[4] = true;
  input_xyz[7] = {std::numeric_limits<Real>::quiet_NaN(), 0, 0};
  HydrogenInputContext<Real, Int, CPU> input{
      view(input_types, {1, 5}),
      view(input_xyz, {1, 5, 2}),
      view(missing, {1, 5, 2}),
      view(input_parent, {1, 2}),
      view(ideal, {1, 2}),
      Real(.5),
      view(retained, {1, 5, 2})};
  launch_hydrogen_input<tmol_test::SerialOps, CPU>(mgr, input);
  for (int a = 0; a < 10; ++a) assert(retained[a] == (a == 1));
}
int main() {
  check<float, int32_t>();
  check<double, int32_t>();
  check<float, int64_t>();
  check<double, int64_t>();
}
