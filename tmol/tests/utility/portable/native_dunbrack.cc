#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
#include <tmol/score/dunbrack/potentials/dunbrack_pose_kernels.hh>
#include "native_test_ops.hh"
#include <array>
#include <algorithm>
#include <tmol/score/common/pose_energy_accumulation.hh>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
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
using namespace tmol::score::dunbrack::potentials;
auto forward32 =
    &dunbrack_pose_forward<SerialOps, tmol::Device::CPU, float, int>;
auto backward32 =
    &dunbrack_pose_backward<SerialOps, tmol::Device::CPU, float, int>;
auto forward64 =
    &dunbrack_pose_forward<SerialOps, tmol::Device::CPU, double, int>;
auto backward64 =
    &dunbrack_pose_backward<SerialOps, tmol::Device::CPU, double, int>;

auto indices32 =
    &dunbrack_pose_forward<SerialOps, tmol::Device::CPU, double, int, int32_t>;

// Scratch must be overwritten before consumption. Exercise all three score
// channels, real and undefined torsions, mirrored defaults, padded blocks and
// residue types without a Dunbrack assignment. No framework allocator is used.
template <class Real, class EnergyReal = Real>
std::vector<EnergyReal> evaluate(bool poison, bool derivs, bool block_pairs) {
  using namespace tmol_test;
  using tmol::portable::Vec;
  using Quad = tmol::portable::Mat<Real, 4, 3>;
  constexpr int nr = 6, nd = 5, na = nr * 7;
  Real const fill = poison ? std::numeric_limits<Real>::quiet_NaN() : Real(0);
  int const ifill = poison ? 1000007 : 0;
  Vec<Real, 3> xyz[na];
  for (int i = 0; i < na; ++i) {
    xyz[i][0] = Real(0.7 * (i % 7));
    xyz[i][1] = Real(std::sin(i * 0.9));
    xyz[i][2] = Real(std::cos(i * 0.7));
  }
  int offsets[nr] = {0, 7, 14, 21, 28, 35};
  int first[nr] = {0, 1, 2, 3, 4, -1};
  int types[nr] = {0, 1, 2, -1, 1, 0};
  Vec<int, 2> connections[nr];
  for (auto& v : connections) v = Vec<int, 2>::Constant(-1);
  int downstream[3] = {-1, -1, -1};
  int ndih[3] = {3, 4, 0}, tables[3] = {0, 0, -1};
  int rotind[3] = {0, 0, -1}, semiind[3] = {-1, 0, -1};
  int nchi[3] = {1, 2, 0}, nrotchi[3] = {1, 1, 0};
  int zero[3] = {0, 0, 0}, mirrored[3] = {0, 1, 0};
  int rotmap[3] = {0, 0, 0};
  tmol::UnresolvedAtomID<int> uaids[3 * nd * 4];
  for (int t = 0; t < 3; ++t)
    for (int d = 0; d < nd; ++d)
      for (int a = 0; a < 4; ++a) uaids[(t * nd + d) * 4 + a] = {d + a, -1, -1};
  // Partially resolved phi must reset all four indices and its derivative.
  uaids[(1 * nd) * 4 + 2] = {-1, -1, -1};
  Real prob[16], means[16], sdev[16], semi[64];
  for (int i = 0; i < 16; ++i) {
    prob[i] = Real(1 + 0.03 * i);
    means[i] = Real(0.2 + 0.02 * i);
    sdev[i] = Real(0.8 + 0.01 * i);
  }
  for (int i = 0; i < 64; ++i) semi[i] = Real(2 + 0.02 * i);
  Vec<int64_t, 2> sizes2[1], strides2[1];
  Vec<int64_t, 3> sizes3[1], strides3[1];
  Vec<Real, 2> start2[1], step2[1], period2[1];
  Vec<Real, 3> start3[1], step3[1], period3[1];
  for (int i = 0; i < 3; ++i) {
    sizes3[0][i] = 4;
    strides3[0][i] = i == 0 ? 16 : (i == 1 ? 4 : 1);
    start3[0][i] = Real(-M_PI);
    step3[0][i] = Real(M_PI / 2);
    period3[0][i] = Real(2 * M_PI);
    if (i < 2) {
      sizes2[0][i] = 4;
      strides2[0][i] = i == 0 ? 4 : 1;
      start2[0][i] = start3[0][i];
      step2[0][i] = step3[0][i];
      period2[0][i] = period3[0][i];
    }
  }
  Vec<int, 4> atoms[nr * nd];
  Real angles[nr * nd];
  Quad angle_derivs[nr * nd], term_derivs[nr * 3];
  int assignments[nr], semiassignments[nr];
  for (auto& v : atoms) v = Vec<int, 4>::Constant(ifill);
  for (auto& v : angles) v = fill;
  for (auto& v : angle_derivs) v = Quad::Constant(fill);
  for (auto& v : term_derivs) v = Quad::Constant(fill);
  for (auto& v : assignments) v = ifill;
  for (auto& v : semiassignments) v = ifill;
  int nv = block_pairs ? 3 : 1;
  std::vector<EnergyReal> energy(3 * 2 * nv * nv, 0);
  Vec<Real, 3> gradient[3 * na];
  for (auto& v : gradient) v = Vec<Real, 3>::Zero();
  tmol::ContextManager mgr;
  dunbrack_pose_forward<::SerialOps, tmol::Device::CPU, Real, int>(
      mgr,
      view(xyz, {na}),
      view(offsets, {nr}),
      view(first, {2, 3}),
      view(types, {2, 3}),
      view(connections, {2, 3, 1}),
      view(downstream, {3, 1, 1}),
      view(prob, {1, 4, 4}),
      view(sizes2, {1}),
      view(strides2, {1}),
      view(means, {1, 4, 4}),
      view(sdev, {1, 4, 4}),
      view(sizes2, {1}),
      view(strides2, {1}),
      view(start2, {1}),
      view(step2, {1}),
      view(period2, {1}),
      view(rotmap, {3}),
      view(rotmap, {3}),
      view(semi, {1, 4, 4, 4}),
      view(sizes3, {1}),
      view(strides3, {1}),
      view(start3, {1}),
      view(step3, {1}),
      view(period3, {1}),
      view(ndih, {3}),
      view(uaids, {3, nd, 4}),
      view(tables, {3}),
      view(rotind, {3}),
      view(semiind, {3}),
      view(nchi, {3}),
      view(nrotchi, {3}),
      view(zero, {3}),
      view(zero, {3}),
      view(zero, {3}),
      view(zero, {3}),
      view(mirrored, {3}),
      block_pairs,
      derivs,
      view(energy.data(), {3, 2, nv, nv}),
      view(gradient, {3, na}),
      view(atoms, {nr, nd}),
      view(angles, {nr, nd}),
      view(angle_derivs, {nr, nd}),
      view(term_derivs, {nr, 3}),
      view(assignments, {nr}),
      view(semiassignments, {nr}));
  // Every channel has real nonzero science; a skipped evaluation cannot pass.
  for (int c = 0; c < 3; ++c) {
    Real total = 0;
    for (int i = 0; i < 2 * nv * nv; ++i) total += energy[c * 2 * nv * nv + i];
    if (!(total > 0)) throw std::runtime_error("inactive Dunbrack channel");
  }
  Real gradient_norm = 0;
  for (auto const& v : gradient)
    for (int k = 0; k < 3; ++k) {
      energy.push_back(v[k]);
      gradient_norm += std::abs(v[k]);
    }
  if (derivs && !block_pairs && !(gradient_norm > 0))
    throw std::runtime_error("inactive Dunbrack derivative");
  return energy;
}

template <class Real>
void check_workspace() {
  for (bool derivs : {false, true})
    for (bool block_pairs : {false, true}) {
      auto clean = evaluate<Real>(false, derivs, block_pairs);
      auto poisoned = evaluate<Real>(true, derivs, block_pairs);
      if (clean != poisoned)
        throw std::runtime_error("Dunbrack consumed uninitialized scratch");
      for (auto value : poisoned)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite result");
    }
}
void check_wide_accumulation() {
  auto wide = evaluate<float, double>(false, true, false);
  auto narrow = evaluate<float>(false, true, false);
  auto blocks = evaluate<float>(false, false, true);
  for (int term = 0; term < 3; ++term)
    for (int pose = 0; pose < 2; ++pose) {
      long double sum = 0;
      for (int cell = 0; cell < 9; ++cell)
        sum += blocks[(term * 2 + pose) * 9 + cell];
      if (wide[term * 2 + pose] != sum)
        throw std::runtime_error("wide total changed per-block float physics");
    }
  for (std::size_t i = 6; i < wide.size(); ++i)
    if (wide[i] != narrow[i])
      throw std::runtime_error("energy storage changed coordinate derivatives");

  // Completed float contributions must survive cancellation independently of
  // the order in which GPU workgroups arrive at a pose total.
  std::array<float, 3> contributions{-16777216.f, 1.f, 16777216.f};
  do {
    double total = 0;
    for (float contribution : contributions)
      emit_dunbrack_energy<tmol::Device::CPU>(
          tmol_test::view(&total, {1, 1}), 0, 0, 0, 0, false, contribution);
    if (total != 1) throw std::runtime_error("lost float energy contribution");
  } while (std::next_permutation(contributions.begin(), contributions.end()));
}

int main() {
  check_workspace<float>();
  check_workspace<double>();
  check_wide_accumulation();
}
