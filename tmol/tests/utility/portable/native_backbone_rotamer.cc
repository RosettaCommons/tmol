#include <tmol/score/backbone_torsion/potentials/backbone_rotamer_kernels.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <cstdint>
using namespace tmol::score::backbone_torsion::potentials;
using tmol_test::SerialOps;
using tmol_test::view;
template <typename Int>
void check() {
  tmol::ContextManager mgr;
  Int counts[] = {2, 3, 0, 0}, types[] = {0, 0, 0, -1};
  Int upper[] = {1}, first[] = {0, 2, 5, 5};
  tmol::portable::Vec<Int, 2> connections[8];
  for (auto& c : connections) c = tmol::portable::Vec<Int, 2>::Constant(-1);
  connections[1][0] = 1;
  connections[3][0] = 2;
  int64_t sizes[4] = {}, offsets[] = {0, 6, 6, 6};
  tmol::TView<Int, 2, tmol::Device::CPU> group_view{};
  auto count = [&] {
    backbone_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
        mgr,
        view(counts, {1, 4}),
        view(types, {1, 4}),
        view(connections, {1, 4, 2}),
        view(upper, {1}),
        view(sizes, {1, 4}),
        group_view);
  };
  count();
  assert(sizes[0] == 6 && sizes[1] == 0 && sizes[2] == 0 && sizes[3] == 0);
  Int owners[6] = {}, pairs[18] = {};
  auto fill = [&](int entries) {
    backbone_rotamer_fill_native<SerialOps, tmol::Device::CPU, double, Int>(
        mgr,
        view(first, {1, 4}),
        view(counts, {1, 4}),
        view(types, {1, 4}),
        view(connections, {1, 4, 2}),
        view(upper, {1}),
        view(offsets, {1, 4}),
        view(owners, {entries}),
        view(pairs, {3, entries}),
        group_view);
  };
  fill(6);
  for (int i = 0; i < 6; ++i) {
    assert(pairs[i] == 0);
    assert(pairs[6 + i] == i / 3);
    assert(pairs[12 + i] == 2 + i % 3);
  }
  Int groups[] = {5, 5, -1, -1};
  group_view = view(groups, {1, 4});
  counts[1] = 2;
  count();
  assert(sizes[0] == 2 && sizes[1] == 0);
  fill(2);
  assert(pairs[0] == 0 && pairs[1] == 0);
  assert(pairs[2] == 0 && pairs[3] == 1);
  assert(pairs[4] == 2 && pairs[5] == 3);
  group_view = {};
  counts[0] = counts[1] = 50000;
  count();
  assert(sizes[0] == INT64_C(2500000000));
  counts[0] = 2;
  counts[1] = 3;
  connections[3][0] = 0;
  sizes[0] = sizes[1] = 0;
  count();
  assert(sizes[0] == 6 && sizes[1] == 0);
  fill(6);
  for (int i = 0; i < 6; ++i) {
    assert(pairs[6 + i] == i / 3);
    assert(pairs[12 + i] == 2 + i % 3);
  }
  backbone_rotamer_count_native<SerialOps, tmol::Device::CPU, double, Int>(
      mgr,
      view(counts, {0, 4}),
      view(types, {0, 4}),
      view(connections, {0, 4, 2}),
      view(upper, {1}),
      view(sizes, {0, 4}));
}
template <typename Real, typename Int>
void instantiate() {
  auto volatile f = &backbone_rotamer_score_native<
      false,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  auto volatile b = &backbone_rotamer_score_native<
      true,
      SerialOps,
      tmol::Device::CPU,
      Real,
      Int>;
  (void)f;
  (void)b;
}
// Connected third-member identity is independent of the stored first type.
template <typename Int>
void check_third_member() {
  using tmol::Device;
  using tmol::UnresolvedAtomID;
  using tmol::score::common::resolve_rotamer_pair_atom_from_uaid;
  Int first[] = {0, 2, 4}, first_types[] = {0, 0, 0};
  Int types[] = {0, 1, 0, 0, 0, 0}, offsets[] = {0, 3, 6, 9, 12, 15};
  Int downstream[] = {0, 0, 1, 2};
  Int groups[] = {7, 7, 7};
  tmol::portable::Vec<Int, 2> conn[6];
  for (auto& c : conn) c = tmol::portable::Vec<Int, 2>::Constant(-1);
  conn[2][0] = 0;
  conn[2][1] = 1;
  auto resolve = [&](int r1, int r2, bool correlated) {
    return resolve_rotamer_pair_atom_from_uaid<Int, Device::CPU>(
        UnresolvedAtomID<Int>{-1, 0, 0},
        r1,
        r2,
        1,
        2,
        0,
        0,
        view(offsets, {6}),
        view(first, {1, 3}),
        view(first_types, {1, 3}),
        view(conn, {1, 3, 2}),
        view(downstream, {2, 2, 1}),
        correlated ? view(groups, {1, 3}) : tmol::TView<Int, 2, Device::CPU>{},
        view(types, {6}));
  };
  assert(resolve(3, 5, false) == 0);
  assert(resolve(3, 5, true) == 5);  // selected third type1 uses atom2
  groups[1] = -1;  // third member follows the other pair endpoint
  assert(resolve(2, 5, true) == 5);
  assert(resolve(3, 4, true) == 0);
  groups[2] = -1;
  assert(resolve(3, 5, true) == 0);  // independent fixed neighbor
  first[0] = -1;
  assert(resolve(3, 5, true) == -1);
}
// Verify nonzero Rama energy and all weighted coordinate derivatives against
// independently selected physical atoms, including an alternate third type.
template <typename Real, typename Int>
void check_correlated_science() {
  using tmol::Device;
  using V3 = tmol::portable::Vec<Real, 3>;
  using V2 = tmol::portable::Vec<Int, 2>;
  using UAID = tmol::UnresolvedAtomID<Int>;
  tmol::ContextManager mgr;
  V3 xyz[18];
  for (int i = 0; i < 18; ++i) {
    xyz[i][0] = Real(std::sin(i * 0.73) + i * 0.1);
    xyz[i][1] = Real(std::cos(i * 0.49));
    xyz[i][2] = Real(std::sin(i * 0.37 + 0.4));
  }
  Int offsets[] = {0, 3, 6, 9, 12, 15};
  Int first[] = {0, 2, 4}, first_types[] = {0, 0, 0};
  Int blocks[] = {0, 0, 1, 1, 2, 2}, types[] = {0, 1, 0, 0, 0, 0};
  Int downstream[] = {0, 0, 1, 2};
  Int rama_inds[] = {0, 0, 0, 0}, omega_inds[] = {-1, -1, -1, -1};
  Int upper[] = {1, 1}, pro[] = {0, 0};
  Int groups[] = {7, 7, 7}, indices[] = {0, 3, 5};
  V2 conn[6];
  for (auto& c : conn) c = V2::Constant(-1);
  conn[2][0] = 0;
  conn[2][1] = 1;
  conn[3][0] = 2;
  conn[3][1] = 0;
  UAID torsions[24];
  for (auto& u : torsions) u = UAID{-1, -1, -1};
  torsions[0] = UAID{-1, 0, 0};
  torsions[1] = UAID{0, -1, -1};
  torsions[2] = UAID{1, -1, -1};
  torsions[3] = UAID{2, -1, -1};
  torsions[4] = UAID{0, -1, -1};
  torsions[5] = UAID{1, -1, -1};
  torsions[6] = UAID{2, -1, -1};
  torsions[7] = UAID{-1, 1, 0};
  Real coeffs[64];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      coeffs[i * 8 + j] = Real(std::sin(i * 0.8) + 0.6 * std::cos(j * 0.7));
  RamaTableParams<Real> params[1];
  params[0].bbstarts[0] = params[0].bbstarts[1] = Real(-3.141592653589793);
  params[0].bbsteps[0] = params[0].bbsteps[1] = Real(0.7853981633974483);
  Real energy[2] = {}, weights[] = {Real(1.7), Real(-0.4)};
  V3 deriv[36];
  for (auto& v : deriv) v = V3::Zero();
  auto eval = [&]<bool Backward>() {
    backbone_rotamer_score_native<Backward, SerialOps, Device::CPU, Real, Int>(
        mgr,
        view(xyz, {18}),
        view(offsets, {6}),
        view(first, {1, 3}),
        view(first_types, {1, 3}),
        view(blocks, {6}),
        view(types, {6}),
        view(conn, {1, 3, 2}),
        view(downstream, {2, 2, 1}),
        view(rama_inds, {2, 2}),
        view(omega_inds, {2, 2}),
        view(upper, {2}),
        view(pro, {2}),
        view(torsions, {2, 12}),
        view(coeffs, {1, 8, 8}),
        view(params, {1}),
        {},
        {},
        view(indices, {3, 1}),
        view(energy, {2, 1}),
        view(deriv, {2, 18}),
        view(weights, {2, 1}),
        true,
        false,
        view(groups, {1, 3}));
  };
  for (int scenario = 0; scenario < 2; ++scenario) {
    // Both own-group and other-endpoint-group routing must select atom5.
    if (scenario) {
      groups[1] = -1;
      indices[1] = 2;
    }
    int own = offsets[indices[1]], next = offsets[indices[2]];
    auto physical_energy = [&] {
      CoordQuad<Real> phi, psi;
      int pa[] = {5, own, own + 1, own + 2},
          qa[] = {own, own + 1, own + 2, next};
      for (int i = 0; i < 4; ++i) {
        tmol::portable::row_ref(phi, i) = xyz[pa[i]];
        tmol::portable::row_ref(psi, i) = xyz[qa[i]];
      }
      return rama_V<Device::CPU, Real, Int>(
          phi,
          psi,
          view(coeffs, {8, 8}),
          tmol::portable::vec_from<Real, 2>(params[0].bbstarts),
          tmol::portable::vec_from<Real, 2>(params[0].bbsteps));
    };
    energy[0] = energy[1] = 0;
    for (auto& v : deriv) v = V3::Zero();
    eval.template operator()<false>();
    assert(std::abs(energy[0] - physical_energy()) < Real(2e-6));
    assert(std::abs(energy[0]) > Real(0.01));
    eval.template operator()<true>();
    Real h = sizeof(Real) == 4 ? Real(0.002) : Real(1e-5);
    Real tolerance = sizeof(Real) == 4 ? Real(0.0015) : Real(2e-7);
    for (int i = 0; i < 18; ++i)
      for (int j = 0; j < 3; ++j) {
        Real saved = xyz[i][j];
        xyz[i][j] = saved + h;
        Real plus = physical_energy();
        xyz[i][j] = saved - h;
        Real minus = physical_energy();
        xyz[i][j] = saved;
        Real fd = weights[0] * (plus - minus) / (2 * h);
        assert(std::abs(deriv[i][j] - fd) < tolerance);
        assert(deriv[18 + i][j] == 0);
      }
    assert(
        deriv[0].norm() == 0);  // unused first neighbor cannot receive gradient
    assert(deriv[5].norm() > Real(0.01));
  }
}

int main() {
  check_correlated_science<float, int32_t>();
  check_correlated_science<double, int32_t>();
  check_correlated_science<float, int64_t>();
  check_correlated_science<double, int64_t>();
  check_third_member<int32_t>();
  check_third_member<int64_t>();
  check<int32_t>();
  check<int64_t>();
  instantiate<float, int32_t>();
  instantiate<double, int32_t>();
  instantiate<float, int64_t>();
  instantiate<double, int64_t>();
}
