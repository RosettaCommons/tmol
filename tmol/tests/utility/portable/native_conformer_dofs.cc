#include <tmol/pack/rotamer/conformer_dofs.hh>
#include <tmol/score/common/tview_binder.hh>
#include "native_test_ops.hh"
#include <cassert>
#include <limits>

using tmol_test::view;
using namespace tmol::pack::rotamer;
using Binder = tmol::score::common::TViewBinder<tmol::Device::CPU>;

template <typename Real, typename Int>
void check() {
  tmol::ContextManager mgr;
  Int source_types[] = {0, 0}, source_offsets[] = {0, 0};
  Int target_types[] = {1, 1}, target_offsets[] = {2, 4};
  Int fingerprints[] = {0, -1}, source_atoms[] = {0, 1, -1, -1};
  Int target_atoms[] = {-1, -1, 1, -1}, kfo[] = {1, 0, 0, 1};
  Real source[27] = {}, target[63];
  std::fill(target, target + 63, Real(-7));
  for (int i = 0; i < 9; ++i) source[18 + i] = Real(i);
  source[22] = std::numeric_limits<Real>::quiet_NaN();
  source[23] = std::numeric_limits<Real>::infinity();
  source[24] = -std::numeric_limits<Real>::infinity();
  MainchainDofCopy<Binder::template View, Real, Int> c{
      view(source_types, {2}),
      view(source_offsets, {2}),
      view(target_types, {2}),
      view(target_offsets, {2}),
      view(fingerprints, {2}),
      view(source_atoms, {2, 2}),
      view(target_atoms, {1, 1, 2, 2}),
      view(kfo, {2, 2}),
      view(source, {3, 9}),
      view(target, {7, 9}),
      0};
  launch_copy_mainchain_dofs<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr, c);
  for (int row = 0; row < 7; ++row)
    for (int dof = 0; dof < 9; ++dof) {
      bool copied = (row == 4 || row == 6) && (dof < 4 || dof > 6);
      assert(target[row * 9 + dof] == (copied ? Real(dof) : Real(-7)));
    }
  Int conformers[] = {1, 0};
  int32_t atoms[] = {1, -1, 0, 1};
  Real chi[] = {
      Real(.5), std::numeric_limits<Real>::quiet_NaN(), Real(1), Real(2)};
  Real corrections[] = {0, 0, Real(.125), Real(.25)};
  ChiDofAssignment<Binder::template View, Real, Int> a{
      view(target_types, {2}),
      view(target_offsets, {2}),
      view(conformers, {2}),
      view(atoms, {2, 2}),
      view(chi, {2, 2}),
      view(kfo, {2, 2}),
      view(corrections, {2, 2}),
      view(target, {7, 9})};
  launch_assign_chi_dofs<tmol_test::SerialOps, tmol::Device::CPU, Real>(mgr, a);
  assert(target[6 * 9 + 3] == Real(.25));
  assert(target[3 * 9 + 3] == Real(.875));
  assert(target[4 * 9 + 3] == Real(1.75));
  assert(target[3] == Real(-7) && chi[0] == Real(.5) && chi[3] == Real(2));
  a.atoms = view(atoms, {0, 0});
  launch_assign_chi_dofs<tmol_test::SerialOps, tmol::Device::CPU, Real>(mgr, a);
  c.source_types =
      view(source_types, {int64_t(std::numeric_limits<int>::max())});
  bool overflow = false;
  try {
    launch_copy_mainchain_dofs<tmol_test::SerialOps, tmol::Device::CPU, Real>(
        mgr, c);
  } catch (std::overflow_error const&) {
    overflow = true;
  }
  assert(overflow);
}
template <typename Real, typename Int>
void check_rows() {
  tmol::ContextManager mgr;
  Int source_offsets[] = {0, 4, 0, 1, 0}, target_offsets[] = {0, 1, 0, 4, 0};
  Int packed_offsets[] = {0, 0, 2, 2, 5, 5};
  for (int width : {3, 9})
    for (int shift : {0, 1}) {
      Real source[90], target[90];
      for (int i = 0; i < 90; ++i) {
        source[i] = Real(i);
        target[i] = Real(-7);
      }
      source[5 * width + 1] = std::numeric_limits<Real>::quiet_NaN();
      ConformerRowCopy<Binder::template View, Real, Int> c{
          view(source_offsets, {5}),
          view(target_offsets, {5}),
          view(packed_offsets, {6}),
          view(source, {10, width}),
          view(target, {10, width})};
      launch_copy_conformer_rows<tmol_test::SerialOps, tmol::Device::CPU>(
          mgr, c, 5, shift);
      int destinations[] = {1, 2, 4, 5, 6}, origins[] = {4, 5, 1, 2, 3};
      for (int row = 0; row < 10; ++row)
        for (int component = 0; component < width; ++component) {
          Real expected = Real(-7);
          for (int i = 0; i < 5; ++i)
            if (row == destinations[i] + shift)
              expected = source[(origins[i] + shift) * width + component];
          Real actual = target[row * width + component];
          assert(
              actual == expected
              || (std::isnan(actual) && std::isnan(expected)));
        }
      launch_copy_conformer_rows<tmol_test::SerialOps, tmol::Device::CPU>(
          mgr, c, 0, shift);
    }
}

template <typename Real, typename Int>
void check_correction() {
  tmol::ContextManager mgr;
  constexpr double pi = 3.14159265358979323846;
  double xyz[4][3] = {{1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 1, 1}};
  assert(std::abs(conformer_coordinate_dihedral(xyz) + pi / 2) < 1e-15);
  // A proper rotation and translation preserve the signed quarter turn.
  for (auto& point : xyz) {
    double x = point[0];
    point[0] = -point[1] + 11;
    point[1] = x - 7;
    point[2] += 3;
  }
  assert(std::abs(conformer_coordinate_dihedral(xyz) + pi / 2) < 1e-15);
  Int types[] = {0}, offsets[] = {2}, conformers[] = {0};
  int32_t atoms[] = {3, -1, 2};
  Int chi_atoms[16];
  std::fill(chi_atoms, chi_atoms + 16, Int(-1));
  for (int i = 0; i < 4; ++i) chi_atoms[12 + i] = Int(i);
  Int kfo[] = {3, 0, 2, 1};
  Real chi[] = {0, 7, 8}, coords[18] = {}, dofs[63];
  std::fill(dofs, dofs + 63, Real(.25));
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j) coords[(i + 2) * 3 + j] = Real(xyz[i][j]);
  ChiDofCorrection<Binder::template View, Real, Int> c{
      view(types, {1}),
      view(offsets, {1}),
      view(conformers, {1}),
      view(atoms, {1, 3}),
      view(chi, {1, 3}),
      view(coords, {6, 3}),
      view(chi_atoms, {1, 4, 4}),
      view(kfo, {1, 4}),
      view(dofs, {7, 9})};
  launch_correct_chi_dofs<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr, c);
  for (int i = 0; i < 63; ++i) {
    Real expected = i == 4 * 9 + 3 ? Real(.25) + Real(pi / 2) : Real(.25);
    assert(
        std::abs(dofs[i] - expected)
        <= 2 * std::numeric_limits<Real>::epsilon());
  }
  assert(chi[0] == Real(0) && chi[1] == Real(7));
  c.atoms = view(atoms, {0, 0});
  launch_correct_chi_dofs<tmol_test::SerialOps, tmol::Device::CPU, Real>(
      mgr, c);
}
int main() {
  check_correction<float, int32_t>();
  check_correction<float, int64_t>();
  check_correction<double, int32_t>();
  check_correction<double, int64_t>();

  check_rows<float, int32_t>();
  check_rows<float, int64_t>();
  check_rows<double, int32_t>();
  check_rows<double, int64_t>();
  check<float, int32_t>();
  check<float, int64_t>();
  check<double, int32_t>();
  check<double, int64_t>();
}
