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
int main() {
  check<float, int32_t>();
  check<float, int64_t>();
  check<double, int32_t>();
  check<double, int64_t>();
}
