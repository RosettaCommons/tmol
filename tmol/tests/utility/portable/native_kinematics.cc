#include <tmol/kinematics/compiled/native_kinematics.hh>
#include "native_test_ops.hh"
#include <cmath>
#include <stdexcept>

using namespace tmol::kinematics;
using tmol_test::view;

template <class Real, class A, class B>
void same_transform(A const& a, B const& b) {
  Real tolerance = sizeof(Real) == 4 ? Real(2e-5) : Real(1e-11);
  if (!a.allFinite() || !b.allFinite()
      || (a - b).cwiseAbs().maxCoeff() > tolerance)
    throw std::runtime_error("kinematic transform round trip failed");
}

template <class Real>
void check() {
  constexpr auto D = tmol::Device::CPU;
  using Math = common<D, Real, int>;
  // Ordinary and near-collinear frame atoms. The established collinearity
  // repair must produce finite, invertible frames for disconnected jumps too.
  KinCoord<Real> xyz[4];
  xyz[0] << 0, 0, 0;
  xyz[1] << 1, 0, 0;
  xyz[2] << 2, Real(1e-4), 0;
  xyz[3] << Real(0.5), Real(1.2), Real(0.7);
  int parent[] = {0, 0, 1, 0};
  int fx[] = {0, 1, 2, 3}, fy[] = {0, 0, 1, 1};
  int fz[] = {0, 2, 0, 2}, kind[] = {ROOT, JUMP, JUMP, JUMP};
  KinTransform<Real> frames[4];
  KinDofs<Real> dofs[4];
  tmol::ContextManager mgr;
  inverse_kinematics_native<tmol_test::SerialOps, D, Real, int>(
      mgr,
      view(xyz, {4}),
      view(parent, {4}),
      view(fx, {4}),
      view(fy, {4}),
      view(fz, {4}),
      view(kind, {4}),
      view(frames, {4}),
      view(dofs, {4}));
  if (!dofs[0].isZero()) throw std::runtime_error("root DOFs are nonzero");
  for (int i = 1; i < 4; ++i) {
    KinTransform<Real> reconstructed =
        Math::jumpTransform(dofs[i]) * frames[parent[i]];
    same_transform<Real>(reconstructed, frames[i]);
    if ((reconstructed.template block<1, 3>(3, 0).transpose() - xyz[i]).norm()
        > Real(2e-5))
      throw std::runtime_error("inverse kinematics changed coordinates");
  }
  for (int i = 0; i < 11; ++i) {
    KinDofs<Real> bond = KinDofs<Real>::Zero();
    bond[DOF_phip] = Real(-2.1 + i * 0.37);
    bond[DOF_theta] = Real(0.4 + i * 0.13);
    bond[DOF_D] = Real(1.1 + i * 0.07);
    bond[DOF_phic] = Real(0.2 + i * 0.19);
    auto transform = Math::bondTransform(bond);
    same_transform<Real>(
        Math::bondTransform(Math::invBondTransform(transform)), transform);
  }
  // Empty batches do not access either their inputs or uninitialized scratch.
  inverse_kinematics_native<tmol_test::SerialOps, D, Real, int>(
      mgr,
      view(xyz, {0}),
      view(parent, {0}),
      view(fx, {0}),
      view(fy, {0}),
      view(fz, {0}),
      view(kind, {0}),
      view(frames, {0}),
      view(dofs, {0}));
}

int main() {
  check<float>();
  check<double>();
}
