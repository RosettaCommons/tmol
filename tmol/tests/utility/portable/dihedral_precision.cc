#include <tmol/score/common/geom.hh>
#include <cmath>
#include <cstdio>
#include <initializer_list>

using tmol::portable::Vec;
using tmol::score::common::dihedral_angle;

template <typename Real>
Vec<Real, 3> vec(double x, double y, double z) {
  Vec<Real, 3> result;
  result[0] = Real(x);
  result[1] = Real(y);
  result[2] = Real(z);
  return result;
}

template <typename Real>
int check() {
  const double pi = std::acos(-1.0);
  const double tolerance = sizeof(Real) == 4 ? 4e-7 : 2e-14;
  double largest = 0;
  for (double theta :
       {1e-7, -1e-7, 1e-4, -1e-4, 0.7, -2.3, pi - 1e-4, -pi + 1e-4}) {
    auto I = vec<Real>(1, 0, 0), J = vec<Real>(0, 0, 0), K = vec<Real>(0, 0, 1);
    auto L = vec<Real>(std::cos(theta), std::sin(theta), 1);
    auto both = dihedral_angle<Real>::V_dV(I, J, K, L);
    Real value = dihedral_angle<Real>::V(I, J, K, L);
    double error = std::abs(std::remainder(double(value) - theta, 2 * pi));
    largest = std::fmax(largest, error);
    if (error > tolerance || value != both.V) {
      std::fprintf(
          stderr,
          "torsion %g returned %.12g (error %g)\n",
          theta,
          double(value),
          error);
      return 1;
    }
    // The fourth atom rotates on a unit circle about the central bond.
    // Its angle derivative is exactly one, independently of implementation.
    auto tangent = vec<Real>(-std::sin(theta), std::cos(theta), 0);
    if (std::abs(double(both.dV_dL.dot(tangent)) - 1) > 4 * tolerance) return 2;
    auto sum = both.dV_dI + both.dV_dJ + both.dV_dK + both.dV_dL;
    if (sum.norm() > 4 * tolerance) return 3;
    // A rigid rotation and translation preserve the physical torsion.
    auto rotate = [](Vec<Real, 3> p) { return vec<Real>(p[2], p[0], p[1]); };
    Real rotated =
        dihedral_angle<Real>::V(rotate(I), rotate(J), rotate(K), rotate(L));
    if (std::abs(double(rotated) - double(value)) > tolerance) return 4;
  }
  auto I = vec<Real>(1, 0, 0), J = vec<Real>(0, 0, 0), K = vec<Real>(0, 0, 1),
       L = vec<Real>(-1, 0, 1);
  // Keep the existing branch convention at an exactly planar trans geometry.
  if (dihedral_angle<Real>::V(I, J, K, L) >= 0) return 5;
  std::printf(
      "%zu-bit largest angular error %.12g\n", sizeof(Real) * 8, largest);
  return 0;
}

int main() {
  int result = check<float>();
  return result ? result : check<double>();
}
