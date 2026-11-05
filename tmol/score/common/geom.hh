#pragma once

#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

using tmol::portable::Vec;

#define def auto TMOL_HOST_DEVICE_FUNC

#define Real3 Vec<Real, 3>

// Generic over the 3-vector type so that terms can move from Eigen to
// tmol::portable::Vec one at a time: both spellings supply the operators used
// here, and each caller gets back whichever type it passed in.
template <typename Real>
struct distance {
  template <typename V3>
  struct V_dV_T_of {
    Real V;
    V3 dV_dA;
    V3 dV_dB;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB); }
  };

  typedef V_dV_T_of<Vec<Real, 3>> V_dV_T;

  template <typename V3>
  static def V(V3 A, V3 B) -> Real {
    return (A - B).norm();
  }

  template <typename V3>
  static def V_dV(V3 A, V3 B) -> V_dV_T_of<V3> {
    V3 delta = (A - B);
    Real V = delta.norm();

    if (V != 0) {
      // One reciprocal rather than six divisions. A division on Metal is an
      // IEEE-correct multi-instruction sequence under MTLMathModeSafe, and
      // this runs per atom pair in the derivative kernels.
      Real const inv_V = Real(1) / V;
      return {V, delta * inv_V, -delta * inv_V};
    } else {
      // Correct for nan, gradient is discontinuous across dist = 0
      return {V, V3({0.0, 0.0, 0.0}), V3({0.0, 0.0, 0.0})};
    }
  }
};

template <typename Real>
struct interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB); }
  };

  static def V(Real3 A, Real3 B) -> Real {
    auto CR = A.cross(B);
    auto z_unit = CR.normalized();

    auto A_norm = A.norm();
    auto B_norm = B.norm();

    return 2
           * tmol::portable::math::atan2(
               CR.dot(z_unit), A_norm * B_norm + A.dot(B));
  }

  static def V_dV(Real3 A, Real3 B) -> V_dV_T {
    auto CR = A.cross(B);
    auto z_unit = CR.normalized();

    auto A_norm = A.norm();
    auto B_norm = B.norm();
    // Two reciprocals rather than twelve divisions; see distance::V_dV.
    Real const inv_A_norm = Real(1) / A_norm;
    Real const inv_B_norm = Real(1) / B_norm;

    return {
        2
            * tmol::portable::math::atan2(
                CR.dot(z_unit), A_norm * B_norm + A.dot(B)),
        (A * inv_A_norm).cross(z_unit) * inv_A_norm,
        -(B * inv_B_norm).cross(z_unit) * inv_B_norm};
  }
};

template <typename Real>
struct pt_interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;
    Vec<Real, 3> dV_dC;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB, dV_dC); }
  };
  static def V(Real3 A, Real3 B, Real3 C) -> Real {
    Real3 BA = A - B;
    Real3 BC = C - B;

    return interior_angle<Real>::V(BA, BC);
  }

  static def V_dV(Real3 A, Real3 B, Real3 C) -> V_dV_T {
    Real3 BA = A - B;
    Real3 BC = C - B;

    auto angle = interior_angle<Real>::V_dV(BA, BC);
    return {angle.V, angle.dV_dA, -(angle.dV_dA + angle.dV_dB), angle.dV_dB};
  }
};

template <typename Real>
struct cos_interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB); }
  };

  static def V(Real3 A, Real3 B) -> Real {
    return A.dot(B) / (A.norm() * B.norm());
  }

  static def V_dV(Real3 A, Real3 B) -> V_dV_T {
    auto A_norm = A.norm();
    auto B_norm = B.norm();
    auto AB_norm = A.norm() * B.norm();

    auto cosAB = (A.dot(B) / AB_norm);

    return {
        cosAB,
        -cosAB * A / (A_norm * A_norm) + B / AB_norm,
        -cosAB * B / (B_norm * B_norm) + A / AB_norm};
  }
};

template <typename Real>
struct pt_cos_interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;
    Vec<Real, 3> dV_dC;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB, dV_dC); }
  };

  static def V(Real3 A, Real3 B, Real3 C) -> Real {
    Real3 BA = A - B;
    Real3 BC = C - B;

    return cos_interior_angle<Real>::V(BA, BC);
  }

  static def V_dV(Real3 A, Real3 B, Real3 C) -> V_dV_T {
    Real3 BA = A - B;
    Real3 BC = C - B;
    Real V;
    Real3 dV_dBA, dV_dBC;

    auto angle = cos_interior_angle<Real>::V_dV(BA, BC);

    return {angle.V, angle.dV_dA, -(angle.dV_dA + angle.dV_dB), angle.dV_dB};
  }
};

template <typename Real>
struct dihedral_angle {
  // atan2 retains the small signed sine near planar cis/trans geometries,
  // where acos of a float32 cosine can round a nonzero angle to zero/pi.
  // The analytic Blondel/Karplus coordinate derivatives below are unchanged.
  static def from_normals(Real3 A, Real3 B, Real3 G) -> Real {
    Real norm_G = G.norm();
    Real sine = norm_G == 0 ? Real(0) : -G.dot(A.cross(B)) / norm_G;
    // Preserve the historical negative-pi branch for exactly planar trans
    // geometry, including either sign of a zero scalar triple product.
    if (sine == 0) sine = -Real(0);
    return tmol::portable::math::atan2(sine, A.dot(B));
  }

  struct V_dV_T {
    Real V;
    Real3 dV_dI;
    Real3 dV_dJ;
    Real3 dV_dK;
    Real3 dV_dL;

    def astuple() -> auto { return make_tuple(V, dV_dI, dV_dJ, dV_dK, dV_dL); }
  };

  static def V(Real3 I, Real3 J, Real3 K, Real3 L) -> Real {
    // Blondel A, Karplus M. New formulation for derivatives of torsion angles
    // and improper torsion angles in molecular mechanics: Elimination of
    // singularities. J Comput Chem. 1996;17: 1132–1141.
    auto F = I - J;
    auto G = J - K;
    auto H = L - K;

    auto A = F.cross(G);
    auto B = H.cross(G);

    return from_normals(A, B, G);
  }

  static def V_dV(Real3 I, Real3 J, Real3 K, Real3 L) -> V_dV_T {
    // Blondel A, Karplus M. New formulation for derivatives of torsion angles
    // and improper torsion angles in molecular mechanics: Elimination of
    // singularities. J Comput Chem. 1996;17: 1132–1141.
    auto F = I - J;
    auto G = J - K;
    auto H = L - K;

    auto A = F.cross(G);
    auto B = H.cross(G);

    auto V = from_normals(A, B, G);

    return {
        V,
        -(G.norm() / A.dot(A)) * A,
        G.norm() / A.dot(A) * A + F.dot(G) / (A.dot(A) * G.norm()) * A
            - (H.dot(G) / (B.dot(B) * G.norm())) * B,
        -G.norm() / B.dot(B) * B - F.dot(G) / (A.dot(A) * G.norm()) * A
            + (H.dot(G) / (B.dot(B) * G.norm())) * B,
        G.norm() / B.dot(B) * B};
  }

  struct Gradient {
    Real3 dI, dJ, dK, dL;
  };

  // Directional derivative of the Blondel/Karplus gradient above. This avoids
  // finite differences and shares torsion curvature with every native caller.
  static def gradient_jvp(
      Real3 I,
      Real3 J,
      Real3 K,
      Real3 L,
      Real3 dI,
      Real3 dJ,
      Real3 dK,
      Real3 dL) -> Gradient {
    auto F = I - J, G = J - K, H = L - K;
    auto dF = dI - dJ, dG = dJ - dK, dH = dL - dK;
    auto A = F.cross(G), B = H.cross(G);
    auto dA = dF.cross(G) + F.cross(dG), dB = dH.cross(G) + H.cross(dG);
    Real u = G.norm(), a = A.dot(A), b = B.dot(B);
    Real du = G.dot(dG) / u, da = 2 * A.dot(dA), db = 2 * B.dot(dB);
    Real p = u / a, q = F.dot(G) / (a * u), r = H.dot(G) / (b * u), t = u / b;
    Real dp = (du - p * da) / a, dt = (du - t * db) / b;
    Real dq = (dF.dot(G) + F.dot(dG) - q * (da * u + a * du)) / (a * u);
    Real dr = (dH.dot(G) + H.dot(dG) - r * (db * u + b * du)) / (b * u);
    return {
        -dp * A - p * dA,
        (dp + dq) * A + (p + q) * dA - dr * B - r * dB,
        -dq * A - q * dA + (dr - dt) * B + (r - t) * dB,
        dt * B + t * dB};
  }
};

#undef Real3
#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
