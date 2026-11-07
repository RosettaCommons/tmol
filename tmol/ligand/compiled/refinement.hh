#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tmol {
namespace ligand {

// CPU distance-geometry input. Array storage belongs to the caller. Weights
// come from the common chemistry preparation; this header owns no defaults.
template <typename Real>
struct RefinementInput {
  const Real* coordinates;
  std::int64_t n_atoms, n_dims, n_exact, n_lower, n_upper;
  std::int64_t n_components, n_component_atoms, n_chiral;
  const std::int32_t* pairs;
  const Real* distances;
  const std::int32_t* component_offsets;
  const std::int32_t* component_atoms;
  const std::int32_t* chiral_atoms;
  const Real* chiral_sign;
  const Real* weights;  // exact, bound, plane, chiral, chiral margin
  bool four_dimensional;
  Real fourth_weight;
};

enum class RefinementStatus {
  success = 0,
  invalid_shape = 1,
  invalid_index = 2,
  eigensolver_failure = 3
};

template <typename Real>
RefinementStatus validate_refinement_input(const RefinementInput<Real>& p) {
  if (p.n_atoms < 0 || p.n_atoms > std::numeric_limits<std::int32_t>::max()
      || p.n_dims < 2 || p.n_dims > 4 || p.n_exact < 0 || p.n_lower < 0
      || p.n_upper < 0 || p.n_components < 0 || p.n_component_atoms < 0
      || p.n_chiral < 0 || p.weights == nullptr
      || (p.n_atoms && p.coordinates == nullptr)
      || (p.four_dimensional && p.n_dims != 4)
      || (p.n_dims < 3 && (p.n_components || p.n_chiral))) {
    return RefinementStatus::invalid_shape;
  }
  const auto max_count = std::numeric_limits<std::int32_t>::max();
  if (p.n_exact > max_count || p.n_lower > max_count || p.n_upper > max_count
      || p.n_components > max_count || p.n_component_atoms > max_count
      || p.n_chiral > max_count) return RefinementStatus::invalid_shape;
  const auto n_pairs = p.n_exact + p.n_lower + p.n_upper;
  if ((n_pairs && (p.pairs == nullptr || p.distances == nullptr))
      || (p.n_components && p.component_offsets == nullptr)
      || (p.n_component_atoms && p.component_atoms == nullptr)
      || (p.n_chiral && (p.chiral_atoms == nullptr || p.chiral_sign == nullptr))
      || (p.n_components == 0 && p.n_component_atoms != 0)) {
    return RefinementStatus::invalid_shape;
  }
  auto valid_atom = [&](std::int32_t atom) { return atom >= 0 && atom < p.n_atoms; };
  for (std::int64_t i = 0; i < 2 * n_pairs; ++i) {
    if (!valid_atom(p.pairs[i])) return RefinementStatus::invalid_index;
  }
  if (p.n_components) {
    if (p.component_offsets[0] != 0
        || p.component_offsets[p.n_components] != p.n_component_atoms) {
      return RefinementStatus::invalid_shape;
    }
    for (std::int64_t c = 0; c < p.n_components; ++c) {
      if (p.component_offsets[c] < 0
          || p.component_offsets[c + 1] <= p.component_offsets[c]
          || p.component_offsets[c + 1] > p.n_component_atoms) {
        return RefinementStatus::invalid_shape;
      }
    }
  }
  for (std::int64_t i = 0; i < p.n_component_atoms; ++i) {
    if (!valid_atom(p.component_atoms[i])) return RefinementStatus::invalid_index;
  }
  for (std::int64_t i = 0; i < 4 * p.n_chiral; ++i) {
    if (!valid_atom(p.chiral_atoms[i])) return RefinementStatus::invalid_index;
  }
  return RefinementStatus::success;
}

// All physics and analytic derivatives are evaluated in the coordinate dtype.
// Coordinates are [atom,dimension], with 2D supported for planar embedding.
// The caller validates buffer extents/dtypes; the common kernel validates the
// declared topology and returns an explicit failure instead of a partial score.
template <typename Real>
RefinementStatus refinement_value_gradient(
    const RefinementInput<Real>& p, Real& value, Real* gradient) {
  auto status = validate_refinement_input(p);
  if (status != RefinementStatus::success) return status;
  if (p.n_atoms && gradient == nullptr) return RefinementStatus::invalid_shape;
  if (p.n_atoms) std::fill_n(gradient, p.n_atoms * p.n_dims, Real(0));
  value = Real(0);
  using Vec = Eigen::Matrix<Real, 3, 1>;
  using Mat = Eigen::Matrix<Real, 3, 3>;
  auto point = [&](std::int32_t atom) -> Vec {
    const Real* x = p.coordinates + atom * p.n_dims;
    return Vec(x[0], x[1], x[2]);
  };
  auto scatter = [&](std::int32_t atom, const Vec& derivative) {
    Real* g = gradient + atom * p.n_dims;
    for (int d = 0; d < 3; ++d) g[d] += derivative[d];
  };

  Real pair_loss[3] = {Real(0), Real(0), Real(0)};
  const auto n_pairs = p.n_exact + p.n_lower + p.n_upper;
  for (std::int64_t k = 0; k < n_pairs; ++k) {
    const auto i = p.pairs[2 * k], j = p.pairs[2 * k + 1];
    Real delta[4], squared = Real(0);
    for (int d = 0; d < p.n_dims; ++d) {
      delta[d] = p.coordinates[i * p.n_dims + d] - p.coordinates[j * p.n_dims + d];
      squared += delta[d] * delta[d];
    }
    const Real distance = std::sqrt(std::max(squared, Real(1e-12)));
    const int group = k < p.n_exact ? 0 : (k < p.n_exact + p.n_lower ? 1 : 2);
    Real error = distance - p.distances[k];
    if (group == 1) error = std::min(error, Real(0));
    if (group == 2) error = std::max(error, Real(0));
    pair_loss[group] += error * error;
    const Real weight = p.weights[group == 0 ? 0 : 1];
    if (squared >= Real(1e-12)) {
      const Real scale = (Real(2) * error * weight) / distance;
      for (int d = 0; d < p.n_dims; ++d) {
        const Real derivative = scale * delta[d];
        gradient[i * p.n_dims + d] += derivative;
        gradient[j * p.n_dims + d] -= derivative;
      }
    }
  }
  value = p.weights[0] * pair_loss[0];
  value += p.weights[1] * pair_loss[1];
  value += p.weights[1] * pair_loss[2];

  for (std::int64_t c = 0; c < p.n_components; ++c) {
    const auto begin = p.component_offsets[c], end = p.component_offsets[c + 1];
    const Real count = Real(end - begin);
    Vec mean = Vec::Zero();
    for (auto i = begin; i < end; ++i) mean += point(p.component_atoms[i]);
    mean /= count;
    Mat covariance = Mat::Zero();
    for (auto i = begin; i < end; ++i) {
      const Vec centered = point(p.component_atoms[i]) - mean;
      covariance += centered * centered.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Mat> solver(covariance);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
      return RefinementStatus::eigensolver_failure;
    }
    const Vec normal = solver.eigenvectors().col(0);
    value += p.weights[2] * solver.eigenvalues()[0];
    // Include the derivative through centering. This also preserves scatter
    // semantics if the same atom is present in multiple planar components.
    Vec mean_derivative = Vec::Zero();
    for (auto i = begin; i < end; ++i) {
      const Vec centered = point(p.component_atoms[i]) - mean;
      mean_derivative += Real(2) * p.weights[2] * centered.dot(normal) * normal;
    }
    mean_derivative /= count;
    for (auto i = begin; i < end; ++i) {
      const Vec centered = point(p.component_atoms[i]) - mean;
      const Vec derivative = Real(2) * p.weights[2] * centered.dot(normal) * normal;
      scatter(p.component_atoms[i], derivative - mean_derivative);
    }
  }

  Real chiral_loss = Real(0);
  for (std::int64_t k = 0; k < p.n_chiral; ++k) {
    const auto* atom = p.chiral_atoms + 4 * k;
    const Vec origin = point(atom[0]);
    const Vec a = point(atom[1]) - origin;
    const Vec b = point(atom[2]) - origin;
    const Vec c = point(atom[3]) - origin;
    const Vec da = b.cross(c), db = c.cross(a), dc = a.cross(b);
    const Real volume = a.dot(da);
    const Real hinge = std::max(p.weights[4] - p.chiral_sign[k] * volume, Real(0));
    chiral_loss += hinge * hinge;
    const Real scale = -Real(2) * p.weights[3] * p.chiral_sign[k] * hinge;
    scatter(atom[1], scale * da);
    scatter(atom[2], scale * db);
    scatter(atom[3], scale * dc);
    scatter(atom[0], -scale * (da + db + dc));
  }
  value += p.weights[3] * chiral_loss;
  if (p.four_dimensional && p.fourth_weight > Real(0)) {
    Real fourth_loss = Real(0);
    for (std::int64_t i = 0; i < p.n_atoms; ++i) {
      const Real fourth = p.coordinates[i * p.n_dims + 3];
      fourth_loss += fourth * fourth;
      gradient[i * p.n_dims + 3] += Real(2) * p.fourth_weight * fourth;
    }
    value += p.fourth_weight * fourth_loss;
  }
  return RefinementStatus::success;
}

}  // namespace ligand
}  // namespace tmol
