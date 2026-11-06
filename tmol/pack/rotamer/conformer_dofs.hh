#pragma once

#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/utility/portable/device.hh>
#include <cmath>
#include <tmol/score/common/accumulate.hh>

namespace tmol::pack::rotamer {
// Static topology and caller-owned DOFs. All indices address compact forests
// including their virtual-root row only where explicitly added below.
template <template <typename, int> class View, typename Real, typename Int>
struct MainchainDofCopy {
  View<Int, 1> source_types, source_offsets, target_types, target_offsets;
  View<Int, 1> source_fingerprint;
  View<Int, 2> source_atoms;
  View<Int, 4> target_atoms;
  View<Int, 2> kinematic_index;
  View<Real, 2> source_dofs, target_dofs;
  int sampler;
};

template <typename Real, typename Ctx>
TMOL_DEVICE_FUNC void copy_mainchain_dof(Ctx const& c, int index) {
  int const width = c.source_atoms.size(1);
  int const row = index / width, atom = index % width;
  auto source_type = c.source_types[row];
  auto fingerprint = c.source_fingerprint[source_type];
  if (fingerprint < 0) return;
  auto source_atom = c.source_atoms[source_type][atom];
  auto target_type = c.target_types[row];
  auto target_atom = c.target_atoms[c.sampler][fingerprint][target_type][atom];
  if (source_atom < 0 || target_atom < 0) return;
  auto source = c.kinematic_index[source_type][source_atom];
  auto target = c.kinematic_index[target_type][target_atom];
  if (source < 0 || target < 0) return;
  source += c.source_offsets[row] + 1;
  target += c.target_offsets[row] + 1;
  for (int dof = 0; dof < 9; ++dof) {
    Real value = c.source_dofs[source][dof];
    // Missing side chains can leave dependent mainchain hydrogens unresolved.
    // Preserve the already initialized ideal DOF for each nonfinite component.
    if (std::isfinite(value)) c.target_dofs[target][dof] = value;
  }
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Context,
    typename Ctx>
void launch_copy_mainchain_dofs(Context& context, Ctx c) {
  LAUNCH_BOX_128;
  int count = score::common::checked_dispatch_product(
      c.source_types.size(0),
      c.source_atoms.size(1),
      "conformer mainchain DOFs");
  Ops<D>::template forall<launch_t>(
      context, count, [=] TMOL_DEVICE_FUNC(int i) {
        copy_mainchain_dof<Real>(c, i);
      });
}

template <template <typename, int> class View, typename Real, typename Int>
struct ChiDofAssignment {
  View<Int, 1> types, offsets, conformers;
  View<int32_t, 2> atoms;
  View<Real, 2> chi;
  View<Int, 2> kinematic_index;
  View<Real, 2> ring_corrections, dofs;
};

template <typename Real, typename Ctx>
TMOL_DEVICE_FUNC void assign_chi_dof(Ctx const& c, int index) {
  int const width = c.atoms.size(1);
  int const row = index / width, column = index % width;
  int const atom = c.atoms[row][column];
  if (atom < 0) return;
  auto conformer = c.conformers[row];
  auto type = c.types[conformer];
  auto destination = c.kinematic_index[type][atom] + c.offsets[conformer] + 1;
  c.dofs[destination][3] = c.chi[row][column] - c.ring_corrections[type][atom];
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Context,
    typename Ctx>
void launch_assign_chi_dofs(Context& context, Ctx c) {
  LAUNCH_BOX_128;
  int count = score::common::checked_dispatch_product(
      c.atoms.size(0), c.atoms.size(1), "conformer chi DOFs");
  Ops<D>::template forall<launch_t>(
      context, count, [=] TMOL_DEVICE_FUNC(int i) {
        assign_chi_dof<Real>(c, i);
      });
}
// Copy complete current conformers without materializing two per-atom index
// arrays. Packed offsets delimit only the selected conformers; source and
// target offsets address their respective coordinate or internal-DOF layouts.
template <template <typename, int> class View, typename Real, typename Int>
struct ConformerRowCopy {
  View<Int, 1> source_offsets, target_offsets, packed_offsets;
  View<Real, 2> source_values, target_values;
};

struct ConformerRowIndices {
  int64_t source, target;
};
template <typename Ctx>
TMOL_DEVICE_FUNC ConformerRowIndices
conformer_row_lookup(Ctx const& c, int row, int root_shift) {
  int begin = 0, end = c.source_offsets.size(0);
  // Find the selected conformer containing this packed atom. Equal offsets
  // skip empty conformers, including leading and trailing empty segments.
  while (begin + 1 < end) {
    int middle = begin + (end - begin) / 2;
    if (c.packed_offsets[middle] <= row)
      begin = middle;
    else
      end = middle;
  }
  auto local = row - c.packed_offsets[begin];
  int64_t source = c.source_offsets[begin] + local + root_shift;
  int64_t target = c.target_offsets[begin] + local + root_shift;
  return {source, target};
}

template <typename Ctx>
TMOL_DEVICE_FUNC void copy_conformer_row(
    Ctx const& c, int row, int root_shift) {
  auto indices = conformer_row_lookup(c, row, root_shift);
  auto source = indices.source, target = indices.target;
  for (int component = 0; component < c.target_values.size(1); ++component)
    c.target_values[target][component] = c.source_values[source][component];
}

template <template <Device> class Ops, Device D, typename Context, typename Ctx>
void launch_copy_conformer_rows(
    Context& context, Ctx c, int64_t n_rows, int root_shift) {
  LAUNCH_BOX_128;
  int count =
      score::common::checked_dispatch_size(n_rows, "conformer row copy");
  Ops<D>::template forall<launch_t>(
      context, count, [=] TMOL_DEVICE_FUNC(int i) {
        copy_conformer_row(c, i, root_shift);
      });
}

template <template <typename, int> class View, typename Int>
struct ConformerRowIndexPlan {
  View<Int, 1> source_offsets, target_offsets, packed_offsets;
  View<int64_t, 2> indices;
};
template <template <Device> class Ops, Device D, typename Context, typename Ctx>
void launch_conformer_row_indices(Context& context, Ctx c, int root_shift) {
  LAUNCH_BOX_128;
  int n = score::common::checked_dispatch_size(
      c.indices.size(1), "conformer row indices");
  Ops<D>::template forall<launch_t>(context, n, [=] TMOL_DEVICE_FUNC(int row) {
    auto i = conformer_row_lookup(c, row, root_shift);
    c.indices[0][row] = i.source;
    c.indices[1][row] = i.target;
  });
}

// Projected-vector atan2 definition used by tmol.numeric.coord_dihedrals.
// The existing implementation credits the Praxeolitic construction:
// https://stackoverflow.com/questions/20305272
TMOL_HOST_DEVICE_FUNC double conformer_coordinate_dihedral(
    double const xyz[4][3]) {
  double ba[3], bc[3], cd[3], axis[3], v[3], w[3];
  double length2 = 0, ba_axis = 0, cd_axis = 0;
  for (int i = 0; i < 3; ++i) {
    ba[i] = xyz[0][i] - xyz[1][i];
    bc[i] = xyz[2][i] - xyz[1][i];
    cd[i] = xyz[3][i] - xyz[2][i];
    length2 += bc[i] * bc[i];
  }
  double length = std::sqrt(length2);
  for (int i = 0; i < 3; ++i) {
    axis[i] = bc[i] / length;
    ba_axis += ba[i] * axis[i];
    cd_axis += cd[i] * axis[i];
  }
  for (int i = 0; i < 3; ++i) {
    v[i] = ba[i] - ba_axis * axis[i];
    w[i] = cd[i] - cd_axis * axis[i];
  }
  double x = 0, y = 0;
  for (int i = 0; i < 3; ++i) {
    int j = (i + 1) % 3, k = (i + 2) % 3;
    x += v[i] * w[i];
    y += (axis[j] * v[k] - axis[k] * v[j]) * w[i];
  }
  return std::atan2(y, x);
}

template <template <typename, int> class View, typename Real, typename Int>
struct ChiDofCorrection {
  View<Int, 1> types, offsets, conformers;
  View<int32_t, 2> atoms;
  View<Real, 2> chi, trial_coords;
  View<Int, 3> chi_atoms;
  View<Int, 2> kinematic_index;
  View<Real, 2> dofs;
};

template <Device D, typename Real, typename Ctx>
TMOL_DEVICE_FUNC void correct_chi_dof(Ctx const& c, int index) {
  int width = c.atoms.size(1), row = index / width, column = index % width;
  int atom = c.atoms[row][column];
  if (atom < 0) return;
  auto conformer = c.conformers[row], type = c.types[conformer];
  auto offset = c.offsets[conformer];
  double xyz[4][3];
  for (int point = 0; point < 4; ++point) {
    auto local = c.chi_atoms[type][atom][point];
    if (local < 0) return;
    for (int component = 0; component < 3; ++component)
      xyz[point][component] = double(c.trial_coords[offset + local][component]);
  }
  // Float32 reconstruction historically computes geometry in double and
  // rounds the measured angle before forming its double-precision delta.
  // Float64 reconstruction retains the measured double precision.
  Real measured = Real(conformer_coordinate_dihedral(xyz));
  double delta = double(c.chi[row][column]) - double(measured);
  constexpr double pi = 3.14159265358979323846;
  double wrapped = std::fmod(delta + pi, 2 * pi);
  if (wrapped < 0) wrapped += 2 * pi;
  delta = wrapped - pi;
  auto destination = c.kinematic_index[type][atom] + offset + 1;
  Real correction = Real(delta);
  score::common::accumulate<D, Real>::add(c.dofs[destination][3], correction);
}

template <
    template <Device> class Ops,
    Device D,
    typename Real,
    typename Context,
    typename Ctx>
void launch_correct_chi_dofs(Context& context, Ctx c) {
  LAUNCH_BOX_128;
  int count = score::common::checked_dispatch_product(
      c.atoms.size(0), c.atoms.size(1), "conformer chi correction");
  Ops<D>::template forall<launch_t>(
      context, count, [=] TMOL_DEVICE_FUNC(int i) {
        correct_chi_dof<D, Real>(c, i);
      });
}

}  // namespace tmol::pack::rotamer
