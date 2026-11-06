#pragma once

#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/utility/portable/device.hh>
#include <cmath>

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

template <typename Ctx>
TMOL_DEVICE_FUNC void copy_conformer_row(
    Ctx const& c, int row, int root_shift) {
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
  auto source = c.source_offsets[begin] + local + root_shift;
  auto target = c.target_offsets[begin] + local + root_shift;
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

}  // namespace tmol::pack::rotamer
