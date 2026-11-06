#pragma once
#include <tmol/io/details/compiled/leaf_atoms.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/gen_coord.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <cmath>
namespace tmol::io::details::compiled {
// Common chemistry/coordinates; native launches borrow all storage. Callers
// zero the output to retain padded-coordinate and gradient accumulation rules.
template <typename Real, typename Int, Device Dev>
struct LeafAtomContext {
  TView<Vec<Real, 3>, 2, Dev> orig_coords;
  TView<bool, 3, Dev> orig_coords_atom_missing;
  TView<bool, 2, Dev> pose_stack_atom_missing;
  TView<Int, 2, Dev> pose_stack_block_coord_offset;
  TView<Int, 2, Dev> pose_stack_block_type;
  TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections;
  TView<Int, 1, Dev> block_type_n_atoms;
  TView<Int, 3, Dev> block_type_atom_downstream_of_conn;
  TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors;
  TView<Real, 3, Dev> block_type_atom_icoors;
  TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup;
  TView<Real, 3, Dev> block_type_atom_icoors_backup;
};

template <
    bool Backward,
    template <Device> class DeviceOps,
    Device Dev,
    typename Real,
    typename Int,
    typename Context>
void launch_pose_leaf_atoms(
    Context& mgr,
    LeafAtomContext<Real, Int, Dev> c,
    TView<Vec<Real, 3>, 2, Dev> dE_d_new_coords,
    TView<Vec<Real, 3>, 2, Dev> output) {
  int const n_poses = c.orig_coords.size(0);
  int const max_n_pose_atoms = c.orig_coords.size(1);
  int const max_n_blocks = c.pose_stack_block_type.size(1);
  int const max_n_conn = c.pose_stack_inter_block_connections.size(2);
  int const n_block_types = c.block_type_n_atoms.size(0);
  int const max_n_block_atoms = c.block_type_atom_downstream_of_conn.size(2);

  assert(c.orig_coords_atom_missing.size(0) == n_poses);
  assert(c.orig_coords_atom_missing.size(1) == max_n_blocks);
  assert(c.orig_coords_atom_missing.size(2) == max_n_block_atoms);
  assert(c.pose_stack_atom_missing.size(0) == n_poses);
  assert(c.pose_stack_atom_missing.size(1) == max_n_pose_atoms);
  assert(c.pose_stack_block_coord_offset.size(0) == n_poses);
  assert(c.pose_stack_block_type.size(0) == n_poses);
  assert(c.pose_stack_inter_block_connections.size(0) == n_poses);
  assert(c.pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(c.block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(c.block_type_atom_downstream_of_conn.size(1) == max_n_conn);
  assert(c.block_type_atom_ancestors.size(0) == n_block_types);
  assert(c.block_type_atom_ancestors.size(1) == max_n_block_atoms);
  assert(
      c.block_type_atom_ancestors.size(2)
      == 3);  // parent, grandparent, great grandparent
  assert(c.block_type_atom_icoors.size(0) == n_block_types);
  assert(c.block_type_atom_icoors.size(1) == max_n_block_atoms);
  assert(c.block_type_atom_icoors.size(2) == 3);  // phi, theta, D
  assert(c.block_type_atom_ancestors_backup.size(0) == n_block_types);
  assert(c.block_type_atom_ancestors_backup.size(1) == max_n_block_atoms);
  assert(
      c.block_type_atom_ancestors_backup.size(2)
      == 3);  // parent, grandparent, great grandparent
  assert(c.block_type_atom_icoors_backup.size(0) == n_block_types);
  assert(c.block_type_atom_icoors_backup.size(1) == max_n_block_atoms);
  assert(c.block_type_atom_icoors_backup.size(2) == 3);  // phi, theta, D

  LAUNCH_BOX_32;

  auto f_coord_builder = ([c,
                           max_n_blocks,
                           max_n_block_atoms,
                           output,
                           dE_d_new_coords] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / (max_n_blocks * max_n_block_atoms);
    ind = ind - (pose_ind * max_n_blocks * max_n_block_atoms);
    int const block_ind = ind / max_n_block_atoms;
    int const atom_ind = ind % max_n_block_atoms;
    int const block_type = c.pose_stack_block_type[pose_ind][block_ind];
    if (block_type < 0) {
      return;
    }
    int const n_block_type_atoms = c.block_type_n_atoms[block_type];
    if (atom_ind >= n_block_type_atoms) {
      return;
    }
    auto const block_offset =
        c.pose_stack_block_coord_offset[pose_ind][block_ind];

    // is this an atom we should build coordinates for?
    if (c.orig_coords_atom_missing[pose_ind][block_ind][atom_ind]) {
      auto geom = select_leaf_ancestors<Int, Real, Dev>::find_ancestors(
          pose_ind,
          block_ind,
          atom_ind,
          block_type,
          c.pose_stack_block_coord_offset,
          c.pose_stack_block_type,
          c.pose_stack_inter_block_connections,
          c.block_type_atom_downstream_of_conn,
          c.block_type_atom_ancestors,
          c.block_type_atom_icoors,
          c.block_type_atom_ancestors_backup,
          c.block_type_atom_icoors_backup,
          c.pose_stack_atom_missing);

      // An atom that cannot be placed is left NaN rather than at the origin
      // new_coords was allocated with; the caller checks for NaN and must not
      // be handed a plausible-looking coordinate in its place.
      Vec<Real, 3> const unbuildable = Vec<Real, 3>::Constant(Real(NAN));

      if (geom.anc0 == -1) {
        if constexpr (!Backward)
          output[pose_ind][block_offset + atom_ind] = unbuildable;
        return;
      }

      Vec<Real, 3> coord0 = c.orig_coords[pose_ind][geom.anc0];
      Vec<Real, 3> coord1 = c.orig_coords[pose_ind][geom.anc1];
      Vec<Real, 3> coord2 = c.orig_coords[pose_ind][geom.anc2];

      // an ancestor that is itself missing leaves this atom unbuildable
      if (std::isnan(coord0[0]) || std::isnan(coord0[1])
          || std::isnan(coord0[2]) || std::isnan(coord1[0])
          || std::isnan(coord1[1]) || std::isnan(coord1[2])
          || std::isnan(coord2[0]) || std::isnan(coord2[1])
          || std::isnan(coord2[2])) {
        if constexpr (!Backward)
          output[pose_ind][block_offset + atom_ind] = unbuildable;
        return;
      }

      if constexpr (Backward) {
        auto coord_derivs = score::common::build_coordinate<Real>::dV(
            coord0, coord1, coord2, geom.D, geom.theta, geom.phi);

        Vec<Real, 3> dE_dH = dE_d_new_coords[pose_ind][block_offset + atom_ind];

        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            output[pose_ind][geom.anc0], coord_derivs.dp * dE_dH);
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            output[pose_ind][geom.anc1], coord_derivs.dgp * dE_dH);
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            output[pose_ind][geom.anc2], coord_derivs.dggp * dE_dH);

      } else {
        Vec<Real, 3> new_coord = score::common::build_coordinate<Real>::V(
            coord0, coord1, coord2, geom.D, geom.theta, geom.phi);

        output[pose_ind][block_offset + atom_ind] = new_coord;
      }
    } else {
      if constexpr (Backward) {
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            output[pose_ind][block_offset + atom_ind],
            dE_d_new_coords[pose_ind][block_offset + atom_ind]);
      } else {
        output[pose_ind][block_offset + atom_ind] =
            c.orig_coords[pose_ind][block_offset + atom_ind];
      }
    }
  });
  int const n_atoms = score::common::checked_dispatch_product(
      score::common::checked_count_product(
          n_poses, max_n_blocks, "leaf construction blocks"),
      max_n_block_atoms,
      "leaf construction atoms");
  DeviceOps<Dev>::template forall<launch_t>(mgr, n_atoms, f_coord_builder);
}
}  // namespace tmol::io::details::compiled
