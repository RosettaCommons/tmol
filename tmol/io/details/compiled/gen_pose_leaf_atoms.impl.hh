#pragma once
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/io/details/compiled/leaf_atom_kernels.hh>
namespace tmol::io::details::compiled {
template <
    template <Device> class DeviceOps,
    Device Dev,
    typename Real,
    typename Int>
struct GeneratePoseLeafAtoms {
  static auto forward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 2, Dev> orig_coords,
      TView<bool, 3, Dev> orig_coords_atom_missing,
      TView<bool, 2, Dev> pose_stack_atom_missing,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // n-bt x max-n-ats x 3 x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,

      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors,

      // TEMP! Handle the case when an atom's coordinate depends on
      // an un-resolvable atom, e.g., "down" for an N-terminal atom
      // n-bt x max-n-ats x 3 x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup,
      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors_backup

      ) -> TPack<Vec<Real, 3>, 2, Dev> {
    auto result = TPack<Vec<Real, 3>, 2, Dev>::zeros(
        {orig_coords.size(0), orig_coords.size(1)});
    LeafAtomContext<Real, Int, Dev> c{
        orig_coords,
        orig_coords_atom_missing,
        pose_stack_atom_missing,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_n_atoms,
        block_type_atom_downstream_of_conn,
        block_type_atom_ancestors,
        block_type_atom_icoors,
        block_type_atom_ancestors_backup,
        block_type_atom_icoors_backup};
    launch_pose_leaf_atoms<false, DeviceOps, Dev, Real, Int>(
        mgr, c, result.view, result.view);
    return result;
  }
  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 2, Dev> dE_d_new_coords,
      TView<Vec<Real, 3>, 2, Dev> new_coords,
      TView<Vec<Real, 3>, 2, Dev> orig_coords,
      TView<bool, 3, Dev> orig_coords_atom_missing,
      TView<bool, 2, Dev> pose_stack_atom_missing,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // n-bt x max-n-ats x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,

      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors,

      // TEMP! Handle the case when an atom's coordinate depends on
      // an un-resolvable atom, e.g., "down" for an N-terminal atom
      // n-bt x max-n-ats x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup,
      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors_backup

      ) -> TPack<Vec<Real, 3>, 2, Dev> {
    auto result = TPack<Vec<Real, 3>, 2, Dev>::zeros(
        {orig_coords.size(0), orig_coords.size(1)});
    LeafAtomContext<Real, Int, Dev> c{
        orig_coords,
        orig_coords_atom_missing,
        pose_stack_atom_missing,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_n_atoms,
        block_type_atom_downstream_of_conn,
        block_type_atom_ancestors,
        block_type_atom_icoors,
        block_type_atom_ancestors_backup,
        block_type_atom_icoors_backup};
    launch_pose_leaf_atoms<true, DeviceOps, Dev, Real, Int>(
        mgr, c, dE_d_new_coords, result.view);
    return result;
  }
};
}  // namespace tmol::io::details::compiled
