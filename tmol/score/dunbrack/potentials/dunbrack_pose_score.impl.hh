#pragma once

#include <tmol/utility/portable/matrix.hh>
#include <type_traits>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/unresolved_atom.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/dunbrack/potentials/dunbrack_pose_score.hh>

// Operator definitions; safe for CPU compilation

#include "dunbrack_pose_kernels.hh"

#include "potentials.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using tmol::portable::Vec;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    TView<Int, 1, D> block_is_mirrored,
    bool output_block_pair_energies,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);
  assert(block_is_mirrored.size(0) == n_block_types);

  int n_V = output_block_pair_energies ? max_n_blocks : 1;
  TPack<Real, 4, D> V_t = TPack<Real, 4, D>::zeros({3, n_poses, n_V, n_V});
  // Block-pair autograd recomputes coordinate derivatives in backward.
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({3, 0});

  // Both CUDA modes fully write every scratch element they consume.
  // Keep CPU initialization while avoiding separate CUDA memset launches.
  auto dihedral_atom_inds_t =
      D == Device::CPU
          ? TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih})
          : TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::empty({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = D == Device::CPU
                               ? TPack<Real, 2, D>::zeros({n_rots, max_n_dih})
                               : TPack<Real, 2, D>::empty({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  // Score-only and block-pair forward calls never consume derivative scratch.
  int const n_deriv_rots = accumulate_derivs ? n_rots : 0;
  auto dihedral_deriv_t =
      D == Device::CPU && accumulate_derivs
          ? TPack<CoordQuad<Real>, 2, D>::zeros({n_deriv_rots, max_n_dih})
          : TPack<CoordQuad<Real>, 2, D>::empty({n_deriv_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto rotameric_rottable_assignment_t =
      D == Device::CPU ? TPack<Int, 1, D>::zeros({n_rots})
                       : TPack<Int, 1, D>::empty({n_rots});
  auto rotameric_rottable_assignment = rotameric_rottable_assignment_t.view;

  auto semirotameric_rottable_assignment_t =
      D == Device::CPU ? TPack<Int, 1, D>::zeros({n_rots})
                       : TPack<Int, 1, D>::empty({n_rots});
  auto semirotameric_rottable_assignment =
      semirotameric_rottable_assignment_t.view;

  // These three terms consume their derivatives sequentially within each
  // rotamer's thread. Reuse one buffer after each contribution is accumulated.
  auto dterm_dtor_xyz_t =
      D == Device::CPU && accumulate_derivs
          ? TPack<CoordQuad<Real>, 2, D>::zeros({n_deriv_rots, 3})
          : TPack<CoordQuad<Real>, 2, D>::empty({n_deriv_rots, 3});
  auto dterm_dtor_xyz = dterm_dtor_xyz_t.view;

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  dunbrack_pose_forward<DeviceDispatch, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      rotameric_neglnprob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,
      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,
      block_n_dihedrals,
      block_dih_uaids,
      block_rotamer_table_set,
      block_rotameric_index,
      block_semirotameric_index,
      block_n_chi,
      block_n_rotameric_chi,
      block_probability_table_offset,
      block_mean_table_offset,
      block_rotamer_index_to_table_index,
      block_semirotameric_tableset_offset,
      block_is_mirrored,
      output_block_pair_energies,
      compute_derivs,
      V,
      dV_dx,
      dihedral_atom_inds,
      dihedral_values,
      dihedral_deriv,
      dterm_dtor_xyz,
      rotameric_rottable_assignment,
      semirotameric_rottable_assignment);
  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    TView<Int, 1, D> block_is_mirrored,

    TView<Real, 4, D> dTdV  // n_terms x n_poses x max_n_blocks x max_n_blocks
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);
  assert(block_is_mirrored.size(0) == n_block_types);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});

  // Each CUDA work item fully initializes the scratch for its rotamer; retain
  // the prior zero-initialized allocation on CPU.
  auto dihedral_atom_inds_t =
      D == Device::CPU
          ? TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih})
          : TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::empty({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = D == Device::CPU
                               ? TPack<Real, 2, D>::zeros({n_rots, max_n_dih})
                               : TPack<Real, 2, D>::empty({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      D == Device::CPU
          ? TPack<tmol::portable::Mat<Real, DIH_N_ATOMS, 3>, 2, D>::zeros(
                {n_rots, max_n_dih})
          : TPack<tmol::portable::Mat<Real, DIH_N_ATOMS, 3>, 2, D>::empty(
                {n_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  // Each term consumes its derivatives before the next term overwrites them.
  auto dterm_dtor_xyz_t =
      D == Device::CPU ? TPack<CoordQuad<Real>, 2, D>::zeros({n_rots, 3})
                       : TPack<CoordQuad<Real>, 2, D>::empty({n_rots, 3});
  auto dterm_dtor_xyz = dterm_dtor_xyz_t.view;

  auto dV_dx = dV_dx_t.view;

  dunbrack_pose_backward<DeviceDispatch, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      rotameric_neglnprob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,
      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,
      block_n_dihedrals,
      block_dih_uaids,
      block_rotamer_table_set,
      block_rotameric_index,
      block_semirotameric_index,
      block_n_chi,
      block_n_rotameric_chi,
      block_probability_table_offset,
      block_mean_table_offset,
      block_rotamer_index_to_table_index,
      block_semirotameric_tableset_offset,
      block_is_mirrored,
      dTdV,
      dV_dx,
      dihedral_atom_inds,
      dihedral_values,
      dihedral_deriv,
      dterm_dtor_xyz);
  return dV_dx_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    TView<Int, 1, D> block_is_mirrored,
    bool output_block_pair_energies,

    bool compute_derivs

    ) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);
  assert(block_is_mirrored.size(0) == n_block_types);

  TPack<Real, 2, D> V_t;
  TPack<Int, 2, D> dispatch_indices_t;
  if (output_block_pair_energies) {
    V_t = TPack<Real, 2, D>::zeros({3, n_rots});
    dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_rots});
  } else {
    V_t = TPack<Real, 2, D>::zeros({3, n_poses});
    dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_poses});
  }
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({3, 0});

  // Scratch is written before use; derivative terms reuse one buffer.
  auto dihedral_atom_inds_t =
      TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::empty({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = TPack<Real, 2, D>::empty({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  int const n_deriv_rots = accumulate_derivs ? n_rots : 0;
  auto dihedral_deriv_t =
      TPack<CoordQuad<Real>, 2, D>::empty({n_deriv_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto dterm_dtor_xyz_t =
      TPack<CoordQuad<Real>, 2, D>::empty({n_deriv_rots, 3});
  auto dterm_dtor_xyz = dterm_dtor_xyz_t.view;

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  dunbrack_pose_forward<DeviceDispatch, D, Real, Int, int64_t, 2>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      rotameric_neglnprob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,
      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,
      block_n_dihedrals,
      block_dih_uaids,
      block_rotamer_table_set,
      block_rotameric_index,
      block_semirotameric_index,
      block_n_chi,
      block_n_rotameric_chi,
      block_probability_table_offset,
      block_mean_table_offset,
      block_rotamer_index_to_table_index,
      block_semirotameric_tableset_offset,
      block_is_mirrored,
      output_block_pair_energies,
      compute_derivs,
      V,
      dV_dx,
      dihedral_atom_inds,
      dihedral_values,
      dihedral_deriv,
      dterm_dtor_xyz,
      {},
      {},
      pose_ind_for_rot,
      block_ind_for_rot,
      block_type_ind_for_rot,
      dispatch_indices);
  return {V_t, dV_dx_t, dispatch_indices_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    TView<Int, 1, D> block_is_mirrored,

    TView<Real, 2, D> dTdV  // n_terms x n_rotamers
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);
  assert(block_is_mirrored.size(0) == n_block_types);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});

  auto dihedral_atom_inds_t =
      TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = TPack<Real, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      TPack<tmol::portable::Mat<Real, DIH_N_ATOMS, 3>, 2, D>::zeros(
          {n_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  // Each term consumes its derivatives before the next term overwrites them.
  auto dterm_dtor_xyz_t = TPack<CoordQuad<Real>, 2, D>::zeros({n_rots, 3});
  auto dterm_dtor_xyz = dterm_dtor_xyz_t.view;

  auto dV_dx = dV_dx_t.view;

  dunbrack_pose_backward<DeviceDispatch, D, Real, Int, int64_t, 2>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      rotameric_neglnprob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,
      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,
      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,
      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,
      block_n_dihedrals,
      block_dih_uaids,
      block_rotamer_table_set,
      block_rotameric_index,
      block_semirotameric_index,
      block_n_chi,
      block_n_rotameric_chi,
      block_probability_table_offset,
      block_mean_table_offset,
      block_rotamer_index_to_table_index,
      block_semirotameric_tableset_offset,
      block_is_mirrored,
      dTdV,
      dV_dx,
      dihedral_atom_inds,
      dihedral_values,
      dihedral_deriv,
      dterm_dtor_xyz,
      pose_ind_for_rot,
      block_ind_for_rot,
      block_type_ind_for_rot);
  return dV_dx_t;
}

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
