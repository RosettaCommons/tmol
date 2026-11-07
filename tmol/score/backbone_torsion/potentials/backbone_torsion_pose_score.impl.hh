#pragma once
#include <tmol/score/backbone_torsion/potentials/backbone_rotamer_kernels.hh>

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/unresolved_atom.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>

#include <tmol/score/backbone_torsion/potentials/params.hh>
#include <tmol/score/backbone_torsion/potentials/potentials.hh>
#include <tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.hh>

// Operator definitions; safe for CPU compilation

#include "backbone_torsion_pose_kernels.hh"

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto BackboneTorsionPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> pose_ind_for_atom,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> pose_ind_for_rot,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Int, 1, Dev> n_rots_for_pose,
    TView<Int, 1, Dev> rot_offset_for_pose,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_lower_conn_ind,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
    bool output_block_pair_energies,
    bool compute_derivs)
    -> std::tuple<TPack<Real, 4, Dev>, TPack<Vec<Real, 3>, 2, Dev>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);
  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);
  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);
  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);
  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);
  int const n_V = output_block_pair_energies ? max_n_blocks : 1;
  auto V_t = TPack<Real, 4, Dev>::zeros({2, n_poses, n_V, n_V});
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dxyz_t = accumulate_derivs
                       ? TPack<Vec<Real, 3>, 2, Dev>::zeros({2, n_atoms})
                       : TPack<Vec<Real, 3>, 2, Dev>::empty({2, 0});
  backbone_torsion_pose_forward<DeviceDispatch, Dev, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_lower_conn_ind,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_torsion_atoms,
      rama_tables,
      rama_table_params,
      omega_tables,
      omega_table_params,
      output_block_pair_energies,
      compute_derivs,
      V_t.view,
      dV_dxyz_t.view);
  return {V_t, dV_dxyz_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto BackboneTorsionPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> pose_ind_for_atom,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> pose_ind_for_rot,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Int, 1, Dev> n_rots_for_pose,
    TView<Int, 1, Dev> rot_offset_for_pose,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_lower_conn_ind,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,

    TView<Real, 4, Dev> dTdV  // nterms x nposes x (len) x (len)
    ) -> TPack<Vec<Real, 3>, 2, Dev> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);
  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);
  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);
  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);
  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);
  auto dV_dxyz_t = TPack<Vec<Real, 3>, 2, Dev>::zeros({2, n_atoms});
  backbone_torsion_pose_backward<DeviceDispatch, Dev, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_lower_conn_ind,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_torsion_atoms,
      rama_tables,
      rama_table_params,
      omega_tables,
      omega_table_params,
      dTdV,
      dV_dxyz_t.view);
  return dV_dxyz_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto BackboneTorsionRotamerScoreDispatch<DeviceDispatch, Dev, Real, Int>::
    forward(
        ContextManager& mgr,
        // common params
        TView<Vec<Real, 3>, 1, Dev> rot_coords,
        TView<Int, 1, Dev> rot_coord_offset,
        TView<Int, 1, Dev> pose_ind_for_atom,
        TView<Int, 2, Dev> first_rot_for_block,
        TView<Int, 2, Dev> first_rot_block_type,
        TView<Int, 1, Dev> block_ind_for_rot,
        TView<Int, 1, Dev> pose_ind_for_rot,
        TView<Int, 1, Dev> block_type_ind_for_rot,
        TView<Int, 1, Dev> n_rots_for_pose,
        TView<Int, 1, Dev> rot_offset_for_pose,
        TView<Int, 2, Dev> n_rots_for_block,
        TView<Int, 2, Dev> rot_offset_for_block,
        Int max_n_rots_per_pose,

        TView<Int, 2, Dev> pose_stack_block_type,
        TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

        TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

        TView<Int, 2, Dev> block_type_rama_table,
        TView<Int, 2, Dev> block_type_omega_table,
        TView<Int, 1, Dev> block_type_lower_conn_ind,
        TView<Int, 1, Dev> block_type_upper_conn_ind,
        TView<Int, 1, Dev> block_type_is_pro,

        TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

        TView<Real, 3, Dev> rama_tables,
        TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
        TView<Real, 4, Dev> omega_tables,
        TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
        bool output_block_pair_energies,
        bool topology_only,
        TView<Int, 2, Dev> lockstep_group_for_block)
        -> std::tuple<
            TPack<Real, 2, Dev>,
            TPack<Vec<Real, 3>, 2, Dev>,
            TPack<Int, 2, Dev>> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conn = pose_stack_inter_block_connections.size(2);
  int const n_block_types = block_type_atom_downstream_of_conn.size(0);
  int const max_n_atoms_per_block_type =
      block_type_atom_downstream_of_conn.size(1);
  int const n_rama_tables = rama_tables.size(0);
  int const n_omega_tables = omega_tables.size(0);

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

  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);
  assert(block_type_lower_conn_ind.size(0) == n_block_types);
  assert(block_type_upper_conn_ind.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);

  LAUNCH_BOX_32;
  // Define nt
  CTA_LAUNCH_T_PARAMS;

  // RamaPrePro is a pseudo-1body term that's really a 2-body term:
  // the score for residue i changes if residue i+1 becomes proline.
  // Thus we will first count the number
  // sum(i<N-1, n_rots[i]*n_rots[i+1]) + n_rots[N-1] for each Pose
  // and allocate that many scores to cacluate, then calculate
  // them in the second step.

  auto n_energies_for_block_t =
      TPack<int64_t, 2, Dev>::zeros({n_poses, max_n_blocks});
  auto n_energies_for_block = n_energies_for_block_t.view;
  int const n_pose_block_cells = common::checked_dispatch_product(
      n_poses, max_n_blocks, "backbone-torsion count dispatch");
  backbone_rotamer_count_native<DeviceDispatch, Dev, Real, Int>(
      mgr,
      n_rots_for_block,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_upper_conn_ind,
      n_energies_for_block,
      lockstep_group_for_block);

  auto n_energies_for_block_offset_t =
      TPack<int64_t, 2, Dev>::zeros({n_poses, max_n_blocks});
  auto n_energies_for_block_offset = n_energies_for_block_offset_t.view;
  int64_t const n_dispatch_total_64 =
      DeviceDispatch<Dev>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          n_energies_for_block.data(),
          n_energies_for_block_offset.data(),
          n_pose_block_cells,
          tmol::portable::plus_t<int64_t>());
  int const n_dispatch_total = score::common::checked_dispatch_size(
      n_dispatch_total_64, "backbone-torsion output dispatch");

  TPack<Real, 2, Dev> V_t;
  auto dispatch_indices_t = TPack<Int, 2, Dev>::zeros({3, n_dispatch_total});
  if (topology_only) {
    V_t = TPack<Real, 2, Dev>::empty({2, 0});
  } else if (output_block_pair_energies) {
    V_t = TPack<Real, 2, Dev>::zeros({2, n_dispatch_total});
  } else {
    V_t = TPack<Real, 2, Dev>::zeros({2, n_poses});
  }
  // Per-pair backward recomputes weighted derivatives from the saved inputs.
  bool const accumulate_derivs = !topology_only && !output_block_pair_energies;
  auto dV_dxyz_t = accumulate_derivs
                       ? TPack<Vec<Real, 3>, 2, Dev>::zeros({2, n_atoms})
                       : TPack<Vec<Real, 3>, 2, Dev>::empty({2, 0});

  // Which block each dispatch index belongs to. Walking the sparse list is
  // what keeps this bounded: a dense (block x rot1 x rot2) grid would be
  // quadratic in the largest block's rotamer count and overflow an int
  // once a block carries a few thousand rotamers.
  TPack<Int, 1, Dev> block_for_dispatch_t =
      DeviceDispatch<Dev>::template load_balancing_search<launch_t>(
          mgr,
          n_dispatch_total,
          n_energies_for_block_offset.data(),
          n_poses * max_n_blocks);
  auto block_for_dispatch = block_for_dispatch_t.view;

  auto V = V_t.view;
  auto dV_dxyz = dV_dxyz_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  backbone_rotamer_fill_native<DeviceDispatch, Dev, Real, Int>(
      mgr,
      first_rot_for_block,
      n_rots_for_block,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_upper_conn_ind,
      n_energies_for_block_offset,
      block_for_dispatch,
      dispatch_indices,
      lockstep_group_for_block);

  if (topology_only) {
    return {V_t, dV_dxyz_t, dispatch_indices_t};
  }

  backbone_rotamer_score_native<false, DeviceDispatch, Dev, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      block_ind_for_rot,
      block_type_ind_for_rot,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_torsion_atoms,
      rama_tables,
      rama_table_params,
      omega_tables,
      omega_table_params,
      dispatch_indices,
      V,
      dV_dxyz,
      {},
      output_block_pair_energies,
      accumulate_derivs,
      lockstep_group_for_block);

  return {V_t, dV_dxyz_t, dispatch_indices_t};
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto BackboneTorsionRotamerScoreDispatch<DeviceDispatch, Dev, Real, Int>::
    backward(
        ContextManager& mgr,
        // common params
        TView<Vec<Real, 3>, 1, Dev> rot_coords,
        TView<Int, 1, Dev> rot_coord_offset,
        TView<Int, 1, Dev> pose_ind_for_atom,
        TView<Int, 2, Dev> first_rot_for_block,
        TView<Int, 2, Dev> first_rot_block_type,
        TView<Int, 1, Dev> block_ind_for_rot,
        TView<Int, 1, Dev> pose_ind_for_rot,
        TView<Int, 1, Dev> block_type_ind_for_rot,
        TView<Int, 1, Dev> n_rots_for_pose,
        TView<Int, 1, Dev> rot_offset_for_pose,
        TView<Int, 2, Dev> n_rots_for_block,
        TView<Int, 2, Dev> rot_offset_for_block,
        Int max_n_rots_per_pose,

        TView<Int, 2, Dev> pose_stack_block_type,
        TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

        TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

        TView<Int, 2, Dev> block_type_rama_table,
        TView<Int, 2, Dev> block_type_omega_table,
        TView<Int, 1, Dev> block_type_lower_conn_ind,
        TView<Int, 1, Dev> block_type_upper_conn_ind,
        TView<Int, 1, Dev> block_type_is_pro,

        TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

        TView<Real, 3, Dev> rama_tables,
        TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
        TView<Real, 4, Dev> omega_tables,
        TView<RamaTableParams<Real>, 1, Dev> omega_table_params,

        TView<Int, 2, Dev> dispatch_indices,  // from forward pass
        TView<Real, 2, Dev> dTdV,  // nterms x nposes x (1|len) x (1|len)
        TView<Int, 2, Dev> lockstep_group_for_block)
        -> TPack<Vec<Real, 3>, 2, Dev> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conn = pose_stack_inter_block_connections.size(2);
  int const n_block_types = block_type_atom_downstream_of_conn.size(0);
  int const max_n_atoms_per_block_type =
      block_type_atom_downstream_of_conn.size(1);
  int const n_rama_tables = rama_tables.size(0);
  int const n_omega_tables = omega_tables.size(0);

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

  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);
  assert(block_type_lower_conn_ind.size(0) == n_block_types);
  assert(block_type_upper_conn_ind.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);

  auto dV_dxyz_t = TPack<Vec<Real, 3>, 2, Dev>::zeros({2, n_atoms});

  auto dV_dxyz = dV_dxyz_t.view;

  backbone_rotamer_score_native<true, DeviceDispatch, Dev, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      block_ind_for_rot,
      block_type_ind_for_rot,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_torsion_atoms,
      rama_tables,
      rama_table_params,
      omega_tables,
      omega_table_params,
      dispatch_indices,
      {},
      dV_dxyz,
      dTdV,
      true,
      true,
      lockstep_group_for_block);

  return dV_dxyz_t;
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
