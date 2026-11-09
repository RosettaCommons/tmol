#pragma once

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include "params_types.hh"
#include "potentials.hh"

// Shared disulfide whole-pose energies and coordinate derivatives.
// Callers provide zeroed output views; execution is selected by DeviceDispatch.
namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {
using tmol::portable::Vec;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void disulfide_pose_forward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    bool output_block_pair_energies,
    TView<Real, 4, D> V,
    TView<Vec<Real, 3>, 2, D> dV_dx) {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);
  LAUNCH_BOX_32;

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = ind % max_n_blocks;

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    if (rot_ind1 < 0) {
      return;
    }
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    if (block_type1 < 0) {
      return;
    }
    for (int conn_ind1 = 0; conn_ind1 < max_n_conns; conn_ind1++) {
      if (disulfide_conns[block_type1][conn_ind1]) {
        int const block_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [0];
        int const conn_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [1];
        if (block_ind2 < block_ind1) {
          // Only count disulfides to upper residues;
          // coincidentally handles the case when block_ind2 == -1
          continue;
        }
        int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
        int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
        if (rot_ind2 < 0) {
          continue;
        }

        // one parameter row per block type; a d-amino acid's row carries
        //    negated dihedral means
        const auto& params1 = global_params[block_type1];
        const auto& params2 = global_params[block_type2];

        int atom_offset1 = rot_coord_offset[rot_ind1];
        int atom_offset2 = rot_coord_offset[rot_ind2];

        // Skip if the other end isn't capable of a disulfide
        // bond -- imagine a CYS with a chemical bond to some
        // kind of funky group; it could happen!
        if (!disulfide_conns[block_type2][conn_ind2]) {
          return;
        }

        // Get the 6 atoms that we need for the disulfides
        auto block1_CA_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][2];
        auto block1_CB_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][1];
        auto block1_S_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][0];
        auto block2_S_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][0];
        auto block2_CB_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][1];
        auto block2_CA_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][2];

        int const Vind1 = output_block_pair_energies ? block_ind1 : 0;
        int const Vind2 = output_block_pair_energies ? block_ind2 : 0;
        // Calculate score and derivatives and put them in the out tensors
        accumulate_disulfide<Real, D>(
            rot_coords,
            block1_CA_ind,
            block1_CB_ind,
            block1_S_ind,
            block2_S_ind,
            block2_CB_ind,
            block2_CA_ind,
            params1,
            params2,
            dV_dx,
            Real(1),
            &V[0][pose_ind][Vind1][Vind2]);
      }
    }
  });

  DeviceDispatch<D>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, eval_energies);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void disulfide_pose_backward(
    ContextManager& mgr,
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<DisulfideGlobalParams<Real>, 1, D> global_params,

    TView<Real, 4, D> dTdV,
    TView<Vec<Real, 3>, 2, D> dV_dx) {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = disulfide_conns.size(1);
  LAUNCH_BOX_32;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = ind / max_n_blocks;
    int const block_ind1 = (ind % max_n_blocks);

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    if (rot_ind1 < 0) {
      return;
    }
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];
    if (block_type1 < 0) {
      return;
    }
    for (int conn_ind1 = 0; conn_ind1 < max_n_conns; conn_ind1++) {
      if (disulfide_conns[block_type1][conn_ind1]) {
        int const block_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [0];
        int const conn_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [1];
        if (block_ind2 < block_ind1) {
          // Only count disulfides to upper residues;
          // coincidentally handles the case when block_ind2 == -1
          continue;
        }
        if (dTdV[0][pose_ind][block_ind1][block_ind2] == 0) {
          continue;
        }
        int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
        int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
        if (rot_ind2 < 0) {
          continue;
        }

        // one parameter row per block type; a d-amino acid's row carries
        //    negated dihedral means
        const auto& params1 = global_params[block_type1];
        const auto& params2 = global_params[block_type2];

        int atom_offset1 = rot_coord_offset[rot_ind1];
        int atom_offset2 = rot_coord_offset[rot_ind2];

        if (!disulfide_conns[block_type2][conn_ind2]) {
          return;
        }
        // Get the 6 atoms that we need for the disulfides
        auto block1_CA_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][2];
        auto block1_CB_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][1];
        auto block1_S_ind =
            atom_offset1
            + block_type_atom_downstream_of_conn[block_type1][conn_ind1][0];
        auto block2_S_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][0];
        auto block2_CB_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][1];
        auto block2_CA_ind =
            atom_offset2
            + block_type_atom_downstream_of_conn[block_type2][conn_ind2][2];

        // Calculate score and derivatives and put them in the out tensors
        accumulate_disulfide<Real, D>(
            rot_coords,
            block1_CA_ind,
            block1_CB_ind,
            block1_S_ind,
            block2_S_ind,
            block2_CB_ind,
            block2_CA_ind,
            params1,
            params2,
            dV_dx,
            dTdV[0][pose_ind][block_ind1][block_ind2],
            nullptr);
      }
    }
  });

  DeviceDispatch<D>::template forall_grouped<launch_t>(
      mgr, n_poses, max_n_blocks, eval_derivs);
}

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
