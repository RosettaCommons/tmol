#pragma once
#include <tmol/score/backbone_torsion/potentials/backbone_torsion_pose_kernels.hh>
#include <tmol/score/common/rotamer_correlation.hh>

namespace tmol::score::backbone_torsion::potentials {
// Caller-owned native sparse preparation and science. Count buffers and score/
// derivative outputs start at zero. Exclusive scan and allocation remain with
// the caller; count offsets are int64, preserving the mature sparse traversal.
template <bool Weighted, typename Vector, typename Real>
TMOL_HOST_DEVICE_FUNC auto backbone_rotamer_derivative(
    Vector value, Real weight) {
  if constexpr (Weighted)
    return value * weight;
  else
    return value;
}
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
void backbone_rotamer_count_native(
    ContextManager& mgr,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<int64_t, 2, Dev> n_energies_for_block,
    TView<Int, 2, Dev> lockstep_group_for_block = {}) {
  using tmol::score::common::accumulate;
  LAUNCH_BOX_32;
  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const n_pose_block_cells = common::checked_dispatch_product(
      n_poses, max_n_blocks, "backbone-torsion count dispatch");
  auto count_n_rotamer_energies = ([=] TMOL_DEVICE_FUNC(int index) {
    // Look at each residue and make sure that it has both upper and lower
    // neighbors and then write down the number of rotamer pair energies to
    // evaluate as n_rots_i * n_rots_upper
    int const pose_ind = index / max_n_blocks;
    int const block_ind = index % max_n_blocks;

    int const pose_block_type = pose_stack_block_type[pose_ind][block_ind];
    if (pose_block_type == -1) {
      return;
    }
    int const n_rots = n_rots_for_block[pose_ind][block_ind];

    int const upper_conn = block_type_upper_conn_ind[pose_block_type];
    if (upper_conn < 0) {
      return;
    }
    int const lower_conn = block_type_upper_conn_ind[pose_block_type];
    if (lower_conn < 0) {
      return;
    }
    int const lower_nbr_block_ind =
        pose_stack_inter_block_connections[pose_ind][block_ind][lower_conn][0];
    if (lower_nbr_block_ind < 0) {
      // no lower neighbor --> no backbone torsion term
      return;
    }

    int const upper_nbr_block_ind =
        pose_stack_inter_block_connections[pose_ind][block_ind][upper_conn][0];
    if (upper_nbr_block_ind < 0) {
      return;
    }
    // A two-residue cycle has two directed peptide bonds but only one
    // unordered conformer pair. Its scoring entry evaluates both directions.
    if (block_ind > upper_nbr_block_ind) {
      int const other_type =
          pose_stack_block_type[pose_ind][upper_nbr_block_ind];
      int const other_upper = block_type_upper_conn_ind[other_type];
      if (other_upper >= 0
          && pose_stack_inter_block_connections[pose_ind][upper_nbr_block_ind]
                                               [other_upper][0]
                 == block_ind) {
        return;
      }
    }
    // the upper neighbor is a real residue: we will score
    // the n_rots_i x n_rots_j rotamers for this pair.
    // This will properly include upper neighbors for circular peptides,
    // too.
    int const upper_n_rots = n_rots_for_block[pose_ind][upper_nbr_block_ind];

    bool const coupled = common::same_rotamer_group(
        lockstep_group_for_block, pose_ind, block_ind, upper_nbr_block_ind);
    n_energies_for_block[pose_ind][block_ind] =
        int64_t(n_rots) * (coupled && upper_n_rots > 0 ? 1 : upper_n_rots);
  });
  DeviceDispatch<Dev>::template forall<launch_t>(
      mgr, n_pose_block_cells, count_n_rotamer_energies);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
void backbone_rotamer_fill_native(
    ContextManager& mgr,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> n_rots_for_block,
    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<int64_t, 2, Dev> n_energies_for_block_offset,
    TView<Int, 1, Dev> block_for_dispatch,
    TView<Int, 2, Dev> dispatch_indices,
    TView<Int, 2, Dev> lockstep_group_for_block = {}) {
  using tmol::score::common::accumulate;
  LAUNCH_BOX_32;
  int const max_n_blocks = first_rot_for_block.size(1);
  auto mark_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose_and_block = block_for_dispatch[index];
    int const pose_ind = pose_and_block / max_n_blocks;
    int const block1_ind = pose_and_block % max_n_blocks;

    int const pose_block_type = pose_stack_block_type[pose_ind][block1_ind];
    int const upper_conn = block_type_upper_conn_ind[pose_block_type];
    int const upper_nbr_block_ind =
        pose_stack_inter_block_connections[pose_ind][block1_ind][upper_conn][0];
    int const upper_n_rots = n_rots_for_block[pose_ind][upper_nbr_block_ind];

    // invert the sparse index this block's pairs were counted into
    int const within =
        index - n_energies_for_block_offset[pose_ind][block1_ind];
    bool const coupled = common::same_rotamer_group(
        lockstep_group_for_block, pose_ind, block1_ind, upper_nbr_block_ind);
    int const local_rot1_ind = coupled ? within : within / upper_n_rots;
    int const local_rot2_ind =
        coupled ? within : within - local_rot1_ind * upper_n_rots;

    int const rot1_ind =
        first_rot_for_block[pose_ind][block1_ind] + local_rot1_ind;
    int const rot2_ind =
        first_rot_for_block[pose_ind][upper_nbr_block_ind] + local_rot2_ind;

    dispatch_indices[0][index] = pose_ind;
    // Sparse graph entries stay upper triangular, including closure bonds.
    dispatch_indices[1][index] = min(rot1_ind, rot2_ind);
    dispatch_indices[2][index] = max(rot1_ind, rot2_ind);
  });
  DeviceDispatch<Dev>::template forall<launch_t>(
      mgr, dispatch_indices.size(1), mark_dispatch_indices);
}

template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
void backbone_rotamer_score_native(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,
    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,
    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,
    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,
    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
    TView<Int, 2, Dev> dispatch_indices,
    TView<Real, 2, Dev> V,
    TView<Vec<Real, 3>, 2, Dev> dV_dxyz,
    TView<Real, 2, Dev> dTdV,
    bool output_block_pair_energies,
    bool accumulate_derivs,
    TView<Int, 2, Dev> lockstep_group_for_block = {}) {
  using tmol::score::common::accumulate;
  LAUNCH_BOX_32;
  auto rama_omega_func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = dispatch_indices[0][ind];

    int rot_ind1 = dispatch_indices[1][ind];
    int rot_ind2 = dispatch_indices[2][ind];

    int block_ind1 = block_ind_for_rot[rot_ind1];
    int block_ind2 = block_ind_for_rot[rot_ind2];

    int block_type1 = block_type_ind_for_rot[rot_ind1];
    int block_type2 = block_type_ind_for_rot[rot_ind2];

    // correction for cyclic peptides
    int const stored_upper_conn = block_type_upper_conn_ind[block_type1];
    if (stored_upper_conn < 0
        || pose_stack_inter_block_connections[pose_ind][block_ind1]
                                             [stored_upper_conn][0]
               != block_ind2) {
      int const swap_rot = rot_ind1;
      int const swap_block = block_ind1;
      int const swap_block_type = block_type1;
      rot_ind1 = rot_ind2;
      block_ind1 = block_ind2;
      block_type1 = block_type2;
      rot_ind2 = swap_rot;
      block_ind2 = swap_block;
      block_type2 = swap_block_type;
    }

    // Where will we write the output?
    // In block-pair-scoring mode, we store one energy per rotamer;
    // In non-block-pair-scoring mode, we store one energy per pose
    // (using atomic_add to accumulate for each block/rotamer in the pose)
    int const V_ind = Backward || output_block_pair_energies ? ind : pose_ind;

    auto score_direction = [&]() {
      int const upper_nbr_is_pro = block_type_is_pro[block_type2];
      int const rama_table_ind =
          block_type_rama_table[block_type1][upper_nbr_is_pro];
      int const omega_table_ind =
          block_type_omega_table[block_type1][upper_nbr_is_pro];

      int const rot_offset1 = rot_coord_offset[rot_ind1];

      bool valid_phipsi = true;
      Vec<Int, 4> phi_ats;
      for (int i = 0; i < 4; ++i) {
        UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i];
        phi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
            i_at,
            rot_ind1,
            rot_ind2,
            block_ind1,
            block_ind2,
            block_type2,
            pose_ind,
            rot_coord_offset,
            first_rot_for_block,
            first_rot_block_type,
            pose_stack_inter_block_connections,
            block_type_atom_downstream_of_conn,
            lockstep_group_for_block,
            block_type_ind_for_rot);
        valid_phipsi &= phi_ats[i] != -1;
      }
      CoordQuad<Real> phi_coords;
      if (valid_phipsi) {
        for (int i = 0; i < 4; ++i) {
          tmol::portable::row_ref(phi_coords, i) = rot_coords[phi_ats[i]];
        }
      }

      Vec<Int, 4> psi_ats;
      for (int i = 0; i < 4; ++i) {
        UnresolvedAtomID<Int> i_at =
            block_type_torsion_atoms[block_type1][i + 4];
        psi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
            i_at,
            rot_ind1,
            rot_ind2,
            block_ind1,
            block_ind2,
            block_type2,
            pose_ind,
            rot_coord_offset,
            first_rot_for_block,
            first_rot_block_type,
            pose_stack_inter_block_connections,
            block_type_atom_downstream_of_conn,
            lockstep_group_for_block,
            block_type_ind_for_rot);
        valid_phipsi &= psi_ats[i] != -1;
      }
      CoordQuad<Real> psi_coords;
      if (valid_phipsi) {
        for (int i = 0; i < 4; ++i) {
          tmol::portable::row_ref(psi_coords, i) = rot_coords[psi_ats[i]];
        }
      }

      Real const rama_weight = Backward ? dTdV[0][ind] : Real(1);
      // accumulate rama
      if (valid_phipsi && rama_table_ind >= 0) {
        auto rama = rama_V_dV<Dev, Real, Int>(
            phi_coords,
            psi_coords,
            rama_tables[rama_table_ind],
            tmol::portable::vec_from<Real, 2>(
                rama_table_params[rama_table_ind].bbstarts),
            tmol::portable::vec_from<Real, 2>(
                rama_table_params[rama_table_ind].bbsteps));
        if (!Backward)
          accumulate<Dev, Real>::add(V[0][V_ind], common::get<0>(rama));
        if (Backward || accumulate_derivs) {
          for (int j = 0; j < 4; ++j) {
            accumulate<Dev, Vec<Real, 3>>::add(
                dV_dxyz[0][phi_ats[j]],
                backbone_rotamer_derivative<Backward>(
                    common::get<1>(rama).row(j), rama_weight));
            accumulate<Dev, Vec<Real, 3>>::add(
                dV_dxyz[0][psi_ats[j]],
                backbone_rotamer_derivative<Backward>(
                    common::get<2>(rama).row(j), rama_weight));
          }
        }
      }

      if (omega_table_ind < 0) {
        return;
      }

      CoordQuad<Real> omega_coords;
      Vec<Int, 4> omega_ats;
      for (int i = 0; i < 4; ++i) {
        // TO DO: replace "8" in the lookup below
        UnresolvedAtomID<Int> i_at =
            block_type_torsion_atoms[block_type1][i + 8];

        // Check to see if the omega uaids are actually defined for this block
        // type. If the atom offset [0] or the connection index [1] are both -1,
        // this is a sentinel for an undefined omega.
        if (i_at.atom_id == -1 && i_at.conn_id == -1)
          return;  // no omega and we already calculated rama

        omega_ats[i] = resolve_rotamer_pair_atom_from_uaid(
            i_at,
            rot_ind1,
            rot_ind2,
            block_ind1,
            block_ind2,
            block_type2,
            pose_ind,
            rot_coord_offset,
            first_rot_for_block,
            first_rot_block_type,
            pose_stack_inter_block_connections,
            block_type_atom_downstream_of_conn,
            lockstep_group_for_block,
            block_type_ind_for_rot);

        // omega_atom_ind == -1 -> UAID resolution failed
        if (omega_ats[i] == -1)
          return;  // no omega and we already calculated rama
      }

      for (int i = 0; i < 4; ++i) {
        tmol::portable::row_ref(omega_coords, i) = rot_coords[omega_ats[i]];
      }

      Real const omega_weight = Backward ? dTdV[1][ind] : Real(1);
      if (valid_phipsi && omega_table_ind >= 0) {
        auto omega = omega_bbdep_V_dV<Dev, Real, Int>(
            phi_coords,
            psi_coords,
            omega_coords,
            omega_tables[omega_table_ind][0],
            omega_tables[omega_table_ind][1],
            tmol::portable::vec_from<Real, 2>(
                omega_table_params[omega_table_ind].bbstarts),
            tmol::portable::vec_from<Real, 2>(
                omega_table_params[omega_table_ind].bbsteps),
            32.8);
        if (!Backward)
          accumulate<Dev, Real>::add(V[1][V_ind], common::get<0>(omega));
        if (Backward || accumulate_derivs) {
          for (int j = 0; j < 4; ++j) {
            // omega : [V, dVdphi, dVdpsi, dVdomega]
            accumulate<Dev, Vec<Real, 3>>::add(
                dV_dxyz[1][phi_ats[j]],
                backbone_rotamer_derivative<Backward>(
                    common::get<1>(omega).row(j), omega_weight));
            accumulate<Dev, Vec<Real, 3>>::add(
                dV_dxyz[1][psi_ats[j]],
                backbone_rotamer_derivative<Backward>(
                    common::get<2>(omega).row(j), omega_weight));
            accumulate<Dev, Vec<Real, 3>>::add(
                dV_dxyz[1][omega_ats[j]],
                backbone_rotamer_derivative<Backward>(
                    common::get<3>(omega).row(j), omega_weight));
          }
        }
      } else {
        // if rama is undefined, fall back to old version
        auto omega = omega_V_dV<Dev, Real, Int>(omega_coords, 32.8);
        if (!Backward)
          accumulate<Dev, Real>::add(V[1][V_ind], common::get<0>(omega));
        if (Backward || accumulate_derivs) {
          for (int j = 0; j < 4; ++j) {
            // omega : [V, dVdomega]
            accumulate<Dev, Vec<Real, 3>>::add(
                dV_dxyz[1][omega_ats[j]],
                backbone_rotamer_derivative<Backward>(
                    common::get<1>(omega).row(j), omega_weight));
          }
        }
      }
    };
    score_direction();
    int const reverse_upper = block_type_upper_conn_ind[block_type2];
    if (block_ind1 != block_ind2 && reverse_upper >= 0
        && pose_stack_inter_block_connections[pose_ind][block_ind2]
                                             [reverse_upper][0]
               == block_ind1) {
      int const swap_rot = rot_ind1;
      int const swap_block = block_ind1;
      int const swap_type = block_type1;
      rot_ind1 = rot_ind2;
      block_ind1 = block_ind2;
      block_type1 = block_type2;
      rot_ind2 = swap_rot;
      block_ind2 = swap_block;
      block_type2 = swap_type;
      score_direction();
    }
  });
  DeviceDispatch<Dev>::template forall<launch_t>(
      mgr, dispatch_indices.size(1), rama_omega_func);
}

}  // namespace tmol::score::backbone_torsion::potentials
