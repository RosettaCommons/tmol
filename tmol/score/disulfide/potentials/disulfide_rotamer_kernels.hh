#pragma once
#include <tmol/score/disulfide/potentials/disulfide_pose_kernels.hh>
#include <tmol/score/common/rotamer_correlation.hh>
namespace tmol::score::disulfide::potentials {
// Caller-owned sparse entries with explicit connection identity. Counts must
// start at zero; offsets are exclusive and candidate_dispatch has been checked
// for integer overflow. Score/gradient storage must be zeroed; sparse scores
// have one column per dispatch entry. Per-block-type mirrored parameters and
// non-disulfide connection partners retain the mature scientific behavior.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void disulfide_rotamer_count_native(
    ContextManager& mgr,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 2, D> n_rots_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<int64_t, 1, D> n_energies_for_rot,
    TView<Int, 2, D> lockstep_group_for_block = {}) {
  LAUNCH_BOX_32;
  int const max_n_conns = disulfide_conns.size(1);
  auto count_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    // count how many other rotamers this rotamer has disulfide bonds to
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    if (pose_ind == -1 || block_ind == -1) {
      return;
    }

    int const rot_block_type = block_type_ind_for_rot[rot_ind];
    if (rot_block_type == -1) {
      return;
    }
    int64_t n_energies = 0;
    for (int conn_index = 0; conn_index < max_n_conns; conn_index++) {
      if (disulfide_conns[rot_block_type][conn_index]) {
        int const other_block_index =
            pose_stack_inter_block_connections[pose_ind][block_ind][conn_index]
                                              [0];

        if (other_block_index < block_ind) {
          // Only count disulfides to upper residues
          // as block_ind >= 0, this will ALSO handle the
          // scenario when other_block_index == -1
          continue;
        }
        int const other_block_n_rots =
            n_rots_for_block[pose_ind][other_block_index];
        n_energies += other_block_n_rots > 0
                              && common::same_rotamer_group(
                                  lockstep_group_for_block,
                                  pose_ind,
                                  block_ind,
                                  other_block_index)
                          ? 1
                          : other_block_n_rots;
      }
    }
    if (n_energies != 0) {
      n_energies_for_rot[rot_ind] = n_energies;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_energies_for_rot.size(0), count_dispatch_indices);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void disulfide_rotamer_fill_native(
    ContextManager& mgr,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 2, D> n_rots_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<bool, 2, D> disulfide_conns,
    TView<int64_t, 1, D> n_energies_for_rot,
    TView<int64_t, 1, D> n_energies_for_rot_offset,
    TView<Int, 2, D> dispatch_indices,
    TView<Int, 2, D> conns_for_dispatch_indices,
    int max_n_energies_for_rot,
    int candidate_dispatch,
    TView<Int, 2, D> lockstep_group_for_block = {}) {
  LAUNCH_BOX_32;
  int const max_n_conns = disulfide_conns.size(1);
  auto mark_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const rot_ind1 = ind / max_n_energies_for_rot;
    // which disulfide connection to upper neighbors is this?
    int const upper_rot_ind = ind % max_n_energies_for_rot;
    int local_rot_ind2 = upper_rot_ind;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    if (pose_ind == -1 || block_ind1 == -1) {
      return;
    }
    int const n_energies = n_energies_for_rot[rot_ind1];
    if (upper_rot_ind >= n_energies) {
      // either n_energies is 0 because this rotamer has
      // no RPEs to evaluate or we have some upper neighbor
      // energies to calculate, but fewer than the maximum
      // and thus we are "of the end" of the list.
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];

    int const offset_for_conn = 0;
    for (int conn_ind1 = 0; conn_ind1 < max_n_conns; conn_ind1++) {
      if (disulfide_conns[block_type1][conn_ind1]) {
        int const block_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [0];
        if (block_ind2 < block_ind1) {
          // Only count disulfides to upper residues
          continue;
        }
        int const conn_ind2 =
            pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                              [1];

        int const other_n_rots = n_rots_for_block[pose_ind][block_ind2];
        bool const coupled = common::same_rotamer_group(
            lockstep_group_for_block, pose_ind, block_ind1, block_ind2);
        int const n_rots2 = coupled && other_n_rots > 0 ? 1 : other_n_rots;
        if (local_rot_ind2 < n_rots2) {
          // We have figured out which upper rotamer this
          // index is referring to
          int const rot_ind2 =
              first_rot_for_block[pose_ind][block_ind2]
              + (coupled ? rot_ind1 - first_rot_for_block[pose_ind][block_ind1]
                         : local_rot_ind2);
          int const sparse_index =
              n_energies_for_rot_offset[rot_ind1] + upper_rot_ind;

          dispatch_indices[0][sparse_index] = pose_ind;
          dispatch_indices[1][sparse_index] = rot_ind1;
          dispatch_indices[2][sparse_index] = rot_ind2;
          conns_for_dispatch_indices[0][sparse_index] = conn_ind1;
          conns_for_dispatch_indices[1][sparse_index] = conn_ind2;
          break;
        } else {
          // this code is only reachable if we have some block
          // type wherein there are two or more disulfides -- imagine
          // something like valine but two sulfurs instead of two methyls
          // -- and residue i has disulfide bonds to both residues
          // j and k; so we evaluate i's rotamer's energies with j's
          // rotamers and also and i's rotamer's energies with k's
          // rotamers. Well, we have walked off the end of the list for
          // residue j, so we must instead go to the next connection
          // for residue k.
          local_rot_ind2 -= n_rots2;
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, candidate_dispatch, mark_dispatch_indices);
}

template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void disulfide_rotamer_score_native(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<bool, 2, D> disulfide_conns,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,
    TView<DisulfideGlobalParams<Real>, 1, D> global_params,
    TView<Int, 2, D> dispatch_indices,
    TView<Int, 2, D> conns_for_dispatch_indices,
    TView<Real, 2, D> V,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    TView<Real, 2, D> dTdV) {
  LAUNCH_BOX_32;
  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int dispatch_ind) {
    int const pose_ind = dispatch_indices[0][dispatch_ind];
    int const rot_ind1 = dispatch_indices[1][dispatch_ind];
    int const rot_ind2 = dispatch_indices[2][dispatch_ind];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    // one parameter row per block type; a d-amino acid's row carries
    //    negated dihedral means
    const auto& params1 = global_params[block_type1];
    const auto& params2 = global_params[block_type2];

    int atom_offset1 = rot_coord_offset[rot_ind1];
    int atom_offset2 = rot_coord_offset[rot_ind2];
    int conn_ind1 = conns_for_dispatch_indices[0][dispatch_ind];
    int conn_ind2 = conns_for_dispatch_indices[1][dispatch_ind];

    // We are scoring only one connection!

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
        Backward ? dTdV[0][dispatch_ind] : Real(1),
        Backward ? nullptr : &V[0][dispatch_ind]);
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, dispatch_indices.size(1), eval_energies);
}

}  // namespace tmol::score::disulfide::potentials
