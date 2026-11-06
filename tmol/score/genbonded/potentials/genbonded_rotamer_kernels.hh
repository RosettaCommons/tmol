#pragma once
#include <tmol/score/genbonded/potentials/genbonded_pose_kernels.hh>
namespace tmol::score::genbonded::potentials {
// Sparse connected/intra-residue enumeration and physics. The caller supplies
// zeroed counts, score/gradient outputs, exclusive offsets and work-item
// owners. Lockstep groups and explicit source parameter identities remain in
// this core.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void genbonded_rotamer_count_native(
    ContextManager& mgr,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> lockstep_group_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<int64_t, 1, D> n_output_intxns_for_rot_conn) {
  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  auto count_intxns_for_rot_conn = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rot_ind = index / (max_n_conns + 1);
    int const conn_ind = index % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    int const block_type_ind = block_type_ind_for_rot[rot_ind];
    if (block_type_ind == -1) return;

    if (conn_ind == max_n_conns) {
      n_output_intxns_for_rot_conn[index] = 1;
    } else {
      int const other_block_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][0];
      if (other_block_ind == -1) return;
      int const other_block_n_rots =
          n_rots_for_block[pose_ind][other_block_ind];
      if (block_ind < other_block_ind) {
        // blocks that sample in lockstep only ever coexist at matching
        // rotamer indices, so this rotamer pairs with exactly one of theirs
        int const g1 = lockstep_group_for_block[pose_ind][block_ind];
        int const g2 = lockstep_group_for_block[pose_ind][other_block_ind];
        n_output_intxns_for_rot_conn[index] =
            (g1 >= 0 && g1 == g2) ? 1 : other_block_n_rots;
      }
    }
  });
  DeviceOps<D>::template forall<launch_t>(
      mgr, n_output_intxns_for_rot_conn.size(0), count_intxns_for_rot_conn);
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void genbonded_rotamer_fill_native(
    ContextManager& mgr,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 2, D> lockstep_group_for_block,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<int64_t, 1, D> n_output_intxns_for_rot_conn_offset,
    TView<Int, 1, D> rotconn_for_output_intxn,
    TView<Int, 2, D> dispatch_indices) {
  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  auto record_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rotconn_ind = rotconn_for_output_intxn[index];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    dispatch_indices[0][index] = pose_ind;
    dispatch_indices[1][index] = rot_ind1;
    int rot_ind2;
    if (conn_ind1 == max_n_conns) {
      rot_ind2 = rot_ind1;
    } else {
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const rotconn_offset =
          n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const local_rot_ind2 = index - rotconn_offset;
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const g1 = lockstep_group_for_block[pose_ind][block_ind1];
      int const g2 = lockstep_group_for_block[pose_ind][block_ind2];
      int const offset2 =
          (g1 >= 0 && g1 == g2)
              ? rot_ind1 - first_rot_for_block[pose_ind][block_ind1]
              : local_rot_ind2;
      rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + offset2;
    }
    dispatch_indices[2][index] = rot_ind2;
  });
  DeviceOps<D>::template forall<launch_t>(
      mgr, dispatch_indices.size(1), record_dispatch_indices);
}

template <
    bool Backward,
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void genbonded_rotamer_score_native(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Vec<Int, 5>, 1, D> gen_intra_subgraphs,
    TView<Int, 1, D> gen_intra_subgraph_offsets,
    TView<Vec<Real, 5>, 1, D> gen_intra_params,
    TView<Vec<Int, GB_MAX_HIER_DEPTH>, 2, D> gen_atom_type_hierarchy,
    TView<Int, 2, D> gen_atom_is_rosetta,
    TView<Int, 3, D> gen_connection_bond_bins,
    TView<Int, 2, D> gen_conn_scored_elsewhere,
    TView<Int, 2, D> gen_source_atom_index,
    TView<Int, 1, D> gen_source_block_type_index,
    TView<Vec<Int, 6>, 1, D> gen_inter_torsion_hash_keys,
    TView<Vec<Real, 5>, 1, D> gen_inter_torsion_hash_values,
    TView<Vec<Int, 5>, 1, D> gen_inter_improper_hash_keys,
    TView<Vec<Real, 5>, 1, D> gen_inter_improper_hash_values,
    TView<Int, 1, D> rotconn_for_output_intxn,
    TView<Int, 2, D> dispatch_indices,
    TView<Real, 2, D> V,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    TView<Real, 2, D> dTdV,
    bool output_block_pair_energies,
    bool accumulate_derivs) {
  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);
  auto eval_torsions_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real torsion_score = 0.0;
    Real const block_weight = Backward ? dTdV[0][cta] : Real(1);

    auto score_torsion =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          auto eval = gbtorsion_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1],
              prm[2],
              prm[3],
              prm[4]);
          accumulate_torsion_result<Real, Int, D>(
              torsion_score,
              eval,
              atoms,
              Backward || accumulate_derivs,
              dV_dx[0],
              block_weight);
        });

    auto score_improper =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          auto eval = gbimproper_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1]);
          accumulate_torsion_result<Real, Int, D>(
              torsion_score,
              eval,
              atoms,
              Backward || accumulate_derivs,
              dV_dx[0],
              block_weight);
        });

    if (conn_ind1 == max_n_conns) {
      // Intra-block: iterate gen_intra_subgraphs with tag dispatch.
      int const intra_start = gen_intra_subgraph_offsets[block_type1];
      int const intra_end = (block_type1 + 1 == n_block_types)
                                ? n_total_intra
                                : gen_intra_subgraph_offsets[block_type1 + 1];
      int const n_intra = intra_end - intra_start;

      auto eval_intra = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_intra; i += nt) {
          int const idx = intra_start + i;
          Vec<Int, 5> sg = gen_intra_subgraphs[idx];
          if (sg[0] == -1) continue;
          Vec<Int, 4> local_sg;
          local_sg[0] = sg[1];
          local_sg[1] = sg[2];
          local_sg[2] = sg[3];
          local_sg[3] = sg[4];
          Vec<Int, 4> atoms =
              atom_local_to_global_indices(local_sg, rot_coord_offset1);
          Vec<Real, 5> prm = gen_intra_params[idx];
          if (sg[0] == 0) {
            score_torsion(atoms, prm);
          } else {
            score_improper(atoms, prm);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_intra);

    } else {
      // Inter-block: look up torsion params via hierarchy hash.
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      // which rotamer of block 2 this work unit pairs with was decided when
      // the interactions were enumerated; lockstep pairs are not a simple
      // walk over that block's rotamers, so read it rather than recompute it
      int const rot_ind2 = dispatch_indices[2][cta];
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];
      Int const bond_type_int =
          gen_connection_bond_bins[block_type1][conn_ind1][0];
      bool const conn_claimed =
          gen_conn_scored_elsewhere[block_type1][conn_ind1]
          || gen_conn_scored_elsewhere[block_type2][conn_ind2];

      auto eval_inter = ([&] TMOL_DEVICE_FUNC(int tid) {
        int const n_pairs = MAX_PATHS_FROM_CONN * MAX_PATHS_FROM_CONN;
        for (int pair = tid; pair < n_pairs; pair += nt) {
          int const path_A_idx = pair / MAX_PATHS_FROM_CONN;
          int const path_B_idx = pair % MAX_PATHS_FROM_CONN;
          Vec<Int, 3> pathA =
              atom_paths_from_conn[block_type1][conn_ind1][path_A_idx];
          Vec<Int, 3> pathB =
              atom_paths_from_conn[block_type2][conn_ind2][path_B_idx];
          if (pathA[0] == -1 || pathB[0] == -1) continue;

          Vec<Int, 4> atoms;
          bool source_parameter = false;
          int param_index = inter_block_torsion_parameter<Int, D>(
              pathA,
              pathB,
              block_type1,
              block_type2,
              rot_coord_offset1,
              rot_coord_offset2,
              bond_type_int,
              gen_connection_bond_bins[block_type1][conn_ind1][path_A_idx + 1],
              gen_connection_bond_bins[block_type2][conn_ind2][path_B_idx + 1],
              conn_claimed,
              gen_atom_type_hierarchy,
              gen_atom_is_rosetta,
              gen_source_atom_index,
              gen_source_block_type_index,
              gen_intra_subgraphs,
              gen_intra_subgraph_offsets,
              gen_inter_torsion_hash_keys,
              n_block_types,
              n_total_intra,
              atoms,
              source_parameter);
          if (param_index >= 0) {
            score_torsion(
                atoms,
                source_parameter ? gen_intra_params[param_index]
                                 : gen_inter_torsion_hash_values[param_index]);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_inter);

      auto eval_inter_improper = ([&] TMOL_DEVICE_FUNC(int tid) {
        // Either connection atom can be the three-coordinate center, so both
        // sides are evaluated. The loop strides over the workgroup because a
        // single-threaded one would otherwise only ever see side 0.
        for (int side = tid; side < 2; side += nt) {
          Vec<Int, 4> atoms;
          int val_idx = inter_block_improper_for_side<Int, D>(
              side,
              block_type1,
              block_type2,
              conn_ind1,
              conn_ind2,
              rot_coord_offset1,
              rot_coord_offset2,
              atom_paths_from_conn,
              gen_atom_type_hierarchy,
              gen_atom_is_rosetta,
              gen_source_atom_index,
              gen_inter_improper_hash_keys,
              atoms);
          if (val_idx >= 0) {
            score_improper(atoms, gen_inter_improper_hash_values[val_idx]);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_inter_improper);
    }

    if (!Backward) {
      auto reduce_and_write = ([&] TMOL_DEVICE_FUNC(int tid) {
        Real const cta_score = DeviceOps<D>::template reduce_in_workgroup<nt>(
            torsion_score, shared, tmol::portable::plus_t<Real>());
        if (tid == 0 && cta_score != 0.0) {
          int const out_idx =
              output_block_pair_energies ? cta : dispatch_indices[0][cta];
          accumulate<D, Real>::add(V[0][out_idx], cta_score);
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(reduce_and_write);
    }
  });
  if (!Backward && output_block_pair_energies)
    DeviceOps<D>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_torsions_for_interaction);
  else
    DeviceOps<D>::template foreach_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_torsions_for_interaction);
}

}  // namespace tmol::score::genbonded::potentials
