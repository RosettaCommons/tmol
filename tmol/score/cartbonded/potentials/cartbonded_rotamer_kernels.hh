#pragma once
#include <tmol/score/cartbonded/potentials/cartbonded_pose_kernels.hh>
namespace tmol::score::cartbonded::potentials {
// Caller-owned connected entries and five per-term outputs/gradients. Preserve
// connection/fragment parameter namespaces and per-entry upstream weights.
template <
    bool Backward,
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void cartbonded_rotamer_score_native(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Int, 2, D> atom_unique_ids,
    TView<Int, 2, D> atom_wildcard_ids,
    TView<Int, 2, D> atom_is_rosetta,
    TView<Int, 1, D> block_type_is_fragment,
    TView<Int, 2, D> atom_cross_ids,
    TView<Vec<Int, 5>, 1, D> connection_hash_keys,
    TView<Vec<Int, 2>, 1, D> connection_spans,
    TView<Vec<Int, 5>, 1, D> connection_paths,
    TView<Vec<Int, 5>, 1, D> hash_keys,
    TView<Vec<Real, 7>, 1, D> hash_values,
    TView<Vec<Int, 4>, 1, D> cart_subgraphs,
    TView<Int, 1, D> cart_subgraph_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,
    TView<Int, 1, D> rotconn_for_output_intxn,
    TView<Int, 2, D> dispatch_indices,
    TView<Real, 2, D> V,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    TView<Real, 2, D> dTdV,
    bool output_block_pair_energies,
    bool accumulate_derivs,
    bool compute_derivs) {
  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    // Only one element of this union: the shared memory array for
    // the reduction, which itself only will take any space if NT > 32
    // as the reduction otherwise uses only warp shuffle operations!
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;  // can't have an empty union, so, we'll have an integer
                 // placeholder.
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real length_score = 0.0;
    Real angle_score = 0.0;
    Real torsion_score = 0.0;
    Real improper_torsion_score = 0.0;
    Real hxyl_torsion_score = 0.0;

    auto score_subgraph = ([&] TMOL_DEVICE_FUNC(
                               Vec<Int, 4> atoms, Int param_index) {
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[atoms[0]];
      Vec<Real, 3> atom2 = rot_coords[atoms[1]];

      subgraph_type type = get_subgraph_type(atoms);

      int score_type = params[0];
      Real const block_weight = Backward ? dTdV[score_type][cta] : Real(1);

      // Real score;
      switch (type) {
        case subgraph_type::length: {
          if (Backward || compute_derivs) {
            auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
            accumulate_result<Real, Int, 2, D>(
                length_score,
                eval,
                atoms.template head<2>(),
                Backward || accumulate_derivs,
                dV_dx[score_type],
                block_weight);
          } else {
            length_score += cblength_V(atom1, atom2, params[2], params[1]);
          }

          break;
        }
        case subgraph_type::angle: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          if (Backward || compute_derivs) {
            auto eval = cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
            accumulate_result<Real, Int, 3, D>(
                angle_score,
                eval,
                atoms.template head<3>(),
                Backward || accumulate_derivs,
                dV_dx[score_type],
                block_weight);
          } else {
            angle_score += cbangle_V(atom1, atom2, atom3, params[2], params[1]);
          }

          break;
        }
        case subgraph_type::torsion: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          Vec<Real, 3> atom4 = rot_coords[atoms[3]];
          Real& tor_score = (score_type == 3)   ? improper_torsion_score
                            : (score_type == 4) ? hxyl_torsion_score
                                                : torsion_score;
          if (Backward || compute_derivs) {
            auto eval = cbtorsion_V_dV(
                atom1,
                atom2,
                atom3,
                atom4,
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6]);
            accumulate_result<Real, Int, 4, D>(
                tor_score,
                eval,
                atoms.template head<4>(),
                Backward || accumulate_derivs,
                dV_dx[score_type],
                block_weight);
          } else {
            tor_score += cbtorsion_V(
                atom1,
                atom2,
                atom3,
                atom4,
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6]);
          }

          break;
        }
      }
    });

    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_subgraphs; i += nt) {
          int const subgraph_index = subgraph_offset + i;
          int const param_index = cart_subgraph_param_indices[subgraph_index];
          if (param_index != -1) {
            Vec<Int, 4> subgraph_atom_indices = atom_local_to_global_indices(
                cart_subgraphs[subgraph_index], rot_coord_offset1);
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
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

      // Use by-reference capture here so that we can write to the length,
      // angle, and torsion_score variables
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        evaluate_connection_paths<Int, D, nt>(
            tid,
            block_type1,
            block_type2,
            rot_coord_offset1,
            rot_coord_offset2,
            conn_ind1,
            conn_ind2,
            atom_paths_from_conn,
            block_type_is_fragment,
            atom_is_rosetta,
            atom_unique_ids,
            atom_wildcard_ids,
            atom_cross_ids,
            hash_keys,
            connection_hash_keys,
            connection_spans,
            connection_paths,
            score_subgraph);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);

      auto eval_inter_res_impropers = ([&] TMOL_DEVICE_FUNC(int tid) {
        evaluate_connection_impropers<Int, D, nt>(
            tid,
            block_type1,
            block_type2,
            rot_coord_offset1,
            rot_coord_offset2,
            conn_ind1,
            conn_ind2,
            atom_paths_from_conn,
            atom_is_rosetta,
            atom_unique_ids,
            atom_wildcard_ids,
            atom_cross_ids,
            hash_keys,
            score_subgraph);
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_impropers);
    }

    if (!Backward) {
      // Now let's accumulate the scores into main memory
      auto reduce_energies = ([&] TMOL_DEVICE_FUNC(int tid) {
        Real const cta_length_score =
            DeviceDispatch<D>::template reduce_in_workgroup<nt>(
                length_score, shared, tmol::portable::plus_t<Real>());
        Real const cta_angle_score =
            DeviceDispatch<D>::template reduce_in_workgroup<nt>(
                angle_score, shared, tmol::portable::plus_t<Real>());
        Real const cta_torsion_score =
            DeviceDispatch<D>::template reduce_in_workgroup<nt>(
                torsion_score, shared, tmol::portable::plus_t<Real>());
        Real const cta_improper_torsion_score =
            DeviceDispatch<D>::template reduce_in_workgroup<nt>(
                improper_torsion_score, shared, tmol::portable::plus_t<Real>());
        Real const cta_hxyl_torsion_score =
            DeviceDispatch<D>::template reduce_in_workgroup<nt>(
                hxyl_torsion_score, shared, tmol::portable::plus_t<Real>());

        if (tid == 0) {
          int const output_index =
              (output_block_pair_energies) ? cta : pose_ind;
          if (cta_length_score != 0.0) {
            accumulate<D, Real>::add(V[0][output_index], cta_length_score);
          }
          if (cta_angle_score != 0.0) {
            accumulate<D, Real>::add(V[1][output_index], cta_angle_score);
          }
          if (cta_torsion_score != 0.0) {
            accumulate<D, Real>::add(V[2][output_index], cta_torsion_score);
          }
          if (cta_improper_torsion_score != 0.0) {
            accumulate<D, Real>::add(
                V[3][output_index], cta_improper_torsion_score);
          }
          if (cta_hxyl_torsion_score != 0.0) {
            accumulate<D, Real>::add(
                V[4][output_index], cta_hxyl_torsion_score);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
    }
  });
  if (Backward || !output_block_pair_energies)
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_subgraphs_for_interaction);
  else
    DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_subgraphs_for_interaction);
}

}  // namespace tmol::score::cartbonded::potentials
