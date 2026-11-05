#pragma once

#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/matrix.hh>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/connection.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/hash_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include "params.hh"
#include "potentials.hh"
#include "genbonded_pose_score.hh"
#include "genbonded_pose_kernels.hh"

// mgpu operators for warp/CTA reductions

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

using namespace tmol::score::common;

using tmol::portable::Vec;

#define Real3 Vec<Real, 3>

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedPoseScoreDispatch<DeviceOps, D, Real, Int>::forward(
    ContextManager& mgr,
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

    bool output_block_pair_energies,
    bool compute_derivs)
    -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_V = output_block_pair_energies ? first_rot_for_block.size(1) : 1;
  auto V_t = TPack<Real, 4, D>::zeros({1, n_poses, n_V, n_V});
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({1, 0});
  genbonded_pose_forward<DeviceOps, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      gen_intra_subgraphs,
      gen_intra_subgraph_offsets,
      gen_intra_params,
      gen_atom_type_hierarchy,
      gen_atom_is_rosetta,
      gen_connection_bond_bins,
      gen_conn_scored_elsewhere,
      gen_source_atom_index,
      gen_source_block_type_index,
      gen_inter_torsion_hash_keys,
      gen_inter_torsion_hash_values,
      gen_inter_improper_hash_keys,
      gen_inter_improper_hash_values,
      output_block_pair_energies,
      compute_derivs,
      V_t.view,
      dV_dx_t.view);
  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedPoseScoreDispatch<DeviceOps, D, Real, Int>::backward(
    ContextManager& mgr,
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

    TView<Real, 4, D> dTdV) -> TPack<Vec<Real, 3>, 2, D> {
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, rot_coords.size(0)});
  genbonded_pose_backward<DeviceOps, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      gen_intra_subgraphs,
      gen_intra_subgraph_offsets,
      gen_intra_params,
      gen_atom_type_hierarchy,
      gen_atom_is_rosetta,
      gen_connection_bond_bins,
      gen_conn_scored_elsewhere,
      gen_source_atom_index,
      gen_source_block_type_index,
      gen_inter_torsion_hash_keys,
      gen_inter_torsion_hash_values,
      gen_inter_improper_hash_keys,
      gen_inter_improper_hash_values,
      dTdV,
      dV_dx_t.view);
  return dV_dx_t;
}

// ===========================================================================
// GenBondedRotamerScoreDispatch::forward
// ===========================================================================
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedRotamerScoreDispatch<DeviceOps, D, Real, Int>::forward(
    ContextManager& mgr,
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
    // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
    // lockstep, so only matching rotamer indices ever coexist
    TView<Int, 2, D> lockstep_group_for_block,
    Int max_n_rots_per_pose,

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

    bool output_block_pair_energies,
    bool compute_derivs,
    bool topology_only)
    -> std::tuple<
        TPack<Real, 2, D>,
        TPack<Vec<Real, 3>, 2, D>,
        TPack<Int, 2, D>,
        TPack<int64_t, 1, D>,
        TPack<Int, 1, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);

  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Convention: conn == max_n_conns => intra-rotamer interactions.
  int const max_n_interactions = score::common::checked_dispatch_product(
      n_rots, max_n_conns + 1, "generic-bonded count dispatch");

  auto n_output_intxns_for_rot_conn_t =
      TPack<int64_t, 1, D>::zeros({max_n_interactions});
  auto n_output_intxns_for_rot_conn = n_output_intxns_for_rot_conn_t.view;
  auto n_output_intxns_for_rot_conn_offset_t =
      TPack<int64_t, 1, D>::zeros({max_n_interactions});
  auto n_output_intxns_for_rot_conn_offset =
      n_output_intxns_for_rot_conn_offset_t.view;

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
      mgr, max_n_interactions, count_intxns_for_rot_conn);

  int64_t const n_output_intxns_total_64 =
      DeviceOps<D>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          n_output_intxns_for_rot_conn.data(),
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions,
          tmol::portable::plus_t<int64_t>());
  int const n_output_intxns_total = score::common::checked_dispatch_size(
      n_output_intxns_total_64, "generic-bonded output dispatch");
  TPack<Int, 1, D> rotconn_for_output_intxn_t =
      DeviceOps<D>::template load_balancing_search<launch_t>(
          mgr,
          n_output_intxns_total,
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions);
  auto rotconn_for_output_intxn = rotconn_for_output_intxn_t.view;

  int const n_V =
      topology_only
          ? 0
          : (output_block_pair_energies ? n_output_intxns_total : n_poses);
  auto V_t = TPack<Real, 2, D>::zeros({1, n_V});
  bool const accumulate_derivs =
      !topology_only && compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({1, 0});
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_output_intxns_total});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

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
      mgr, n_output_intxns_total, record_dispatch_indices);

  if (topology_only) {
    return {
        V_t,
        dV_dx_t,
        dispatch_indices_t,
        n_output_intxns_for_rot_conn_offset_t,
        rotconn_for_output_intxn_t};
  }

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
              torsion_score, eval, atoms, accumulate_derivs, dV_dx[0], 1.0);
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
              torsion_score, eval, atoms, accumulate_derivs, dV_dx[0], 1.0);
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
  });
  if (!output_block_pair_energies) {
    DeviceOps<D>::template foreach_workgroup<launch_t>(
        mgr, n_output_intxns_total, eval_torsions_for_interaction);
  } else {
    DeviceOps<D>::template foreach_independent_workgroup<launch_t>(
        mgr, n_output_intxns_total, eval_torsions_for_interaction);
  }

  return {
      V_t,
      dV_dx_t,
      dispatch_indices_t,
      n_output_intxns_for_rot_conn_offset_t,
      rotconn_for_output_intxn_t,
  };
}

// ===========================================================================
// GenBondedRotamerScoreDispatch::backward
// ===========================================================================
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedRotamerScoreDispatch<DeviceOps, D, Real, Int>::backward(
    ContextManager& mgr,
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

    TView<Int, 2, D> dispatch_indices,
    TView<int64_t, 1, D> n_output_intxns_for_rot_conn_offset,
    TView<Int, 1, D> rotconn_for_output_intxn,
    TView<Real, 2, D> dTdV) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_output_intxns_total = dispatch_indices.size(1);

  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dx_t.view;

  auto eval_torsions_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];
    Real const block_weight = dTdV[0][cta];

    auto score_torsion =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          Real dummy = 0.0;
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
              dummy, eval, atoms, true, dV_dx[0], block_weight);
        });

    auto score_improper =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          Real dummy = 0.0;
          auto eval = gbimproper_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1]);
          accumulate_torsion_result<Real, Int, D>(
              dummy, eval, atoms, true, dV_dx[0], block_weight);
        });

    if (conn_ind1 == max_n_conns) {
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
  });
  DeviceOps<D>::template foreach_workgroup<launch_t>(
      mgr, n_output_intxns_total, eval_torsions_for_interaction);

  return dV_dx_t;
}

#undef Real3

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
