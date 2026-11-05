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
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/hash_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/cartbonded/potentials/cartbonded_pose_score.hh>

// Operator definitions; safe for CPU compilation

#include <chrono>

#include "potentials.hh"
#include "cartbonded_pose_kernels.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

using tmol::portable::Vec;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
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

    // What is the index of the first intra-block subgraph for a block type
    // among all the subgraphs for all intra-block subgraphs?
    TView<Int, 1, D> cart_subgraph_offsets,

    // How many intra-block subgraphs of the three types (lengths, angles, &
    // torsions) are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,

    // int max_subgraphs_per_block,
    bool output_block_pair_energies,

    bool compute_derivs

    )
    -> std::tuple<
        TPack<Real, 4, D>,  // V_t: n-terms x n_poses x [n_blocks x n_blocks or
                            // 1 x 1]
        TPack<Vec<Real, 3>, 2, D>  // dV_dx_t,
        > {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  assert(pose_ind_for_atom.size(0) == n_atoms);
  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);
  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);
  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);
  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);
  int const n_V = output_block_pair_energies ? first_rot_for_block.size(1) : 1;
  auto V_t = TPack<Real, 4, D>::zeros({5, n_poses, n_V, n_V});
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({5, 0});
  cartbonded_pose_forward<DeviceDispatch, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      atom_unique_ids,
      atom_wildcard_ids,
      atom_is_rosetta,
      block_type_is_fragment,
      atom_cross_ids,
      connection_hash_keys,
      connection_spans,
      connection_paths,
      hash_keys,
      hash_values,
      cart_subgraphs,
      cart_subgraph_offsets,
      cart_subgraph_type_counts,
      cart_subgraph_type_offsets,
      cart_subgraph_param_indices,
      output_block_pair_energies,
      compute_derivs,
      V_t.view,
      dV_dx_t.view);
  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
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

    // What is the index of the first intra-block subgraph for a block type
    // among all the subgraphs for all intra-block subgraphs?
    TView<Int, 1, D> cart_subgraph_offsets,

    // How many intra-block subgraphs of the three types (lengths, angles, &
    // torsions) are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,

    TView<Real, 4, D> dTdV  // nterms x n-poses x max_n_blocks x max_n_blocks
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  assert(pose_ind_for_atom.size(0) == n_atoms);
  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);
  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);
  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);
  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, rot_coords.size(0)});
  cartbonded_pose_backward<DeviceDispatch, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      atom_unique_ids,
      atom_wildcard_ids,
      atom_is_rosetta,
      block_type_is_fragment,
      atom_cross_ids,
      connection_hash_keys,
      connection_spans,
      connection_paths,
      hash_keys,
      hash_values,
      cart_subgraphs,
      cart_subgraph_offsets,
      cart_subgraph_type_counts,
      cart_subgraph_type_offsets,
      cart_subgraph_param_indices,
      dTdV,
      dV_dx_t.view);
  return dV_dx_t;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
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
    // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
    // lockstep, so only matching rotamer indices ever coexist
    TView<Int, 2, D> lockstep_group_for_block,
    Int max_n_rots_per_pose,

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

    // What is the index of the first intra-block subgraph for a block type
    // among all the subgraphs for all intra-block subgraphs?
    TView<Int, 1, D> cart_subgraph_offsets,

    // How many intra-block subgraphs of the three types (lengths, angles, &
    // torsions) are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,

    // int max_subgraphs_per_block,
    bool output_block_pair_energies,

    bool compute_derivs,
    bool topology_only)
    -> std::tuple<
        TPack<Real, 2, D>,          // V_t,
        TPack<Vec<Real, 3>, 2, D>,  // dV_dx_t,
        TPack<Int, 2, D>,           // dispatch_indices_t,
        TPack<int64_t, 1, D>,       // n_output_intxns_for_rot_conn_offset,
        TPack<Int, 1, D>            // rotconn_for_output_intxn,
        > {
  using tmol::score::common::get_connection_spanning_subgraphs_offset;
  using tmol::score::common::get_n_connection_spanning_subgraphs;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  int const n_max_atoms_per_block = atom_unique_ids.size(1);

  assert(pose_ind_for_atom.size(0) == n_atoms);
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

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_cross_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);
  assert(atom_cross_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);

  // Algorithm:
  // 1. Count the number of rotamer-single and rotamer-pairs
  // we will need to dispatch over, and for each interaction, how many
  // bonds, angles, and torsions we'll calculate
  // 2. Then use load-balancing search for these three interactions
  // so we can assign one thread per atom-tuple
  // 3. Eval lenghts
  // 4. Eval angles
  // 5. Eval torsions

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Convention: conn == max_n_conns ==> the intra-rotamer set of cart energies
  int const max_n_interactions = score::common::checked_dispatch_product(
      n_rots, max_n_conns + 1, "Cartesian-bonded count dispatch");

  // Here we only count the rotconns to "upper" neighbors
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
    if (block_type_ind == -1) {
      // Not a real residue!
      return;
    }

    bool const is_intra_conn = conn_ind == max_n_conns;

    if (is_intra_conn) {
      n_output_intxns_for_rot_conn[index] = 1;
    } else {
      int const other_block_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][0];
      int const other_conn_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][1];

      if (other_block_ind == -1) {
        return;
      }
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

  DeviceDispatch<D>::template forall<launch_t>(
      mgr, max_n_interactions, count_intxns_for_rot_conn);

  // Scan and LBS on n output intxns: figure out how many rotamer pair
  // interactions there are and which pair each work unit should be assigned to.
  int64_t const n_output_intxns_total_64 =
      DeviceDispatch<D>::template scan_and_return_total<
          tmol::portable::scan_type::exclusive>(
          mgr,
          n_output_intxns_for_rot_conn.data(),
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions,
          tmol::portable::plus_t<int64_t>());
  int const n_output_intxns_total = score::common::checked_dispatch_size(
      n_output_intxns_total_64, "Cartesian-bonded output dispatch");
  TPack<Int, 1, D> rotconn_for_output_intxn_t =
      DeviceDispatch<D>::template load_balancing_search<launch_t>(
          mgr,
          n_output_intxns_total,
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions);
  auto rotconn_for_output_intxn = rotconn_for_output_intxn_t.view;

  // Allocate the tensors to which we will write our outputs
  int const n_V =
      topology_only
          ? 0
          : (output_block_pair_energies ? n_output_intxns_total : n_poses);
  auto V_t = TPack<Real, 2, D>::zeros({5, n_V});
  bool const accumulate_derivs =
      !topology_only && compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms})
                     : TPack<Vec<Real, 3>, 2, D>::empty({5, 0});
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_output_intxns_total});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  auto record_dispatch_indices_for_output_intxns =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const rotconn_ind = rotconn_for_output_intxn[index];
        int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
        int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
        int const pose_ind = pose_ind_for_rot[rot_ind1];
        dispatch_indices[0][index] = pose_ind;
        dispatch_indices[1][index] = rot_ind1;

        int rot_ind2;
        if (conn_ind1 == max_n_conns) {
          // intra-residue:
          rot_ind2 = rot_ind1;
        } else {
          // inter-residue
          int const block_ind1 = block_ind_for_rot[rot_ind1];
          int const rotconn_offset =
              n_output_intxns_for_rot_conn_offset[rotconn_ind];
          int const local_rot_ind2 = index - rotconn_offset;
          int const block_ind2 =
              pose_stack_inter_block_connections[pose_ind][block_ind1]
                                                [conn_ind1][0];

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
  // Record the rotamer pair indices for the output interactions; though we will
  // only use them downstream if we are in output_block_pair_energies mode
  // std::cout << "record_dispatch_indices_for_intxns" << std::endl;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_output_intxns_total, record_dispatch_indices_for_output_intxns);

  if (topology_only) {
    return {
        V_t,
        dV_dx_t,
        dispatch_indices_t,
        n_output_intxns_for_rot_conn_offset_t,
        rotconn_for_output_intxn_t};
  }

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

      // Real score;
      switch (type) {
        case subgraph_type::length: {
          if (compute_derivs) {
            auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
            accumulate_result<Real, Int, 2, D>(
                length_score,
                eval,
                atoms.template head<2>(),
                accumulate_derivs,
                dV_dx[score_type],
                1.0);
          } else {
            length_score += cblength_V(atom1, atom2, params[2], params[1]);
          }

          break;
        }
        case subgraph_type::angle: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          if (compute_derivs) {
            auto eval = cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
            accumulate_result<Real, Int, 3, D>(
                angle_score,
                eval,
                atoms.template head<3>(),
                accumulate_derivs,
                dV_dx[score_type],
                1.0);
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
          if (compute_derivs) {
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
                accumulate_derivs,
                dV_dx[score_type],
                1.0);
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
        int const output_index = (output_block_pair_energies) ? cta : pose_ind;
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
          accumulate<D, Real>::add(V[4][output_index], cta_hxyl_torsion_score);
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
  });
  if (!output_block_pair_energies) {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_subgraphs_for_interaction);
  } else {
    DeviceDispatch<D>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_subgraphs_for_interaction);
  }

  return {
      V_t,
      dV_dx_t,
      dispatch_indices_t,
      n_output_intxns_for_rot_conn_offset_t,
      rotconn_for_output_intxn_t,
  };
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
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

    // What is the index of the first intra-block subgraph for a block type
    // among all the subgraphs for all intra-block subgraphs?
    TView<Int, 1, D> cart_subgraph_offsets,

    // How many intra-block subgraphs of the three types (lengths, angles, &
    // torsions) are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,
    TView<Int, 1, D> cart_subgraph_param_indices,

    TView<Int, 2, D> dispatch_indices,
    TView<int64_t, 1, D> n_output_intxns_for_rot_conn_offset,
    TView<Int, 1, D> rotconn_for_output_intxn,

    TView<Real, 2, D> dTdV  // nterms x n-dispatch
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  int const n_max_atoms_per_block = atom_unique_ids.size(1);

  assert(pose_ind_for_atom.size(0) == n_atoms);
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

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_cross_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);
  assert(atom_cross_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  int const n_output_intxns_total = dispatch_indices.size(1);
  // We only call backward if we are in output_block_pair_energies mode,
  // so we will just go directly to allocating V_t w/ n_intxns_total size
  // int const n_V = output_block_pair_energies ? n_intxns_total : n_poses;
  auto V_t = TPack<Real, 2, D>::zeros({5, n_output_intxns_total});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});
  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
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

    auto score_subgraph =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
          Vec<Real, 7> params = hash_values[param_index];

          Vec<Real, 3> atom1 = rot_coords[atoms[0]];
          Vec<Real, 3> atom2 = rot_coords[atoms[1]];

          subgraph_type type = get_subgraph_type(atoms);

          int score_type = params[0];
          Real block_weight = dTdV[score_type][cta];

          // Real score;
          switch (type) {
            case subgraph_type::length: {
              auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
              accumulate_result<Real, Int, 2, D>(
                  length_score,
                  eval,
                  atoms.template head<2>(),
                  true,
                  dV_dx[score_type],
                  block_weight);

              break;
            }
            case subgraph_type::angle: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              auto eval =
                  cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
              accumulate_result<Real, Int, 3, D>(
                  angle_score,
                  eval,
                  atoms.template head<3>(),
                  true,
                  dV_dx[score_type],
                  block_weight);

              break;
            }
            case subgraph_type::torsion: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              Vec<Real, 3> atom4 = rot_coords[atoms[3]];
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
              Real& tor_score = (score_type == 3)   ? improper_torsion_score
                                : (score_type == 4) ? hxyl_torsion_score
                                                    : torsion_score;
              accumulate_result<Real, Int, 4, D>(
                  tor_score,
                  eval,
                  atoms.template head<4>(),
                  true,
                  dV_dx[score_type],
                  block_weight);

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
  });

  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      mgr, dispatch_indices.size(1), eval_subgraphs_for_interaction);

  return dV_dx_t;

}  // backward

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
