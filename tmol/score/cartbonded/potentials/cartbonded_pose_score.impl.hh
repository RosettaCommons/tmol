#pragma once
#include <tmol/score/cartbonded/potentials/cartbonded_rotamer_kernels.hh>
#include <tmol/score/common/connected_rotamer_dispatch.hh>

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
  // Keep pair physics in Real; only concurrent CUDA float sums use double.
  using Accum = std::conditional_t<
      D == Device::CUDA && std::is_same_v<Real, float>,
      double,
      Real>;
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
  auto V_t = TPack<Accum, 4, D>::zeros({5, n_poses, n_V, n_V});
  bool const accumulate_derivs = compute_derivs && !output_block_pair_energies;
  auto dV_dx_t = accumulate_derivs
                     ? TPack<Vec<Accum, 3>, 2, D>::zeros({5, n_atoms})
                     : TPack<Vec<Accum, 3>, 2, D>::empty({5, 0});
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
  return {V_t.template cast<Real>(), dV_dx_t.template cast<Vec<Real, 3>>()};
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
  // Keep pair physics in Real; only concurrent CUDA float sums use double.
  using Accum = std::conditional_t<
      D == Device::CUDA && std::is_same_v<Real, float>,
      double,
      Real>;
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
  auto dV_dx_t = TPack<Vec<Accum, 3>, 2, D>::zeros({5, rot_coords.size(0)});
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
  return dV_dx_t.template cast<Vec<Real, 3>>();
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

  tmol::score::common::connected_rotamer_count_native<DeviceDispatch,D,Real,Int>(mgr,block_ind_for_rot,pose_ind_for_rot,block_type_ind_for_rot,n_rots_for_block,lockstep_group_for_block,pose_stack_inter_block_connections,n_output_intxns_for_rot_conn);


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

  tmol::score::common::connected_rotamer_fill_native<DeviceDispatch,D,Real,Int>(mgr,first_rot_for_block,block_ind_for_rot,pose_ind_for_rot,lockstep_group_for_block,pose_stack_inter_block_connections,n_output_intxns_for_rot_conn_offset,rotconn_for_output_intxn,dispatch_indices);


  if (topology_only) {
    return {
        V_t,
        dV_dx_t,
        dispatch_indices_t,
        n_output_intxns_for_rot_conn_offset_t,
        rotconn_for_output_intxn_t};
  }

  cartbonded_rotamer_score_native<false,DeviceDispatch,D,Real,Int>(mgr,rot_coords,rot_coord_offset,block_ind_for_rot,pose_ind_for_rot,block_type_ind_for_rot,pose_stack_inter_block_connections,atom_paths_from_conn,atom_unique_ids,atom_wildcard_ids,atom_is_rosetta,block_type_is_fragment,atom_cross_ids,connection_hash_keys,connection_spans,connection_paths,hash_keys,hash_values,cart_subgraphs,cart_subgraph_offsets,cart_subgraph_param_indices,rotconn_for_output_intxn,dispatch_indices,V,dV_dx,{},output_block_pair_energies,accumulate_derivs,compute_derivs);

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

  cartbonded_rotamer_score_native<true,DeviceDispatch,D,Real,Int>(mgr,rot_coords,rot_coord_offset,block_ind_for_rot,pose_ind_for_rot,block_type_ind_for_rot,pose_stack_inter_block_connections,atom_paths_from_conn,atom_unique_ids,atom_wildcard_ids,atom_is_rosetta,block_type_is_fragment,atom_cross_ids,connection_hash_keys,connection_spans,connection_paths,hash_keys,hash_values,cart_subgraphs,cart_subgraph_offsets,cart_subgraph_param_indices,rotconn_for_output_intxn,dispatch_indices,{},dV_dx,dTdV,true,true,true);

  return dV_dx_t;

}  // backward

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
