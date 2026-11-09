#pragma once
#include <tmol/score/common/pose_energy_accumulation.hh>
#include <tmol/score/genbonded/potentials/genbonded_rotamer_kernels.hh>

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
  return score::common::with_pose_energy_accumulation<D, Real>(
      output_block_pair_energies, [&](auto precision) {
        using AccumReal = decltype(precision);
        auto V_t = TPack<AccumReal, 4, D>::zeros({1, n_poses, n_V, n_V});
        bool const accumulate_derivs =
            compute_derivs && !output_block_pair_energies;
        auto dV_dx_t = accumulate_derivs
                           ? TPack<Vec<AccumReal, 3>, 2, D>::zeros({1, n_atoms})
                           : TPack<Vec<AccumReal, 3>, 2, D>::empty({1, 0});
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
        return std::make_tuple(
            V_t.template cast<Real>(), dV_dx_t.template cast<Vec<Real, 3>>());
      });
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

  genbonded_rotamer_count_native<DeviceOps, D, Real, Int>(
      mgr,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_block,
      lockstep_group_for_block,
      pose_stack_inter_block_connections,
      n_output_intxns_for_rot_conn);

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

  genbonded_rotamer_fill_native<DeviceOps, D, Real, Int>(
      mgr,
      first_rot_for_block,
      block_ind_for_rot,
      pose_ind_for_rot,
      lockstep_group_for_block,
      pose_stack_inter_block_connections,
      n_output_intxns_for_rot_conn_offset,
      rotconn_for_output_intxn,
      dispatch_indices);

  if (topology_only) {
    return {
        V_t,
        dV_dx_t,
        dispatch_indices_t,
        n_output_intxns_for_rot_conn_offset_t,
        rotconn_for_output_intxn_t};
  }

  genbonded_rotamer_score_native<false, DeviceOps, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
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
      rotconn_for_output_intxn,
      dispatch_indices,
      V,
      dV_dx,
      {},
      output_block_pair_energies,
      accumulate_derivs);

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

  genbonded_rotamer_score_native<true, DeviceOps, D, Real, Int>(
      mgr,
      rot_coords,
      rot_coord_offset,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
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
      rotconn_for_output_intxn,
      dispatch_indices,
      {},
      dV_dx,
      dTdV,
      true,
      true);

  return dV_dx_t;
}

#undef Real3

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
