// Device-agnostic common numeric routines for kinematics.

#pragma once

#include "kinematics_math.hh"
#include <Eigen/Core>
#include <tmol/utility/tensor/TensorUtilEigen.h>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/unresolved_atom.hh>

#include <pybind11/pybind11.h>

namespace tmol {
namespace kinematics {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct KinForestFromStencil {
  static auto get_kfo_indices_for_atoms(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Int, 1, D> block_type_n_atoms,
      TView<bool, 2, D> block_type_atom_is_real)
      -> std::tuple<TPack<Int, 2, D>, TPack<Int, 2, D>, TPack<Int, 3, D>>;

  static auto get_kfo_atom_parents(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,                 // P x L
      TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
      TView<Int, 2, D> pose_stack_ff_parent,                  // P x L
      // TView<Int, 2, D> pose_stack_ff_conn_to_parent,          // P x L
      TView<Int, 3, D> pose_stack_block_in_and_first_out,  // P x L x 2
      TView<Int, 3, D> block_type_parents,                 // T x O x A
      TView<Int, 2, D> kfo_2_orig_mapping,                 // K x 3
      TView<Int, 3, D> atom_kfo_index,                     // P x L x A
      TView<Int, 1, D> block_type_jump_atom,               // T
      TView<Int, 1, D> block_type_n_conn,                  // T
      TView<Int, 2, D> block_type_conn_atom)
      -> std::tuple<TPack<Int, 1, D>, TPack<Int, 1, D>>;

  static auto get_children(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,              // x
      TView<Int, 3, D> pose_stack_block_in_and_first_out,  // x
      TView<Int, 2, D> kfo_2_orig_mapping,                 // x
      TView<Int, 1, D> kfo_parent_atoms,                   // x
      TView<Int, 1, D> block_type_n_conn                   // x
      )
      -> std::tuple<
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<bool, 1, D>>;

  static auto get_id_and_frame_xyz(
      ContextManager& mgr,
      int64_t const max_n_pose_atoms,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> kfo_2_orig_mapping,  // K x 3
      TView<Int, 1, D> parents,             // K
      TView<Int, 1, D> child_list_span,     // K+1
      TView<Int, 1, D> child_list,          // K
      TView<bool, 1, D> is_atom_jump        // K
      )
      -> std::tuple<
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<bool, 2, D>>;

  static auto calculate_ff_edge_delays(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_coord_offset,  // P x L
      TView<Int, 2, D> pose_stack_block_type,          // x - P x L
      TView<Int, 3, Device::CPU> ff_edges_cpu,  // y - P x E x 4 -- 0: type, 1:
                                                // start, 2: stop, 3: jump ind
      TView<Int, 5, D> block_type_kts_conn_info,  // y - T x I x O x C x 2 -- 2
                                                  // is for gen (0) and scan (1)
      TView<Int, 5, D> block_type_nodes_for_gens,  // y - T x I x O x G x N
      TView<Int, 5, D>
          block_type_scan_path_seg_starts,  // y - T x I x O x G x S
      // CPU copies: the scheduling below runs on the host and must read the
      // generation each child departs its parent block in
      TView<Int, 2, Device::CPU> pose_stack_block_type_cpu,  // P x L
      TView<Int, 4, Device::CPU>
          pose_stack_inter_res_conn_cpu,  // P x L x C x 2
      TView<Int, 5, Device::CPU>
          block_type_kts_conn_info_cpu,  // T x I x O x C x 2
      TView<Int, 2, Device::CPU> block_type_polymeric_conn_index_cpu,  // T x 2
      TView<Int, 1, Device::CPU> block_type_n_conn_cpu                 // T
      )
      -> std::tuple<
          TPack<Int, 2, Device::CPU>,  // dfs_order_of_ff_edges_t
          TPack<Int, 1, Device::CPU>,  // n_ff_edges_t
          TPack<Int, 2, Device::CPU>,  // ff_edge_parent_t
          TPack<Int, 2, Device::CPU>,  // first_ff_edge_for_block_cpu_t
          TPack<Int, 2, Device::CPU>,  // pose_stack_ff_parent_t
          TPack<Int, 2, Device::CPU>,  // max_gen_depth_of_ff_edge_t
          TPack<Int, 2, Device::CPU>,  // first_child_of_ff_edge_t
          TPack<Int, 2, Device::CPU>,  // delay_for_edge_t
          TPack<Int, 1, Device::CPU>   // toposort_order_of_edges_t
          >;

  static auto get_jump_atom_indices(
      ContextManager& mgr,
      TView<Int, 3, D>
          ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
      TView<Int, 2, D> pose_stack_block_type,  // P x L
      TView<Int, 1, D> block_type_jump_atom    // T
      ) -> std::tuple<TPack<Int, 3, D>, TPack<Int, 2, D>>;

  static auto get_block_parent_connectivity_from_toposort(
      ContextManager& mgr,
      TView<Int, 2, D> pose_stack_block_type,                 // P x L
      TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
      TView<Int, 2, D> pose_stack_ff_parent,
      TView<Int, 2, D> dfs_order_of_ff_edges,
      TView<Int, 1, D> n_ff_edges,               // P
      TView<Int, 3, D> ff_edges,                 // P x E x 4
      TView<Int, 2, D> first_ff_edge_for_block,  // P x L
      // TView<Int, 2, D> max_n_gens_for_ff_edge, // P x E
      TView<Int, 2, D> first_child_of_ff_edge,   // P x E
      TView<Int, 2, D> delay_for_edge,           // P x E
      TView<Int, 1, D> toposort_order_of_edges,  // (P*E)
      TView<Int, 1, D> block_type_n_conn,        // T
      TView<Int, 2, D>
          block_type_polymeric_conn_index  // T x 2 - 2 is for "down" and "up"
                                           // connections.
      ) -> TPack<Int, 3, D>;

  static auto get_scans2(
      ContextManager& mgr,
      int64_t const max_n_atoms_per_pose,
      TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
      TView<Int, 2, D> pose_stack_block_type,                 // P x L
      TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
      TView<Int, 3, D>
          ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
      int64_t const max_delay,
      TView<Int, 2, D> delay_for_edge,            // P x E
      TView<Int, 1, D> topo_sort_index_for_edge,  // (P*E)
      TView<Int, 2, D> first_ff_edge_for_block,   // P x L
      TView<Int, 2, D> pose_stack_ff_parent,      // P x L
      // TView<Int, 2, D> pose_stack_ff_conn_to_parent,       // P x L
      TView<Int, 3, D> pose_stack_block_in_and_first_out,  // P x L x 2
      TView<Int, 3, D> block_type_parents,                 // T x O x A
      TView<Int, 2, D> kfo_2_orig_mapping,                 // K x 3
      TView<Int, 3, D> atom_kfo_index,                     // P x L x A
      TView<Int, 1, D> block_type_jump_atom,               // T
      TView<Int, 1, D> block_type_n_conn,                  // T
      TView<Int, 2, D>
          block_type_polymeric_conn_index,  // T x 2 - 2 is for "down" and "up"
                                            // connections.
      TView<Int, 3, D> block_type_n_gens,   // T x I x O
      TView<Int, 5, D> block_type_kts_conn_info,     // T x I x O x C x 2 - 2 is
                                                     // for gen (0) and scan (1)
      TView<Int, 5, D> block_type_nodes_for_gens,    // T x I x O x G x N
      TView<Int, 4, D> block_type_n_scan_path_segs,  // T x I x O x G
      TView<Int, 5, D> block_type_scan_path_seg_starts,    // T x I x O x G x S
      TView<bool, 5, D> block_type_scan_path_seg_is_real,  // T x I x O x G x S
      TView<bool, 5, D>
          block_type_scan_path_seg_is_inter_block,      // T x I x O x G x S
      TView<Int, 5, D> block_type_scan_path_seg_length  // T x I x O x G x S
      )
      -> std::tuple<
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 2, D>,
          TPack<Int, 1, D>,
          TPack<Int, 1, D>,
          TPack<Int, 2, D>>;

  static auto create_minimizer_map(
      ContextManager& mgr,
      TView<Int, 1, D> kinforest_id,  // K
      int64_t const max_n_atoms_per_pose,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 3, D> pose_stack_block_in_and_first_out,  // P x L x 2
      TView<Int, 3, D> pose_stack_atom_for_jump,           // P x E x 2
      TView<Int, 2, D> pose_stack_atom_for_root_jump,      // P x L
      TView<bool, 2, D> keep_dof_fixed,                    // K x 9
      TView<Int, 1, D> bt_n_named_torsions,
      TView<UnresolvedAtomID<Int>, 3, D> bt_uaid_for_torsion,
      TView<bool, 2, D> bt_named_torsion_is_mc,
      TView<Int, 2, D> bt_which_mcsc_torsion_for_named_torsion,
      TView<Int, 3, D> bt_atom_downstream_of_conn,
      bool move_all_jumps,
      bool move_all_root_jumps,
      bool move_all_mcs,
      bool move_all_scs,
      bool move_all_named_torsions,
      bool non_ideal,
      TView<bool, 2, D> move_jumps,
      TView<bool, 2, D> move_jumps_mask,
      TView<bool, 2, D> move_root_jumps,
      TView<bool, 2, D> move_root_jumps_mask,
      TView<bool, 2, D> move_mcs,
      TView<bool, 2, D> move_mcs_mask,
      TView<bool, 2, D> move_scs,
      TView<bool, 2, D> move_scs_mask,
      TView<bool, 2, D> move_named_torsions,
      TView<bool, 2, D> move_named_torsions_mask,
      TView<bool, 3, D> move_jump_dof,
      TView<bool, 3, D> move_jump_dof_mask,
      TView<bool, 3, D> move_root_jump_dof,
      TView<bool, 3, D> move_root_jump_dof_mask,
      TView<bool, 3, D> move_mc,
      TView<bool, 3, D> move_mc_mask,
      TView<bool, 3, D> move_sc,
      TView<bool, 3, D> move_sc_mask,
      TView<bool, 3, D> move_named_torsion,
      TView<bool, 3, D> move_named_torsion_mask,
      TView<bool, 4, D> move_atom_dof,
      TView<bool, 4, D> move_atom_dof_mask) -> TPack<bool, 2, D>;
};

#undef Dofs
#undef HomogeneousTransform
#undef QuatTranslation
#undef Coord

}  // namespace kinematics
}  // namespace tmol
