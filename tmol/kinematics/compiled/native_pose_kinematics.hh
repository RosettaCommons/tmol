#pragma once
// Mature pose topology assembly. Allocation and dispatch belong to the caller;
// chemistry and traversal algorithms are shared by every framework adapter.
#include "kinematics_math.hh"
#include "params_types.hh"
#include <cstddef>
#include <tuple>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/unresolved_atom_types.hh>

namespace tmol {
namespace kinematics {
// Pack<T,N,D> owns contiguous storage, exposes .view, and supplies zeros,
// ones and full factories. Returned packs must retain their allocation.
// Context is forwarded unchanged to the caller's typed DeviceDispatch.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Int,
    template <typename, size_t, tmol::Device> class Pack,
    typename Context>
struct NativeKinForestFromStencil {
  // Assemble a validated FoldForest and prepared palette in one native call.
  // CPU mirrors describe the same immutable topology as the device views.
  // Pack::from_cpu supplies the explicit preparation-time transfer policy.
  static auto assemble_pose(
      Context& mgr,
      int64_t max_n_pose_atoms,
      TView<Int, 2, D> block_coord_offset,
      TView<Int, 2, D> block_type,
      TView<Int, 4, D> inter_residue_connections,
      TView<Int, 3, D> ff_edges,
      TView<Int, 1, D> block_type_n_atoms,
      TView<bool, 2, D> block_type_atom_is_real,
      TView<Int, 1, D> block_type_n_conn,
      TView<Int, 2, D> block_type_conn_atom,
      TView<Int, 2, D> block_type_polymeric_conn,
      TView<Int, 1, D> jump_atom,
      TView<Int, 3, D> parents,
      TView<Int, 3, D> dof_type,
      TView<Int, 3, D> n_gens,
      TView<Int, 5, D> conn_info,
      TView<Int, 5, D> nodes_for_gen,
      TView<Int, 4, D> n_scan_path_segs,
      TView<Int, 5, D> scan_path_seg_starts,
      TView<bool, 5, D> scan_path_seg_is_real,
      TView<bool, 5, D> scan_path_seg_is_inter_block,
      TView<Int, 5, D> scan_path_seg_lengths,
      TView<Int, 3, Device::CPU> ff_edges_cpu,
      TView<Int, 2, Device::CPU> block_type_cpu,
      TView<Int, 4, Device::CPU> connections_cpu,
      TView<Int, 5, Device::CPU> conn_info_cpu,
      TView<Int, 2, Device::CPU> polymeric_conn_cpu,
      TView<Int, 1, Device::CPU> n_conn_cpu)
      -> std::tuple<
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 2, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 2, D>,
          Pack<Int, 3, D>,
          Pack<bool, 2, D>,
          Pack<Int, 3, D>,
          Pack<Int, 2, D>>;

  static auto get_kfo_indices_for_atoms(
      Context& mgr,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Int, 1, D> block_type_n_atoms,
      TView<bool, 2, D> block_type_atom_is_real)
      -> std::tuple<Pack<Int, 2, D>, Pack<Int, 2, D>, Pack<Int, 3, D>>;

  static auto get_kfo_atom_parents(
      Context& mgr,
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
      -> std::tuple<Pack<Int, 1, D>, Pack<Int, 1, D>>;

  static auto get_children(
      Context& mgr,
      TView<Int, 2, D> pose_stack_block_type,              // x
      TView<Int, 3, D> pose_stack_block_in_and_first_out,  // x
      TView<Int, 2, D> kfo_2_orig_mapping,                 // x
      TView<Int, 1, D> kfo_parent_atoms,                   // x
      TView<Int, 1, D> block_type_n_conn                   // x
      )
      -> std::tuple<
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<bool, 1, D>>;

  static auto get_id_and_frame_xyz(
      Context& mgr,
      int64_t const max_n_pose_atoms,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> kfo_2_orig_mapping,  // K x 3
      TView<Int, 1, D> parents,             // K
      TView<Int, 1, D> child_list_span,     // K+1
      TView<Int, 1, D> child_list,          // K
      TView<bool, 1, D> is_atom_jump        // K
      )
      -> std::tuple<
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<bool, 2, D>>;

  static auto calculate_ff_edge_delays(
      Context& mgr,
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
          Pack<Int, 2, Device::CPU>,  // dfs_order_of_ff_edges_t
          Pack<Int, 1, Device::CPU>,  // n_ff_edges_t
          Pack<Int, 2, Device::CPU>,  // ff_edge_parent_t
          Pack<Int, 2, Device::CPU>,  // first_ff_edge_for_block_cpu_t
          Pack<Int, 2, Device::CPU>,  // pose_stack_ff_parent_t
          Pack<Int, 2, Device::CPU>,  // max_gen_depth_of_ff_edge_t
          Pack<Int, 2, Device::CPU>,  // first_child_of_ff_edge_t
          Pack<Int, 2, Device::CPU>,  // delay_for_edge_t
          Pack<Int, 1, Device::CPU>   // toposort_order_of_edges_t
          >;

  static auto get_jump_atom_indices(
      Context& mgr,
      TView<Int, 3, D>
          ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
      TView<Int, 2, D> pose_stack_block_type,  // P x L
      TView<Int, 1, D> block_type_jump_atom    // T
      ) -> std::tuple<Pack<Int, 3, D>, Pack<Int, 2, D>>;

  static auto get_block_parent_connectivity_from_toposort(
      Context& mgr,
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
      ) -> Pack<Int, 3, D>;

  static auto get_scans2(
      Context& mgr,
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
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 2, D>,
          Pack<Int, 1, D>,
          Pack<Int, 1, D>,
          Pack<Int, 2, D>>;

  static auto create_minimizer_map(
      Context& mgr,
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
      TView<bool, 4, D> move_atom_dof_mask) -> Pack<bool, 2, D>;
};

}  // namespace kinematics
}  // namespace tmol
